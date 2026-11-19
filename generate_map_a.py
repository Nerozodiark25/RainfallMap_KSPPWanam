""" 
============================================
AI Flood Early Warning System with Uncertainty
Author      : James Albert Kaunang
Version     : 2.36
Last updated: 2026-11-19
============================================
Description:
csv data format: date (dd/mm/yyyy), rainfall (mm), discharge (m³/s) (case sensitive)
process: load csv -> train & tune models -> input 7-day rainfall -> predict flood levels with uncertainty
feature: LSTM with MC Dropout + Random Forest ensemble, Optuna hyperparameter tuning,
         live training dashboard, ensemble weight calibration, Model Fit diagnostics
===========================================
Changelog v2.3:
- Added rolling-window Adaptive Conformal Inference for MC Dropout intervals
- Consistent units: all discharge labels now use m³/s (no bare "m")
- Added "Show Model Fit" button for visual diagnostics (time series, scatter, residuals)
- Improved prediction progress logging
Previous:
- L2 regularization hyperparameter tuning
- Best-epoch validation loss tracking + restore best weights
- Save/load model functionality
- Ensemble weight calibration
===========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras import regularizers
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.constants import END
from datetime import datetime, timedelta
import threading
import random
import time
import queue
import hashlib
from collections import deque
import plotly.io as pio
import tempfile
import webbrowser
import joblib
import shutil
import zipfile
from optuna.visualization import plot_contour
from optuna.visualization import plot_parallel_coordinate

class LivePlotCallback(tf.keras.callbacks.Callback):
    """Keras callback to stream training/validation loss per epoch to the GUI via a queue."""
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tr = logs.get('loss')
        val = logs.get('val_loss')
        self.epochs.append(epoch + 1)
        self.train_losses.append(tr if tr is not None else float('nan'))
        self.val_losses.append(val if val is not None else float('nan'))
        try:
            self.app.plot_queue.put(('epoch', list(self.epochs), list(self.train_losses), list(self.val_losses)))
        except Exception as e:
            print("LivePlotCallback enqueue failed:", e)


class StopTrainingCallback(tf.keras.callbacks.Callback):
    """Callback to stop Keras training when the app requests cancellation.
    Posts a log message to the app queue when stopping."""
    def __init__(self, app):
        super().__init__()
        self.app = app

    def on_epoch_end(self, epoch, logs=None):
        try:
            if getattr(self.app, 'stop_training', False):
                try:
                    self.model.stop_training = True
                except Exception:
                    pass
                try:
                    self.app.plot_queue.put(('log', f"Training cancelled by user (epoch {epoch+1})"))
                except Exception:
                    pass
        except Exception:
            pass


class RollingACI:
    """Rolling-window adaptive conformal correction for a base prediction interval."""

    def __init__(self, window=180, gamma=0.01, target_alpha=0.05,
                 alpha=0.05, scores=None):
        self.window = int(window)
        self.gamma = float(gamma)
        self.target_alpha = float(target_alpha)
        self.alpha = float(alpha)
        self.alpha_min = 0.01
        self.alpha_max = 0.25
        if self.window < 1:
            raise ValueError("ACI window must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("ACI gamma must be in (0, 1]")
        if not 0.0 < self.target_alpha < 1.0:
            raise ValueError("ACI target alpha must be in (0, 1)")
        if not self.alpha_min <= self.alpha <= self.alpha_max:
            raise ValueError("ACI alpha is outside the supported range")
        self.scores = deque((float(s) for s in (scores or [])), maxlen=self.window)

    def quantile(self):
        if len(self.scores) < self.window:
            return 0.0
        ordered = np.sort(np.asarray(self.scores, dtype=np.float64))
        rank = int(np.ceil((len(ordered) + 1) * (1.0 - self.alpha)))
        rank = min(len(ordered), max(1, rank))
        return float(ordered[rank - 1])

    def interval(self, lower, upper):
        lower = float(lower)
        upper = float(upper)
        width = max(upper - lower, 1e-6)
        q = self.quantile()
        return max(0.0, lower - q * width), upper + q * width, q

    def observe(self, raw_lower, raw_upper, observed, calibrated_lower, calibrated_upper):
        width = max(float(raw_upper) - float(raw_lower), 1e-6)
        score = max(0.0, float(raw_lower) - float(observed),
                    float(observed) - float(raw_upper)) / width
        miss = float(observed) < float(calibrated_lower) or float(observed) > float(calibrated_upper)
        self.scores.append(float(score))
        self.alpha = float(np.clip(
            self.alpha + self.gamma * (self.target_alpha - float(miss)),
            self.alpha_min, self.alpha_max))
        return score, bool(miss)

    def to_state(self):
        return {
            'window': self.window,
            'gamma': self.gamma,
            'target_alpha': self.target_alpha,
            'alpha': self.alpha,
            'scores': list(self.scores),
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            window=state['window'],
            gamma=state['gamma'],
            target_alpha=state.get('target_alpha', 0.05),
            alpha=state['alpha'],
            scores=state.get('scores', []))


class FloodPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Daily Rainfall-Runoff Prediction System (LSTM-RF Ensemble with Uncertainty)")
        self.root.geometry("1650x1080")
        self.root.minsize(1450, 900)

        self.df = None
        self.scaler_rain = MinMaxScaler()
        self.scaler_level = MinMaxScaler()
        self.lstm_model = None
        self.rf_model = None
        self.study = None
        self.sequence_cache = {}
        self.tune_seq_var = ttk.BooleanVar(value=True)
        self.continue_optuna_var = ttk.BooleanVar(value=True)
        self.optuna_study_name = "daily_rainfall_runoff"
        # Search mode: focused (recommended) | full | manual
        self.search_mode_var = ttk.StringVar(value="focused")
        # Manual LSTM hyperparameters (skip Optuna when enabled)
        self.use_manual_params_var = ttk.BooleanVar(value=False)
        self.manual_units1_var = ttk.IntVar(value=128)
        self.manual_units2_var = ttk.IntVar(value=64)
        self.manual_dropout_var = ttk.DoubleVar(value=0.20)
        self.manual_lr_var = ttk.DoubleVar(value=1e-3)
        self.manual_batch_var = ttk.IntVar(value=32)
        self.manual_hdelta_var = ttk.DoubleVar(value=0.05)
        self.manual_l2_var = ttk.DoubleVar(value=0.0)
        self.manual_seq_var = ttk.IntVar(value=30)
        self.seq_min_var = ttk.IntVar(value=1)
        self.seq_max_var = ttk.IntVar(value=14)
        self.elevated_var = ttk.DoubleVar(value=20.0)
        self.moderate_var = ttk.DoubleVar(value=50.0)
        self.high_var = ttk.DoubleVar(value=100.0)
        self.mc_samples_var = ttk.IntVar(value=1000)
        self.use_aci_var = ttk.BooleanVar(value=True)
        self.aci_window_var = ttk.IntVar(value=180)
        self.aci_gamma_var = ttk.DoubleVar(value=0.01)
        self.aci_calibrators = {}
        self.aci_mode = None
        self.aci_calibrated_through = None
        self._val_unc_cache = None
        self._val_unc_export = None
        self._val_unc_stats = None
        self.ensemble_lstm_var = ttk.DoubleVar(value=0.85)
        self.calibrate_ensemble_var = ttk.BooleanVar(value=True)
        self.use_peak_weight_var = ttk.BooleanVar(value=True)
        self.peak_weight_max_var = ttk.DoubleVar(value=5.0)
        self.stop_training = False
        self.training_thread = None
        self.tuning_start_time = None
        self.last_predictions = None
        self.last_rf_inputs = None  # flattened RF inputs per forecast step (for SHAP)

        self.train_X = self.train_y = self.val_X = self.val_y = None

        # Feature toggles (extra engineered features beyond rainfall & discharge)
        self.use_cum3_var = ttk.BooleanVar(value=False)
        self.use_cum7_var = ttk.BooleanVar(value=False)
        self.use_cum14_var = ttk.BooleanVar(value=False)
        self.use_rate_var = ttk.BooleanVar(value=False)
        self.use_lag1_var = ttk.BooleanVar(value=False)
        self.use_wetseason_var = ttk.BooleanVar(value=False)
        # Antecedent Precipitation Index (API) toggle and decay factor
        self.use_api_var = ttk.BooleanVar(value=False)
        self.api_decay_var = ttk.DoubleVar(value=0.82)
        # Seq2Seq toggle
        self.use_seq2seq_var = ttk.BooleanVar(value=False)   # Default = recursive
        # Concurrent: include target-day rainfall, mask its discharge (matches rain-forecast use)
        self.same_month_var = ttk.BooleanVar(value=True)

        self.display_name_map = {
            'rainfall': 'Rainfall (mm)',
            'Discharge': 'Discharge (m³/s)',
            'rain_cum3': '3-day Cum Rain (mm)',
            'rain_cum7': '7-day Cum Rain (mm)',
            'rain_cum14': '14-day Cum Rain (mm)',
            'disc_diff': 'Rate of Rise (m³/s per day)',
            'disc_lag1': 'Lag-1 Discharge (m³/s)',
            'wet_season': 'Wet Season Flag',
            'api': 'Antecedent Precipitation Index (API)'
        }
        try:
            self.use_cum3_var.trace_add('write', lambda *a: self._on_feature_toggle_change())
            self.use_cum7_var.trace_add('write', lambda *a: self._on_feature_toggle_change())
            self.use_cum14_var.trace_add('write', lambda *a: self._on_feature_toggle_change())
            self.use_rate_var.trace_add('write', lambda *a: self._on_feature_toggle_change())
            self.use_lag1_var.trace_add('write', lambda *a: self._on_feature_toggle_change())
            self.use_wetseason_var.trace_add('write', lambda *a: self._on_feature_toggle_change())
            self.use_api_var.trace_add('write', lambda *a: self._on_feature_toggle_change())  
            self.api_decay_var.trace_add('write', lambda *a: self._on_feature_toggle_change())
            self.aci_window_var.trace_add('write', lambda *a: self._reset_aci_calibration())
            self.aci_gamma_var.trace_add('write', lambda *a: self._reset_aci_calibration())
            self.mc_samples_var.trace_add('write', lambda *a: self._reset_aci_calibration())
            self.elevated_var.trace_add('write', lambda *a: self._reset_aci_calibration())
            self.moderate_var.trace_add('write', lambda *a: self._reset_aci_calibration())
            self.high_var.trace_add('write', lambda *a: self._reset_aci_calibration())
            self.use_aci_var.trace_add('write', lambda *a: self._reset_aci_calibration())
            self.use_seq2seq_var.trace_add('write', lambda *a: self._reset_aci_calibration())
            self.same_month_var.trace_add('write', lambda *a: self._reset_aci_calibration())
        except Exception:
            pass

        self.plot_queue = queue.Queue()

        self._closing = False
        self._after_id = None

        self.setup_ui()
        self._after_id = self.root.after(100, self.process_plot_queue)
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception:
            pass
        try:
            def _safe_report_callback_exception(exc, val, tb):
                try:
                    import traceback
                    traceback.print_exception(exc, val, tb)
                except Exception:
                    pass
            try:
                self.root.report_callback_exception = _safe_report_callback_exception
            except Exception:
                pass
        except Exception:
            pass

    def setup_ui(self):
        main = ttk.Panedwindow(self.root, orient=HORIZONTAL)
        main.pack(fill=BOTH, expand=True, padx=15, pady=15)

        # LEFT PANEL
        left = ttk.Frame(main)
        main.add(left, weight=1)
        try:
            main.pane(left, minsize=480)   # minimum size for left pane
        except Exception:
            pass
        title = ttk.Label(left, text="LSTM-RF Ensemble Rainfall-Runoff Model", font=("Helvetica", 21, "bold"), bootstyle="primary")
        title.pack(pady=20)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="Load CSV Data", bootstyle=SUCCESS, width=25, command=self.load_file).pack(side=LEFT, padx=8)
        self.train_btn = ttk.Button(btn_frame, text="Train & Tune Models", bootstyle="danger-outline", width=30, command=self.start_training)
        self.train_btn.pack(side=LEFT, padx=8)
        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", bootstyle=WARNING, command=self.cancel_training, state=DISABLED)
        self.cancel_btn.pack(side=LEFT, padx=8)
        # Model persistence / inspection buttons (separate row under main controls)
        model_btn_frame1 = ttk.Frame(left)
        model_btn_frame1.pack(pady=(6, 0))
        self.save_model_btn = ttk.Button(model_btn_frame1, text="Save Models", bootstyle="primary-outline", width=15, command=self.save_models)
        self.save_model_btn.pack(side=LEFT, padx=5)
        self.load_model_btn = ttk.Button(model_btn_frame1, text="Load Models", bootstyle="light-outline", width=15, command=self.load_models)
        self.load_model_btn.pack(side=LEFT, padx=5)
        self.inspect_model_btn = ttk.Button(model_btn_frame1, text="Inspect Models", bootstyle="info-outline", width=15, command=self.inspect_models)
        self.inspect_model_btn.pack(side=LEFT, padx=5)
        model_btn_frame2 = ttk.Frame(left)
        model_btn_frame2.pack(pady=(6, 0))
        self.fit_btn = ttk.Button(model_btn_frame2, text="Show Model Fit", bootstyle="success-outline", width=15, command=self.show_model_fit, state=DISABLED)
        self.fit_btn.pack(side=LEFT, padx=5)
        self.val_unc_btn = ttk.Button(
            model_btn_frame2, text="Val Uncertainty", bootstyle="warning-outline", width=15,
            command=self.show_val_uncertainty, state=DISABLED)
        self.val_unc_btn.pack(side=LEFT, padx=5)
        self.status = ttk.Label(left, text="No data loaded", bootstyle="secondary", font=("Helvetica", 11))
        self.status.pack(pady=10)

        # SETTINGS
        settings_frame = ttk.Labelframe(left, text="Model & Hyperparameter Settings", bootstyle="info", padding=15)
        settings_frame.pack(fill=X, pady=15, padx=20)

        self.show_settings = ttk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Show Advanced Settings", variable=self.show_settings,
                        bootstyle="round-toggle", command=self.toggle_settings).pack(anchor=W, padx=5, pady=5)

        self.settings_canvas = ttk.Canvas(settings_frame, height=360)
        self.settings_scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=self.settings_canvas.yview, bootstyle="round")
        self.settings_inner = ttk.Frame(self.settings_canvas)
        self.settings_inner.bind("<Configure>", lambda e: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.create_window((0, 0), window=self.settings_inner, anchor="nw", tags="settings_inner_window")
        self.settings_canvas.configure(yscrollcommand=self.settings_scrollbar.set)
        def _settings_adapt_width(event):
            try:
                self.settings_canvas.itemconfigure("settings_inner_window", width=event.width)
                self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all"))
            except Exception:
                pass
        self.settings_canvas.bind("<Configure>", _settings_adapt_width)

        def _settings_wheel(event):
            try:
                delta = getattr(event, "delta", 0)
                if delta:
                    self.settings_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
                elif getattr(event, "num", None) == 4:
                    self.settings_canvas.yview_scroll(-1, "units")
                elif getattr(event, "num", None) == 5:
                    self.settings_canvas.yview_scroll(1, "units")
            except Exception:
                pass

        def _bind_settings_wheel(w):
            try:
                w.bind("<MouseWheel>", _settings_wheel)
                w.bind("<Button-4>", _settings_wheel)
                w.bind("<Button-5>", _settings_wheel)
            except Exception:
                pass

        _bind_settings_wheel(self.settings_canvas)
        _bind_settings_wheel(self.settings_inner)

        row1 = ttk.Frame(self.settings_inner)
        row1.pack(fill=X, pady=8)
        # Sequence length (may be tuned by Optuna if enabled)
        ttk.Label(row1, text="Optuna Trials:", font=("Helvetica", 10, "bold")).pack(side=LEFT)
        ttk.Label(row1, text="   Sequence Length:", font=("Helvetica", 10, "bold")).pack(side=LEFT, padx=(30,0))
        self.seq_var = ttk.IntVar(value=7)
        self.trials_var = ttk.IntVar(value=20)
        self.seq_spin = ttk.Spinbox(row1, from_=7, to=365, textvariable=self.seq_var, width=10)
        self.seq_spin.pack(side=RIGHT, padx=10)
        self.trials_spin = ttk.Spinbox(row1, from_=10, to=100, increment=5, textvariable=self.trials_var, width=10)
        self.trials_spin.pack(side=RIGHT, padx=10)

        # Search mode — encodes tuning best practice without expert workflow
        search_row = ttk.Frame(self.settings_inner)
        search_row.pack(fill=X, padx=(4,10), pady=(4,2))
        ttk.Label(search_row, text="Search mode:", font=("Helvetica", 10, "bold")).pack(side=LEFT)
        rb_focused = ttk.Radiobutton(
            search_row, text="Recommended", variable=self.search_mode_var, value="focused",
            bootstyle="toolbutton", command=self._on_search_mode_change)
        rb_focused.pack(side=LEFT, padx=(8,4))
        rb_full = ttk.Radiobutton(
            search_row, text="Full search", variable=self.search_mode_var, value="full",
            bootstyle="toolbutton", command=self._on_search_mode_change)
        rb_full.pack(side=LEFT, padx=4)
        rb_manual = ttk.Radiobutton(
            search_row, text="Manual only", variable=self.search_mode_var, value="manual",
            bootstyle="toolbutton", command=self._on_search_mode_change)
        rb_manual.pack(side=LEFT, padx=4)
        self.search_mode_hint = ttk.Label(
            self.settings_inner,
            text="Recommended: tune lr, dropout, units, and sequence length; batch/Hδ/L2 fixed.",
            font=("Helvetica", 9), foreground="#888888")
        self.search_mode_hint.pack(anchor=W, padx=8, pady=(0,6))
        self._attach_tooltip(
            rb_focused,
            "BEST PRACTICE (default).\n"
            "Tunes: learning rate, dropout, LSTM units, and sequence length within the selected range.\n"
            "Keeps batch size, Huber δ, and L2 fixed.\n"
            "Fewer knobs → less random-looking results with the same trial count."
        )
        self._attach_tooltip(
            rb_full,
            "Tunes all hyperparameters in the search space\n"
            "(including batch, Hδ, L2, and sequence if enabled).\n"
            "Use when exploring widely; needs more trials."
        )
        self._attach_tooltip(
            rb_manual,
            "Skip Optuna entirely.\n"
            "Uses Edit params… values and trains the final model only."
        )

        optuna_mem_row = ttk.Frame(self.settings_inner)
        optuna_mem_row.pack(fill=X, padx=(4,10), pady=(0,6))
        self.optuna_remember_cb = ttk.Checkbutton(
            optuna_mem_row, text="Remember Optuna trials (continue across runs)",
            variable=self.continue_optuna_var, bootstyle="round-toggle")
        self.optuna_remember_cb.pack(side=LEFT, padx=(0,8))
        self.optuna_reset_btn = ttk.Button(
            optuna_mem_row, text="Reset Optuna history", bootstyle="warning-outline",
            command=self.reset_optuna_history)
        self.optuna_reset_btn.pack(side=LEFT, padx=6)
        ttk.Label(optuna_mem_row, text="(saves to SQLite next to the script)", font=("Helvetica", 9),
                  foreground="#888888").pack(side=LEFT, padx=(4,0))
        self._attach_tooltip(
            self.optuna_remember_cb,
            "ON: Keep past trials and continue searching (same data & similar search space).\n"
            "Good when: same station/CSV, same features, same ranges — just adding more trials.\n"
            "OFF: This run does not load or save history.\n"
            "Auto-reset: loading a new CSV clears history so old trials do not mix with new data.\n"
            "Manual reset: use 'Reset Optuna history' after big changes (features, concurrent mode,\n"
            "peak weight, units/lr ranges)."
        )
        self._attach_tooltip(
            self.optuna_reset_btn,
            "Wipe saved Optuna trials now.\n"
            "Use when changing search ranges, features, or modeling options\n"
            "so the sampler does not reuse results from a different setup."
        )

        # Sequence tuning options (min/max + toggle)
        seq_opts = ttk.Frame(self.settings_inner)
        seq_opts.pack(fill=X, padx=(4,10), pady=(0,8))
        self.tune_seq_cb = ttk.Checkbutton(
            seq_opts, text="Tune Sequence Length", variable=self.tune_seq_var, bootstyle="round-toggle")
        self.tune_seq_cb.pack(side=LEFT, padx=(0,8))
        ttk.Label(seq_opts, text="Min:").pack(side=LEFT)
        self.seq_min_spin = ttk.Spinbox(seq_opts, from_=1, to=365, textvariable=self.seq_min_var, width=8)
        self.seq_min_spin.pack(side=LEFT, padx=4)
        ttk.Label(seq_opts, text="Max:").pack(side=LEFT, padx=(8,0))
        self.seq_max_spin = ttk.Spinbox(seq_opts, from_=2, to=720, textvariable=self.seq_max_var, width=8)
        self.seq_max_spin.pack(side=LEFT, padx=4)
        # Seq2Seq toggle
        seq2seq_row = ttk.Frame(self.settings_inner)
        seq2seq_row.pack(fill=X, padx=(4,10), pady=(0,8))
        ttk.Checkbutton(seq2seq_row, text="Use Seq2Seq Decoder (True 7-day forecast)",
                   variable=self.use_seq2seq_var, bootstyle="round-toggle").pack(side=LEFT, padx=(0,6))
        ttk.Label(seq2seq_row, text="(One-shot prediction, less error accumulation)",
             font=("Helvetica", 9), foreground="#888888").pack(side=LEFT)

        same_row = ttk.Frame(self.settings_inner)
        same_row.pack(fill=X, padx=(4,10), pady=(0,8))
        ttk.Checkbutton(same_row, text="Concurrent prediction (use target-day rainfall)",
                        variable=self.same_month_var, bootstyle="round-toggle").pack(side=LEFT, padx=(0,6))
        ttk.Label(same_row, text="(Recommended when you have rainfall forecast for the days you predict)",
                  font=("Helvetica", 9), foreground="#888888").pack(side=LEFT)

        ttk.Separator(self.settings_inner, orient=HORIZONTAL).pack(fill=X, pady=10)
        ttk.Label(self.settings_inner, text="LSTM Hyperparameter Search Space", font=("Helvetica", 10, "bold")).pack(anchor=W)

        grid = ttk.Frame(self.settings_inner)
        grid.pack(fill=X, pady=5)

        ttk.Label(grid, text="Units (min-max):").grid(row=0, column=0, sticky=W, padx=5, pady=3)
        self.units_min = ttk.IntVar(value=100)
        self.units_max = ttk.IntVar(value=256)
        ttk.Spinbox(grid, from_=32, to=512, textvariable=self.units_min, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(grid, text="→").grid(row=0, column=2)
        ttk.Spinbox(grid, from_=64, to=1024, textvariable=self.units_max, width=8).grid(row=0, column=3, padx=5)

        ttk.Label(grid, text="Dropout (min-max):").grid(row=1, column=0, sticky=W, padx=5, pady=3)
        self.drop_min = ttk.DoubleVar(value=0.05)
        self.drop_max = ttk.DoubleVar(value=0.35)
        ttk.Spinbox(grid, from_=0.0, to=0.5, increment=0.05, textvariable=self.drop_min, width=8).grid(row=1, column=1, padx=5)
        ttk.Label(grid, text="→").grid(row=1, column=2)
        ttk.Spinbox(grid, from_=0.1, to=0.7, increment=0.05, textvariable=self.drop_max, width=8).grid(row=1, column=3, padx=5)

        ttk.Label(grid, text="Learning Rate (1e-):").grid(row=2, column=0, sticky=W, padx=5, pady=3)
        self.lr_min_exp = ttk.IntVar(value=5)
        self.lr_max_exp = ttk.IntVar(value=2)
        ttk.Spinbox(grid, from_=6, to=1, textvariable=self.lr_min_exp, width=6).grid(row=2, column=1, padx=5)
        ttk.Label(grid, text="→ 1e-").grid(row=2, column=2)
        ttk.Spinbox(grid, from_=5, to=1, textvariable=self.lr_max_exp, width=6).grid(row=2, column=3, padx=5)

        ttk.Label(grid, text="Batch Sizes:").grid(row=3, column=0, sticky=W, padx=5, pady=3)
        self.batch_options = ttk.Entry(grid, width=30)
        self.batch_options.insert(0, "8, 16, 32, 64")
        self.batch_options.grid(row=3, column=1, columnspan=3, padx=5, sticky=W)

        # L2 regularization (min-max) - user-adjustable
        ttk.Label(grid, text="L2 (min-max):").grid(row=4, column=0, sticky=W, padx=5, pady=3)
        self.l2_min = ttk.DoubleVar(value=0.0001)
        self.l2_max = ttk.DoubleVar(value=0.001)
        ttk.Spinbox(grid, from_=0.0, to=0.01, increment=0.001, textvariable=self.l2_min, width=8).grid(row=4, column=1, padx=5)
        ttk.Label(grid, text="→").grid(row=4, column=2)
        ttk.Spinbox(grid, from_=0.0, to=0.01, increment=0.001, textvariable=self.l2_max, width=8).grid(row=4, column=3, padx=5)
        # Log-scale checkbox for L2 sampling if used it is likely to be far from minimum and maximum value specified
        self.l2_log_var = ttk.BooleanVar(value=False)
        ttk.Checkbutton(grid, text="Log-scale L2", variable=self.l2_log_var, bootstyle="round-toggle").grid(row=4, column=4, padx=10)
        # L2 toggle (allow disabling L2 entirely)
        self.use_l2_var = ttk.BooleanVar(value=False)
        ttk.Checkbutton(grid, text="Use L2", variable=self.use_l2_var, bootstyle="round-toggle").grid(row=4, column=5, padx=10)      

        ttk.Separator(self.settings_inner, orient=HORIZONTAL).pack(fill=X, pady=10)
        ttk.Label(self.settings_inner, text="Random Forest Settings", font=("Helvetica", 10, "bold")).pack(anchor=W)
        rf_row = ttk.Frame(self.settings_inner)
        rf_row.pack(fill=X, pady=5)
        ttk.Label(rf_row, text="n_estimators:").pack(side=LEFT)
        self.rf_estimators = ttk.IntVar(value=500)
        ttk.Spinbox(rf_row, from_=100, to=2000, increment=100, textvariable=self.rf_estimators, width=10).pack(side=RIGHT)
        ttk.Label(rf_row, text="   max_depth:").pack(side=LEFT, padx=(30,0))
        self.rf_depth = ttk.IntVar(value=300)
        ttk.Spinbox(rf_row, from_=0, to=100, textvariable=self.rf_depth, width=10).pack(side=RIGHT)
        ttk.Label(rf_row, text=" (0 = unlimited)").pack(side=RIGHT, padx=5)
        LSTM_row = ttk.Frame(self.settings_inner)
        LSTM_row.pack(fill=X, pady=5)
        ttk.Label(LSTM_row, text="Final LSTM Max Epochs:", font=("Helvetica", 9)).pack(side=LEFT)
        self.final_epochs = ttk.IntVar(value=300)
        ttk.Spinbox(LSTM_row, from_=100, to=1000, increment=50, textvariable=self.final_epochs, width=10).pack(side=RIGHT)

        # Manual hyperparameters — compact row; details live in a dialog
        man_row = ttk.Frame(self.settings_inner)
        man_row.pack(fill=X, pady=(10,4))
        ttk.Checkbutton(
            man_row, text="Skip Optuna — use manual LSTM params",
            variable=self.use_manual_params_var, bootstyle="round-toggle",
            command=self._update_manual_params_label
        ).pack(side=LEFT, padx=(0,8))
        self.manual_edit_btn = ttk.Button(
            man_row, text="Edit params…", bootstyle="secondary-outline", width=12,
            command=self.open_manual_params_dialog)
        self.manual_edit_btn.pack(side=LEFT, padx=4)
        self.manual_params_label = ttk.Label(
            self.settings_inner, text="", font=("Helvetica", 9), foreground="#888888")
        self.manual_params_label.pack(anchor=W, padx=4)
        self._update_manual_params_label()
        self._on_search_mode_change()

        # Ensemble weight control
        ens_row = ttk.Frame(self.settings_inner)
        ens_row.pack(fill=X, pady=(8,6))
        ttk.Label(ens_row, text="LSTM Ensemble Weight:", font=("Helvetica", 10, "bold")).pack(side=LEFT)
        ttk.Spinbox(ens_row, from_=0.0, to=1.0, increment=0.01, textvariable=self.ensemble_lstm_var, width=8).pack(side=LEFT, padx=8)
        ttk.Label(ens_row, text="(RF weight = 1 - LSTM)").pack(side=LEFT, padx=8)
        ttk.Checkbutton(ens_row, text="Calibrate weight (grid-search)", variable=self.calibrate_ensemble_var, bootstyle="round-toggle").pack(side=LEFT, padx=12)

        peak_row = ttk.Frame(self.settings_inner)
        peak_row.pack(fill=X, pady=(4,6))
        ttk.Checkbutton(peak_row, text="Peak-weighted training",
                        variable=self.use_peak_weight_var, bootstyle="round-toggle").pack(side=LEFT, padx=(0,8))
        ttk.Label(peak_row, text="Max weight:").pack(side=LEFT, padx=(8,0))
        ttk.Spinbox(peak_row, from_=1.0, to=100.0, increment=1.0,
                    textvariable=self.peak_weight_max_var, width=6).pack(side=LEFT, padx=4)
        ttk.Label(peak_row, text="(weight ≈ 1 + Q/Qmean, clipped to max)", font=("Helvetica", 9),
                  foreground="#888888").pack(side=LEFT, padx=(6,0))

        # Risk thresholds UI (dynamic)
        thr_row = ttk.Frame(self.settings_inner)
        thr_row.pack(fill=X, pady=(8,6))
        ttk.Label(thr_row, text="Risk Thresholds (m³/s):", font=("Helvetica", 10, "bold")).pack(side=LEFT)
        ttk.Button(thr_row, text="Suggest from data", bootstyle="info-outline",
                   command=self.suggest_risk_thresholds).pack(side=LEFT, padx=(12, 4))
        thr_row2 = ttk.Frame(self.settings_inner)
        thr_row2.pack(fill=X, pady=(4,6))
        ttk.Label(thr_row2, text="  Elevated >").pack(side=LEFT, padx=(8,0))
        ttk.Spinbox(thr_row2, from_=0.0, to=50.0, increment=0.1, textvariable=self.elevated_var, width=8).pack(side=LEFT, padx=4)
        ttk.Label(thr_row2, text="  Moderate >").pack(side=LEFT, padx=(8,0))
        ttk.Spinbox(thr_row2, from_=0.0, to=50.0, increment=0.1, textvariable=self.moderate_var, width=8).pack(side=LEFT, padx=4)
        ttk.Label(thr_row2, text="  High >").pack(side=LEFT, padx=(8,0))
        ttk.Spinbox(thr_row2, from_=0.0, to=50.0, increment=0.1, textvariable=self.high_var, width=8).pack(side=LEFT, padx=4)

        try:
            self.elevated_var.trace_add('write', lambda *a: self.validate_thresholds())
            self.moderate_var.trace_add('write', lambda *a: self.validate_thresholds())
            self.high_var.trace_add('write', lambda *a: self.validate_thresholds())
        except Exception:
            pass

        mc_row = ttk.Frame(self.settings_inner)
        mc_row.pack(fill=X, pady=(6,8))
        ttk.Label(mc_row, text="MC Dropout Samples:", font=("Helvetica", 10, "bold")).pack(side=LEFT)
        ttk.Spinbox(mc_row, from_=10, to=2000, increment=10, textvariable=self.mc_samples_var, width=10).pack(side=LEFT, padx=8)

        aci_row = ttk.Frame(self.settings_inner)
        aci_row.pack(fill=X, pady=(4,8))
        ttk.Checkbutton(
            aci_row, text="Apply rolling ACI to 95% intervals",
            variable=self.use_aci_var, bootstyle="round-toggle").pack(side=LEFT, padx=(0,8))
        aci_row2 = ttk.Frame(self.settings_inner)
        aci_row2.pack(fill=X, pady=(4,8))
        ttk.Label(aci_row2, text="Window (observations):").pack(side=LEFT, padx=(6,0))
        ttk.Spinbox(
            aci_row2, from_=180, to=365, increment=30,
            textvariable=self.aci_window_var, width=7).pack(side=LEFT, padx=4)
        ttk.Label(aci_row2, text="Adaptation γ:").pack(side=LEFT, padx=(8,0))
        ttk.Spinbox(
            aci_row2, from_=0.001, to=0.1, increment=0.001,
            textvariable=self.aci_gamma_var, width=7).pack(side=LEFT, padx=4)
        ttk.Label(
            aci_row2, text="Run Validation Uncertainty to initialize/update calibration",
            font=("Helvetica", 9), foreground="#888888").pack(side=LEFT, padx=(8,0))

        # Engineered feature toggles (defaults: only rainfall & discharge used)
        feat_row = ttk.Frame(self.settings_inner)
        feat_row.pack(fill=X, pady=(6,8))
        ttk.Label(feat_row, text="Engineered Features:", font=("Helvetica", 10, "bold")).pack(anchor=W)
        cb_row1 = ttk.Frame(feat_row)
        cb_row1.pack(fill=X, pady=4)
        ttk.Checkbutton(cb_row1, text="3-day Cumulative Rain", variable=self.use_cum3_var, bootstyle="round-toggle").pack(side=LEFT, padx=4)
        ttk.Checkbutton(cb_row1, text="7-day Cumulative Rain", variable=self.use_cum7_var, bootstyle="round-toggle").pack(side=LEFT, padx=4)
        ttk.Checkbutton(cb_row1, text="14-day Cumulative Rain", variable=self.use_cum14_var, bootstyle="round-toggle").pack(side=LEFT, padx=4)
        ttk.Checkbutton(cb_row1, text="Rate of Rise", variable=self.use_rate_var, bootstyle="round-toggle").pack(side=LEFT, padx=4)
        cb_row2 = ttk.Frame(feat_row)
        cb_row2.pack(fill=X, pady=4)
        ttk.Checkbutton(cb_row2, text="Lag-1 Discharge", variable=self.use_lag1_var, bootstyle="round-toggle").pack(side=LEFT, padx=4)
        ttk.Checkbutton(cb_row2, text="Wet Season Flag", variable=self.use_wetseason_var, bootstyle="round-toggle").pack(side=LEFT, padx=4)
        ttk.Checkbutton(cb_row2, text="Include API (Antecedent Precip. Index)", variable=self.use_api_var, bootstyle="round-toggle").pack(side=LEFT, padx=8)
        api_ctrl = ttk.Frame(feat_row)
        api_ctrl.pack(fill=X, pady=(4,0))
        ttk.Label(api_ctrl, text="API Decay (k):", font=("Helvetica", 9)).pack(side=LEFT)
        ttk.Spinbox(api_ctrl, from_=0.5, to=0.99, increment=0.01, textvariable=self.api_decay_var, width=8).pack(side=LEFT, padx=6)
        ttk.Label(api_ctrl, text="(API_t = k*API_{t-1} + rainfall_t)", font=("Helvetica", 9), foreground="#888888").pack(side=LEFT, padx=8)

        # pack the scrollable settings canvas and its scrollbar
        self.settings_canvas.pack(side="left", fill=X, expand=True, pady=10, padx=5)
        self.settings_scrollbar.pack(side="right", fill="y", pady=10)

        # PREDICTION
        pred_box = ttk.Labelframe(left, text="7-Day Rainfall Forecast (comma-separated)", padding=15)
        pred_box.pack(fill=X, pady=20, padx=20)
        self.rain_input = ttk.Entry(pred_box, font=("Helvetica", 12))
        self.rain_input.pack(fill=X, pady=5)
        self.rain_input.insert(0, "0, 10, 25, 50, 30, 5, 0")

        pred_btn_frame = ttk.Frame(pred_box)
        pred_btn_frame.pack(pady=10)
        self.predict_btn = ttk.Button(pred_btn_frame, text="Start Prediction", bootstyle=DANGER, width=18, command=self.predict_flood)
        self.predict_btn.pack(side=LEFT, padx=5)
        self.export_btn = ttk.Button(pred_btn_frame, text="Export Results", bootstyle="success", width=18, command=self.export_to_excel, state=DISABLED)
        self.export_btn.pack(side=LEFT, padx=5)
        self.forecast_graph_btn = ttk.Button(
            pred_btn_frame, text="Open Forecast Graph", bootstyle="info-outline", width=18,
            command=self.show_forecast_graph, state=DISABLED)
        self.forecast_graph_btn.pack(side=LEFT, padx=5)
        self.rf_shap_btn = ttk.Button(
            pred_btn_frame, text="RF SHAP", bootstyle="warning-outline", width=12,
            command=self.show_rf_shap, state=DISABLED)
        self.rf_shap_btn.pack(side=LEFT, padx=5)

        self.result_text = scrolledtext.ScrolledText(left, height=18, font=("Helvetica", 11), bg="#112233", fg="white")
        self.result_text.pack(fill=BOTH, expand=True, padx=20, pady=10)
        try:
            self.result_text.tag_config('bar', foreground='#00ff99')
            self.result_text.tag_config('xai', foreground='#cccccc')
            self.result_text.tag_config('xai_hdr', foreground='#88ccff', font=("Helvetica", 11, "bold"))
            self.result_text.tag_config('day', foreground='#00ffff', font=("Helvetica", 11, "bold"))
            # Additional tags for colored XAI pieces
            self.result_text.tag_config('feat_name', foreground='#a6e22e')
            self.result_text.tag_config('feat_val', foreground='#ffd700')
            self.result_text.tag_config('past_day', foreground='#61dafb')
            self.result_text.tag_config('risk_high', foreground='#ff4444')
            self.result_text.tag_config('risk_med', foreground='#ff9900')
            self.result_text.tag_config('risk_elevated', foreground='#ffcc66')
            self.result_text.tag_config('risk_low', foreground='#00ff99')
            # Predicted value and PI tags
            self.result_text.tag_config('pred_val', foreground='#7CFC00')
            self.result_text.tag_config('pi_val', foreground='#ffd27f')
            # Confidence tags
            self.result_text.tag_config('confidence_high', foreground='#00ff99')
            self.result_text.tag_config('confidence_med', foreground='#ffcc66')
            self.result_text.tag_config('confidence_low', foreground='#ff6666')
            # Probability tags
            self.result_text.tag_config('prob_high', foreground="#ce3333")
            self.result_text.tag_config('prob_med', foreground='#ff9900')
            self.result_text.tag_config('prob_elev', foreground='#ffcc66')
            self.result_text.tag_config('xai_sep', foreground='#444444')
        except Exception:
            pass

        # RIGHT PANEL (scrollable dashboard similar to v2.4)
        right_frame = ttk.Labelframe(main, text="Live Training Dashboard", padding=15)
        main.add(right_frame, weight=2)

        canvas = ttk.Canvas(right_frame, background="#1e1e1e")
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview, bootstyle="round")
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", tags="inner_frame")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def adapt_inner_width(event):
            try:
                canvas.itemconfigure("inner_frame", width=event.width)
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass
        canvas.bind("<Configure>", adapt_inner_width)

        right = scrollable_frame
        try:
            self.root.after(1000, lambda: main.sashpos(0, int(main.winfo_width() * 0.41)))
        except Exception:
            pass

        ttk.Label(right, text="Overall Progress", font=("Helvetica", 12, "bold")).pack(anchor=W, padx=10)
        self.overall_pb = ttk.Progressbar(right, length=520, mode='determinate', bootstyle="success-striped")
        self.overall_pb.pack(pady=8, padx=10, fill=X)
        # Small text label to show current progress state
        self.progress_label = ttk.Label(right, text="Idle", font=("Helvetica", 10), foreground="#cccccc")
        self.progress_label.pack(anchor=W, padx=10)

        ttk.Label(right, text="Optuna Hyperparameter Search", font=("Helvetica", 12, "bold")).pack(anchor=W, padx=10, pady=(15,0))
        self.lstm_pb = ttk.Progressbar(right, length=520, mode='determinate', bootstyle="info-striped")
        self.lstm_pb.pack(pady=8, padx=10, fill=X)

        stats_frame = ttk.Frame(right)
        stats_frame.pack(fill=X, padx=10, pady=10)
        self.trial_label = ttk.Label(stats_frame, text="Trials: 0/0", font=("Helvetica", 11))
        self.trial_label.pack(side=LEFT)
        self.best_label = ttk.Label(stats_frame, text=" | Best NSE: -", font=("Helvetica", 11, "bold"), foreground="#00ff99")
        self.best_label.pack(side=LEFT, padx=20)
        self.time_label = ttk.Label(stats_frame, text=" | Time: 0s", font=("Helvetica", 10), foreground="#cccccc")
        self.time_label.pack(side=RIGHT)

        self.viz_btn_quick = ttk.Button(stats_frame, text="Show Optuna Visualization", bootstyle="success-outline", width=30, command=self.show_optuna_viz)
        self.viz_btn_quick.pack(side=RIGHT, padx=8)
        self.viz_btn_quick.config(state=DISABLED)

        self.current_trial_label = ttk.Label(right, text="Ready", font=("Consolas", 10), foreground="#88ccff")
        self.current_trial_label.pack(anchor=W, padx=10, pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={'height_ratios': [1, 1]}, tight_layout=True)
        try:
            self.fig.subplots_adjust(hspace=0.45)
        except Exception:
            pass

        self.ax1.set_title("Live Validation Loss (Trial Best)", fontsize=12, color="#008cff", pad=10)
        self.ax1.set_xlabel("Trial")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True, alpha=0.3, color="#333333")
        self.loss_line, = self.ax1.plot([], [], 'o-', color='#00ff99', linewidth=3, markersize=8, label="Best so far")
        self.current_dot, = self.ax1.plot([], [], 'o', color='#ff4444', markersize=12, label="Current trial")
        self.ax1.legend(fontsize=10)

        self.ax2.set_title("LSTM Training Loss (per epoch)", fontsize=12, color="#008cff", pad=6)
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Loss")
        self.ax2.grid(True, alpha=0.3, color="#333333")
        self.train_line, = self.ax2.plot([], [], '-o', color='#1f77b4', linewidth=2, markersize=5, label="Train loss")
        self.val_line, = self.ax2.plot([], [], '-o', color='#ff7f0e', linewidth=2, markersize=5, label="Val loss")
        self.ax2.legend(fontsize=10)

        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(pady=10, fill=X)

        log_frame = ttk.Labelframe(right, text="Training Log", padding=10)
        log_frame.pack(fill=BOTH, expand=True, pady=10)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Consolas", 10), bg="#1e1e1e", fg="#00ff99")
        self.log_text.pack(fill=BOTH, expand=True)
        self.log("System ready...")

    def explain_prediction(self, seq_scaled):
        """
        Gradient-based XAI: Which past days & features drive the prediction?
        """
        seq_tensor = tf.convert_to_tensor(seq_scaled[np.newaxis, ...], dtype=tf.float32)
        seq_tensor = tf.Variable(seq_tensor)

        # Detect if the LSTM model expects seq2seq inputs (encoder + decoder)
        model = getattr(self, 'lstm_model', None)
        is_seq2seq = False
        try:
            if model is not None and isinstance(model.inputs, list) and len(model.inputs) == 2:
                is_seq2seq = True
        except Exception:
            is_seq2seq = False

        with tf.GradientTape() as tape:
            try:
                if is_seq2seq:
                    # build a zero decoder input matching the model's decoder horizon
                    try:
                        horizon = int(model.inputs[1].shape[1]) if model.inputs[1].shape[1] is not None else 7
                    except Exception:
                        horizon = 7
                    decoder_zero = tf.zeros((1, horizon, 1), dtype=tf.float32)
                    pred = model([seq_tensor, decoder_zero], training=False)
                    # take last timestep's discharge
                    predicted_disc_diff_scaled = pred[0, -1, 0]
                else:
                    pred = model(seq_tensor, training=False)
                    predicted_disc_diff_scaled = pred[0, 0]
            except Exception:
                # fallback: try single-input call
                pred = model(seq_tensor, training=False)
                try:
                    predicted_disc_diff_scaled = pred[0, 0]
                except Exception:
                    predicted_disc_diff_scaled = tf.reduce_mean(pred)

        grads = tape.gradient(predicted_disc_diff_scaled, seq_tensor)[0]
        
        grads = np.abs(grads.numpy())

        timestep_importance = np.mean(grads, axis=1)

        try:
            feature_names = list(self.feature_columns)
        except Exception:
            feature_names = [
                'rainfall', 'discharge', 'rain_cum3', 'rain_cum7', 'rain_cum14',
                'disc_diff', 'disc_lag1', 'wet_season', 'api'
            ]

        feature_importance = np.mean(grads, axis=0)

        feat_map = {}
        for i in range(min(len(feature_names), len(feature_importance))):
            raw = feature_names[i]
            pretty = self.display_name_map.get(raw, raw)
            try:
                feat_map[pretty] = float(feature_importance[i])
            except Exception:
                feat_map[pretty] = float(np.array(feature_importance[i]).astype(float))

        return {
            'timestep': timestep_importance,
            'feature': feat_map,
            'raw_grads': grads
        }

    def toggle_settings(self):
        if self.show_settings.get():
            try:
                self.settings_canvas.pack(side="left", fill=X, expand=True, pady=10, padx=5)
                self.settings_scrollbar.pack(side="right", fill="y", pady=10)
                try:
                    self.settings_canvas.yview_moveto(0)
                except Exception:
                    pass
            except Exception:
                pass
        else:
            try:
                self.settings_canvas.pack_forget()
                self.settings_scrollbar.pack_forget()
            except Exception:
                pass

    def log(self, msg, color=None):
        # `color` kept for backward compatibility with callers; currently unused
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            self.log_text.insert(END, f"[{timestamp}] {msg}\n")
            self.log_text.see(END)
            try:
                self.root.update_idletasks()
            except Exception:
                pass
        except Exception:
            # If log widget not available, fall back to stdout
            try:
                print(f"[{timestamp}] {msg}")
            except Exception:
                pass

    def process_plot_queue(self):
        try:
            while True:
                msg = self.plot_queue.get_nowait()
                if not isinstance(msg, tuple) or len(msg) == 0:
                    continue
                mtype = msg[0]
                if mtype == 'epoch':
                    try:
                        _, epochs, train_losses, val_losses = msg
                        self.train_line.set_data(epochs, train_losses)
                        self.val_line.set_data(epochs, val_losses)
                        self.ax2.relim()
                        self.ax2.autoscale_view()
                        self.canvas.draw()
                        self.canvas.flush_events()
                    except Exception as e:
                        print(f"Epoch plot update error: {e}")
                elif mtype == 'update_stats':
                    try:
                        _, stats = msg
                        completed = stats.get('completed')
                        n_trials = stats.get('n_trials')
                        best_mse = stats.get('best_mse')
                        elapsed_s = stats.get('elapsed_s')
                        if completed is not None and n_trials is not None:
                            self.trial_label.config(text=f"Trials: {completed}/{n_trials}")
                            self.lstm_pb['value'] = (completed / n_trials) * 100 if n_trials else 0
                        if best_mse is not None:
                            try:
                                best_nse = 1.0 - float(best_mse)
                                self.best_label.config(
                                    text=f" | Best NSE: {best_nse:.3f}", foreground="#00ff99")
                            except Exception:
                                pass
                        if elapsed_s is not None:
                            self.time_label.config(text=f" | Time: {elapsed_s}s")
                    except Exception as e:
                        print(f"Stats update error: {e}")
                elif mtype == 'progress':
                    try:
                        _, cur, total, ptext = msg
                        try:
                            if total and total > 0:
                                self.overall_pb['maximum'] = total
                                self.overall_pb['value'] = cur
                            else:
                                try:
                                    self.overall_pb['value'] = float(cur)
                                except Exception:
                                    self.overall_pb['value'] = 0
                        except Exception:
                            pass
                        if getattr(self, 'progress_label', None) is not None:
                            try:
                                self.progress_label.config(text=str(ptext))
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Progress update error: {e}")
                elif mtype == 'log':
                    try:
                        _, text = msg
                        try:
                            self.log(str(text))
                        except Exception:
                            print(str(text))
                    except Exception as e:
                        print(f"Log message error: {e}")
                elif mtype == 'clear_epoch_plot':
                    try:
                        self.train_line.set_data([], [])
                        self.val_line.set_data([], [])
                        self.ax2.relim()
                        self.ax2.autoscale_view()
                        self.canvas.draw()
                    except Exception:
                        pass
                elif mtype == 'trial' or mtype == 'plot':
                    try:
                        _, x, best_y, curr_x, curr_y = msg
                        self.loss_line.set_data(x, best_y)
                        self.current_dot.set_data(curr_x, curr_y)
                        self.ax1.relim()
                        self.ax1.autoscale_view()
                        self.canvas.draw()
                        self.canvas.flush_events()
                    except Exception as e:
                        print(f"Trial plot update error (queue message): {e}")
                elif mtype == 'trial_history':
                    try:
                        _, x, best_y = msg
                        self.loss_line.set_data(x, best_y)
                        self.current_dot.set_data([], [])
                        self.ax1.relim()
                        self.ax1.autoscale_view()
                        self.canvas.draw()
                        self.canvas.flush_events()
                    except Exception as e:
                        print(f"Trial history plot update error: {e}")
                elif mtype == 'trial_point':
                    try:
                        _, trial_number, trial_value, best_value = msg
                        x = list(self.loss_line.get_xdata())
                        best_y = list(self.loss_line.get_ydata())
                        x.append(trial_number)
                        best_y.append(best_value)
                        self.loss_line.set_data(x, best_y)
                        self.current_dot.set_data([trial_number], [trial_value])
                        self.ax1.relim()
                        self.ax1.autoscale_view()
                        self.canvas.draw()
                        self.canvas.flush_events()
                    except Exception as e:
                        print(f"Trial point plot update error: {e}")
                elif mtype == 'current_trial':
                    try:
                        _, text = msg
                        if getattr(self, 'current_trial_label', None) is not None:
                            try:
                                self.current_trial_label.config(text=text)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Current trial update error: {e}")
                else:
                    try:
                        x, best_y, curr_x, curr_y = msg
                        self.loss_line.set_data(x, best_y)
                        self.current_dot.set_data(curr_x, curr_y)
                        self.ax1.relim()
                        self.ax1.autoscale_view()
                        self.canvas.draw()
                        self.canvas.flush_events()
                    except Exception as e:
                        print(f"Unhandled plot queue message: {msg} ({e})")
                if isinstance(msg, tuple) and len(msg) > 0 and msg[0] == 'enable_viz':
                    try:
                        if getattr(self, 'viz_btn_quick', None) is not None:
                            try:
                                self.viz_btn_quick.config(state=NORMAL)
                            except Exception:
                                pass
                        if getattr(self, 'viz_btn', None) is not None:
                            try:
                                self.viz_btn.config(state=NORMAL)
                            except Exception:
                                pass
                        if getattr(self, 'viz_btn_top', None) is not None:
                            try:
                                self.viz_btn_top.config(state=NORMAL)
                            except Exception:
                                pass
                        try:
                            self.log("Optuna visualization enabled")
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Could not enable viz buttons: {e}")
                    continue
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Plot update error: {e}")
        finally:
            try:
                if not getattr(self, '_closing', False):
                    self._after_id = self.root.after(100, self.process_plot_queue)
            except Exception:
                self._after_id = None

    
    def load_file(self):
        path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        try:
            self.log(f"Loading {os.path.basename(path)}...")
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
            n_before = len(df)
            df = df.dropna().sort_values('date').reset_index(drop=True)
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                self.log(f"Dropped {n_dropped} row(s) with missing values ({n_before} → {len(df)}).")
            self.df = df
            cached = getattr(self, '_val_unc_cache', None)
            try:
                keep_cached_results = (
                    cached is not None and
                    cached.get('data_signature') == self._validation_data_signature())
            except Exception:
                keep_cached_results = False
            if not keep_cached_results:
                self._reset_aci_calibration()
            self.seq_length = self.seq_var.get()

            # ==================== FEATURE ENGINEERING ====================
            df_feat = df.copy()
            # Cumulative rainfall
            df_feat['rain_cum3'] = df_feat['rainfall'].rolling(window=3, min_periods=1).sum()
            df_feat['rain_cum7'] = df_feat['rainfall'].rolling(window=7, min_periods=1).sum()
            df_feat['rain_cum14'] = df_feat['rainfall'].rolling(window=14, min_periods=1).sum()
            # Discharge dynamics
            df_feat['disc_diff'] = df_feat['discharge'].diff().fillna(0)
            df_feat['disc_lag1'] = df_feat['discharge'].shift(1).bfill()
            # Seasonal flag
            df_feat['wet_season'] = df_feat['date'].dt.month.isin([6,7,8,9,10,11]).astype(float)
            # Antecedent Precipitation Index (API)
            try:
                decay = float(self.api_decay_var.get()) if getattr(self, 'api_decay_var', None) is not None else 0.9
            except Exception:
                decay = 0.9
            api_vals = []
            prev = 0.0
            for r in df_feat['rainfall'].fillna(0).values:
                cur = decay * prev + float(r)
                api_vals.append(cur)
                prev = cur
            df_feat['api'] = api_vals
            df_feat = df_feat.fillna(0)

            # Active feature set from toggles
            feature_columns = ['rainfall', 'discharge']
            if self.use_cum3_var.get():   feature_columns.append('rain_cum3')
            if self.use_cum7_var.get():   feature_columns.append('rain_cum7')
            if self.use_cum14_var.get():  feature_columns.append('rain_cum14')
            if self.use_rate_var.get():   feature_columns.append('disc_diff')
            if self.use_lag1_var.get():   feature_columns.append('disc_lag1')
            if self.use_wetseason_var.get(): feature_columns.append('wet_season')
            if self.use_api_var.get():    feature_columns.append('api')
            self.feature_columns = feature_columns

            # ==================== TIME-BASED SPLIT ====================
            if len(df) == 0:
                raise ValueError("No valid rows in the CSV after parsing dates and dropping NaNs")
            split_idx = int(0.8 * len(df))
            if split_idx >= len(df):
                split_idx = max(0, len(df) - 1)
            split_date = df['date'].iloc[split_idx]
            train_df = df_feat[df_feat['date'] < split_date].copy()
            val_df   = df_feat[df_feat['date'] >= split_date].copy()

            self.log(
                f"Time-based split: {split_date.strftime('%Y-%m-%d')} "
                f"Train: {len(train_df)} days  Val: {len(val_df)} days"
            )
            self.train_df = train_df
            self.val_df   = val_df

            # ==================== TRAIN-ONLY SCALERS ====================
            self.scaler_features = MinMaxScaler()
            self.scaler_features.fit(train_df[feature_columns].values.astype(np.float32))

            self.scaler_level = MinMaxScaler()
            self.scaler_level.fit(train_df['discharge'].values.reshape(-1, 1).astype(np.float32))

            # Rainfall-only scaler (for decoder inputs / user-provided future rain)
            try:
                self.scaler_rain = MinMaxScaler()
                self.scaler_rain.fit(train_df['rainfall'].values.reshape(-1, 1).astype(np.float32))
            except Exception:
                pass

            # Create train/val sequences (create_sequences_for() uses self.scaler_features.transform internally)
            self.train_X, self.train_y = self.create_sequences_for(train_df, self.seq_length, feature_columns)
            self.val_X,   self.val_y   = self.create_sequences_for(val_df,   self.seq_length, feature_columns)

            if len(self.val_X) == 0:
                self.log("Warning: Validation set empty → using 80/20 index fallback", "orange")
                X_full, y_full = self.create_sequences_for(df_feat, self.seq_length, feature_columns)
                split_idx = int(0.8 * len(X_full))
                self.train_X, self.val_X = X_full[:split_idx], X_full[split_idx:]
                self.train_y, self.val_y = y_full[:split_idx], y_full[split_idx:]

            try:
                self.log(f"Active features: {', '.join(feature_columns)}")
            except Exception:
                pass

            self.log(f"Ready → Train: {len(self.train_X):,} sequences  Val: {len(self.val_X):,} sequences")
            self.status.config(text=f"Loaded {len(df):,} days → Time-split + Features Active!", bootstyle="success")
            # New dataset → drop previous Optuna memory so trials are not mixed
            self._clear_optuna_history(silent=True, reason="new data loaded")
            self._load_persistent_val_unc_cache()

        except Exception as e:
            self.log(f"Load failed: {str(e)}", color="danger")  # or "red"
            messagebox.showerror("Load Error", f"Could not load file:\n{e}")


    def _compute_peak_weights(self, y_scaled):
        """Sample weights that emphasise high-flow (peak) periods.

        Rank-based scheme (stable vs a few extreme outliers):
          - low flows ≈ weight 1
          - highest flows approach max_weight
          - quadratic in percentile so upper tail is emphasised more than mean flow
        Falls back to 1 + Q/Q_mean if ranking fails.
        """
        y = np.asarray(y_scaled, dtype=np.float64)
        if y.ndim >= 2:
            y = y[:, -1]
        y = y.ravel()
        try:
            y_m = self.scaler_level.inverse_transform(y.reshape(-1, 1)).ravel()
        except Exception:
            y_m = y
        try:
            w_max = float(self.peak_weight_max_var.get())
        except Exception:
            w_max = 5.0
        w_max = max(1.0, w_max)
        try:
            # percentile rank in [0, 1]
            order = np.argsort(np.argsort(y_m))
            pct = order / max(len(y_m) - 1, 1)
            # quadratic: mild weight in bulk, stronger near peaks
            w = 1.0 + (w_max - 1.0) * (pct ** 2)
            w = np.clip(w, 1.0, w_max).astype(np.float32)
        except Exception:
            y_mean = float(np.mean(y_m)) + 1e-6
            w = 1.0 + (y_m / y_mean)
            w = np.clip(w, 1.0, w_max).astype(np.float32)
        return w

    def _discharge_feature_indices(self, feature_columns):
        """Indices of features that would leak target discharge if left on the last timestep."""
        leak = []
        for name in ('discharge', 'disc_diff', 'disc_lag1'):
            if name in feature_columns:
                leak.append(feature_columns.index(name))
        return leak

    def create_sequences_for(self, df_part, seq_length, feature_columns, forecast_horizon=7):
        """Supports recursive, concurrent (target-day rain), and seq2seq modes."""
        try:
            use_seq2seq = getattr(self, 'use_seq2seq_var', ttk.BooleanVar(value=False)).get()
            concurrent = (not use_seq2seq) and bool(getattr(self, 'same_month_var', ttk.BooleanVar(value=False)).get())

            if use_seq2seq:
                min_len = seq_length + forecast_horizon
            elif concurrent:
                min_len = seq_length
            else:
                min_len = seq_length + 1
            if len(df_part) < min_len:
                return np.array([]), np.array([])

            data_scaled = self.scaler_features.transform(df_part[feature_columns].values.astype(np.float32))
            X_part, y_part = [], []
            disc_idx = feature_columns.index('discharge') if 'discharge' in feature_columns else 1
            leak_idx = self._discharge_feature_indices(feature_columns)

            if use_seq2seq:
                end = len(data_scaled) - forecast_horizon + 1
                for i in range(seq_length, end):
                    X_part.append(data_scaled[i - seq_length:i])
                    y_part.append(data_scaled[i:i + forecast_horizon, disc_idx])
            elif concurrent:
                for i in range(seq_length - 1, len(data_scaled)):
                    window = data_scaled[i - seq_length + 1:i + 1].copy()
                    for j in leak_idx:
                        window[-1, j] = 0.0
                    X_part.append(window)
                    y_part.append(data_scaled[i, disc_idx])
            else:
                for i in range(seq_length, len(data_scaled)):
                    X_part.append(data_scaled[i - seq_length:i])
                    y_part.append(data_scaled[i, disc_idx])

            return np.array(X_part), np.array(y_part)
        except Exception as e:
            print("create_sequences_for error:", e)
            return np.array([]), np.array([])
    
    def build_seq2seq_model(self, seq_length, n_features, units1, units2, dropout, l2=0.0, horizon=7):
        """Builds Encoder-Decoder Seq2Seq model"""
        # Encoder
        encoder_inputs = Input(shape=(seq_length, n_features))
        x = LSTM(units1, return_sequences=True, kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None)(encoder_inputs)
        x = Dropout(dropout)(x)
        attn = MultiHeadAttention(num_heads=8, key_dim=32, dropout=dropout)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)
        encoder_out, state_h, state_c = LSTM(units2, return_state=True, 
                                           kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None)(x)

        # Decoder - receives future 7-day rainfall
        decoder_inputs = Input(shape=(horizon, 1))   # future rainfall only
        decoder_lstm = LSTM(units2, return_sequences=True)
        decoder_out = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
        decoder_out = Dense(1)(decoder_out)   # output 7 steps of discharge

        model = Model([encoder_inputs, decoder_inputs], decoder_out)
        return model    

    def _get_seq_length(self, fallback):
        """Return a safe integer sequence length. If stored seq_length is invalid/NaN, return fallback."""
        try:
            val = getattr(self, 'seq_length', None)
            if val is None:
                return int(fallback)
            if isinstance(val, float) and np.isnan(val):
                return int(fallback)
            return int(val)
        except Exception:
            try:
                return int(fallback)
            except Exception:
                return 1

    def _on_feature_toggle_change(self, *args):
        """Handler called when user toggles engineered features.
        If data is loaded, rebuild scalers and sequences immediately; otherwise prompt user to load data.
        """
        try:
            if getattr(self, '_loading_model_metadata', False):
                return
            if getattr(self, 'df', None) is None:
                try:
                    self.log("Feature selection changed — no data loaded; opening file dialog to load data.")
                except Exception:
                    pass
                try:
                    self.load_file()
                except Exception:
                    try:
                        self.log("Auto-load failed; please load CSV manually.")
                    except Exception:
                        pass
                return
            self.apply_feature_selection()
            self.log("Feature selection applied — scalers and sequences rebuilt.")
        except Exception as e:
            try:
                self.log(f"Applying feature selection failed: {e}")
            except Exception:
                pass

    
    def apply_feature_selection(self):
        """Recompute engineered features, fit scalers and recreate train/val sequences
        using the currently selected feature toggles. This uses the already-loaded
        `self.df` and does not prompt the user for files.
        """
        if getattr(self, 'df', None) is None:
            return

        self._reset_aci_calibration()
        df = self.df.copy()
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        except Exception:
            pass

        # ---------- Rebuild engineered features ----------
        df_feat = df.copy()
        df_feat['rain_cum3']  = df_feat['rainfall'].rolling(window=3,  min_periods=1).sum()
        df_feat['rain_cum7']  = df_feat['rainfall'].rolling(window=7,  min_periods=1).sum()
        df_feat['rain_cum14'] = df_feat['rainfall'].rolling(window=14, min_periods=1).sum()
        df_feat['disc_diff'] = df_feat['discharge'].diff().fillna(0)
        df_feat['disc_lag1'] = df_feat['discharge'].shift(1).bfill()
        df_feat['wet_season'] = df_feat['date'].dt.month.isin([6,7,8,9,10,11]).astype(float)

        # API recompute (only if toggle included)
        try:
            decay = float(self.api_decay_var.get()) if getattr(self, 'api_decay_var', None) is not None else 0.9
        except Exception:
            decay = 0.9
        api_vals = []
        prev = 0.0
        for r in df_feat['rainfall'].fillna(0).values:
            cur = decay * prev + float(r)
            api_vals.append(cur)
            prev = cur
        df_feat['api'] = api_vals

        df_feat = df_feat.fillna(0)

        # Active feature set from toggles
        feature_columns = ['rainfall', 'discharge']
        if self.use_cum3_var.get():      feature_columns.append('rain_cum3')
        if self.use_cum7_var.get():      feature_columns.append('rain_cum7')
        if self.use_cum14_var.get():     feature_columns.append('rain_cum14')
        if self.use_rate_var.get():      feature_columns.append('disc_diff')
        if self.use_lag1_var.get():      feature_columns.append('disc_lag1')
        if self.use_wetseason_var.get(): feature_columns.append('wet_season')
        if self.use_api_var.get():       feature_columns.append('api')

        self.feature_columns = feature_columns
        try:
            self.log(f"Active features: {', '.join(feature_columns)}")
        except Exception:
            pass

        # ---------- Time-based split ----------
        try:
            if len(df) == 0:
                raise ValueError("No valid rows available for splitting")
            split_idx = int(0.8 * len(df))
            if split_idx >= len(df):
                split_idx = max(0, len(df) - 1)
            split_date = df['date'].iloc[split_idx]
            train_df = df_feat[df_feat['date'] < split_date].copy()
            val_df   = df_feat[df_feat['date'] >= split_date].copy()
        except Exception:
            train_df = df_feat.copy()
            val_df   = df_feat.copy()

        self.train_df = train_df
        self.val_df   = val_df

        # ==================== TRAIN-ONLY SCALERS (NO LEAKAGE) ====================
        try:
            self.scaler_features = MinMaxScaler()
            self.scaler_features.fit(train_df[feature_columns].values.astype(np.float32))
        except Exception:
            pass
        try:
            self.scaler_level = MinMaxScaler()
            self.scaler_level.fit(train_df['discharge'].values.reshape(-1, 1).astype(np.float32))
        except Exception:
            pass
        try:
            # fit rainfall scaler when features are (re)computed
            self.scaler_rain = MinMaxScaler()
            self.scaler_rain.fit(train_df['rainfall'].values.reshape(-1, 1).astype(np.float32))
        except Exception:
            pass

        # ---------- Recreate sequences ----------
        seq_length = getattr(self, 'seq_length', None) or self.seq_var.get()
        try:
            self.train_X, self.train_y = self.create_sequences_for(train_df, self.seq_length, feature_columns)
            self.val_X,   self.val_y   = self.create_sequences_for(val_df,   self.seq_length, feature_columns)
        except Exception:
            self.train_X = self.train_y = self.val_X = self.val_y = None

        # Clear any cached sequences tied to old scalers/settings
        try:
            self.sequence_cache = {}
        except Exception:
            pass

        try:
            if getattr(self, 'df', None) is not None and getattr(self, 'status', None) is not None:
                self.status.config(text=f"Features applied — {len(self.df):,} days loaded", bootstyle="success")
        except Exception:
            pass


    def _set_widget_state(self, widget, enabled):
        try:
            if widget is None:
                return
            widget.configure(state=NORMAL if enabled else DISABLED)
        except Exception:
            try:
                widget.configure(state='normal' if enabled else 'disabled')
            except Exception:
                pass

    def _on_search_mode_change(self):
        """Update UI controls so the chosen search mode is visible and guided."""
        try:
            mode = str(self.search_mode_var.get())

            if mode == "focused":
                self.use_manual_params_var.set(False)
                self.tune_seq_var.set(True)
                self._set_widget_state(getattr(self, 'trials_spin', None), True)
                self._set_widget_state(getattr(self, 'seq_spin', None), True)
                self._set_widget_state(getattr(self, 'tune_seq_cb', None), True)
                self._set_widget_state(getattr(self, 'seq_min_spin', None), True)
                self._set_widget_state(getattr(self, 'seq_max_spin', None), True)
                self._set_widget_state(getattr(self, 'optuna_remember_cb', None), True)
                self._set_widget_state(getattr(self, 'optuna_reset_btn', None), True)
                self._set_widget_state(getattr(self, 'manual_edit_btn', None), True)
                try:
                    self.manual_edit_btn.configure(bootstyle="secondary-outline")
                except Exception:
                    pass
                self.search_mode_hint.config(
                    text="Recommended → set sequence Min/Max + Trials, then Train. "
                         "Tunes lr / dropout / units / sequence (batch, Hδ & L2 fixed).")

            elif mode == "full":
                self.use_manual_params_var.set(False)
                self._set_widget_state(getattr(self, 'trials_spin', None), True)
                self._set_widget_state(getattr(self, 'seq_spin', None), True)
                self._set_widget_state(getattr(self, 'tune_seq_cb', None), True)
                self._set_widget_state(getattr(self, 'seq_min_spin', None), True)
                self._set_widget_state(getattr(self, 'seq_max_spin', None), True)
                self._set_widget_state(getattr(self, 'optuna_remember_cb', None), True)
                self._set_widget_state(getattr(self, 'optuna_reset_btn', None), True)
                self._set_widget_state(getattr(self, 'manual_edit_btn', None), True)
                try:
                    self.manual_edit_btn.configure(bootstyle="secondary-outline")
                except Exception:
                    pass
                self.search_mode_hint.config(
                    text="Full search → optionally enable Tune Sequence Length, set ranges below, then Train.")

            else:  # manual
                self.use_manual_params_var.set(True)
                self.tune_seq_var.set(False)
                self._set_widget_state(getattr(self, 'trials_spin', None), False)
                self._set_widget_state(getattr(self, 'seq_spin', None), True)
                self._set_widget_state(getattr(self, 'tune_seq_cb', None), False)
                self._set_widget_state(getattr(self, 'seq_min_spin', None), False)
                self._set_widget_state(getattr(self, 'seq_max_spin', None), False)
                self._set_widget_state(getattr(self, 'optuna_remember_cb', None), False)
                self._set_widget_state(getattr(self, 'optuna_reset_btn', None), False)
                self._set_widget_state(getattr(self, 'manual_edit_btn', None), True)
                try:
                    self.manual_edit_btn.configure(bootstyle="success")
                except Exception:
                    pass
                self.search_mode_hint.config(
                    text="Manual only → click Edit params… (highlighted green), set values, then Train. No Optuna trials.")

            self._update_manual_params_label()
        except Exception:
            pass

    def _apply_best_to_manual(self, best):
        """Copy Optuna best params into manual fields so the user can retrain without search."""
        try:
            if not isinstance(best, dict):
                return
            if 'units1' in best:
                self.manual_units1_var.set(int(best['units1']))
            if 'units2' in best:
                self.manual_units2_var.set(int(best['units2']))
            if 'dropout' in best:
                self.manual_dropout_var.set(float(best['dropout']))
            if 'lr' in best:
                self.manual_lr_var.set(float(best['lr']))
            if 'batch_size' in best:
                self.manual_batch_var.set(int(best['batch_size']))
            if 'Hdelta' in best:
                self.manual_hdelta_var.set(float(best['Hdelta']))
            if 'l2' in best:
                self.manual_l2_var.set(float(best['l2']))
            if 'seq_length' in best:
                self.manual_seq_var.set(int(best['seq_length']))
            self._update_manual_params_label()
            self.log("Best params copied to Manual (Edit params…) for easy retrain later.")
        except Exception as e:
            try:
                self.log(f"Could not copy best → manual params: {e}")
            except Exception:
                pass

    def _update_manual_params_label(self):
        """Refresh the compact summary under the manual-params toggle."""
        try:
            if not bool(self.use_manual_params_var.get()):
                self.manual_params_label.config(
                    text="Optuna will search hyperparameters (or continue saved study).")
                return
            self.manual_params_label.config(
                text=(
                    f"Manual: u1={self.manual_units1_var.get()}, u2={self.manual_units2_var.get()}, "
                    f"drop={self.manual_dropout_var.get():.2f}, lr={self.manual_lr_var.get():.1e}, "
                    f"batch={self.manual_batch_var.get()}, Hδ={self.manual_hdelta_var.get():.3f}, "
                    f"L2={self.manual_l2_var.get():.4g}, seq={self.manual_seq_var.get()}"
                )
            )
        except Exception:
            pass

    def _get_manual_best_params(self):
        """Dict shaped like Optuna best_params for final model training."""
        return {
            'units1': int(self.manual_units1_var.get()),
            'units2': int(self.manual_units2_var.get()),
            'dropout': float(self.manual_dropout_var.get()),
            'lr': float(self.manual_lr_var.get()),
            'batch_size': int(self.manual_batch_var.get()),
            'Hdelta': float(self.manual_hdelta_var.get()),
            'l2': float(self.manual_l2_var.get()),
            'seq_length': int(self.manual_seq_var.get()),
        }

    def _scrollable_container(self, parent):
        """Create a vertical scrollable area; returns (inner_frame, canvas)."""
        outer = ttk.Frame(parent)
        outer.pack(fill=BOTH, expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, bootstyle="round")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        inner = ttk.Frame(canvas, padding=8)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(win_id, width=max(1, event.width))

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _wheel(event):
            delta = getattr(event, "delta", 0)
            if delta:
                canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")

        def _bind_wheel(w):
            w.bind("<MouseWheel>", _wheel)
            w.bind("<Button-4>", _wheel)
            w.bind("<Button-5>", _wheel)

        _bind_wheel(canvas)
        _bind_wheel(inner)

        def _bind_recursive(widget):
            _bind_wheel(widget)
            for ch in widget.winfo_children():
                _bind_recursive(ch)

        inner.bind("<Map>", lambda e: _bind_recursive(inner))
        return inner, canvas

    def open_manual_params_dialog(self):
        """Small dialog to set known LSTM hyperparameters without cluttering the main panel."""
        win = ttk.Toplevel(self.root)
        win.title("Manual LSTM Parameters")
        win.geometry("460x480")
        win.minsize(360, 320)
        win.transient(self.root)
        try:
            win.grab_set()
        except Exception:
            pass

        frm, _canvas = self._scrollable_container(win)

        ttk.Label(
            frm,
            text="Use when you already know good hyperparameters\n"
                 "and want to train the final model without Optuna trials.",
            font=("Helvetica", 9), foreground="#888888", justify=LEFT
        ).pack(anchor=W, pady=(0, 10))

        grid = ttk.Frame(frm)
        grid.pack(fill=X)

        def row(r, label, var, from_, to_, increment=None, width=12):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky=W, pady=4, padx=(0, 8))
            kwargs = dict(from_=from_, to=to_, textvariable=var, width=width)
            if increment is not None:
                kwargs['increment'] = increment
            ttk.Spinbox(grid, **kwargs).grid(row=r, column=1, sticky=W, pady=4)

        try:
            self.manual_seq_var.set(int(self.seq_var.get()))
        except Exception:
            pass

        row(0, "Units 1:", self.manual_units1_var, 16, 1024)
        row(1, "Units 2:", self.manual_units2_var, 8, 512)
        row(2, "Dropout:", self.manual_dropout_var, 0.0, 0.8, 0.01)
        row(3, "Learning rate:", self.manual_lr_var, 1e-6, 1e-1, 1e-4)
        row(4, "Batch size:", self.manual_batch_var, 4, 256)
        row(5, "Huber Hdelta:", self.manual_hdelta_var, 0.001, 1.0, 0.005)
        row(6, "L2:", self.manual_l2_var, 0.0, 0.1, 0.0001)
        row(7, "Sequence length:", self.manual_seq_var, 1, 365)

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=X, pady=(16, 0))

        def load_from_optuna():
            try:
                if getattr(self, 'study', None) is None or not getattr(self.study, 'best_params', None):
                    messagebox.showinfo("Optuna", "No Optuna best params available yet.\nRun tuning once, or enter values manually.")
                    return
                bp = self.study.best_params
                self.manual_units1_var.set(int(bp.get('units1', self.manual_units1_var.get())))
                self.manual_units2_var.set(int(bp.get('units2', self.manual_units2_var.get())))
                self.manual_dropout_var.set(float(bp.get('dropout', self.manual_dropout_var.get())))
                self.manual_lr_var.set(float(bp.get('lr', self.manual_lr_var.get())))
                self.manual_batch_var.set(int(bp.get('batch_size', self.manual_batch_var.get())))
                self.manual_hdelta_var.set(float(bp.get('Hdelta', bp.get('delta', self.manual_hdelta_var.get()))))
                self.manual_l2_var.set(float(bp.get('l2', self.manual_l2_var.get())))
                self.manual_seq_var.set(int(bp.get('seq_length', self.manual_seq_var.get())))
                self.log("Manual params loaded from last Optuna best.")
            except Exception as e:
                messagebox.showerror("Optuna", f"Could not load best params:\n{e}")

        def on_ok():
            self.use_manual_params_var.set(True)
            self._update_manual_params_label()
            try:
                win.destroy()
            except Exception:
                pass

        ttk.Button(btn_row, text="Load from last Optuna best", bootstyle="info-outline",
                   command=load_from_optuna).pack(side=LEFT)
        ttk.Button(btn_row, text="OK", bootstyle="success", width=10, command=on_ok).pack(side=RIGHT, padx=4)
        ttk.Button(btn_row, text="Cancel", bootstyle="secondary", width=10,
                   command=win.destroy).pack(side=RIGHT, padx=4)

    def start_training(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load CSV first!")
            return
        self.stop_training = False
        self.train_btn.config(state=DISABLED)
        self.cancel_btn.config(state=NORMAL)
        self.overall_pb['value'] = 0
        self.lstm_pb['value'] = 0
        self.trial_label.config(text="Trials: 0/0")
        self.best_label.config(text=" | Best NSE: -")
        self.time_label.config(text=" | Time: 0s")
        self.current_trial_label.config(text="Optuna Optimization in Progress...")
        self.ax1.clear()
        self.ax1.set_title("Live Validation Loss (Trial Best)", fontsize=15, color="#00ff99", pad=10)
        self.ax1.set_xlabel("Trial")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True, alpha=0.3, color="#333333")
        self.loss_line, = self.ax1.plot([], [], 'o-', color='#00ff99', linewidth=3, markersize=8, label="Best so far")
        self.current_dot, = self.ax1.plot([], [], 'o', color='#ff4444', markersize=12, label="Current trial")
        self.ax1.legend(fontsize=10)

        try:
            self.ax2.clear()
            self.ax2.set_title("LSTM Training Loss (per epoch)", fontsize=13, color="#00ff99", pad=6)
            self.ax2.set_xlabel("Epoch")
            self.ax2.set_ylabel("Loss")
            self.ax2.grid(True, alpha=0.3, color="#333333")
            self.train_line, = self.ax2.plot([], [], '-o', color='#1f77b4', linewidth=2, markersize=5, label="Train loss")
            self.val_line, = self.ax2.plot([], [], '-o', color='#ff7f0e', linewidth=2, markersize=5, label="Val loss")
            self.ax2.legend(fontsize=10)
        except Exception:
            pass
        self.canvas.draw()

        self.training_thread = threading.Thread(target=self.train_models, daemon=True)
        self.training_thread.start()

    def cancel_training(self):
        self.stop_training = True
        self.log("Cancellation requested...")

    def _ask_user_yesno(self, title, message):
        """Thread-safe Yes/No dialog (safe to call from the training worker)."""
        result = {"value": False}
        done = threading.Event()

        def _ask():
            try:
                result["value"] = bool(messagebox.askyesno(title, message, parent=self.root))
            except Exception:
                result["value"] = False
            finally:
                done.set()

        try:
            self.root.after(0, _ask)
            done.wait(timeout=300)
        except Exception:
            result["value"] = False
        return bool(result["value"])

    def _attach_tooltip(self, widget, text):
        """Simple hover tooltip for a widget."""
        tip = {"win": None}

        def show(_event=None):
            if tip["win"] is not None:
                return
            try:
                x = widget.winfo_rootx() + 20
                y = widget.winfo_rooty() + widget.winfo_height() + 8
                win = tk.Toplevel(widget)
                win.wm_overrideredirect(True)
                win.wm_geometry(f"+{x}+{y}")
                try:
                    win.attributes("-topmost", True)
                except Exception:
                    pass
                lbl = tk.Label(
                    win, text=text, justify=LEFT, background="#ffffe0",
                    relief="solid", borderwidth=1, font=("Helvetica", 9),
                    padx=8, pady=6, wraplength=420
                )
                lbl.pack()
                tip["win"] = win
            except Exception:
                tip["win"] = None

        def hide(_event=None):
            win = tip.get("win")
            tip["win"] = None
            if win is not None:
                try:
                    win.destroy()
                except Exception:
                    pass

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _app_base_dir(self):
        """Writable folder next to the script, or next to the .exe when frozen."""
        try:
            if getattr(sys, 'frozen', False):
                # PyInstaller / cx_Freeze: save beside the executable (not in temp _MEIPASS)
                return os.path.dirname(os.path.abspath(sys.executable))
            return os.path.dirname(os.path.abspath(__file__))
        except Exception:
            return os.getcwd()

    def _optuna_storage_path(self):
        """SQLite file path for persistent Optuna studies."""
        return os.path.join(self._app_base_dir(), "optuna_daily_study.db")

    def _clear_optuna_history(self, silent=True, reason=None):
        """Delete persistent Optuna study data. silent=True skips popup (used on data load)."""
        study_name = getattr(self, 'optuna_study_name', 'daily_rainfall_runoff')
        storage_path = self._optuna_storage_path()
        cleared = False
        try:
            if os.path.isfile(storage_path):
                storage = f"sqlite:///{storage_path}"
                try:
                    optuna.delete_study(study_name=study_name, storage=storage)
                    cleared = True
                except Exception:
                    try:
                        os.remove(storage_path)
                        cleared = True
                    except Exception:
                        pass
            self.study = None
            if cleared:
                msg = "Optuna history cleared"
                if reason:
                    msg += f" ({reason})"
                self.log(msg + ".")
            elif not silent:
                self.log("No Optuna history file found — nothing to reset.")
            return cleared
        except Exception as e:
            self.log(f"Clear Optuna history failed: {e}")
            return False

    def _create_or_load_optuna_study(self):
        """Create a new Optuna study or continue from SQLite if enabled."""
        direction = "minimize"
        study_name = getattr(self, 'optuna_study_name', 'daily_rainfall_runoff')
        continue_prev = bool(getattr(self, 'continue_optuna_var', ttk.BooleanVar(value=True)).get())
        if not continue_prev:
            self.log("Optuna memory OFF — this run will not load or save prior trials.")
            return optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        storage_path = self._optuna_storage_path()
        storage = f"sqlite:///{storage_path}"
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=direction,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            self.log(f"Optuna storage: {storage_path}")
            return study
        except Exception as e:
            self.log(f"Optuna storage failed ({e}); falling back to in-memory study.")
            return optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))

    def reset_optuna_history(self):
        """Manual wipe — use when changing parameters / features / setup."""
        try:
            self._clear_optuna_history(silent=False, reason="manual reset")
            messagebox.showinfo(
                "Optuna",
                "Optuna history cleared.\n\n"
                "Next training starts fresh.\n"
                "Use this after changing search ranges, features, or model options."
            )
        except Exception as e:
            try:
                messagebox.showerror("Optuna", f"Could not reset history:\n{e}")
            except Exception:
                pass
            self.log(f"Reset Optuna history failed: {e}")

    def _cleanup_before_training(self):
        """Release previous models / TF graphs / caches so repeated trains don't bloat RAM."""
        try:
            self.log("Cleaning up previous training session...")
        except Exception:
            pass
        try:
            self.sequence_cache = {}
        except Exception:
            pass
        for attr in ('lstm_model', 'rf_model'):
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    del obj
                setattr(self, attr, None)
            except Exception:
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        try:
            import gc
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception:
            pass
        try:
            plt.close('all')
        except Exception:
            pass

    def train_models(self):
        # Global seeds for reproducibility (RF also uses random_state=42)
        try:
            import random as _random
            _random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
        except Exception:
            pass
        try:
            self._reset_aci_calibration()
            self._cleanup_before_training()
            # Always honour the current Sequence Length from the UI (same as monthly)
            try:
                ui_seq = int(self.seq_var.get())
                if ui_seq < 1:
                    ui_seq = 1
                if getattr(self, 'seq_length', None) != ui_seq:
                    self.seq_length = ui_seq
                    self.log(f"Sequence length set from UI → {self.seq_length} days")
                    if getattr(self, 'train_df', None) is not None and getattr(self, 'feature_columns', None) is not None:
                        self.train_X, self.train_y = self.create_sequences_for(
                            self.train_df, self.seq_length, self.feature_columns)
                        self.val_X, self.val_y = self.create_sequences_for(
                            self.val_df, self.seq_length, self.feature_columns)
                        self.sequence_cache = {}
                        self.log(f"Sequences rebuilt → Train: {len(self.train_X)}, Val: {len(self.val_X)}")
                else:
                    # Still refresh seq_length from UI even if equal (keeps state consistent)
                    self.seq_length = ui_seq
            except Exception as e:
                self.log(f"Could not apply UI sequence length: {e}")

            self.log("Starting Optuna optimization...")
            n_trials = self.trials_var.get()
            self.tuning_start_time = time.time()
            self.log("Optuna trial started...")
            trial_plot_x = []
            trial_plot_best = []

            def objective(trial):
                if self.stop_training:
                    raise optuna.exceptions.TrialPruned()

                search_mode = str(getattr(self, 'search_mode_var', ttk.StringVar(value='focused')).get())
                focused = (search_mode == 'focused')

                try:
                    if bool(self.tune_seq_var.get()):
                        seq_length = trial.suggest_int("seq_length", int(self.seq_min_var.get()), int(self.seq_max_var.get()))
                    else:
                        seq_length = int(self.seq_var.get())
                except Exception:
                    seq_length = int(getattr(self, 'seq_length', self.seq_var.get()))

                key = int(seq_length)
                if key not in self.sequence_cache:
                    X_tr, y_tr = self.create_sequences_for(self.train_df, key, self.feature_columns)
                    X_val, y_val = self.create_sequences_for(self.val_df, key, self.feature_columns)
                    if X_tr.size == 0 or X_val.size == 0:
                        raise optuna.exceptions.TrialPruned()
                    self.sequence_cache[key] = (X_tr, y_tr, X_val, y_val)

                X_tr, y_tr, X_val, y_val = self.sequence_cache[key]

                units1_min = max(16, self.units_min.get())
                units1_max = max(32, self.units_max.get())
                units1 = trial.suggest_int("units1", units1_min, units1_max)
                units2 = trial.suggest_int("units2", units1_min, units1)
                # Quantized floats so search is finite/readable (not full float noise)
                drop_lo = float(self.drop_min.get()) if hasattr(self, 'drop_min') else 0.1
                drop_hi = float(self.drop_max.get()) if hasattr(self, 'drop_max') else 0.4
                if drop_lo > drop_hi:
                    drop_lo, drop_hi = drop_hi, drop_lo
                dropout = trial.suggest_float("dropout", drop_lo, drop_hi, step=0.001)  # 3 decimals

                lr_lo = max(1e-6, 10 ** -(self.lr_min_exp.get()))
                lr_hi = max(1e-4, 10 ** -(self.lr_max_exp.get()))
                if lr_lo > lr_hi:
                    lr_lo, lr_hi = lr_hi, lr_lo
                lr = trial.suggest_float("lr", lr_lo, lr_hi, log=True)
                lr = float(f"{lr:.6g}")  # ~6 significant figures (natural for log-scale LR)
                # Floor LR: extremely small values can luck into low trial loss but underfit peaks
                lr = max(lr, 1e-4)

                try:
                    raw = self.batch_options.get()
                    batch_candidates = [int(x.strip()) for x in raw.split(',') if x.strip().isdigit()]
                except Exception:
                    batch_candidates = [16, 32, 64]
                if not batch_candidates:
                    batch_candidates = [16, 32, 64]

                if focused:
                    # Fixed secondary knobs → search concentrates on units / dropout / lr
                    batch = int(sorted(batch_candidates)[len(batch_candidates) // 2])
                    Hdelta = 0.03
                    l2 = 0.0
                else:
                    batch = trial.suggest_categorical("batch_size", batch_candidates)
                    Hdelta = trial.suggest_float("Hdelta", 0.01, 0.05, step=0.001)  # 3 decimals

                    try:
                        l2_low = float(self.l2_min.get()) if hasattr(self, 'l2_min') else 0.0
                        l2_high = float(self.l2_max.get()) if hasattr(self, 'l2_max') else 0.0
                    except Exception:
                        l2_low, l2_high = 0.0, 0.0

                    try:
                        if l2_low and l2_high and l2_low > l2_high:
                            l2_low, l2_high = l2_high, l2_low
                    except Exception:
                        pass

                    try:
                        use_l2_ui = bool(self.use_l2_var.get()) if hasattr(self, 'use_l2_var') else True
                    except Exception:
                        use_l2_ui = True

                    if use_l2_ui and l2_high and (l2_low < l2_high):
                        try:
                            use_log_ui = bool(self.l2_log_var.get()) if hasattr(self, 'l2_log_var') else (l2_low > 0)
                        except Exception:
                            use_log_ui = (l2_low > 0)

                        use_log = use_log_ui and (l2_low > 0)
                        try:
                            l2 = trial.suggest_float("l2", l2_low, l2_high, log=use_log)
                        except Exception:
                            l2 = trial.suggest_float("l2", l2_low, l2_high, log=False)
                        try:
                            l2 = max(l2_low, min(l2, l2_high))
                            l2 = float(f"{l2:.6g}") if l2 > 0 else 0.0
                        except Exception:
                            pass
                    else:
                        l2 = 0.0

                try:
                    tf.keras.backend.clear_session()
                except Exception:
                    pass

                inp_shape = (X_tr.shape[1], X_tr.shape[2])
                inputs = Input(shape=inp_shape)

                x = LSTM(units1, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None)(inputs)
                x = Dropout(dropout)(x)

                attn_output = MultiHeadAttention(num_heads=8, key_dim=32, dropout=dropout)(x, x)
                x = Add()([x, attn_output])
                x = LayerNormalization()(x)

                x = LSTM(units2, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None)(x)
                x = Dropout(dropout)(x)

                outputs = Dense(1, kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None)(x)

                model = Model(inputs, outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.Huber(delta = Hdelta))

                try:
                    n_trials_local = getattr(self, 'trials_var', None).get() if getattr(self, 'trials_var', None) is not None else '??'
                    txt_start = (f"Starting trial {trial.number+1}/{n_trials_local}: seq={seq_length} "
                                f"units={units1};{units2} drop={dropout:.3f} lr={lr:.1e} batch={batch} Hdelta={Hdelta:.3f}")
                    self.plot_queue.put(('log', txt_start))
                except Exception:
                    pass

                try:
                    elapsed_now = int(time.time() - self.tuning_start_time) if getattr(self, 'tuning_start_time', None) else 0
                    self.plot_queue.put(('update_stats', {
                        'completed': trial.number + 1,
                        'n_trials': n_trials,
                        'best_mse': None,
                        'elapsed_s': elapsed_now,
                    }))
                except Exception:
                    pass

                lc = LivePlotCallback(self)
                st_cb = StopTrainingCallback(self)
                self.plot_queue.put(('clear_epoch_plot',))

                fit_callbacks = [EarlyStopping(patience=30, restore_best_weights=True),
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
                                 lc, st_cb]

                sw_tr = None
                if getattr(self, 'use_peak_weight_var', None) and self.use_peak_weight_var.get():
                    try:
                        sw_tr = self._compute_peak_weights(y_tr)
                    except Exception:
                        sw_tr = None

                model.fit(X_tr, y_tr,
                          epochs=150, batch_size=batch,
                          validation_data=(X_val, y_val),
                          sample_weight=sw_tr,
                          verbose=1,
                          callbacks=fit_callbacks)

                # --- Trial score: 1 - NSE on discharge (m³/s) so ranking matches Model Fit ---
                try:
                    hist_vals = model.history.history.get('val_loss', [])
                    val_mse = float(min(hist_vals)) if hist_vals else float('inf')
                except Exception:
                    val_mse = float('inf')

                try:
                    pred_s = model.predict(X_val, verbose=0)
                    yt = np.asarray(y_val, dtype=np.float64)
                    yp = np.asarray(pred_s, dtype=np.float64)
                    if yt.ndim > 1:
                        yt = yt.reshape(yt.shape[0], -1)[:, -1]
                    else:
                        yt = yt.ravel()
                    if yp.ndim > 1:
                        yp = yp.reshape(yp.shape[0], -1)[:, -1]
                    else:
                        yp = yp.ravel()
                    n = min(len(yt), len(yp))
                    yt, yp = yt[:n], yp[:n]
                    try:
                        yt_m = self.scaler_level.inverse_transform(yt.reshape(-1, 1)).ravel()
                        yp_m = self.scaler_level.inverse_transform(yp.reshape(-1, 1)).ravel()
                    except Exception:
                        yt_m, yp_m = yt, yp
                    denom = float(np.sum((yt_m - np.mean(yt_m)) ** 2))
                    if denom > 1e-12:
                        nse = float(1.0 - np.sum((yt_m - yp_m) ** 2) / denom)
                    else:
                        nse = 0.0
                    if not np.isfinite(nse):
                        nse = -10.0
                    val_loss = float(np.clip(1.0 - nse, -5.0, 20.0))
                except Exception:
                    nse = float('nan')
                    val_loss = float(val_mse) if np.isfinite(val_mse) else 20.0

                # === FIXED: Force live validation graph update after every trial ===
                try:
                    completed = trial.number + 1
                    if np.isfinite(val_loss):
                        best_val = min(trial_plot_best[-1] if trial_plot_best else float('inf'), float(val_loss))
                        trial_plot_x.append(completed)
                        trial_plot_best.append(best_val)
                        self.plot_queue.put(('trial_point', completed, float(val_loss), best_val))
                    else:
                        best_val = trial_plot_best[-1] if trial_plot_best else None

                    elapsed = int(time.time() - self.tuning_start_time) if getattr(self, 'tuning_start_time', None) else 0
                    self.plot_queue.put(('update_stats', {
                        'completed': completed,
                        'n_trials': n_trials,
                        'best_mse': best_val,
                        'elapsed_s': elapsed,
                    }))
                except Exception as e:
                    print("Trial plot update failed:", e)

                try:
                    nse_txt = f"{nse:.3f}" if np.isfinite(nse) else "n/a"
                    mse_txt = f"{val_mse:.6f}" if np.isfinite(val_mse) else "n/a"
                    self.plot_queue.put((
                        'log',
                        f"Trial {trial.number+1} seq={seq_length} units: {units1},{units2}, drop: {dropout:.3f}, "
                        f"lr: {lr:.2e}, batch: {batch}, l2: {l2:.6g}, Hdelta: {Hdelta:.3f} "
                        f"| NSE: {nse_txt} (score={val_loss:.4f}) MSE: {mse_txt}"
                    ))
                except Exception:
                    pass

                return val_loss

            search_mode = str(getattr(self, 'search_mode_var', ttk.StringVar(value='focused')).get())
            use_manual = (search_mode == 'manual') or bool(
                getattr(self, 'use_manual_params_var', ttk.BooleanVar(value=False)).get())
            best_trial_number = None
            if use_manual:
                best = self._get_manual_best_params()
                self.log("Skipping Optuna — using manual LSTM parameters.")
                self.log(f"BEST PARAMETERS (manual; no Optuna trial): {best}")
                try:
                    self.current_trial_label.config(text="Manual hyperparameters → final train")
                except Exception:
                    pass
                self.overall_pb['value'] = 40
            else:
                if search_mode == 'focused':
                    self.log("Search mode: Recommended — tuning lr, dropout, units, and sequence length "
                             "(batch / Hδ / L2 fixed).")
                else:
                    self.log("Search mode: Full — tuning all hyperparameters in the search space.")
                self.study = self._create_or_load_optuna_study()
                study_trials = self.study.trials
                n_prev = len(study_trials)
                if n_prev > 0:
                    self.log(f"Optuna: continuing study with {n_prev} previous trial(s); adding {n_trials} more.")
                else:
                    self.log(f"Optuna: starting fresh study ({n_trials} trials).")
                completed_trials = sorted(
                    (t for t in study_trials
                     if t.value is not None and np.isfinite(t.value)),
                    key=lambda t: t.number,
                )
                best_val = float('inf')
                for past_trial in completed_trials:
                    best_val = min(best_val, float(past_trial.value))
                    trial_plot_x.append(past_trial.number + 1)
                    trial_plot_best.append(best_val)
                if trial_plot_x:
                    self.plot_queue.put(('trial_history', list(trial_plot_x), list(trial_plot_best)))
                self.study.optimize(objective, n_trials=n_trials)

                if self.stop_training:
                    # Offer to keep progress: train final model with best trial so far
                    has_best = False
                    try:
                        bt = getattr(self.study, "best_trial", None)
                        bv = getattr(self.study, "best_value", None)
                        has_best = (
                            bt is not None
                            and bv is not None
                            and np.isfinite(float(bv))
                        )
                    except Exception:
                        has_best = False

                    if has_best:
                        try:
                            nse_approx = 1.0 - float(self.study.best_value)
                            msg = (
                                "Optuna was cancelled.\n\n"
                                f"Best trial so far: #{self.study.best_trial.number + 1}\n"
                                f"  score (1−NSE) = {float(self.study.best_value):.4f}\n"
                                f"  NSE ≈ {nse_approx:.3f}\n"
                                f"  params = {dict(self.study.best_params)}\n\n"
                                "Train the final model with these best parameters?\n"
                                "(Choose No to discard and stop.)"
                            )
                        except Exception:
                            msg = (
                                "Optuna was cancelled, but at least one trial finished.\n\n"
                                "Train the final model with the best parameters found so far?"
                            )
                        if self._ask_user_yesno("Train with best so far?", msg):
                            self.stop_training = False
                            self.log(
                                "Cancel → user chose to train final model with best params so far."
                            )
                        else:
                            self.log("Training cancelled (user declined best-so-far train).")
                            return
                    else:
                        self.log("Training cancelled (no completed trial available).")
                        return

                best = dict(self.study.best_params)
                best_trial_number = int(self.study.best_trial.number) + 1
                # Recommended/focused trials do not store batch/Hδ/L2 in trial.params.
                # Sequence length is included when tuning is enabled.
                try:
                    raw = self.batch_options.get()
                    batch_candidates = [int(x.strip()) for x in raw.split(',') if x.strip().isdigit()]
                except Exception:
                    batch_candidates = [16, 32, 64]
                if not batch_candidates:
                    batch_candidates = [32]
                best.setdefault('batch_size', int(sorted(batch_candidates)[len(batch_candidates) // 2]))
                best.setdefault('Hdelta', 0.03)
                best.setdefault('l2', 0.0)
                try:
                    best.setdefault('seq_length', int(self.seq_var.get()))
                except Exception:
                    best.setdefault('seq_length', int(getattr(self, 'seq_length', 30)))
                self.log(
                    f"BEST PARAMETERS (Optuna Trial #{best_trial_number}; "
                    f"final settings may include defaults): {best}")
                self._apply_best_to_manual(best)
                self.overall_pb['value'] = 60

            best_delta = float(best.get('Hdelta', best.get('delta', 0.05)))
            best_seq = int(best.get('seq_length', getattr(self, 'seq_length', self.seq_var.get())))
            if best_seq != getattr(self, 'seq_length', None):
                self.seq_length = best_seq
                self.train_X, self.train_y = self.create_sequences_for(self.train_df, self.seq_length, self.feature_columns)
                self.val_X, self.val_y = self.create_sequences_for(self.val_df, self.seq_length, self.feature_columns)
                self.log(f"Using best seq_length={self.seq_length} for final training")

            use_seq2seq = self.use_seq2seq_var.get()
            u1 = int(best.get('units1', self.units_min.get()))
            u2 = int(best.get('units2', max(16, u1 // 2)))
            drop_final = float(best.get('dropout', 0.2))
            best_l2 = float(best.get('l2', 0.0))

            if use_seq2seq:
                self.log("Training final Seq2Seq Encoder-Decoder model (one-shot 7-day)...")
                self.lstm_model = self.build_seq2seq_model(
                    seq_length=self.seq_length,
                    n_features=self.train_X.shape[2],
                    units1=u1,
                    units2=u2,
                    dropout=drop_final,
                    l2=best_l2,
                    horizon=7
                )
            else:
                self.log("Training final LSTM (recursive mode)...")
                inputs = Input(shape=(self.train_X.shape[1], self.train_X.shape[2]))
                x = LSTM(u1, return_sequences=True, kernel_regularizer=regularizers.l2(best_l2) if best_l2 > 0 else None)(inputs)
                x = Dropout(drop_final)(x)
                attn_output = MultiHeadAttention(num_heads=8, key_dim=32, dropout=drop_final)(x, x)
                x = Add()([x, attn_output])
                x = LayerNormalization()(x)
                x = LSTM(u2, return_sequences=False, kernel_regularizer=regularizers.l2(best_l2) if best_l2 > 0 else None)(x)
                x = Dropout(drop_final)(x)
                outputs = Dense(1, kernel_regularizer=regularizers.l2(best_l2) if best_l2 > 0 else None)(x)
                self.lstm_model = Model(inputs, outputs)

            best_lr = float(best.get('lr', 1e-3))
            best_batch = int(best.get('batch_size', 32))
            self.lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                                    loss=tf.keras.losses.Huber(delta=best_delta))

            final_epochs = self.final_epochs.get()
            lc_final = LivePlotCallback(self)
            st_cb_final = StopTrainingCallback(self)
            callbacks = [EarlyStopping(patience=50, restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=1),
                         lc_final, st_cb_final]

            if use_seq2seq:
                n_train = len(self.train_y)
                n_val   = len(self.val_y)

                # build decoder arrays aligned with the sequence indices used to create X/y
                def build_decoder_array(df_part, seq_length, horizon=7):
                    rains = np.array(df_part['rainfall'].values, dtype=np.float32)
                    dec = []
                    for i in range(seq_length, len(rains) - horizon + 1):
                        dec.append(rains[i:i + horizon])
                    if len(dec) == 0:
                        return np.zeros((0, horizon, 1), dtype=np.float32)
                    arr = np.array(dec, dtype=np.float32)  # shape (n_samples, horizon)
                    flat = arr.reshape(-1, 1)
                    try:
                        scaled_flat = self.scaler_rain.transform(flat)
                    except Exception:
                        scaled_flat = flat
                    return scaled_flat.reshape(-1, horizon, 1)

                decoder_train = build_decoder_array(self.train_df, self.seq_length, horizon=7)
                decoder_val   = build_decoder_array(self.val_df,   self.seq_length, horizon=7)

                min_train = min(len(self.train_X), len(decoder_train))
                min_val   = min(len(self.val_X), len(decoder_val))

                self.log(f"Seq2Seq training samples: {min_train} (train), {min_val} (val)")

                # Ensure inputs passed as concrete numpy arrays and lists to match model input structure
                train_inputs = [np.asarray(self.train_X[:min_train]), np.asarray(decoder_train[:min_train])]
                try:
                    train_targets = np.asarray(self.train_y[:min_train])
                except Exception:
                    train_targets = self.train_y[:min_train]
                val_inputs = [np.asarray(self.val_X[:min_val]), np.asarray(decoder_val[:min_val])]
                val_targets = np.asarray(self.val_y[:min_val])

                sw_final = None
                if getattr(self, 'use_peak_weight_var', None) and self.use_peak_weight_var.get():
                    try:
                        sw_final = self._compute_peak_weights(train_targets)
                    except Exception:
                        sw_final = None
                    self.log("Peak-weighted training ON (high-flow days get higher weight)")
                else:
                    self.log("Peak-weighted training OFF")
                self.lstm_model.fit(
                    train_inputs,
                    train_targets,
                    epochs=final_epochs,
                    batch_size=best_batch,
                    validation_data=(val_inputs, val_targets),
                    sample_weight=sw_final,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                sw_final = None
                if getattr(self, 'use_peak_weight_var', None) and self.use_peak_weight_var.get():
                    try:
                        sw_final = self._compute_peak_weights(self.train_y)
                    except Exception:
                        sw_final = None
                    self.log("Peak-weighted training ON (high-flow days get higher weight)")
                else:
                    self.log("Peak-weighted training OFF")
                self.lstm_model.fit(self.train_X, self.train_y,
                                    epochs=final_epochs,
                                    batch_size=best_batch,
                                    validation_data=(self.val_X, self.val_y),
                                    sample_weight=sw_final,
                                    callbacks=callbacks, 
                                    verbose=1)

            # ===================== VALIDATION PREDICTION =====================
            try:
                use_seq2seq = self.use_seq2seq_var.get()
                if use_seq2seq:
                    # rebuild aligned decoder_val for validation
                    def build_decoder_array_local(df_part, seq_length, horizon=7):
                        rains = np.array(df_part['rainfall'].values, dtype=np.float32)
                        dec = []
                        for i in range(seq_length, len(rains) - horizon + 1):
                            dec.append(rains[i:i + horizon])
                        if len(dec) == 0:
                            return np.zeros((0, horizon, 1), dtype=np.float32)
                        arr = np.array(dec, dtype=np.float32)
                        flat = arr.reshape(-1, 1)
                        try:
                            scaled_flat = self.scaler_rain.transform(flat)
                        except Exception:
                            scaled_flat = flat
                        return scaled_flat.reshape(-1, horizon, 1)

                    decoder_val = build_decoder_array_local(self.val_df, self.seq_length, horizon=7)
                    val_pred_inputs = [np.asarray(self.val_X), np.asarray(decoder_val)]
                    lstm_val_pred_raw = self.lstm_model.predict(val_pred_inputs, verbose=0)
                    lstm_val_pred = lstm_val_pred_raw[:, -1, 0]
                else:
                    lstm_val_pred = self.lstm_model.predict(self.val_X, verbose=0).ravel()

                try:
                    if use_seq2seq:
                        true_val_scaled = np.array(self.val_y)
                        if true_val_scaled.ndim >= 2:
                            true_val_last_scaled = true_val_scaled[:, -1]
                        else:
                            true_val_last_scaled = true_val_scaled.ravel()
                    else:
                        true_val_last_scaled = np.array(self.val_y).ravel()
                    lstm_val_mse_scaled = mean_squared_error(true_val_last_scaled, lstm_val_pred)
                    lstm_val_mae_scaled = mean_absolute_error(true_val_last_scaled, lstm_val_pred)
                except Exception:
                    lstm_val_mse_scaled = lstm_val_mae_scaled = None

                try:
                    lstm_val_pred_m = self.scaler_level.inverse_transform(lstm_val_pred.reshape(-1, 1)).ravel()
                except Exception:
                    lstm_val_pred_m = np.array(lstm_val_pred).ravel()

                # Ensure we compare the same target (last-step discharge for seq2seq)
                try:
                    if getattr(self, 'use_seq2seq_var', None) and self.use_seq2seq_var.get():
                        true_val = np.array(self.val_y)
                        if true_val.ndim >= 2:
                            true_val_last = true_val[:, -1]
                        else:
                            true_val_last = true_val.ravel()
                    else:
                        true_val_last = np.array(self.val_y).ravel()
                    val_y_m = self.scaler_level.inverse_transform(true_val_last.reshape(-1, 1)).ravel()
                except Exception:
                    try:
                        val_y_m = np.array(self.val_y).ravel()
                    except Exception:
                        val_y_m = np.array([])
                lstm_val_mse_m = mean_squared_error(val_y_m, lstm_val_pred_m)
                lstm_val_mae_m = mean_absolute_error(val_y_m, lstm_val_pred_m)
            except Exception as e:
                self.log(f"Validation prediction failed: {e}")
                lstm_val_pred = None
                lstm_val_mse_scaled = lstm_val_mae_scaled = None
                lstm_val_mse_m = lstm_val_mae_m = None

            # ===================== RANDOM FOREST =====================
            self.overall_pb['value'] = 90
            self.log("Training Random Forest...")
            X_flat = self.train_X.reshape(self.train_X.shape[0], -1)
            try:
                if use_seq2seq:
                    ty = np.array(self.train_y)
                    if ty.ndim >= 2:
                        y_train_rf = ty[:, -1]
                    else:
                        y_train_rf = ty.ravel()
                else:
                    y_train_rf = np.array(self.train_y).ravel()
            except Exception:
                y_train_rf = np.array(self.train_y).ravel()
            
            sw_rf = None
            if getattr(self, 'use_peak_weight_var', None) and self.use_peak_weight_var.get():
                try:
                    sw_rf = self._compute_peak_weights(y_train_rf)
                except Exception:
                    sw_rf = None
            try:
                n_est = int(self.rf_estimators.get()) if getattr(self, 'rf_estimators', None) else 500
            except Exception:
                n_est = 500
            try:
                md = int(self.rf_depth.get()) if getattr(self, 'rf_depth', None) else 30
            except Exception:
                md = 30
            if md <= 0:
                md = None  # unlimited
            self.rf_model = RandomForestRegressor(
                n_estimators=max(10, n_est), max_depth=md, n_jobs=-1, random_state=42)
            self.log(f"RandomForest: n_estimators={max(10, n_est)}, max_depth={md if md else 'None'}")
            self.rf_model.fit(X_flat, y_train_rf, sample_weight=sw_rf)

            try:
                X_val_flat = self.val_X.reshape(self.val_X.shape[0], -1)
                rf_val_pred = self.rf_model.predict(X_val_flat)
                # compare against last-step target for seq2seq or 1d target for recursive
                try:
                    if getattr(self, 'use_seq2seq_var', None) and self.use_seq2seq_var.get():
                        true_val = np.array(self.val_y)
                        if true_val.ndim >= 2:
                            true_val_last = true_val[:, -1]
                        else:
                            true_val_last = true_val.ravel()
                    else:
                        true_val_last = np.array(self.val_y).ravel()
                    rf_val_mse_scaled = mean_squared_error(true_val_last, rf_val_pred)
                    rf_val_mae_scaled = mean_absolute_error(true_val_last, rf_val_pred)
                except Exception:
                    rf_val_mse_scaled = rf_val_mae_scaled = None

                rf_val_pred_m = self.scaler_level.inverse_transform(rf_val_pred.reshape(-1, 1)).ravel()
                try:
                    rf_val_mse_m = mean_squared_error(val_y_m, rf_val_pred_m)
                    rf_val_mae_m = mean_absolute_error(val_y_m, rf_val_pred_m)
                except Exception:
                    rf_val_mse_m = rf_val_mae_m = None
            except Exception:
                rf_val_pred = None
                rf_val_mse_scaled = rf_val_mae_scaled = None
                rf_val_mse_m = rf_val_mae_m = None

            # ===================== ENSEMBLE EVALUATION =====================
            self.log("Evaluating ensemble and calibrating weights...")
            ensemble_scaled = None
            ensemble_m = None
            ens_mse_scaled = ens_mae_scaled = ens_mse_m = ens_mae_m = None
            calibrated_weight = None

            try:
                if lstm_val_pred is not None and rf_val_pred is not None:
                    try:
                        w_initial = float(self.ensemble_lstm_var.get())
                    except Exception:
                        w_initial = 0.85

                    do_calib = bool(getattr(self, 'calibrate_ensemble_var', None) and 
                                  getattr(self.calibrate_ensemble_var, 'get', lambda: False)())

                    if do_calib:
                        self.log("Running ensemble weight calibration (grid search on validation set)...")
                        alphas = np.arange(0.0, 1.01, 0.02)        # Finer grid (step = 0.02)
                        best_alpha = w_initial
                        best_score = float('inf')   # lower is better (composite)
                        best_ens_m = None
                        best_mse = float('inf')
                        best_nse = float('nan')

                        # High-flow mask (upper 20%) — peak-aware calibration
                        try:
                            q_thr = float(np.nanpercentile(val_y_m, 80))
                            hi_mask = val_y_m >= q_thr
                            if int(np.sum(hi_mask)) < 5:
                                hi_mask = np.ones_like(val_y_m, dtype=bool)
                        except Exception:
                            hi_mask = np.ones_like(val_y_m, dtype=bool)

                        for alpha in alphas:
                            ens_m_temp = alpha * lstm_val_pred_m + (1.0 - alpha) * rf_val_pred_m
                            mse_all = mean_squared_error(val_y_m, ens_m_temp)
                            try:
                                mse_hi = mean_squared_error(val_y_m[hi_mask], ens_m_temp[hi_mask])
                            except Exception:
                                mse_hi = mse_all
                            # Composite: balance overall fit and high-flow fit
                            # (pure MSE often under-weights peaks and pulls alpha toward RF)
                            score = 0.5 * mse_all + 0.5 * mse_hi
                            if score < best_score:
                                best_score = score
                                best_mse = mse_all
                                best_alpha = float(alpha)
                                best_ens_m = ens_m_temp
                                denom = float(np.sum((val_y_m - np.mean(val_y_m)) ** 2))
                                best_nse = (1.0 - float(np.sum((val_y_m - ens_m_temp) ** 2) / denom)) if denom > 0 else float('nan')

                        w = best_alpha
                        ensemble_m = best_ens_m
                        self.log(
                            f"Calibrated LSTM weight = {w:.3f} "
                            f"(val MSE={best_mse:.6f}, NSE≈{best_nse:.3f}, "
                            f"objective=0.5·MSE_all+0.5·MSE_highQ)"
                        )
                    else:
                        w = w_initial
                        ensemble_m = w * lstm_val_pred_m + (1.0 - w) * rf_val_pred_m
                        self.log(f"Using user-defined ensemble weight = {w:.3f}")

                    # Final ensemble in scaled space
                    ensemble_scaled = w * lstm_val_pred + (1.0 - w) * rf_val_pred

                    calibrated_weight = float(w)
                    try:
                        self.ensemble_lstm_var.set(calibrated_weight)
                    except Exception:
                        pass

                    self.log(f"Final Ensemble → LSTM: {calibrated_weight:.3f} | RF: {1-calibrated_weight:.3f}")

                    # ===================== SAFE METRIC CALCULATION (Fix for seq2seq) =====================
                    try:
                        # Handle both recursive (1D) and seq2seq (2D) targets safely
                        if getattr(self, 'use_seq2seq_var', None) and self.use_seq2seq_var.get():
                            true_val_last = np.asarray(self.val_y)
                            if true_val_last.ndim >= 2:
                                true_val_last = true_val_last[:, -1]   # Use only the last timestep
                            else:
                                true_val_last = true_val_last.ravel()
                        else:
                            true_val_last = np.asarray(self.val_y).ravel()

                        ens_mse_scaled = mean_squared_error(true_val_last, ensemble_scaled)
                        ens_mae_scaled = mean_absolute_error(true_val_last, ensemble_scaled)
                        ens_mse_m = mean_squared_error(val_y_m, ensemble_m)
                        ens_mae_m = mean_absolute_error(val_y_m, ensemble_m)

                        self.log(f"Ensemble (scaled) MSE: {ens_mse_scaled:.6e} | MAE: {ens_mae_scaled:.6e}")
                        self.log(f"Ensemble (m³/s) RMSE: {np.sqrt(ens_mse_m):.4f} | MAE: {ens_mae_m:.4f}")

                    except Exception as e:
                        self.log(f"Warning: Could not compute ensemble metrics: {e}")
                        ens_mse_scaled = ens_mae_scaled = ens_mse_m = ens_mae_m = None

                    # ===================== DIAGNOSTICS =====================
                    try:
                        lstm_rmse = np.sqrt(mean_squared_error(val_y_m, lstm_val_pred_m)) if 'lstm_val_pred_m' in locals() else None
                        rf_rmse = np.sqrt(mean_squared_error(val_y_m, rf_val_pred_m)) if 'rf_val_pred_m' in locals() else None
                        ens_rmse = np.sqrt(ens_mse_m) if ens_mse_m is not None else None

                        if lstm_rmse and rf_rmse and ens_rmse:
                            best_single = min(lstm_rmse, rf_rmse)
                            improvement = (best_single - ens_rmse) / best_single * 100 if best_single > 0 else 0
                            self.log(f"Ensemble improvement over best single model: {improvement:.1f}%")

                        if abs(calibrated_weight - 0.5) > 0.40:
                            self.log(f"WARNING: Calibrated weight is extreme ({calibrated_weight:.3f}). "
                                     "This may indicate overfitting to the validation set.", color="warning")

                        if len(val_y_m) < 120:
                            self.log(f"NOTE: Small validation set ({len(val_y_m)} samples). "
                                     "Ensemble calibration may be unstable or biased.", color="warning")

                    except Exception:
                        pass

                else:
                    self.log("Warning: Ensemble could not be formed (missing LSTM or RF predictions)")

            except Exception as e:
                self.log(f"Ensemble evaluation error: {e}")
                # Fallback
                if lstm_val_pred is not None:
                    ensemble_scaled = lstm_val_pred
                    ensemble_m = lstm_val_pred_m if 'lstm_val_pred_m' in locals() else None
                elif rf_val_pred is not None:
                    ensemble_scaled = rf_val_pred
                    ensemble_m = rf_val_pred_m if 'rf_val_pred_m' in locals() else None

            # === Log model predictions & metrics into the training log (both scaled & physical units) ===
            try:
                # Sample a small number of validation predictions to display (up to 7)
                sample_n = 7
                def safe_list(x):
                    try:
                        return list(map(float, np.asarray(x).ravel()[:sample_n]))
                    except Exception:
                        return []

                lstm_samps_scaled = safe_list(lstm_val_pred) if lstm_val_pred is not None else []
                rf_samps_scaled = safe_list(rf_val_pred) if rf_val_pred is not None else []
                ens_samps_scaled = safe_list(ensemble_scaled) if ("ensemble_scaled" in locals() and ensemble_scaled is not None) else []

                try:
                    lstm_samps_m = safe_list(lstm_val_pred_m)
                except Exception:
                    lstm_samps_m = []
                try:
                    rf_samps_m = safe_list(rf_val_pred_m)
                except Exception:
                    rf_samps_m = []
                try:
                    ens_samps_m = safe_list(ensemble_m)
                except Exception:
                    ens_samps_m = []

                self.plot_queue.put(('log', "TRAINING COMPLETE! Ready for uncertainty-aware predictions."))
                # Log scaled summaries
                try:
                    if lstm_val_mse_scaled is not None:
                        self.log(f"LSTM (scaled)  MSE: {lstm_val_mse_scaled:.6e}  MAE: {lstm_val_mae_scaled:.6e}")
                    if rf_val_mse_scaled is not None:
                        self.log(f"RF   (scaled)  MSE: {rf_val_mse_scaled:.6e}  MAE: {rf_val_mae_scaled:.6e}")
                    if ens_mse_scaled is not None:
                        self.log(f"Ensemble(scaled)MSE: {ens_mse_scaled:.6e}  MAE: {ens_mae_scaled:.6e}")
                except Exception:
                    pass

                # Log physical summaries
                try:
                    if lstm_val_mse_m is not None:
                        self.log(f"LSTM (m³/s)      RMSE: {np.sqrt(lstm_val_mse_m):.4f}  MAE: {lstm_val_mae_m:.4f}")
                    if rf_val_mse_m is not None:
                        self.log(f"RF   (m³/s)      RMSE: {np.sqrt(rf_val_mse_m):.4f}  MAE: {rf_val_mae_m:.4f}")
                    if ens_mse_m is not None:
                        self.log(f"Ensemble (m³/s)  RMSE: {np.sqrt(ens_mse_m):.4f}  MAE: {ens_mae_m:.4f}")
                except Exception:
                    pass

                # Log a few sample predictions for quick inspection
                try:
                    if lstm_samps_scaled:
                        self.log(f"LSTM preds (scaled, sample): {lstm_samps_scaled}")
                    if rf_samps_scaled:
                        self.log(f"RF   preds (scaled, sample): {rf_samps_scaled}")
                    if ens_samps_scaled:
                        self.log(f"Ens  preds (scaled, sample): {ens_samps_scaled}")
                    if lstm_samps_m:
                        self.log(f"LSTM preds (m³/s, sample): {lstm_samps_m}")
                    if rf_samps_m:
                        self.log(f"RF   preds (m³/s, sample): {rf_samps_m}")
                    if ens_samps_m:
                        self.log(f"Ens  preds (m³/s, sample): {ens_samps_m}")
                except Exception:
                    pass
            except Exception:
                # ensure we still mark completion if logging fails
                try:
                    self.plot_queue.put(('log', "TRAINING COMPLETE! (summary logging error)"))
                except Exception:
                    pass

            self.overall_pb['value'] = 100
            try:
                self.status.config(text="Models ready → Predict with uncertainty Analysis!", bootstyle=SUCCESS)
            except Exception:
                pass

            if best_trial_number is not None:
                try:
                    _bv = float(self.study.best_value)
                    _nse = 1.0 - _bv
                    summary_lines = [
                        f"Optuna best trial: #{best_trial_number}",
                        f"Optuna best (1−NSE): {_bv:.6f}  →  NSE ≈ {_nse:.3f}",
                        f"Best params (Trial #{best_trial_number}): {best}",
                    ]
                except Exception:
                    summary_lines = [
                        f"Optuna best trial: #{best_trial_number}",
                        f"Best params (Trial #{best_trial_number}): {best}",
                    ]
            else:
                summary_lines = [
                    "Parameter source: Manual (no Optuna trial)",
                    f"Parameters: {best}",
                ]
            if lstm_val_mse_m is not None:
                summary_lines += [
                    "\nLSTM final (validation):",
                    f"  RMSE (m³/s): {np.sqrt(lstm_val_mse_m):.4f}",
                    f"  MAE (m³/s): {lstm_val_mae_m:.4f}"
                ]
            if rf_val_mse_m is not None:
                summary_lines += [
                    "\nRandomForest final (validation):",
                    f"  RMSE (m³/s): {np.sqrt(rf_val_mse_m):.4f}",
                    f"  MAE (m³/s): {rf_val_mae_m:.4f}"
                ]
            if ens_mse_m is not None:
                summary_lines += [
                    "\nEnsemble final (validation):",
                    f"  RMSE (m³/s): {np.sqrt(ens_mse_m):.4f}",
                    f"  MAE (m³/s): {ens_mae_m:.4f}"
                ]

            for ln in summary_lines:
                self.log(ln)

            try:
                messagebox.showinfo("Training Summary", "\n".join(summary_lines))
            except Exception:
                pass

        except Exception as e:
            if not self.stop_training:
                self.log(f"Error: {e}")
        finally:
            self.train_btn.config(state=NORMAL)
            self.cancel_btn.config(state=DISABLED)
            # Auto-collapse Advanced Settings after training to free UI space
            try:
                if getattr(self, 'show_settings', None) is not None:
                    self.show_settings.set(False)
                    self.toggle_settings()
            except Exception:
                pass
            try:
                if getattr(self, 'study', None):
                    if getattr(self, 'viz_btn_quick', None) is not None:
                        self.viz_btn_quick.config(state=NORMAL)
                    if getattr(self, 'viz_btn', None) is not None:
                        self.viz_btn.config(state=NORMAL)
                    if getattr(self, 'viz_btn_top', None) is not None:
                        self.viz_btn_top.config(state=NORMAL)
            except Exception:
                pass
            try:
                if getattr(self, 'lstm_model', None) is not None or getattr(self, 'rf_model', None) is not None:
                    try:
                        if getattr(self, 'save_model_btn', None) is not None:
                            self.save_model_btn.config(state=NORMAL)
                        if getattr(self, 'inspect_model_btn', None) is not None:
                            self.inspect_model_btn.config(state=NORMAL)
                        if getattr(self, 'fit_btn', None) is not None:
                            self.fit_btn.config(state=NORMAL)
                        if getattr(self, 'val_unc_btn', None) is not None:
                            self.val_unc_btn.config(state=NORMAL)
                        if getattr(self, 'rf_model', None) is not None and getattr(self, 'rf_shap_btn', None) is not None:
                            self.rf_shap_btn.config(state=NORMAL)
                    except Exception:
                        pass
            except Exception:
                pass

    def _model_uncertainty_signature(self):
        if self.lstm_model is None:
            return None
        digest = hashlib.sha256()
        for weight in self.lstm_model.get_weights():
            arr = np.ascontiguousarray(weight)
            digest.update(str(arr.dtype).encode('utf-8'))
            digest.update(repr(arr.shape).encode('utf-8'))
            digest.update(arr.tobytes())
        settings = (
            tuple(getattr(self, 'feature_columns', []) or []),
            self._get_seq_length(getattr(self, 'seq_var', 30)),
            bool(self.use_seq2seq_var.get()),
            bool(self.same_month_var.get()),
            bool(self.use_aci_var.get()),
            int(self.mc_samples_var.get()),
            int(self.aci_window_var.get()),
            float(self.aci_gamma_var.get()),
            float(self.api_decay_var.get()),
            float(self.elevated_var.get()),
            float(self.moderate_var.get()),
            float(self.high_var.get()),
        )
        digest.update(repr(settings).encode('utf-8'))
        for scaler in (self.scaler_features, self.scaler_level, self.scaler_rain):
            for name in ('data_min_', 'data_max_', 'scale_', 'min_'):
                value = getattr(scaler, name, None)
                if value is not None:
                    arr = np.ascontiguousarray(value)
                    digest.update(str(arr.dtype).encode('utf-8'))
                    digest.update(repr(arr.shape).encode('utf-8'))
                    digest.update(arr.tobytes())
        return digest.hexdigest()

    def _validation_data_signature(self, df=None):
        data = self.df if df is None else df
        if data is None:
            return None
        digest = hashlib.sha256()
        for column in ('date', 'rainfall', 'discharge'):
            if column not in data.columns:
                raise ValueError(f"Cannot fingerprint validation data: missing {column!r}")
            values = pd.util.hash_pandas_object(data[column], index=False).to_numpy()
            digest.update(values.tobytes())
        return digest.hexdigest()

    def _is_val_unc_cache_valid(self, cache):
        if (not isinstance(cache, dict) or cache.get('schema') != 2 or
                not isinstance(cache.get('export'), dict) or
                not isinstance(cache.get('stats'), dict) or
                not isinstance(cache.get('model_signature'), str) or
                not isinstance(cache.get('data_signature'), str)):
            return False, "cache format is invalid"
        required_export = {
            'date', 'obs', 'mean', 'raw_lo95', 'raw_hi95', 'lo95', 'hi95', 'width',
            'inside95', 'p_elevated'
        }
        required_stats = {'thr_e', 'thr_m', 'thr_h'}
        if (not required_export.issubset(cache['export']) or
                not required_stats.issubset(cache['stats'])):
            return False, "cache results are incomplete"
        try:
            lengths = [len(cache['export'][key]) for key in required_export]
            if not lengths or len(set(lengths)) != 1:
                return False, "cache result arrays have inconsistent lengths"
        except Exception:
            return False, "cache result arrays are invalid"
        try:
            model_signature = self._model_uncertainty_signature()
            if (model_signature is not None and
                    cache.get('model_signature') != model_signature):
                return False, "model or uncertainty settings have changed"
            data_signature = self._validation_data_signature()
            if (data_signature is not None and
                    cache.get('data_signature') != data_signature):
                return False, "loaded validation data have changed"
            return True, None
        except Exception as e:
            return False, f"cache validation failed: {e}"

    def _val_unc_cache_path(self, model_signature, data_signature):
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        cache_dir = os.path.join(base, 'FPrediction', 'validation_uncertainty')
        return os.path.join(
            cache_dir, f"{model_signature}_{data_signature}.joblib")

    def _store_val_unc_cache(self, cache):
        valid, reason = self._is_val_unc_cache_valid(cache)
        if not valid:
            self.log(f"Validation uncertainty results not cached ({reason}).")
            return
        path = self._val_unc_cache_path(
            cache['model_signature'], cache['data_signature'])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(cache, path, compress=3)
            self.log("Validation uncertainty results saved for reuse.")
        except Exception as e:
            self.log(f"Validation uncertainty cache save failed: {e}")

    def _restore_val_unc_cache(self, cache):
        valid, reason = self._is_val_unc_cache_valid(cache)
        if not valid:
            self.log(f"Saved validation uncertainty results ignored ({reason}).")
            return False
        self._val_unc_cache = cache
        self._val_unc_export = cache['export']
        self._val_unc_stats = cache['stats']
        try:
            states = cache.get('aci_calibrators', {})
            if states:
                self.aci_calibrators = {
                    int(lead): RollingACI.from_state(state)
                    for lead, state in states.items()
                }
                self.aci_mode = cache.get('aci_mode')
                self.aci_calibrated_through = cache.get('aci_calibrated_through')
                self._aci_not_ready_logged = False
        except Exception as e:
            self._reset_aci_calibration()
            self.log(f"ACI calibration cache could not be restored: {e}")
            return False
        return True

    def _load_persistent_val_unc_cache(self):
        if self.lstm_model is None or getattr(self, 'df', None) is None:
            return False
        try:
            model_signature = self._model_uncertainty_signature()
            data_signature = self._validation_data_signature()
            if model_signature is None or data_signature is None:
                return False
            path = self._val_unc_cache_path(model_signature, data_signature)
            if not os.path.exists(path):
                return False
            cache = joblib.load(path)
            return self._restore_val_unc_cache(cache)
        except Exception as e:
            self.log(f"Validation uncertainty cache load failed: {e}")
            return False

    def _open_cached_val_uncertainty(self):
        cache = getattr(self, '_val_unc_cache', None)
        if cache is None:
            return False
        valid, reason = self._is_val_unc_cache_valid(cache)
        if not valid:
            self.log(f"Saved uncertainty results are stale ({reason}); recalculating is required.")
            self._reset_aci_calibration()
            return False

        if not self._restore_val_unc_cache(cache):
            self._reset_aci_calibration()
            return False
        ex = self._val_unc_export
        st = self._val_unc_stats
        self._show_val_uncertainty_window(
            ex['date'], ex['obs'], ex['mean'], ex['raw_lo95'], ex['raw_hi95'],
            ex['lo95'], ex['hi95'], ex['width'], ex['inside95'],
            st['thr_e'], st['thr_m'], st['thr_h'], ex['p_elevated'])
        self.log("Reopened saved validation uncertainty results without rerunning MC Dropout.")
        return True

    def _reset_aci_calibration(self):
        self.aci_calibrators = {}
        self.aci_mode = None
        self.aci_calibrated_through = None
        self._aci_not_ready_logged = False
        self._val_unc_cache = None
        self._val_unc_export = None
        self._val_unc_stats = None

    def _run_aci_series(self, raw_lower, raw_upper, observed, horizon=1):
        """Calibrate one forecast lead using only outcomes available at each issue time."""
        try:
            window = int(self.aci_window_var.get())
            gamma = float(self.aci_gamma_var.get())
        except Exception as e:
            raise ValueError(f"Invalid ACI settings: {e}")
        if window < 30 or window > 730:
            raise ValueError("ACI window must be between 30 and 730 days")
        if window < 180:
            # Short records: allow smaller windows; coverage guarantees are weaker
            pass

        raw_lower = np.maximum(
            0.0, np.asarray(raw_lower, dtype=np.float64).ravel())
        raw_upper = np.maximum(
            0.0, np.asarray(raw_upper, dtype=np.float64).ravel())
        observed = np.asarray(observed, dtype=np.float64).ravel()
        n = min(len(raw_lower), len(raw_upper), len(observed))
        raw_lower, raw_upper, observed = raw_lower[:n], raw_upper[:n], observed[:n]

        calibrator = RollingACI(window=window, gamma=gamma)
        lower = np.empty(n, dtype=np.float64)
        upper = np.empty(n, dtype=np.float64)
        ready = np.zeros(n, dtype=bool)
        quantiles = np.zeros(n, dtype=np.float64)

        for issue_idx in range(n):
            outcome_idx = issue_idx - int(horizon)
            if outcome_idx >= 0:
                calibrator.observe(
                    raw_lower[outcome_idx],
                    raw_upper[outcome_idx],
                    observed[outcome_idx],
                    lower[outcome_idx],
                    upper[outcome_idx])

            lower[issue_idx], upper[issue_idx], quantiles[issue_idx] = calibrator.interval(
                raw_lower[issue_idx], raw_upper[issue_idx])
            ready[issue_idx] = len(calibrator.scores) >= window

        return lower, upper, ready, quantiles, calibrator

    def _apply_aci_forecast_interval(self, lower, upper, lead, use_seq2seq):
        if not bool(self.use_aci_var.get()):
            return float(lower), float(upper), False
        mode = "seq2seq" if use_seq2seq else "recursive"
        calibrator = None
        if self.aci_mode == mode:
            calibrator = self.aci_calibrators.get(int(lead) if use_seq2seq else 1)

        if calibrator is None:
            if not getattr(self, '_aci_not_ready_logged', False):
                self.log("ACI is enabled but no matching validation calibration is available; using raw MC intervals.")
                self._aci_not_ready_logged = True
            return float(lower), float(upper), False

        try:
            if (calibrator.window != int(self.aci_window_var.get()) or
                    not np.isclose(calibrator.gamma, float(self.aci_gamma_var.get()))):
                if not getattr(self, '_aci_not_ready_logged', False):
                    self.log("ACI settings changed; rerun Validation Uncertainty to recalibrate.")
                    self._aci_not_ready_logged = True
                return float(lower), float(upper), False
        except Exception as e:
            self.log(f"ACI settings could not be applied: {e}")
            return float(lower), float(upper), False

        if len(calibrator.scores) < calibrator.window:
            if not getattr(self, '_aci_not_ready_logged', False):
                self.log(
                    f"ACI burn-in incomplete ({len(calibrator.scores)}/{calibrator.window} scores); "
                    "using raw MC intervals.")
                self._aci_not_ready_logged = True
            return float(lower), float(upper), False

        calibrated_lower, calibrated_upper, _ = calibrator.interval(lower, upper)
        return max(0.0, calibrated_lower), calibrated_upper, True

    def mc_dropout_predict(self, seq_scaled, n_samples=100):
        """
        Improved MC Dropout prediction.
        Returns both mean and full sample distribution for better uncertainty quantification.
        """
        if self.lstm_model is None:
            return None, None, None

        preds = []
        seq_tensor = tf.convert_to_tensor(seq_scaled[np.newaxis, ...], dtype=tf.float32)
        
        for _ in range(n_samples):
            pred_tensor = self.lstm_model(seq_tensor, training=True)  # dropout active
            pred = float(pred_tensor.numpy()[0, 0])
            preds.append(pred)
        
        samples = np.array(preds)
        mean_scaled = float(np.mean(samples))
        std_scaled = float(np.std(samples, ddof=1))  # sample standard deviation
        
        return mean_scaled, std_scaled, samples


    def _forecast_anchor_date(self):
        """Last observed date in loaded series (not wall-clock today)."""
        try:
            if getattr(self, 'df', None) is not None and len(self.df) and 'date' in self.df.columns:
                return pd.Timestamp(self.df['date'].iloc[-1]).to_pydatetime()
        except Exception:
            pass
        try:
            if getattr(self, 'raw_df', None) is not None and len(self.raw_df) and 'date' in self.raw_df.columns:
                return pd.Timestamp(self.raw_df['date'].iloc[-1]).to_pydatetime()
        except Exception:
            pass
        return datetime.now()

    def predict_flood(self):
        if not getattr(self, 'lstm_model', None) and not getattr(self, 'rf_model', None):
            messagebox.showwarning("Not Ready", "Train or load at least one model first!")
            return
        if getattr(self, 'df', None) is None:
            messagebox.showwarning("No Data", "Load CSV data first (history required for prediction).")
            return

        try:
            rains = [float(x.strip()) for x in self.rain_input.get().split(",")]
            if len(rains) != 7:
                raise ValueError("Enter exactly 7 comma-separated values")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        self.predict_btn.config(state=DISABLED)
        self.export_btn.config(state=DISABLED)
        self.log("Prediction requested by user...")

        total_days = len(rains)

        # initialize progress UI (make sure bar resets before worker starts)
        try:
            self.overall_pb['maximum'] = total_days
            self.overall_pb['value'] = 0
        except Exception:
            pass
        try:
            self.progress_label.config(text="Starting prediction...")
        except Exception:
            pass

        def worker(rains_list):
            try:
                self._rf_inputs_acc = []
                use_seq2seq = self.use_seq2seq_var.get()
                hist_df = self.df.copy()
                seq_length = self._get_seq_length(getattr(self, 'seq_var', 30))
                current_disc = hist_df['discharge'].iloc[-1]

                predictions = []
                uncertainties = []
                lower_95_list = []
                upper_95_list = []
                raw_lower_95_list = []
                raw_upper_95_list = []
                aci_applied_list = []
                confidences = []
                prob_elevated = []
                prob_moderate = []
                prob_high = []
                explanation_texts = []
                top1_names = top2_names = top3_names = top4_names = []
                top1_vals = top2_vals = top3_vals = top4_vals = []
                past_days = 6
                past_imp_cols = [[] for _ in range(past_days)]

                if use_seq2seq:
                    self.log("=== Using Seq2Seq One-Shot Prediction (all 7 days at once) ===")
                    # Prepare inputs once (scale future rainfall to match training scaler)
                    try:
                        flat_future = np.array(rains_list, dtype=np.float32).reshape(-1, 1)
                        scaled_flat = self.scaler_rain.transform(flat_future)
                        future_rain = scaled_flat.reshape(1, 7, 1)
                    except Exception:
                        future_rain = np.array(rains_list).reshape(1, 7, 1).astype(np.float32)
                    full_seq_scaled = self.scaler_features.transform(
                        hist_df.tail(seq_length)[self.feature_columns].values.astype(np.float32)
                    )
                    seq_input = full_seq_scaled[np.newaxis, ...]   # (1, seq_length, n_features)

                    n_mc = max(10, int(self.mc_samples_var.get()))

                    # MC Dropout for uncertainty
                    mc_samples = []
                    for _ in range(n_mc):
                        pred = self.lstm_model([seq_input, future_rain], training=True)[0].numpy().flatten()
                        mc_samples.append(pred)
                    ens_samples_scaled = np.array(mc_samples)          # (n_mc, 7)
                    mean_scaled = np.mean(ens_samples_scaled, axis=0)

                    for i in range(7):
                        disc_cms = float(self.scaler_level.inverse_transform([[mean_scaled[i]]])[0, 0])
                        ens_meters = self.scaler_level.inverse_transform(ens_samples_scaled[:, i].reshape(-1, 1)).flatten()

                        raw_lower_95 = max(0.0, float(np.percentile(ens_meters, 2.5)))
                        raw_upper_95 = max(0.0, float(np.percentile(ens_meters, 97.5)))
                        lower_95, upper_95, aci_applied = self._apply_aci_forecast_interval(
                            raw_lower_95, raw_upper_95, i + 1, use_seq2seq=True)
                        uncertainty_m = (upper_95 - lower_95) / 2.0

                        # Confidence
                        std_scaled = float(np.std(ens_samples_scaled[:, i], ddof=1))
                        relative_unc = std_scaled / (abs(mean_scaled[i]) + 0.01)
                        conf = max(0, min(100, 100 * np.exp(-relative_unc)))

                        # Risk probabilities
                        high_t = float(self.high_var.get())
                        med_t = float(self.moderate_var.get())
                        elev_t = float(self.elevated_var.get())
                        p_high = float((ens_meters > high_t).mean())
                        p_med = float((ens_meters > med_t).mean())
                        p_elev = float((ens_meters > elev_t).mean())

                        predictions.append(disc_cms)
                        uncertainties.append(uncertainty_m)
                        raw_lower_95_list.append(raw_lower_95)
                        raw_upper_95_list.append(raw_upper_95)
                        lower_95_list.append(lower_95)
                        upper_95_list.append(upper_95)
                        aci_applied_list.append(aci_applied)
                        confidences.append(conf)
                        prob_high.append(p_high)
                        prob_moderate.append(p_med)
                        prob_elevated.append(p_elev)

                        # XAI
                        expl = self.explain_prediction(full_seq_scaled) if self.lstm_model is not None else {}
                        top_features = sorted(expl.get('feature', {}).items(), key=lambda x: x[1], reverse=True)[:4]
                        past_importance = expl.get('timestep', np.zeros(past_days))[:past_days][::-1]

                        expl_obj = {
                            'day_idx': i+1,
                            'date': hist_df['date'].iloc[-1] + timedelta(days=i+1),
                            'disc_cms': disc_cms,
                            'top_features': top_features,
                            'past_importance': past_importance
                        }
                        explanation_texts.append(expl_obj)

                else:
                    # ==================== ORIGINAL RECURSIVE MODE ====================
                    concurrent = bool(getattr(self, 'same_month_var', ttk.BooleanVar(value=False)).get())
                    mode_txt = "Concurrent (target-day rain)" if concurrent else "Next-day"
                    self.plot_queue.put(('log', f"Using recursive prediction ({mode_txt}, one day at a time)"))
                    current_hist = hist_df.tail(seq_length).copy()

                    for day_idx, rain in enumerate(rains_list, 1):
                        next_date = current_hist['date'].iloc[-1] + timedelta(days=1)
                        next_row = pd.DataFrame([{'date': next_date, 'rainfall': rain, 'discharge': current_disc}])

                        temp_df = pd.concat([current_hist, next_row], ignore_index=True)
                        temp_df['rain_cum3']  = temp_df['rainfall'].rolling(3, min_periods=1).sum()
                        temp_df['rain_cum7']  = temp_df['rainfall'].rolling(7, min_periods=1).sum()
                        temp_df['rain_cum14'] = temp_df['rainfall'].rolling(14, min_periods=1).sum()
                        temp_df['disc_diff'] = temp_df['discharge'].diff().fillna(0)
                        temp_df['disc_lag1'] = temp_df['discharge'].shift(1).bfill()
                        temp_df['wet_season'] = temp_df['date'].dt.month.isin([6,7,8,9,10,11]).astype(float)

                        if 'api' in self.feature_columns:
                            decay = float(self.api_decay_var.get()) if hasattr(self, 'api_decay_var') else 0.9
                            api_vals = []
                            prev = 0.0
                            for r in temp_df['rainfall'].fillna(0):
                                cur = decay * prev + float(r)
                                api_vals.append(cur)
                                prev = cur
                            temp_df['api'] = api_vals

                        feature_cols = list(self.feature_columns)
                        full_seq_scaled = self.scaler_features.transform(
                            temp_df[feature_cols].tail(seq_length).values.astype(np.float32)
                        ).copy()
                        if concurrent:
                            for j in self._discharge_feature_indices(feature_cols):
                                full_seq_scaled[-1, j] = 0.0

                        n_mc = max(10, int(self.mc_samples_var.get()))

                        lstm_mean_scaled = lstm_std_scaled = None
                        lstm_samples_scaled = None
                        rf_pred_scaled = None

                        if self.lstm_model is not None:
                            try:
                                lstm_mean_scaled, lstm_std_scaled, lstm_samples_scaled = self.mc_dropout_predict(
                                    full_seq_scaled, n_samples=n_mc)
                            except Exception as e:
                                self.plot_queue.put(('log', f"MC Dropout failed day {day_idx}: {e}"))

                        if self.rf_model is not None:
                            try:
                                _rf_x = full_seq_scaled.flatten().reshape(1, -1)
                                rf_pred_scaled = float(self.rf_model.predict(_rf_x)[0])
                                try:
                                    if not hasattr(self, '_rf_inputs_acc') or self._rf_inputs_acc is None:
                                        self._rf_inputs_acc = []
                                    self._rf_inputs_acc.append(_rf_x.ravel().astype(np.float64).copy())
                                except Exception:
                                    pass
                            except Exception:
                                pass

                        # Robust ensemble weight with clamping
                        try:
                            w = float(self.ensemble_lstm_var.get())
                            w = max(0.0, min(1.0, w))  # Safety clamp
                        except Exception:
                            w = 0.85

                        if day_idx == 1:  # Log only once per prediction
                            self.log(f"Using ensemble weight → LSTM: {w:.3f} | RF: {1-w:.3f}")
                            if bool(self.use_aci_var.get()):
                                self.log(
                                    "ACI uses the one-step normalized score buffer across recursive forecast leads; "
                                    "rerun Validation Uncertainty after adding actual outcomes to refresh it.")

                        if lstm_mean_scaled is not None and rf_pred_scaled is not None:
                            ensemble_mean_scaled = w * lstm_mean_scaled + (1.0 - w) * rf_pred_scaled
                            ens_samples_scaled = (w * lstm_samples_scaled + (1.0 - w) * rf_pred_scaled) if lstm_samples_scaled is not None else np.array([ensemble_mean_scaled])
                        elif lstm_mean_scaled is not None:
                            ensemble_mean_scaled = lstm_mean_scaled
                            ens_samples_scaled = lstm_samples_scaled if lstm_samples_scaled is not None else np.array([lstm_mean_scaled])
                        elif rf_pred_scaled is not None:
                            ensemble_mean_scaled = rf_pred_scaled
                            ens_samples_scaled = np.array([rf_pred_scaled])
                        else:
                            ensemble_mean_scaled = 0.0
                            ens_samples_scaled = np.array([0.0])

                        disc_cms = float(self.scaler_level.inverse_transform([[ensemble_mean_scaled]])[0, 0])

                        # Keep predictive intervals consistent with LSTM-only validation MC Dropout.
                        try:
                            interval_samples_scaled = (
                                lstm_samples_scaled if lstm_samples_scaled is not None
                                else np.array([ensemble_mean_scaled], dtype=np.float64))
                            ens_meters = self.scaler_level.inverse_transform(
                                np.asarray(interval_samples_scaled).reshape(-1, 1)).flatten()
                            raw_lower_95 = max(0.0, float(np.percentile(ens_meters, 2.5)))
                            raw_upper_95 = max(0.0, float(np.percentile(ens_meters, 97.5)))
                            if lstm_samples_scaled is not None:
                                lower_95, upper_95, aci_applied = self._apply_aci_forecast_interval(
                                    raw_lower_95, raw_upper_95, day_idx, use_seq2seq=False)
                            else:
                                lower_95, upper_95 = raw_lower_95, raw_upper_95
                                aci_applied = False
                            uncertainty_m = float((upper_95 - lower_95) / 2.0)
                        except Exception:
                            raw_lower_95 = raw_upper_95 = max(0.0, disc_cms)
                            lower_95 = upper_95 = raw_lower_95
                            uncertainty_m = 0.0
                            aci_applied = False

                        # Confidence
                        try:
                            ensemble_std_scaled = float(np.std(ens_samples_scaled, ddof=1))
                            relative_unc = ensemble_std_scaled / (np.abs(ensemble_mean_scaled) + 0.01)
                            conf = max(0, min(100, 100 * np.exp(-relative_unc)))
                        except Exception:
                            conf = 50.0

                        # Probabilities
                        try:
                            high_t = float(self.high_var.get())
                            med_t = float(self.moderate_var.get())
                            elev_t = float(self.elevated_var.get())
                            p_high = float((ens_meters > high_t).mean())
                            p_med = float((ens_meters > med_t).mean())
                            p_elev = float((ens_meters > elev_t).mean())
                        except Exception:
                            p_high = p_med = p_elev = 0.0

                        predictions.append(disc_cms)
                        uncertainties.append(uncertainty_m)
                        raw_lower_95_list.append(raw_lower_95)
                        raw_upper_95_list.append(raw_upper_95)
                        lower_95_list.append(lower_95)
                        upper_95_list.append(upper_95)
                        aci_applied_list.append(aci_applied)
                        confidences.append(conf)
                        prob_high.append(p_high)
                        prob_moderate.append(p_med)
                        prob_elevated.append(p_elev)

                        # XAI
                        try:
                            expl = self.explain_prediction(full_seq_scaled) if self.lstm_model is not None else {}
                        except Exception:
                            expl = {'timestep': np.zeros(past_days), 'feature': {}}

                        top_features = sorted(expl.get('feature', {}).items(), key=lambda x: x[1], reverse=True)[:4]
                        past_importance = expl.get('timestep', np.zeros(past_days))[:past_days][::-1]

                        expl_obj = {
                            'day_idx': day_idx,
                            'date': next_date,
                            'disc_cms': disc_cms,
                            'top_features': top_features,
                            'past_importance': past_importance
                        }
                        explanation_texts.append(expl_obj)

                        current_disc = disc_cms
                        current_hist = temp_df.copy()

                        self.plot_queue.put(('progress', day_idx, total_days, f"Predicted {day_idx}/{total_days}"))

                # ===================== UNCERTAINTY DIAGNOSTICS =====================
                try:
                    pi_widths = np.array(upper_95_list) - np.array(lower_95_list)
                    avg_pi_width = float(np.mean(pi_widths))
                    self.log(f"Prediction Summary - Avg 95% PI width: ±{avg_pi_width/2:.2f} m³/s")

                    mean_pred = float(np.mean(predictions))
                    if avg_pi_width < 8.0 and mean_pred > 30.0:
                        self.log("WARNING: Uncertainty intervals appear narrow relative to prediction magnitude. "
                                 "Possible under-dispersion in uncertainty estimation.", color="warning")
                except Exception as e:
                    self.log(f"Could not compute uncertainty diagnostics: {e}")

                # ==================== UPDATE UI (common for both modes) ====================
                def finish_ui():
                    try:
                        self.last_predictions = {
                            'Date': [(self._forecast_anchor_date() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(7)],
                            'Rainfall (mm)': rains_list,
                            'Predicted Discharge (m³/s)': [round(p, 3) for p in predictions],
                            'Uncertainty (±m³/s)': [round(u, 3) for u in uncertainties],
                            '95% PI Lower (m³/s)': [round(l, 3) for l in lower_95_list],
                            '95% PI Upper (m³/s)': [round(u, 3) for u in upper_95_list],
                            'Raw MC 95% PI Lower (m³/s)': [
                                round(l, 3) for l in raw_lower_95_list],
                            'Raw MC 95% PI Upper (m³/s)': [
                                round(u, 3) for u in raw_upper_95_list],
                            'Interval Calibration': [
                                'Rolling ACI' if applied else 'Raw MC'
                                for applied in aci_applied_list
                            ],
                            'Confidence (%)': [round(c, 1) for c in confidences],
                            'Risk Level': [self.get_risk_label(p) for p in predictions],
                            'P(>High)': [round(p*100, 1) for p in prob_high],
                            'P(>Moderate)': [round(p*100, 1) for p in prob_moderate],
                            'P(>Elevated)': [round(p*100, 1) for p in prob_elevated],
                            'Explanation': explanation_texts
                        }

                        if top1_names:
                            self.last_predictions.update({
                                'Top1': top1_names, 'Top1_val': top1_vals,
                                'Top2': top2_names, 'Top2_val': top2_vals,
                                'Top3': top3_names, 'Top3_val': top3_vals,
                                'Top4': top4_names, 'Top4_val': top4_vals,
                            })
                            for idx in range(past_days):
                                self.last_predictions[f'PastDay_{idx+1}_imp'] = past_imp_cols[idx]

                        # Display results
                        self.result_text.delete(1.0, END)
                        for i in range(7):
                            try:
                                expl = explanation_texts[i]
                                date = expl['date'].strftime("%Y-%m-%d")
                            except Exception:
                                date = (self._forecast_anchor_date() + timedelta(days=i+1)).strftime("%Y-%m-%d")

                            risk = self.get_risk_label(predictions[i])
                            if "HIGH" in risk:
                                color_tag = "risk_high"
                            elif "MODERATE" in risk:
                                color_tag = "risk_med"
                            elif "ELEVATED" in risk:
                                color_tag = "risk_elevated"
                            else:
                                color_tag = "risk_low"

                            self.result_text.insert(END, f"Day {i+1} ({date})\n", "day")
                            try:
                                self.result_text.insert(END, "   → ")
                                self.result_text.insert(END, f"{predictions[i]:.3f}", 'pred_val')
                                interval_label = "ACI 95% PI" if aci_applied_list[i] else "raw 95% PI"
                                self.result_text.insert(END, f" m³/s    {interval_label}: [")
                                self.result_text.insert(END, f"{lower_95_list[i]:.3f}", 'pi_val')
                                self.result_text.insert(END, " – ")
                                self.result_text.insert(END, f"{upper_95_list[i]:.3f}", 'pi_val')
                                self.result_text.insert(END, "]    ± ")
                                self.result_text.insert(END, f"{uncertainties[i]:.3f}\n", 'pi_val')
                            except Exception:
                                interval_label = "ACI 95% PI" if aci_applied_list[i] else "raw 95% PI"
                                self.result_text.insert(END, f"   → {predictions[i]:.3f} m³/s    {interval_label}: [{lower_95_list[i]:.3f} – {upper_95_list[i]:.3f}]    ± {uncertainties[i]:.3f}\n")

                            # Risk and confidence
                            try:
                                conf_val = float(confidences[i])
                                conf_tag = 'confidence_high' if conf_val >= 75 else 'confidence_med' if conf_val >= 40 else 'confidence_low'
                            except Exception:
                                conf_tag = 'confidence_low'
                            self.result_text.insert(END, "   Risk: ")
                            self.result_text.insert(END, f"{risk}", color_tag)
                            self.result_text.insert(END, "  |  Confidence: ")
                            self.result_text.insert(END, f"{conf_val:.1f}%\n", conf_tag)

                            # Probabilities
                            try:
                                ph = prob_high[i] * 100
                                pm = prob_moderate[i] * 100
                                pe = prob_elevated[i] * 100
                                ht = float(self.high_var.get())
                                mt = float(self.moderate_var.get())
                                et = float(self.elevated_var.get())
                                self.result_text.insert(END, f"   P(>{ht}): {ph:.1f}%   P(>{mt}): {pm:.1f}%   P(>{et}): {pe:.1f}%\n", color_tag)
                            except Exception:
                                pass

                            # XAI
                            expl = explanation_texts[i]
                            try:
                                header = f"\nWhy Day {i+1} ({expl['date'].strftime('%b %d')}) discharge is {expl['disc_cms']:.2f} (m³/s)?\n"
                            except Exception:
                                header = f"\nWhy Day {i+1} discharge is {predictions[i]:.2f} (m³/s)?\n"
                            self.result_text.insert(END, header, "xai_hdr")
                            self.result_text.insert(END, "Strongest drivers:\n", "xai")
                            max_feat_val = expl['top_features'][0][1] if expl['top_features'] else 0.0
                            for feat, val in expl['top_features']:
                                nfilled = int(12 * val / max_feat_val) if max_feat_val > 0 else 0
                                bars = "█" * min(12, nfilled) + "░" * (12 - min(12, nfilled))
                                self.result_text.insert(END, f"  • {feat:12}: ", 'feat_name')
                                self.result_text.insert(END, bars, 'bar')
                                self.result_text.insert(END, f" ({val:.4f})\n", 'feat_val')

                            self.result_text.insert(END, "\nMost influential past days:\n", "xai")
                            past_importance = expl.get('past_importance', np.zeros(6))
                            max_day_val = float(np.max(past_importance)) if len(past_importance) > 0 else 0.0
                            for i_d, val in enumerate(past_importance):
                                date_str = (self._forecast_anchor_date() - timedelta(days=i_d+1)).strftime("%b %d")
                                nfilled = int(10 * val / max_day_val) if max_day_val > 0 else 0
                                bars = "█" * min(10, nfilled) + "░" * (10 - min(10, nfilled))
                                self.result_text.insert(END, f"  ", 'past_day')
                                self.result_text.insert(END, date_str, 'past_day')
                                self.result_text.insert(END, ": ")
                                self.result_text.insert(END, bars, 'bar')
                                self.result_text.insert(END, f" ({val:.4f})\n", 'feat_val')

                        # Compact inline preview (full graph: Open Forecast Graph)
                        plot_frame = ttk.Frame(self.result_text)
                        self.result_text.window_create(END, window=plot_frame)
                        self.result_text.insert(END, "\n\n")
                        fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
                        days = [(self._forecast_anchor_date() + timedelta(days=i+1)).strftime("%b %d") for i in range(7)]
                        ax.plot(days, predictions, 'o-', linewidth=3, markersize=8, color='#00ff99', label="AI Forecast")
                        if any(aci_applied_list):
                            ax.fill_between(
                                days, lower_95_list, upper_95_list, alpha=0.20,
                                color='#77b7df', label='ACI-calibrated 95% PI')
                            ax.fill_between(
                                days, raw_lower_95_list, raw_upper_95_list, alpha=0.48,
                                color='#174a70', label='Raw MC 95% PI')
                        else:
                            ax.fill_between(
                                days, raw_lower_95_list, raw_upper_95_list, alpha=0.48,
                                color='#174a70', label='Raw MC 95% PI')

                        try:
                            ax.axhline(float(self.high_var.get()), color='red', linestyle='--', linewidth=2, label='Flood Level')
                            ax.axhline(float(self.moderate_var.get()), color='orange', linestyle='--', linewidth=1.5, label='Warning Level')
                            ax.axhline(float(self.elevated_var.get()), color="#848D00", linestyle='--', linewidth=1, label='Elevated')
                        except Exception:
                            pass

                        ax.set_title("7-Day Flood Forecast", fontsize=13, color="#00ff99")
                        ax.set_ylabel("Discharge (m³/s)")
                        ax.legend(loc='best', fontsize=8)
                        ax.grid(True, alpha=0.4)
                        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                        try:
                            fig.subplots_adjust(bottom=0.22)
                        except Exception:
                            pass
                        FigureCanvasTkAgg(fig, plot_frame).get_tk_widget().pack(fill=X, expand=False)
                        fig.canvas.draw()

                        self.export_btn.config(state=NORMAL)
                        try:
                            self.forecast_graph_btn.config(state=NORMAL)
                        except Exception:
                            pass
                        try:
                            acc = getattr(self, '_rf_inputs_acc', None)
                            if acc:
                                self.last_rf_inputs = np.asarray(acc, dtype=np.float64)
                                self.rf_shap_btn.config(state=NORMAL)
                            elif self.rf_model is not None and getattr(self, 'val_X', None) is not None:
                                self.rf_shap_btn.config(state=NORMAL)
                        except Exception:
                            pass
                        self.plot_queue.put(('progress', total_days, total_days, 'Prediction complete'))

                    finally:
                        self.predict_btn.config(state=NORMAL)

                self.root.after(0, finish_ui)

            except Exception as e:
                self.plot_queue.put(('log', f"Prediction error: {e}"))
                try:
                    self.predict_btn.config(state=NORMAL)
                except Exception:
                    pass

        threading.Thread(target=worker, args=(rains,), daemon=True).start()

    def _rf_input_feature_names(self):
        """Names for flattened RF inputs: feature[t], feature[t-1], ..."""
        cols = list(getattr(self, 'feature_columns', None) or ['rainfall', 'discharge'])
        try:
            seq = int(getattr(self, 'seq_length', None) or self.seq_var.get())
        except Exception:
            seq = 30
        names = []
        for t in range(seq):
            lag = seq - 1 - t
            for c in cols:
                short = c
                if lag == 0:
                    names.append(f"{short}[t]")
                else:
                    names.append(f"{short}[t-{lag}]")
        return names


    def suggest_risk_thresholds(self):
        """Suggest Elevated / Moderate / High from observed discharge percentiles."""
        series = None
        try:
            if getattr(self, 'df', None) is not None and 'discharge' in self.df.columns:
                series = self.df['discharge'].astype(float).dropna().values
            elif getattr(self, 'raw_df', None) is not None and 'discharge' in self.raw_df.columns:
                series = self.raw_df['discharge'].astype(float).dropna().values
        except Exception:
            series = None
        if series is None or len(series) < 10:
            messagebox.showwarning(
                "No data",
                "Load a CSV with discharge first so thresholds can be estimated.")
            return
        try:
            # Percentile-based (robust for skewed hydrographs)
            p75 = float(np.nanpercentile(series, 75))
            p90 = float(np.nanpercentile(series, 90))
            p95 = float(np.nanpercentile(series, 95))
            # Ensure strictly increasing and not tiny
            elev = max(p75, 0.0)
            mod = max(p90, elev * 1.05 + 1e-6)
            high = max(p95, mod * 1.05 + 1e-6)
            # Round to 1 decimal for UI
            elev, mod, high = round(elev, 1), round(mod, 1), round(high, 1)
            self.elevated_var.set(elev)
            self.moderate_var.set(mod)
            self.high_var.set(high)
            try:
                self.validate_thresholds()
            except Exception:
                pass
            mu = float(np.nanmean(series))
            sd = float(np.nanstd(series))
            self.log(
                f"Suggested risk thresholds from data percentiles: "
                f"Elevated>{elev} (P75), Moderate>{mod} (P90), High>{high} (P95). "
                f"Series mean={mu:.2f}, std={sd:.2f} m³/s (N={len(series)})."
            )
            messagebox.showinfo(
                "Thresholds suggested",
                f"From observed discharge:\n\n"
                f"  Elevated > {elev} m³/s  (75th percentile)\n"
                f"  Moderate > {mod} m³/s  (90th percentile)\n"
                f"  High     > {high} m³/s  (95th percentile)\n\n"
                f"Mean={mu:.2f}, Std={sd:.2f} m³/s\n"
                f"Adjust manually if station flood levels are known.")
        except Exception as e:
            messagebox.showerror("Suggest failed", str(e))


    def show_rf_shap(self):
        """TreeSHAP explanation of the Random Forest (fast). Forecast steps if available, else validation."""
        if self.rf_model is None:
            messagebox.showwarning("No RF model", "Train or load models first (RF required).")
            return
        try:
            import shap
        except ImportError:
            messagebox.showerror(
                "SHAP not installed",
                "Install the package then restart:\n\npip install shap")
            return

        try:
            X_rf = getattr(self, 'last_rf_inputs', None)
            step_labels = None
            source = "forecast"
            if X_rf is None or (hasattr(X_rf, '__len__') and len(X_rf) == 0):
                if getattr(self, 'val_X', None) is None or len(self.val_X) == 0:
                    messagebox.showwarning(
                        "No inputs",
                        "Run Predict with Uncertainty first (recursive mode),\n"
                        "or train so validation sequences exist.")
                    return
                X_rf = self.val_X.reshape(self.val_X.shape[0], -1)
                source = "validation"
                step_labels = [f"Val #{i+1}" for i in range(len(X_rf))]
                self.log("RF SHAP: using validation sequences (no forecast RF inputs stored).")
            else:
                X_rf = np.asarray(X_rf, dtype=np.float64)
                if X_rf.ndim == 1:
                    X_rf = X_rf.reshape(1, -1)
                if self.last_predictions and self.last_predictions.get('Date'):
                    step_labels = list(self.last_predictions['Date'])[:len(X_rf)]
                else:
                    step_labels = [f"Step {i+1}" for i in range(len(X_rf))]

            names = self._rf_input_feature_names()
            if len(names) != X_rf.shape[1]:
                names = [f"f{i}" for i in range(X_rf.shape[1])]

            self.log(f"RF SHAP: computing TreeExplainer on {len(X_rf)} sample(s), {X_rf.shape[1]} features...")
            explainer = shap.TreeExplainer(self.rf_model)
            sv = explainer.shap_values(X_rf)
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv, dtype=np.float64)
            if sv.ndim == 1:
                sv = sv.reshape(1, -1)

            try:
                q_scale = float(self.scaler_level.data_max_[0] - self.scaler_level.data_min_[0])
                if not np.isfinite(q_scale) or q_scale <= 0:
                    q_scale = 1.0
            except Exception:
                q_scale = 1.0
            sv_m = sv * q_scale

            mean_abs = np.mean(np.abs(sv_m), axis=0)
            order = np.argsort(mean_abs)[::-1]
            top_k = int(min(12, len(order)))
            top_idx = [int(i) for i in order[:top_k]]
            labels_top = [str(names[i]) for i in top_idx]
            mean_top = [float(mean_abs[i]) for i in top_idx]

            # Limit signed panels
            n_steps = int(min(len(X_rf), 4))

            win = ttk.Toplevel(self.root)
            win.title("RF SHAP – Random Forest explanation")
            win.geometry("1000x680")
            win.minsize(640, 420)

            body, _sc = self._scrollable_container(win)

            # --- Summary bar chart (mean |SHAP|) ---
            fig1, ax1 = plt.subplots(figsize=(9.5, 3.2))
            y_pos = list(range(top_k))
            ax1.barh(y_pos, list(reversed(mean_top)), color="#2a9d8f")
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(list(reversed(labels_top)), fontsize=9)
            ax1.set_xlabel("Mean |SHAP| (approx. m³/s)")
            ax1.set_title(f"RF feature importance (TreeSHAP) — {source} · top {top_k}", fontsize=12)
            ax1.grid(True, alpha=0.25)
            fig1.tight_layout()
            c1 = FigureCanvasTkAgg(fig1, master=body)
            c1.draw()
            c1.get_tk_widget().pack(fill=X, expand=False, padx=8, pady=6)

            # --- Signed SHAP for a few steps ---
            for s in range(n_steps):
                vals = [float(sv_m[s, i]) for i in top_idx]
                colors = ["#e76f51" if v < 0 else "#264653" for v in vals]
                fig_s, ax_s = plt.subplots(figsize=(9.5, 2.6))
                ax_s.barh(y_pos, list(reversed(vals)), color=list(reversed(colors)))
                ax_s.axvline(0.0, color="black", linewidth=0.8)
                ax_s.set_yticks(y_pos)
                ax_s.set_yticklabels(list(reversed(labels_top)), fontsize=8)
                label = step_labels[s] if step_labels and s < len(step_labels) else f"Step {s+1}"
                ax_s.set_title(f"Signed SHAP — {label}", fontsize=10)
                ax_s.set_xlabel("SHAP (approx. m³/s toward predicted Q)")
                ax_s.grid(True, alpha=0.25)
                fig_s.tight_layout()
                cs = FigureCanvasTkAgg(fig_s, master=body)
                cs.draw()
                cs.get_tk_widget().pack(fill=X, expand=False, padx=8, pady=4)

            note = ttk.Label(
                body,
                text="Note: Explains the Random Forest only (not the LSTM). "
                     "Values are approximate m³/s contributions (scaled×range). "
                     "Red = lowers Q, dark = raises Q.",
                font=("Helvetica", 9), foreground="#888888", wraplength=960)
            note.pack(anchor=W, padx=12, pady=(4, 6))

            btn_row = ttk.Frame(body)
            btn_row.pack(fill=X, padx=10, pady=(0, 12))

            def _save_png():
                path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG image", "*.png")],
                    title="Save RF SHAP figure")
                if not path:
                    return
                try:
                    # Save summary figure
                    fig1.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
                    self.log(f"RF SHAP figure saved: {path}")
                    messagebox.showinfo("Saved", path)
                except Exception as e:
                    messagebox.showerror("Save Failed", str(e))

            ttk.Button(btn_row, text="Save PNG…", bootstyle="success", command=_save_png).pack(side=LEFT, padx=4)
            ttk.Button(btn_row, text="Close", bootstyle="secondary", command=win.destroy).pack(side=LEFT, padx=4)

            try:
                body.update_idletasks()
                _sc.configure(scrollregion=_sc.bbox("all"))
            except Exception:
                pass
            self.log(f"RF SHAP done ({source}, n={len(X_rf)}).")
        except Exception as e:
            self.log(f"RF SHAP failed: {e}")
            messagebox.showerror("RF SHAP Error", str(e))

    def show_forecast_graph(self):
        """Open forecast chart in a clean popup window (like Model Fit), with Save PNG."""
        if not self.last_predictions:
            messagebox.showwarning("No Forecast", "Run Prediction first.")
            return
        try:
            dates = list(self.last_predictions.get('Date', []))
            preds = list(self.last_predictions.get('Predicted Discharge (m³/s)', []))
            lower = list(self.last_predictions.get('95% PI Lower (m³/s)', []))
            upper = list(self.last_predictions.get('95% PI Upper (m³/s)', []))
            raw_lower = list(self.last_predictions.get('Raw MC 95% PI Lower (m³/s)', []))
            raw_upper = list(self.last_predictions.get('Raw MC 95% PI Upper (m³/s)', []))
            calibration = self.last_predictions.get('Interval Calibration', [])
            n = min(len(dates), len(preds))
            if n == 0:
                messagebox.showwarning("No Forecast", "No forecast points available.")
                return
            dates, preds = dates[:n], [float(p) for p in preds[:n]]
            if len(lower) >= n and len(upper) >= n:
                lower = [float(v) for v in lower[:n]]
                upper = [float(v) for v in upper[:n]]
            else:
                lower = upper = None
            has_raw_interval = len(raw_lower) >= n and len(raw_upper) >= n
            if has_raw_interval:
                raw_lower = [float(v) for v in raw_lower[:n]]
                raw_upper = [float(v) for v in raw_upper[:n]]
            else:
                raw_lower = raw_upper = None

            win = ttk.Toplevel(self.root)
            win.title("Forecast Graph – Discharge with Uncertainty")
            win.geometry("1000x620")
            win.minsize(640, 420)

            body, _fc = self._scrollable_container(win)

            fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
            x = list(range(n))
            ax.plot(x, preds, 'o-', linewidth=2.8, markersize=9, color='#00aa66', label='AI Forecast')
            if lower is not None and upper is not None:
                has_aci = any(value == 'Rolling ACI' for value in calibration[:n])
                if has_aci:
                    ax.fill_between(
                        x, lower, upper, alpha=0.20, color='#77b7df',
                        label='ACI-calibrated 95% PI')
                if raw_lower is not None and raw_upper is not None:
                    ax.fill_between(
                        x, raw_lower, raw_upper, alpha=0.48, color='#174a70',
                        label='Raw MC 95% PI')
                elif not has_aci:
                    ax.fill_between(
                        x, lower, upper, alpha=0.48, color='#174a70',
                        label='Raw 95% PI')
            try:
                ax.axhline(float(self.high_var.get()), color='red', linestyle='--', linewidth=2, label='Flood Level')
                ax.axhline(float(self.moderate_var.get()), color='orange', linestyle='--', linewidth=1.5, label='Warning Level')
                ax.axhline(float(self.elevated_var.get()), color='#848D00', linestyle='--', linewidth=1, label='Elevated')
            except Exception:
                pass
            ax.set_xticks(x)
            ax.set_xticklabels(dates, rotation=30, ha='right')
            ax.set_ylabel('Discharge (m³/s)')
            ax.set_title('Rainfall–Runoff Forecast with 95% Uncertainty', fontsize=13)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.35)

            canvas = FigureCanvasTkAgg(fig, master=body)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=8)

            btn_row = ttk.Frame(body)
            btn_row.pack(fill=X, padx=10, pady=(0, 12))

            def _save_png():
                path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
                    title="Save forecast graph")
                if not path:
                    return
                try:
                    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
                    self.log(f"Forecast graph saved: {path}")
                    messagebox.showinfo("Saved", f"Graph saved to:\n{path}")
                except Exception as e:
                    messagebox.showerror("Save Failed", str(e))

            ttk.Button(btn_row, text="Save PNG…", bootstyle="success", command=_save_png).pack(side=LEFT, padx=4)
            ttk.Button(btn_row, text="Close", bootstyle="secondary", command=win.destroy).pack(side=LEFT, padx=4)

            try:
                body.update_idletasks()
                _fc.configure(scrollregion=_fc.bbox("all"))
            except Exception:
                pass
        except Exception as e:
            self.log(f"Open Forecast Graph failed: {e}")
            messagebox.showerror("Forecast Graph", str(e))

    def _validation_uncertainty_export_frame(self):
        ex = getattr(self, '_val_unc_export', None)
        if ex is None:
            return None

        columns = {
            'Date': ex['date'],
            'Observed Discharge (m³/s)': ex['obs'],
            'Predicted Discharge (MC mean, m³/s)': ex['mean'],
            'Raw MC 95% Lower (m³/s)': ex['raw_lo95'],
            'Raw MC 95% Upper (m³/s)': ex['raw_hi95'],
            'ACI 95% Lower (m³/s)': ex['lo95'],
            'ACI 95% Upper (m³/s)': ex['hi95'],
            'ACI Ready': ex['aci_ready'],
            'ACI Quantile': ex['aci_q'],
            'Within Raw 95% PI': ex['inside95_raw'],
            'Within ACI 95% PI': ex['inside95'],
            'P(Q > Elevated)': ex['p_elevated'],
            'P(Q > Moderate)': ex['p_moderate'],
            'P(Q > High)': ex['p_high'],
        }
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError("Validation uncertainty arrays have inconsistent lengths")
        return pd.DataFrame(columns)

    def export_to_excel(self):
        if not self.last_predictions:
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if path:
            cache = getattr(self, '_val_unc_cache', None)
            if cache is not None:
                valid, reason = self._is_val_unc_cache_valid(cache)
                if not valid:
                    self.log(f"Validation uncertainty export cache is stale ({reason}).")
                    self._reset_aci_calibration()
            if getattr(self, '_val_unc_export', None) is None:
                self._load_persistent_val_unc_cache()

            # Normalize `self.last_predictions` so all lists have the same length
            try:
                lp = dict(self.last_predictions)
                # determine target length from 'Date' or from the longest list
                target_n = len(lp.get('Date', [])) if lp.get('Date') else 0
                if target_n == 0:
                    list_lengths = [len(v) for v in lp.values() if isinstance(v, (list, tuple))]
                    target_n = max(list_lengths) if list_lengths else 0

                if target_n == 0:
                    messagebox.showwarning("Export", "No forecast data available to export.")
                    return

                normalized = {}
                for k, v in lp.items():
                    if isinstance(v, (list, tuple)):
                        lst = list(v)
                        # Convert complex objects (e.g., explanation dicts) to strings
                        conv = []
                        for item in lst:
                            if isinstance(item, (str, int, float)):
                                conv.append(item)
                            else:
                                try:
                                    conv.append(str(item))
                                except Exception:
                                    conv.append(repr(item))
                        if len(conv) < target_n:
                            conv.extend([''] * (target_n - len(conv)))
                        elif len(conv) > target_n:
                            conv = conv[:target_n]
                        normalized[k] = conv
                    else:
                        # scalar -> repeat
                        normalized[k] = [v] * target_n

                df = pd.DataFrame(normalized)

            except Exception as e:
                self.log(f"Export preparation failed: {e}")
                messagebox.showerror("Export Error", f"Could not prepare data for export:\n{e}")
                return

            try:
                log_text = self.log_text.get("1.0", "end").strip()
                log_lines = log_text.splitlines() if log_text else []
            except Exception:
                log_lines = []

            log_df = pd.DataFrame({"Log": log_lines})

            try:
                validation_uncertainty_df = self._validation_uncertainty_export_frame()
            except Exception as e:
                self.log(f"Validation uncertainty export preparation failed: {e}")
                messagebox.showerror(
                    "Validation Uncertainty Export Error",
                    f"Could not prepare validation uncertainty data:\n{e}")
                return

            # Prepare XAI sheet if structured XAI fields exist
            xai_df = None
            try:
                if 'Top1' in self.last_predictions:
                    xai_keys = ['Date', 'Top1', 'Top1_val', 'Top2', 'Top2_val', 'Top3', 'Top3_val', 'Top4', 'Top4_val']
                    past_cols = [k for k in self.last_predictions.keys() if k.startswith('PastDay_')]
                    xai_keys.extend(past_cols)
                    xai_data = {k: self.last_predictions.get(k, [''] * len(self.last_predictions.get('Date', []))) for k in xai_keys}
                    xai_df = pd.DataFrame(xai_data)
                elif 'Explanation' in self.last_predictions:
                    xai_df = pd.DataFrame({'Date': self.last_predictions.get('Date', []), 'Explanation': self.last_predictions.get('Explanation', [])})
            except Exception:
                xai_df = None

            # Build forecast chart image from last_predictions (for Excel embed)
            chart_png = None
            try:
                import io
                from openpyxl.drawing.image import Image as XLImage
                dates = list(self.last_predictions.get('Date', []))
                preds = list(self.last_predictions.get('Predicted Discharge (m³/s)', []))
                lower = list(self.last_predictions.get('95% PI Lower (m³/s)', []))
                upper = list(self.last_predictions.get('95% PI Upper (m³/s)', []))
                raw_lower = list(self.last_predictions.get('Raw MC 95% PI Lower (m³/s)', []))
                raw_upper = list(self.last_predictions.get('Raw MC 95% PI Upper (m³/s)', []))
                n = min(len(dates), len(preds))
                if n > 0:
                    dates, preds = dates[:n], preds[:n]
                    if len(lower) >= n and len(upper) >= n:
                        lower, upper = lower[:n], upper[:n]
                    else:
                        lower = upper = None
                    if len(raw_lower) >= n and len(raw_upper) >= n:
                        raw_lower, raw_upper = raw_lower[:n], raw_upper[:n]
                    else:
                        raw_lower = raw_upper = None
                    fig, ax = plt.subplots(figsize=(8.5, 4.2))
                    ax.plot(dates, preds, 'o-', linewidth=2.5, markersize=7, color='#00aa66', label='AI Forecast')
                    if lower is not None and upper is not None:
                        calibration = self.last_predictions.get('Interval Calibration', [])
                        has_aci = any(v == 'Rolling ACI' for v in calibration[:n])
                        if has_aci:
                            ax.fill_between(
                                range(n), lower, upper, alpha=0.20, color='#77b7df',
                                label='ACI-calibrated 95% PI')
                        if raw_lower is not None and raw_upper is not None:
                            ax.fill_between(
                                range(n), raw_lower, raw_upper, alpha=0.48,
                                color='#174a70', label='Raw MC 95% PI')
                        elif not has_aci:
                            ax.fill_between(
                                range(n), lower, upper, alpha=0.48,
                                color='#174a70', label='Raw 95% PI')
                        ax.set_xticks(range(n))
                        ax.set_xticklabels(dates, rotation=30, ha='right')
                    try:
                        ax.axhline(float(self.high_var.get()), color='red', linestyle='--', linewidth=2, label='High')
                        ax.axhline(float(self.moderate_var.get()), color='orange', linestyle='--', linewidth=1.5, label='Moderate')
                        ax.axhline(float(self.elevated_var.get()), color='#848D00', linestyle='--', linewidth=1, label='Elevated')
                    except Exception:
                        pass
                    ax.set_title(f'{n}-Day Discharge Forecast')
                    ax.set_ylabel('Discharge (m³/s)')
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.35)
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    chart_png = buf
            except Exception as e:
                self.log(f"Could not build forecast chart for Excel: {e}")
                chart_png = None

            try:
                with pd.ExcelWriter(path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Forecast', index=False)
                    log_df.to_excel(writer, sheet_name='Training Log', index=False)
                    if validation_uncertainty_df is not None:
                        validation_uncertainty_df.to_excel(
                            writer, sheet_name='Validation Uncertainty', index=False)
                    if xai_df is not None:
                            try:
                                rename_map = {}
                                for i in range(1, 5):
                                    rename_map[f'Top{i}'] = f'Top Driver {i}'
                                    rename_map[f'Top{i}_val'] = f'Top Driver {i} (score)'
                                for c in list(xai_df.columns):
                                    if c.startswith('PastDay_'):
                                        try:
                                            n = int(c.split('_')[1])
                                            rename_map[c] = f'Past Day {n} (importance)'
                                        except Exception:
                                            rename_map[c] = c
                                xai_df = xai_df.rename(columns=rename_map)
                            except Exception:
                                pass
                            xai_df.to_excel(writer, sheet_name='XAI', index=False)
                    if chart_png is not None:
                        try:
                            from openpyxl.drawing.image import Image as XLImage
                            ws = writer.book.create_sheet('Forecast Chart')
                            img = XLImage(chart_png)
                            img.width = 720
                            img.height = 360
                            ws.add_image(img, 'A1')
                            ws['A20'] = 'Forecast chart embedded from the prediction result.'
                        except Exception as e_img:
                            self.log(f"Chart sheet skipped: {e_img}")
                self.log(
                    f"Exported: {os.path.basename(path)}"
                    + (" (with forecast chart)" if chart_png is not None else "")
                    + (" (with validation uncertainty)" if validation_uncertainty_df is not None else ""))
                if validation_uncertainty_df is not None:
                    messagebox.showinfo(
                        "Success",
                        f"Exported forecast and validation uncertainty to:\n{path}")
                else:
                    messagebox.showwarning(
                        "Partial Export",
                        f"Forecast exported to:\n{path}\n\n"
                        "The Validation Uncertainty sheet was omitted because no current "
                        "validation uncertainty result or cache is available. Run Val "
                        "Uncertainty and export again to include it.")
            except Exception as e:
                try:
                    df.to_excel(path, index=False)
                    self.log(f"Exported forecast only (log failed): {os.path.basename(path)}")
                    messagebox.showwarning("Partial Export", f"Forecast exported but training log could not be written:\n{e}")
                except Exception as e2:
                    self.log(f"Export failed: {e2}")
                    messagebox.showerror("Export Failed", f"Could not export to Excel:\n{e2}")

    def save_models(self):
        """Save LSTM, RandomForest, scalers and metadata into a zip bundle."""
        if not getattr(self, 'lstm_model', None) or not getattr(self, 'rf_model', None):
            messagebox.showwarning("Not Ready", "Train models first or load models to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension='.zip', filetypes=[('Model bundle', '*.zip')])
        if not path:
            return
        tmp = tempfile.mkdtemp()
        try:
            lstm_path = os.path.join(tmp, 'lstm.keras') 
            try:
                self.lstm_model.save(lstm_path, include_optimizer=False)  # .keras format
                self.log("LSTM saved as .keras format")
            except Exception as e:
                self.log(f"LSTM save failed: {e}")
                messagebox.showerror("Save Error", "Could not save LSTM model")
                return

            try:
                joblib.dump(self.rf_model, os.path.join(tmp, 'rf.joblib'))
            except Exception as e:
                self.log(f"RF save failed: {e}")
            try:
                joblib.dump(self.scaler_features, os.path.join(tmp, 'scaler_features.joblib'))
                joblib.dump(self.scaler_level, os.path.join(tmp, 'scaler_level.joblib'))
                joblib.dump(self.scaler_rain, os.path.join(tmp, 'scaler_rain.joblib'))
            except Exception as e:
                self.log(f"Scaler save failed: {e}")

            metadata = {
                'feature_columns': getattr(self, 'feature_columns', None),
                'seq_length': getattr(self, 'seq_length', getattr(self, 'seq_var', None).get() if getattr(self, 'seq_var', None) else None),
                'mc_samples': getattr(self, 'mc_samples_var', None).get() if getattr(self, 'mc_samples_var', None) else None,
                'ensemble_alpha': float(getattr(self, 'ensemble_lstm_var', ttk.DoubleVar(value=0.85)).get()) if getattr(self, 'ensemble_lstm_var', None) is not None else 0.85,
                'ensemble_calibrated': bool(getattr(self, 'calibrate_ensemble_var', ttk.BooleanVar(value=False)).get()) if getattr(self, 'calibrate_ensemble_var', None) is not None else False,
                'aci_enabled': bool(self.use_aci_var.get()),
                'aci_window': int(self.aci_window_var.get()),
                'aci_gamma': float(self.aci_gamma_var.get()),
                'api_decay': float(self.api_decay_var.get()),
                'risk_thresholds': (
                    float(self.elevated_var.get()),
                    float(self.moderate_var.get()),
                    float(self.high_var.get()),
                ),
                'use_seq2seq': bool(self.use_seq2seq_var.get()),
                'concurrent': bool(self.same_month_var.get()),
                'aci_interval_version': 2,
                'aci_mode': self.aci_mode,
                'aci_calibrated_through': self.aci_calibrated_through,
                'aci_calibrators': {
                    int(lead): calibrator.to_state()
                    for lead, calibrator in self.aci_calibrators.items()
                }
            }
            try:
                joblib.dump(metadata, os.path.join(tmp, 'metadata.joblib'))
            except Exception as e:
                self.log(f"Metadata save failed: {e}")

            cache = getattr(self, '_val_unc_cache', None)
            if cache is not None:
                valid, reason = self._is_val_unc_cache_valid(cache)
                if valid:
                    try:
                        joblib.dump(
                            cache, os.path.join(tmp, 'validation_uncertainty.joblib'),
                            compress=3)
                    except Exception as e:
                        self.log(f"Validation uncertainty cache save failed: {e}")
                else:
                    self.log(f"Validation uncertainty cache not saved ({reason}).")

            base = path[:-4] if path.lower().endswith('.zip') else path
            shutil.make_archive(base, 'zip', tmp)
            self.log(f"Saved models: {os.path.basename(path)}")
            messagebox.showinfo("Saved", f"Models saved to:\n{path}")
        except Exception as e:
            self.log(f"Save failed: {e}")
            messagebox.showerror("Save Failed", str(e))
        finally:
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass

    def load_models(self):
        """Load a previously saved model bundle (zip) and restore models/scalers/metadata."""
        path = filedialog.askopenfilename(title="Open Model Bundle", filetypes=[('Model bundle', '*.zip'), ('All files', '*.*')])
        if not path:
            return
        tmp = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(tmp)

            md = {}
            self._loading_model_metadata = True
            try:
                mdp = os.path.join(tmp, 'metadata.joblib')
                if os.path.exists(mdp):
                    md = joblib.load(mdp)
                    self.feature_columns = md.get('feature_columns', getattr(self, 'feature_columns', None))
                    feature_set = set(self.feature_columns or [])
                    for feature, variable_name in (
                            ('rain_cum3', 'use_cum3_var'),
                            ('rain_cum7', 'use_cum7_var'),
                            ('rain_cum14', 'use_cum14_var'),
                            ('disc_diff', 'use_rate_var'),
                            ('disc_lag1', 'use_lag1_var'),
                            ('wet_season', 'use_wetseason_var'),
                            ('api', 'use_api_var')):
                        variable = getattr(self, variable_name, None)
                        if variable is not None:
                            variable.set(feature in feature_set)
                    # Restore sequence length safely
                    fallback = 30
                    try:
                        fallback = self.seq_var.get() if hasattr(self, 'seq_var') and self.seq_var else 30
                    except:
                        pass
                    self.seq_length = md.get('seq_length', fallback)
                    try:
                        self.seq_length = int(self.seq_length)
                    except:
                        self.seq_length = fallback
                    if hasattr(self, 'seq_var'):
                        self.seq_var.set(self.seq_length)

                    if md.get('mc_samples') is not None and hasattr(self, 'mc_samples_var'):
                        try:
                            self.mc_samples_var.set(int(md['mc_samples']))
                        except:
                            pass

                    if 'ensemble_alpha' in md and hasattr(self, 'ensemble_lstm_var'):
                        try:
                            self.ensemble_lstm_var.set(float(md['ensemble_alpha']))
                        except:
                            pass
                    if 'ensemble_calibrated' in md and hasattr(self, 'calibrate_ensemble_var'):
                        try:
                            self.calibrate_ensemble_var.set(bool(md['ensemble_calibrated']))
                        except:
                            pass
                    if md.get('aci_window') is not None:
                        self.aci_window_var.set(int(md['aci_window']))
                    if md.get('aci_gamma') is not None:
                        self.aci_gamma_var.set(float(md['aci_gamma']))
                    if md.get('api_decay') is not None:
                        self.api_decay_var.set(float(md['api_decay']))
                    if md.get('risk_thresholds') is not None:
                        elevated, moderate, high = md['risk_thresholds']
                        self.elevated_var.set(float(elevated))
                        self.moderate_var.set(float(moderate))
                        self.high_var.set(float(high))
                    if md.get('aci_enabled') is not None:
                        self.use_aci_var.set(bool(md['aci_enabled']))
                    if md.get('use_seq2seq') is not None:
                        self.use_seq2seq_var.set(bool(md['use_seq2seq']))
                    if md.get('concurrent') is not None:
                        self.same_month_var.set(bool(md['concurrent']))
            except Exception as e:
                self.log(f"Metadata load warning: {e}")
                md = {}
            finally:
                self._loading_model_metadata = False

            keras_path = os.path.join(tmp, 'lstm.keras')
            h5_path = os.path.join(tmp, 'lstm.h5')
            lstm_dir = os.path.join(tmp, 'lstm')

            lstm_loaded = False

            if os.path.exists(keras_path):
                try:
                    self.lstm_model = tf.keras.models.load_model(keras_path, compile=False)
                    self.log("LSTM loaded successfully from .keras format")
                    lstm_loaded = True
                except Exception as e:
                    self.log(f".keras load failed: {e}")

            if not lstm_loaded and os.path.exists(h5_path):
                try:
                    self.lstm_model = tf.keras.models.load_model(h5_path, compile=False)
                    self.log("LSTM loaded from legacy .h5 format (inference only)")
                    lstm_loaded = True
                except Exception as e:
                    self.log(f".h5 load failed: {e}")

            if not lstm_loaded and os.path.isdir(lstm_dir):
                try:
                    self.lstm_model = tf.keras.models.load_model(lstm_dir, compile=False)
                    self.log("LSTM loaded from old SavedModel directory")
                    lstm_loaded = True
                except Exception as e:
                    self.log(f"SavedModel directory load failed: {e}")

            if not lstm_loaded:
                self.log("WARNING: LSTM model failed to load – predictions will use Random Forest only")
                self.lstm_model = None

            # === Load Random Forest ===
            try:
                rf_path = os.path.join(tmp, 'rf.joblib')
                if os.path.exists(rf_path):
                    self.rf_model = joblib.load(rf_path)
                    self.log("Random Forest loaded")
                else:
                    self.log("RF model not found in bundle")
            except Exception as e:
                self.log(f"RF load failed: {e}")

            # === Load scalers ===
            try:
                sf = os.path.join(tmp, 'scaler_features.joblib')
                sl = os.path.join(tmp, 'scaler_level.joblib')
                sr = os.path.join(tmp, 'scaler_rain.joblib')
                if os.path.exists(sf):
                    self.scaler_features = joblib.load(sf)
                if os.path.exists(sl):
                    self.scaler_level = joblib.load(sl)
                if os.path.exists(sr):
                    self.scaler_rain = joblib.load(sr)
                self.log("Scalers loaded")
            except Exception as e:
                self.log(f"Scaler load failed: {e}")

            # Recreate sequences if data is already loaded
            try:
                if getattr(self, 'df', None) is not None and getattr(self, 'feature_columns', None) is not None:
                    self.apply_feature_selection()
            except Exception:
                pass

            try:
                saved_calibrators = md.get('aci_calibrators', {})
                if saved_calibrators and md.get('aci_interval_version') != 2:
                    self.aci_calibrators = {}
                    self.aci_mode = None
                    self.aci_calibrated_through = None
                    self.log(
                        "Saved ACI scores use an older interval definition; "
                        "run Validation Uncertainty to recalibrate.")
                else:
                    self.aci_calibrators = {
                        int(lead): RollingACI.from_state(state)
                        for lead, state in saved_calibrators.items()
                    }
                    self.aci_mode = md.get('aci_mode') if self.aci_calibrators else None
                    self.aci_calibrated_through = (
                        md.get('aci_calibrated_through') if self.aci_calibrators else None)
                    if self.aci_calibrators:
                        self.log(
                            f"Restored ACI calibration through {self.aci_calibrated_through or 'unknown date'} "
                            f"({len(self.aci_calibrators)} lead calibrator(s)).")
                    elif bool(self.use_aci_var.get()):
                        self.log("No saved ACI scores in this bundle; run Validation Uncertainty to calibrate.")
                self._aci_not_ready_logged = False
            except Exception as e:
                self._reset_aci_calibration()
                self.log(f"ACI calibration state could not be restored: {e}")

            cache_path = os.path.join(tmp, 'validation_uncertainty.joblib')
            if os.path.exists(cache_path):
                try:
                    cache = joblib.load(cache_path)
                    if self._restore_val_unc_cache(cache):
                        self.log("Validation uncertainty results restored from model bundle.")
                except Exception as e:
                    self._reset_aci_calibration()
                    self.log(f"Validation uncertainty cache could not be restored: {e}")
            self._load_persistent_val_unc_cache()

            self.status.config(text="Models loaded", bootstyle="success")
            self.log(f"Loaded models from: {os.path.basename(path)}")

            try:
                self.save_model_btn.config(state=NORMAL)
                self.inspect_model_btn.config(state=NORMAL)
                if getattr(self, 'fit_btn', None) is not None:
                    self.fit_btn.config(state=NORMAL)
                if getattr(self, 'val_unc_btn', None) is not None:
                    self.val_unc_btn.config(state=NORMAL)
            except:
                pass

        except Exception as e:
            self.log(f"Load failed: {e}")
            messagebox.showerror("Load Failed", str(e))
        finally:
            try:
                shutil.rmtree(tmp)
            except:
                pass

    def show_val_uncertainty(self):
        """MC Dropout uncertainty on the validation set: coverage table, bands, reliability, CSV."""
        if self._open_cached_val_uncertainty():
            return
        if self.lstm_model is None:
            messagebox.showwarning(
                "LSTM required",
                "Validation uncertainty uses MC Dropout on the LSTM.\n"
                "Train or load an LSTM model first.")
            return
        if getattr(self, 'val_X', None) is None or len(self.val_X) == 0:
            messagebox.showwarning("No Validation Data", "Load data and train models first.")
            return

        try:
            n_mc = max(20, int(self.mc_samples_var.get()))
        except Exception:
            n_mc = 100
        # Cap for responsiveness on long daily val sets (user can raise MC samples)
        n_mc_run = int(min(n_mc, 500))
        if n_mc_run < n_mc:
            self.log(f"Val uncertainty: using {n_mc_run} MC samples (UI capped; MC setting was {n_mc}).")

        try:
            self.val_unc_btn.config(state=DISABLED)
        except Exception:
            pass
        self.log(f"Val uncertainty: MC Dropout on {len(self.val_X)} val samples × {n_mc_run} passes...")

        def worker():
            try:
                self._reset_aci_calibration()
                X = np.asarray(self.val_X, dtype=np.float32)
                use_seq2seq = bool(self.use_seq2seq_var.get())
                y_scaled = np.asarray(self.val_y, dtype=np.float64)
                if use_seq2seq:
                    if y_scaled.ndim != 2 or y_scaled.shape[1] < 7:
                        raise ValueError("Seq2Seq validation targets must contain seven forecast steps")
                    y_scaled = y_scaled[:, :7]
                    n_steps = 7
                    rains = np.asarray(self.val_df['rainfall'].values, dtype=np.float32)
                    seq_len = int(self.seq_length)
                    decoder_rows = [
                        rains[i:i + 7]
                        for i in range(seq_len, len(rains) - 7 + 1)
                    ]
                    if not decoder_rows:
                        raise ValueError("Could not build Seq2Seq validation decoder inputs")
                    decoder_raw = np.asarray(decoder_rows, dtype=np.float32)
                    decoder_val = self.scaler_rain.transform(
                        decoder_raw.reshape(-1, 1)).reshape(-1, 7, 1)
                    n = min(len(X), len(y_scaled), len(decoder_val))
                    X = X[:n]
                    y_scaled = y_scaled[:n]
                    decoder_val = decoder_val[:n]
                else:
                    y_scaled = y_scaled.reshape(-1, 1)
                    n_steps = 1
                    n = min(len(X), len(y_scaled))
                    X = X[:n]
                    y_scaled = y_scaled[:n]
                y_scaled = y_scaled[:, :n_steps]
                y_obs_all = self.scaler_level.inverse_transform(
                    y_scaled.reshape(-1, 1)).reshape(n, n_steps)

                # --- MC Dropout (LSTM) ---
                mc = np.zeros((n_mc_run, n, n_steps), dtype=np.float64)
                for i in range(n_mc_run):
                    model_inputs = [X, decoder_val] if use_seq2seq else X
                    pred = self.lstm_model(model_inputs, training=True)
                    pred = np.asarray(pred).reshape(n, -1)
                    if pred.shape[1] < n_steps:
                        raise ValueError(
                            f"Model returned {pred.shape[1]} outputs; expected {n_steps}")
                    mc[i, :, :] = pred[:, :n_steps]
                    if (i + 1) % 20 == 0 or i == 0:
                        self.plot_queue.put((
                            'log',
                            f"Val uncertainty: MC pass {i+1}/{n_mc_run}"))

                # Inverse-transform samples to m³/s
                mc_m = self.scaler_level.inverse_transform(
                    mc.reshape(-1, 1)).reshape(n_mc_run, n, n_steps)

                # LSTM-only MC Dropout for uncertainty (no RF blend).
                # RF stays for point forecasts / SHAP; blending a deterministic RF
                # prediction into every MC draw artificially shrinks intervals.
                w = 1.0
                self.plot_queue.put((
                    'log',
                    "Val uncertainty: LSTM-only MC Dropout (RF not blended into intervals)."))

                mean_all = np.mean(mc_m, axis=0)
                raw_lo_all = np.maximum(0.0, np.percentile(mc_m, 2.5, axis=0))
                raw_hi_all = np.maximum(0.0, np.percentile(mc_m, 97.5, axis=0))
                aci_lo_all = raw_lo_all.copy()
                aci_hi_all = raw_hi_all.copy()
                aci_ready_all = np.zeros((n, n_steps), dtype=bool)
                aci_quantiles_all = np.zeros((n, n_steps), dtype=np.float64)

                new_calibrators = {}
                for step_idx in range(n_steps):
                    (aci_lo_all[:, step_idx],
                     aci_hi_all[:, step_idx],
                     aci_ready_all[:, step_idx],
                     aci_quantiles_all[:, step_idx],
                     new_calibrators[step_idx + 1]) = self._run_aci_series(
                        raw_lo_all[:, step_idx],
                        raw_hi_all[:, step_idx],
                        y_obs_all[:, step_idx],
                        horizon=step_idx + 1)
                self.aci_calibrators = new_calibrators
                self.aci_mode = "seq2seq" if use_seq2seq else "recursive"
                self.aci_calibrated_through = str(
                    pd.Timestamp(self.val_df['date'].iloc[-1]).date())
                self._aci_not_ready_logged = False
                if use_seq2seq:
                    self.plot_queue.put(('log', "ACI calibration initialized separately for forecast leads 1–7."))
                else:
                    self.plot_queue.put((
                        'log',
                        "ACI calibration initialized from one-step errors; recursive forecasts share this normalized correction."))

                # The plot shows lead one; Seq2Seq also maintains a separate state per lead.
                y_obs = y_obs_all[:, 0]
                mean_m = mean_all[:, 0]
                raw_lo_m = raw_lo_all[:, 0]
                raw_hi_m = raw_hi_all[:, 0]
                lo_m = aci_lo_all[:, 0]
                hi_m = aci_hi_all[:, 0]
                width = (hi_m - lo_m) / 2.0
                mc_display = mc_m[:, :, 0]
                inside95_raw = (y_obs >= raw_lo_m) & (y_obs <= raw_hi_m)
                inside95 = (y_obs >= lo_m) & (y_obs <= hi_m)
                post_burn_mask = aci_ready_all[:, 0]
                cov95_raw = float(np.mean(inside95_raw))
                cov95 = float(np.mean(inside95))
                cov95_post = (float(np.mean(inside95[post_burn_mask]))
                              if np.any(post_burn_mask) else float('nan'))

                lo50 = np.percentile(mc_display, 25, axis=0)
                hi50 = np.percentile(mc_display, 75, axis=0)
                cov50 = float(np.mean((y_obs >= lo50) & (y_obs <= hi50)))
                lo90 = np.percentile(mc_display, 5, axis=0)
                hi90 = np.percentile(mc_display, 95, axis=0)
                cov90 = float(np.mean((y_obs >= lo90) & (y_obs <= hi90)))

                # High-flow stratum
                try:
                    p90 = float(np.nanpercentile(y_obs, 90))
                    hi_mask = y_obs >= p90
                except Exception:
                    p90 = float(np.nanpercentile(y_obs, 90)) if len(y_obs) else 0.0
                    hi_mask = np.zeros(n, dtype=bool)
                if np.any(hi_mask):
                    cov95_hi = float(np.mean(inside95[hi_mask]))
                    width_hi = float(np.mean(width[hi_mask]))
                    n_hi = int(np.sum(hi_mask))
                    hi_post_mask = hi_mask & post_burn_mask
                    cov95_hi_post = (float(np.mean(inside95[hi_post_mask]))
                                     if np.any(hi_post_mask) else float('nan'))
                    n_hi_post = int(np.sum(hi_post_mask))
                else:
                    cov95_hi, width_hi, n_hi = float('nan'), float('nan'), 0
                    cov95_hi_post = float('nan')
                    n_hi_post = 0

                lead_stats = {}
                for step_idx in range(n_steps):
                    lead_inside = (
                        (y_obs_all[:, step_idx] >= aci_lo_all[:, step_idx]) &
                        (y_obs_all[:, step_idx] <= aci_hi_all[:, step_idx]))
                    lead_ready = aci_ready_all[:, step_idx]
                    lead_stats[f"day_{step_idx + 1}"] = {
                        'coverage': (
                            float(np.mean(lead_inside[lead_ready]))
                            if np.any(lead_ready) else float('nan')),
                        'n': int(np.sum(lead_ready)),
                        'raw_coverage': float(np.mean(
                            (y_obs_all[:, step_idx] >= raw_lo_all[:, step_idx]) &
                            (y_obs_all[:, step_idx] <= raw_hi_all[:, step_idx]))),
                    }

                # Threshold probabilities from MC (for reliability)
                try:
                    thr_e = float(self.elevated_var.get())
                    thr_m = float(self.moderate_var.get())
                    thr_h = float(self.high_var.get())
                except Exception:
                    thr_e, thr_m, thr_h = 15.7, 26.5, 37.6

                def _rel_curve(thr):
                    p_hat = np.mean(mc_display > thr, axis=0)
                    y_evt = (y_obs > thr).astype(float)
                    # 10 bins
                    edges = np.linspace(0, 1, 11)
                    centers, obs_freq, counts = [], [], []
                    for a, b in zip(edges[:-1], edges[1:]):
                        if b < 1:
                            msk = (p_hat >= a) & (p_hat < b)
                        else:
                            msk = (p_hat >= a) & (p_hat <= b)
                        c = int(np.sum(msk))
                        counts.append(c)
                        centers.append(0.5 * (a + b))
                        if c > 0:
                            obs_freq.append(float(np.mean(y_evt[msk])))
                        else:
                            obs_freq.append(float('nan'))
                    return np.array(centers), np.array(obs_freq), np.array(counts), p_hat, y_evt

                # Dates for plot
                try:
                    seq_len = int(self.seq_length)
                    concurrent = (not bool(self.use_seq2seq_var.get())) and bool(
                        getattr(self, 'same_month_var', ttk.BooleanVar(value=False)).get())
                    start = (seq_len - 1) if concurrent else seq_len
                    dates = self.val_df['date'].iloc[start:start + n].values
                    if len(dates) < n:
                        dates = np.arange(n)
                except Exception:
                    dates = np.arange(n)

                # Store for CSV
                self._val_unc_export = {
                    'date': dates,
                    'obs': y_obs,
                    'mean': mean_m,
                    'raw_lo95': raw_lo_m,
                    'raw_hi95': raw_hi_m,
                    'lo95': lo_m,
                    'hi95': hi_m,
                    'width': width,
                    'aci_ready': post_burn_mask.astype(int),
                    'aci_q': aci_quantiles_all[:, 0],
                    'inside95_raw': inside95_raw.astype(int),
                    'inside95': inside95.astype(int),
                    'p_elevated': np.mean(mc_display > thr_e, axis=0),
                    'p_moderate': np.mean(mc_display > thr_m, axis=0),
                    'p_high': np.mean(mc_display > thr_h, axis=0),
                }
                self._val_unc_stats = {
                    'n': n, 'n_mc': n_mc_run, 'w': w,
                    'cov50': cov50, 'cov90': cov90, 'cov95_raw': cov95_raw,
                    'cov95': cov95, 'cov95_post': cov95_post,
                    'n_post': int(np.sum(post_burn_mask)),
                    'mean_width': float(np.mean(width)),
                    'cov95_hi': cov95_hi, 'cov95_hi_post': cov95_hi_post,
                    'width_hi': width_hi, 'n_hi': n_hi, 'n_hi_post': n_hi_post, 'p90': p90,
                    'thr_e': thr_e, 'thr_m': thr_m, 'thr_h': thr_h,
                    'aci_mode': self.aci_mode,
                    'aci_window': int(self.aci_window_var.get()),
                    'aci_gamma': float(self.aci_gamma_var.get()),
                    'aci_leads': n_steps,
                    'aci_lead_stats': lead_stats,
                }
                try:
                    self._val_unc_cache = {
                        'schema': 2,
                        'model_signature': self._model_uncertainty_signature(),
                        'data_signature': self._validation_data_signature(),
                        'export': self._val_unc_export,
                        'stats': self._val_unc_stats,
                        'aci_calibrators': {
                            int(lead): calibrator.to_state()
                            for lead, calibrator in self.aci_calibrators.items()
                        },
                        'aci_mode': self.aci_mode,
                        'aci_calibrated_through': self.aci_calibrated_through,
                    }
                    self._store_val_unc_cache(self._val_unc_cache)
                except Exception as e:
                    self._val_unc_cache = None
                    self.plot_queue.put((
                        'log', f"Validation uncertainty results could not be cached: {e}"))

                # Build figure on main thread via queue-like after
                def show_ui():
                    try:
                        self._show_val_uncertainty_window(
                            dates, y_obs, mean_m, raw_lo_m, raw_hi_m, lo_m, hi_m,
                            width, inside95, thr_e, thr_m, thr_h,
                            self._val_unc_export['p_elevated'])
                    finally:
                        try:
                            self.val_unc_btn.config(state=NORMAL)
                        except Exception:
                            pass

                self.root.after(0, show_ui)
                self.plot_queue.put((
                    'log',
                    f"Val uncertainty done — raw 95%={cov95_raw*100:.1f}%, "
                    f"ACI 95%={cov95*100:.1f}% "
                    f"(post-burn-in n={int(np.sum(post_burn_mask))}, "
                    f"coverage={cov95_post*100 if np.isfinite(cov95_post) else float('nan'):.1f}%; "
                    f"high-flow n={n_hi}: {cov95_hi*100:.1f}%), "
                    f"mean half-width={float(np.mean(width)):.2f} m³/s"
                ))
            except Exception as e:
                self.plot_queue.put(('log', f"Val uncertainty failed: {e}"))
                try:
                    self.root.after(0, lambda: messagebox.showerror("Val Uncertainty Error", str(e)))
                except Exception:
                    pass
                try:
                    self.root.after(0, lambda: self.val_unc_btn.config(state=NORMAL))
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _show_val_uncertainty_window(
            self, dates, y_obs, mean_m, raw_lo_m, raw_hi_m, lo_m, hi_m, width,
            inside95, thr_e, thr_m, thr_h, p_elevated):
        """Popup: coverage table, time-series bands, reliability diagram, CSV export."""
        st = getattr(self, '_val_unc_stats', {})
        lead_coverage = "  ".join(
            f"D{lead[4:]}={stats['coverage']*100:.1f}% (n={stats['n']})"
            for lead, stats in st.get('aci_lead_stats', {}).items()
            if np.isfinite(stats['coverage']))
        win = ttk.Toplevel(self.root)
        win.title("Validation Uncertainty (MC Dropout)")
        win.geometry("1100x780")
        win.minsize(700, 500)
        body, sc = self._scrollable_container(win)

        # --- Summary labels ---
        summary = (
            f"N={st.get('n', len(y_obs))}   MC samples={st.get('n_mc', '?')}   "
            f"Raw band=LSTM MC Dropout; outer band=rolling ACI calibration\n"
            f"Raw MC coverage 50%/90%/95%: "
            f"{st.get('cov50', float('nan'))*100:.1f}% / "
            f"{st.get('cov90', float('nan'))*100:.1f}% / "
            f"{st.get('cov95_raw', float('nan'))*100:.1f}%   "
            f"ACI 95% coverage={st.get('cov95', float('nan'))*100:.1f}% "
            f"(post-burn-in n={st.get('n_post', 0)}, "
            f"{st.get('cov95_post', float('nan'))*100:.1f}%)\n"
            f"Lead ACI coverage (post-burn-in): {lead_coverage or 'not enough scores'}\n"
            f"ACI window={st.get('aci_window', '?')} days, γ={st.get('aci_gamma', float('nan')):.3f}; "
            f"mean calibrated half-width={st.get('mean_width', float('nan')):.2f} m³/s\n"
            f"High-flow (Q≥P90={st.get('p90', float('nan')):.1f}): n={st.get('n_hi', 0)}  "
            f"ACI 95% coverage={st.get('cov95_hi', float('nan'))*100:.1f}% "
            f"(post-burn-in n={st.get('n_hi_post', 0)}, "
            f"{st.get('cov95_hi_post', float('nan'))*100:.1f}%)  "
            f"mean half-width={st.get('width_hi', float('nan')):.2f} m³/s"
        )
        ttk.Label(body, text=summary, font=("Consolas", 10), justify=LEFT).pack(
            anchor=W, padx=12, pady=(10, 6))
        ttk.Label(
            body,
            text=("Raw MC band shows dropout variability; ACI calibrates interval coverage "
                  "from validation residuals rather than modeling aleatoric variance separately."),
            wraplength=1040, justify=LEFT, foreground="#555555").pack(
                anchor=W, padx=12, pady=(0, 8))

        # --- Figures ---
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

        # 1) Time series with band (subsample if very long)
        ax1 = axes[0, 0]
        n = len(y_obs)
        if n > 400:
            step = max(1, n // 400)
            idx = np.arange(0, n, step)
        else:
            idx = np.arange(n)
        d = dates[idx] if hasattr(dates, '__getitem__') else idx
        ax1.fill_between(
            idx, lo_m[idx], hi_m[idx], color='#77b7df', alpha=0.22,
            label='ACI-calibrated 95% PI', zorder=1)
        ax1.fill_between(
            idx, raw_lo_m[idx], raw_hi_m[idx], color='#174a70', alpha=0.42,
            label='Raw MC 95% PI', zorder=2)
        ax1.plot(idx, y_obs[idx], 'k-', lw=1.2, label='Observed', alpha=0.85)
        ax1.plot(idx, mean_m[idx], color='#00aa55', lw=1.3, label='MC mean (ens.)')
        ax1.set_title('Validation: observed vs MC predictive interval')
        ax1.set_ylabel('Discharge (m³/s)')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)

        # 2) Coverage bar chart
        ax2 = axes[0, 1]
        names = ['50% raw', '90% raw', '95% raw', '95% ACI', '95% ACI high-Q*']
        vals = [
            st.get('cov50', 0) * 100,
            st.get('cov90', 0) * 100,
            st.get('cov95_raw', 0) * 100,
            st.get('cov95_post', float('nan')) * 100,
            st.get('cov95_hi_post', float('nan')) * 100,
        ]
        nominal = [50, 90, 95, 95, 95]
        x = np.arange(len(names))
        ax2.bar(x, vals, color='#6688cc', edgecolor='k', alpha=0.85, label='Empirical')
        ax2.plot(x, nominal, 'ro-', label='Nominal')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.set_ylabel('Coverage (%)')
        ax2.set_title('Coverage (validation; *post burn-in)')
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3) Reliability for elevated threshold
        ax3 = axes[1, 0]
        p_hat = np.asarray(p_elevated)
        y_evt = (y_obs > thr_e).astype(float)
        edges = np.linspace(0, 1, 11)
        centers, obs_freq, counts = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            msk = (p_hat >= a) & (p_hat < b) if b < 1 else (p_hat >= a) & (p_hat <= b)
            c = int(np.sum(msk))
            counts.append(c)
            centers.append(0.5 * (a + b))
            obs_freq.append(float(np.mean(y_evt[msk])) if c > 0 else np.nan)
        centers = np.array(centers)
        obs_freq = np.array(obs_freq)
        ax3.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
        ax3.plot(centers, obs_freq, 'o-', color='#cc5533', label=f'P(Q>{thr_e:g})')
        for cx, of, c in zip(centers, obs_freq, counts):
            if c > 0 and np.isfinite(of):
                ax3.annotate(str(c), (cx, of), textcoords='offset points', xytext=(0, 6),
                             ha='center', fontsize=7, color='#666')
        ax3.set_xlabel('Predicted probability')
        ax3.set_ylabel('Observed frequency')
        ax3.set_title(f'Reliability diagram — elevated (>{thr_e:g} m³/s)')
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4) Residual vs width / sharpness
        ax4 = axes[1, 1]
        ax4.scatter(width, np.abs(mean_m - y_obs), s=12, alpha=0.4, c='#4466aa', edgecolors='none')
        ax4.set_xlabel('95% half-width (m³/s)')
        ax4.set_ylabel('|error| (m³/s)')
        ax4.set_title('Sharpness vs absolute error')
        ax4.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=body)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=8, pady=6)

        note = ttk.Label(
            body,
            text="ACI uses only scores available before each forecast; the first rolling-window scores are burn-in. "
                 "Coverage bars marked ACI use post-burn-in cases. For Seq2Seq, calibration is separate by lead; "
                 "recursive forecasts use the one-step normalized correction for all leads. "
                 "This validation period is also used for model selection, so these results are diagnostic, "
                 "not an independent coverage guarantee. Intervals use LSTM MC Dropout only; RF point forecasts "
                 "and SHAP remain separate.",
            font=("Helvetica", 9), foreground="#888888", wraplength=1000, justify=LEFT)
        note.pack(anchor=W, padx=12, pady=(0, 6))

        btn_row = ttk.Frame(body)
        btn_row.pack(fill=X, padx=10, pady=(0, 12))

        def _export_csv():
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                title="Export validation uncertainty")
            if not path:
                return
            try:
                ex = self._val_unc_export
                df = pd.DataFrame({
                    'date': ex['date'],
                    'obs_m3s': ex['obs'],
                    'mean_m3s': ex['mean'],
                    'raw_lo95_m3s': ex['raw_lo95'],
                    'raw_hi95_m3s': ex['raw_hi95'],
                    'lo95_m3s': ex['lo95'],
                    'hi95_m3s': ex['hi95'],
                    'half_width_95': ex['width'],
                    'aci_ready': ex['aci_ready'],
                    'aci_q': ex['aci_q'],
                    'inside_95_raw': ex['inside95_raw'],
                    'inside_95': ex['inside95'],
                    'p_elevated': ex['p_elevated'],
                    'p_moderate': ex['p_moderate'],
                    'p_high': ex['p_high'],
                })
                df.to_csv(path, index=False)
                self.log(f"Val uncertainty CSV saved: {path}")
                messagebox.showinfo("Saved", path)
            except Exception as e:
                messagebox.showerror("Export failed", str(e))

        def _save_fig():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png")],
                title="Save uncertainty figure")
            if not path:
                return
            try:
                fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
                self.log(f"Val uncertainty figure saved: {path}")
                messagebox.showinfo("Saved", path)
            except Exception as e:
                messagebox.showerror("Save failed", str(e))

        ttk.Button(btn_row, text="Export CSV…", bootstyle="success", command=_export_csv).pack(side=LEFT, padx=4)
        ttk.Button(btn_row, text="Save figure PNG…", bootstyle="info", command=_save_fig).pack(side=LEFT, padx=4)
        ttk.Button(btn_row, text="Close", bootstyle="secondary", command=win.destroy).pack(side=LEFT, padx=4)

        try:
            body.update_idletasks()
            sc.configure(scrollregion=sc.bbox("all"))
        except Exception:
            pass

    def show_model_fit(self):
        """Visual diagnostic: Observed vs Predicted on the validation set (daily)."""
        if self.lstm_model is None and self.rf_model is None:
            messagebox.showwarning("Not Ready", "Train or load models first.")
            return
        if getattr(self, 'val_X', None) is None or len(self.val_X) == 0:
            messagebox.showwarning("No Validation Data", "Load data and train models first.")
            return

        try:
            use_seq2seq = bool(self.use_seq2seq_var.get()) if hasattr(self, 'use_seq2seq_var') else False
            horizon = 7

            # ---------- Generate predictions on validation set ----------
            true_scaled = np.asarray(self.val_y)
            if true_scaled.ndim >= 2:
                true_scaled = true_scaled[:, -1]
            true_scaled = true_scaled.ravel()
            true_m = self.scaler_level.inverse_transform(true_scaled.reshape(-1, 1)).ravel()

            # LSTM predictions
            lstm_m = None
            if self.lstm_model is not None:
                try:
                    if use_seq2seq:
                        rains = np.array(self.val_df['rainfall'].values, dtype=np.float32)
                        seq_len = int(self.seq_length)
                        dec = []
                        for i in range(seq_len, len(rains) - horizon + 1):
                            dec.append(rains[i:i + horizon])
                        if len(dec) == 0:
                            raise ValueError("Could not build decoder inputs")
                        arr = np.array(dec, dtype=np.float32)
                        flat = arr.reshape(-1, 1)
                        try:
                            scaled_flat = self.scaler_rain.transform(flat)
                        except Exception:
                            scaled_flat = flat
                        decoder_val = scaled_flat.reshape(-1, horizon, 1)
                        n = min(len(self.val_X), len(decoder_val))
                        pred_raw = self.lstm_model.predict(
                            [np.asarray(self.val_X[:n]), np.asarray(decoder_val[:n])], verbose=0)
                        lstm_scaled = pred_raw[:, -1, 0] if pred_raw.ndim == 3 else pred_raw.ravel()
                    else:
                        lstm_scaled = self.lstm_model.predict(self.val_X, verbose=0).ravel()
                    lstm_m = self.scaler_level.inverse_transform(lstm_scaled.reshape(-1, 1)).ravel()
                except Exception as e:
                    self.log(f"LSTM prediction for fit plot failed: {e}")

            # RF predictions
            rf_m = None
            if self.rf_model is not None:
                try:
                    X_flat = self.val_X.reshape(self.val_X.shape[0], -1)
                    rf_scaled = self.rf_model.predict(X_flat)
                    rf_m = self.scaler_level.inverse_transform(rf_scaled.reshape(-1, 1)).ravel()
                except Exception as e:
                    self.log(f"RF prediction for fit plot failed: {e}")

            # Ensemble
            try:
                w = float(self.ensemble_lstm_var.get())
                w = max(0.0, min(1.0, w))
            except Exception:
                w = 0.85

            if lstm_m is not None and rf_m is not None:
                n = min(len(lstm_m), len(rf_m), len(true_m))
                ens_m = w * lstm_m[:n] + (1.0 - w) * rf_m[:n]
                true_m = true_m[:n]
                lstm_m = lstm_m[:n]
                rf_m = rf_m[:n]
            elif lstm_m is not None:
                n = min(len(lstm_m), len(true_m))
                ens_m = lstm_m[:n]
                true_m = true_m[:n]
                lstm_m = lstm_m[:n]
            elif rf_m is not None:
                n = min(len(rf_m), len(true_m))
                ens_m = rf_m[:n]
                true_m = true_m[:n]
                rf_m = rf_m[:n]
            else:
                messagebox.showerror("Error", "No model predictions available.")
                return

            # Metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            rmse = np.sqrt(mean_squared_error(true_m, ens_m))
            mae = mean_absolute_error(true_m, ens_m)
            # Nash-Sutcliffe Efficiency
            try:
                denom = np.sum((true_m - np.mean(true_m)) ** 2)
                nse = 1.0 - (np.sum((true_m - ens_m) ** 2) / denom) if denom > 0 else float('nan')
            except Exception:
                nse = float('nan')
            # Pearson r² (correlation squared) — can differ from NSE when there is bias
            try:
                r_pearson = np.corrcoef(true_m, ens_m)[0, 1]
                r2 = float(r_pearson ** 2) if np.isfinite(r_pearson) else float('nan')
            except Exception:
                r2 = float('nan')

            # Approximate dates for the validation sequence targets
            try:
                seq_len = int(self.seq_length)
                concurrent = (not bool(self.use_seq2seq_var.get())) and bool(
                    getattr(self, 'same_month_var', ttk.BooleanVar(value=False)).get())
                start = (seq_len - 1) if concurrent else seq_len
                dates = self.val_df['date'].iloc[start:start + n].values
                if len(dates) < n:
                    dates = np.arange(n)
            except Exception:
                dates = np.arange(n)

            # ---------- Plot ----------
            win = ttk.Toplevel(self.root)
            win.title("Model Fit Diagnostics – Validation Set")
            win.geometry("1100x750")
            win.minsize(640, 480)

            body, _fit_canvas = self._scrollable_container(win)

            fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

            # 1. Time series
            ax1 = axes[0, 0]
            ax1.plot(dates, true_m, 'k-', linewidth=2.0, label='Observed', alpha=0.85)
            if lstm_m is not None:
                ax1.plot(dates, lstm_m, '--', color='#1f77b4', linewidth=1.5, label='LSTM', alpha=0.8)
            if rf_m is not None:
                ax1.plot(dates, rf_m, '--', color='#ff7f0e', linewidth=1.5, label='Random Forest', alpha=0.8)
            ax1.plot(dates, ens_m, '-', color='#00cc66', linewidth=2.2, label=f'Ensemble (w={w:.2f})')
            ax1.set_title('Observed vs Predicted (Validation)', fontsize=12)
            ax1.set_ylabel('Discharge (m³/s)')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            if hasattr(dates[0], 'strftime'):
                fig.autofmt_xdate(rotation=30)

            # 2. Scatter
            ax2 = axes[0, 1]
            ax2.scatter(true_m, ens_m, c='#00cc66', alpha=0.65, edgecolors='k', linewidths=0.3, s=40)
            mn = min(true_m.min(), ens_m.min())
            mx = max(true_m.max(), ens_m.max())
            ax2.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='1:1 line')
            ax2.set_xlabel('Observed (m³/s)')
            ax2.set_ylabel('Predicted (m³/s)')
            ax2.set_title(f'Scatter (NSE = {nse:.3f}, Pearson r² = {r2:.3f})', fontsize=12)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')

            # 3. Residuals over time
            ax3 = axes[1, 0]
            residuals = ens_m - true_m
            ax3.axhline(0, color='k', linewidth=1)
            ax3.plot(dates, residuals, 'o-', color='#cc3366', markersize=3, linewidth=1)
            ax3.set_title('Residuals (Predicted − Observed)', fontsize=12)
            ax3.set_ylabel('Residual (m³/s)')
            ax3.grid(True, alpha=0.3)
            if hasattr(dates[0], 'strftime'):
                fig.autofmt_xdate(rotation=30)

            # 4. Residual histogram + metrics text
            ax4 = axes[1, 1]
            ax4.hist(residuals, bins=min(30, max(10, n // 5)), color='#6688cc', edgecolor='k', alpha=0.8)
            ax4.axvline(0, color='r', linestyle='--', linewidth=1.5)
            ax4.set_title('Residual Distribution', fontsize=12)
            ax4.set_xlabel('Residual (m³/s)')
            ax4.set_ylabel('Count')
            ax4.grid(True, alpha=0.3)

            metrics_txt = (
                f"Validation metrics (Ensemble)\n"
                f"NSE   = {nse:.3f}\n"
                f"RMSE  = {rmse:.3f} m³/s\n"
                f"MAE   = {mae:.3f} m³/s\n"
                f"r²    = {r2:.3f}\n"
                f"N     = {n}\n"
                f"LSTM weight = {w:.2f}"
            )
            ax4.text(0.97, 0.97, metrics_txt, transform=ax4.transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                     family='monospace')

            canvas = FigureCanvasTkAgg(fig, master=body)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=8, pady=8)
            try:
                body.update_idletasks()
                _fit_canvas.configure(scrollregion=_fit_canvas.bbox("all"))
            except Exception:
                pass

            self.log(f"Model fit diagnostic shown — Val NSE={nse:.3f}, Pearson r²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

        except Exception as e:
            self.log(f"Show Model Fit failed: {e}")
            messagebox.showerror("Fit Diagnostic Error", str(e))

    def inspect_models(self):
        """Show a quick human-readable inspection of the loaded / trained models."""
        try:
            win = ttk.Toplevel(self.root)
            win.title("Model Inspection")
            txt = scrolledtext.ScrolledText(win, width=120, height=40, font=("Consolas", 10))
            txt.pack(fill=BOTH, expand=True)
            txt.insert(END, "Model Inspection Report\n\n")
            txt.insert(END, f"Feature columns: {getattr(self, 'feature_columns', None)}\n")
            txt.insert(END, f"Sequence length: {getattr(self, 'seq_length', None)}\n")
            txt.insert(END, f"MC samples: {getattr(self, 'mc_samples_var', None).get() if getattr(self, 'mc_samples_var', None) else 'N/A'}\n\n")

            try:
                if getattr(self, 'scaler_features', None) is not None:
                    mins = getattr(self.scaler_features, 'data_min_', None)
                    maxs = getattr(self.scaler_features, 'data_max_', None)
                    txt.insert(END, f"Scaler features min: {mins}\n")
                    txt.insert(END, f"Scaler features max: {maxs}\n\n")
                if getattr(self, 'scaler_level', None) is not None:
                    txt.insert(END, f"Scaler level min/max: {getattr(self.scaler_level,'data_min_',None)}/{getattr(self.scaler_level,'data_max_',None)}\n\n")
            except Exception:
                pass

            try:
                if getattr(self, 'lstm_model', None) is not None:
                    txt.insert(END, "LSTM Model Summary:\n")
                    lines = []
                    self.lstm_model.summary(print_fn=lambda s: lines.append(s))
                    for l in lines:
                        txt.insert(END, l + "\n")
                    txt.insert(END, "\n")
                else:
                    txt.insert(END, "LSTM: Not available\n\n")
            except Exception as e:
                txt.insert(END, f"LSTM summary failed: {e}\n\n")

            try:
                if getattr(self, 'rf_model', None) is not None:
                    txt.insert(END, "RandomForest inspection:\n")
                    try:
                        importances = getattr(self.rf_model, 'feature_importances_', None)
                        if importances is not None and getattr(self, 'feature_columns', None) is not None and getattr(self, 'seq_length', None) is not None:
                            n_feat = len(self.feature_columns)
                            seq = int(self.seq_length)
                            if importances.size == seq * n_feat:
                                resh = importances.reshape(seq, n_feat)
                                per_feat = np.mean(resh, axis=0)
                                feat_pairs = list(zip(self.feature_columns, per_feat))
                                feat_pairs.sort(key=lambda x: x[1], reverse=True)
                                txt.insert(END, "Top features (aggregated across timesteps):\n")
                                for f, v in feat_pairs[:20]:
                                    txt.insert(END, f"  - {f}: {v:.6f}\n")
                            else:
                                txt.insert(END, f"RF importances shape mismatch: expected {seq*n_feat}, got {importances.size}\n")
                        else:
                            txt.insert(END, "RF importances not available or missing metadata.\n")
                    except Exception as e:
                        txt.insert(END, f"RF inspection failed: {e}\n")
                    txt.insert(END, "\n")
                else:
                    txt.insert(END, "RandomForest: Not available\n\n")
            except Exception:
                pass

            txt.configure(state='disabled')
        except Exception as e:
            messagebox.showerror("Inspect Failed", str(e))

    def on_close(self):
        """Handle window close: request training stop, wait briefly, then destroy GUI."""
        try:
            self.log("Shutting down application...")
        except Exception:
            pass
        self._closing = True
        self.stop_training = True
        try:
            if self.training_thread and self.training_thread.is_alive():
                self.log("Waiting for training thread to stop...")
                self.training_thread.join(timeout=3)
        except Exception:
            pass
        try:
            if getattr(self, '_after_id', None) is not None:
                try:
                    self.root.after_cancel(self._after_id)
                except Exception:
                    pass
                self._after_id = None
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            try:
                self.root.quit()
            except Exception:
                pass

    def validate_thresholds(self):
        """Ensure elevated < moderate < high. If not, sort and reassign preserving user intent."""
        try:
            e = float(self.elevated_var.get())
            m = float(self.moderate_var.get())
            h = float(self.high_var.get())
            vals = sorted([e, m, h])
            if vals[0] != e:
                try:
                    self.elevated_var.set(vals[0])
                except Exception:
                    pass
            if vals[1] != m:
                try:
                    self.moderate_var.set(vals[1])
                except Exception:
                    pass
            if vals[2] != h:
                try:
                    self.high_var.set(vals[2])
                except Exception:
                    pass
        except Exception:
            pass

    def get_risk_label(self, level: float) -> str:
        """Return risk label based on current UI thresholds."""
        try:
            e = float(self.elevated_var.get())
            m = float(self.moderate_var.get())
            h = float(self.high_var.get())
        except Exception:
            e, m, h = 3.0, 5.0, 8.0
        try:
            level = float(level)
        except Exception:
            return "LOW"
        if level > h:
            return "HIGH RISK"
        if level > m:
            return "MODERATE"
        if level > e:
            return "ELEVATED"
        return "LOW"
    
    def show_optuna_viz(self):
        if not self.study:
            messagebox.showwarning("No Study", "Run training first!")
            return
        params = ["units1", "units2", "dropout", "lr", "batch_size"]
        try:
            trial_param_names = set()
            for t in (self.study.trials or []):
                try:
                    trial_param_names.update((t.params or {}).keys())
                except Exception:
                    pass

            if 'delta' in trial_param_names:
                params.append('delta')
            elif 'Hdelta' in trial_param_names:
                params.append('Hdelta')

            if 'l2' in trial_param_names:
                params.append('l2')
        except Exception:
            pass
        fig = plot_parallel_coordinate(self.study, params=params)
        fig.update_layout(
                title="Optuna Hyperparameter Search Journey",
                height=700,
                paper_bgcolor="#0e0e0e",
                plot_bgcolor="#0e0e0e",
                font=dict(color="#00ff99", size=12)
            )

        tmp1 = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        try:
            pio.write_html(fig, file=tmp1.name, auto_open=False, full_html=True)
            webbrowser.open_new_tab('file://' + os.path.abspath(tmp1.name))
        except Exception:
            try:
                fig.show()
            except Exception:
                messagebox.showwarning("Visualization", "Could not open parallel-coordinate visualization in browser.")

        fig2 = plot_contour(self.study, params=["units1", "units2"])
        fig2.update_layout(
            title="3D Search Space - Best Area in Green",
            paper_bgcolor="#0e0e0e",
            plot_bgcolor="#0e0e0e",
            font=dict(color="#00ff99")
            )

        tmp2 = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        try:
            pio.write_html(fig2, file=tmp2.name, auto_open=False, full_html=True)
            webbrowser.open_new_tab('file://' + os.path.abspath(tmp2.name))
        except Exception:
            try:
                fig2.show()
            except Exception:
                messagebox.showwarning("Visualization", "Could not open contour visualization in browser.")

if __name__ == "__main__":
    app = ttk.Window(themename="darkly")
    predictor = FloodPredictorApp(app)
    app.mainloop()
