# Love it. Let’s build a Tkinter desktop app that:
# - Lets you pick dataset type first: Tabular Data or Images
# - Cleans data automatically before training
# - Trains multiple models, with 50% split or 100% (k-fold CV)
# - Shows metrics and logs
# - Exports trained models (with preprocessing and metadata)
#
# What you’ll get:
# - Tabular data: automatic cleaning (dates → features, missing values, constant columns removal, outlier clipping, encoding, scaling), classification/regression detection, multiple models + optional hyperparameter tuning, metrics, export as .joblib
# - Images: easy transfer learning (MobileNetV2), 50% train/50% validation or 100% k-fold CV, accuracy and report, export as a SavedModel with metadata
#
# How to run:
# 1) Install dependencies
#    - CPU-friendly install:
#      pip install -U pandas numpy scikit-learn joblib pillow
#      pip install -U tensorflow==2.15.*  # or torch if you prefer; this uses TF
#    - On Mac with Apple Silicon:
#      pip install -U tensorflow-macos==2.15.*
#    - Optional for nicer logs:
#      pip install -U rich
#
# 2) Save the code below as app_tk.py
#
# 3) Run
#    python app_tk.py
#
# 4) For images, organize your data like:
#    dataset_root/
#      class_a/ img1.jpg, img2.jpg, ...
#      class_b/ img3.jpg, img4.jpg, ...
#
# Code (single file: app_tk.py):
# ```python
import os
import sys
import json
import time
import queue
import threading
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# ML stack
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# TensorFlow (for images)
try:
    import tensorflow as tf
except Exception:
    tf = None

# Tkinter UI
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

RANDOM_STATE = 42

# ---------------------------
# Utilities / Helpers
# ---------------------------

def infer_task_type(y: pd.Series) -> str:
    try:
        is_numeric = pd.api.types.is_numeric_dtype(y)
    except Exception:
        is_numeric = False
    unique_ratio = y.nunique(dropna=True) / max(1, len(y))
    if is_numeric and (y.nunique(dropna=True) > 20 or unique_ratio > 0.05):
        return "regression"
    return "classification"

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def get_sklearn_version():
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "unknown"

# ---------------------------
# Tabular Preprocessor
# ---------------------------

class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Cleans and encodes tabular data:
    - Parse date-like columns -> year, month, day, dow, hour; drop originals
    - Drop constant columns
    - Clip numeric outliers to [1%, 99%]
    - Impute missing (num=median, cat=most_frequent)
    - One-hot encode categoricals
    - Scale numeric
    Returns numpy array; keeps metadata for export.
    """
    def __init__(self, date_threshold=0.8, clip_quantiles=(0.01, 0.99)):
        self.date_threshold = date_threshold
        self.clip_quantiles = clip_quantiles

    def _detect_date_col(self, s: pd.Series) -> bool:
        try:
            if pd.api.types.is_datetime64_any_dtype(s):
                return True
            sample = s.dropna()
            if len(sample) == 0:
                return False
            sample = sample.sample(min(400, len(sample)), random_state=RANDOM_STATE)
            parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
            frac = parsed.notna().mean()
            return frac >= self.date_threshold
        except Exception:
            return False

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        # Date-like detection
        self.date_cols_ = []
        for col in X.columns:
            if X[col].dtype == 'O' or pd.api.types.is_datetime64_any_dtype(X[col]):
                if self._detect_date_col(X[col]):
                    self.date_cols_.append(col)

        # Expand dates
        for col in self.date_cols_:
            parsed = pd.to_datetime(X[col], errors='coerce', infer_datetime_format=True)
            X[f"{col}_year"] = parsed.dt.year
            X[f"{col}_month"] = parsed.dt.month
            X[f"{col}_day"] = parsed.dt.day
            X[f"{col}_dow"] = parsed.dt.dayofweek
            X[f"{col}_hour"] = parsed.dt.hour

        X = X.drop(columns=self.date_cols_, errors='ignore')

        # Drop constant columns
        nunique = X.nunique(dropna=True)
        self.constant_cols_ = nunique[nunique <= 1].index.tolist()
        X = X.drop(columns=self.constant_cols_, errors='ignore')

        # Determine numeric/categorical
        self.numeric_cols_ = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        # Percentiles for clipping
        self.num_percentiles_ = {}
        for col in self.numeric_cols_:
            s = pd.to_numeric(X[col], errors='coerce')
            lo = s.quantile(self.clip_quantiles[0])
            hi = s.quantile(self.clip_quantiles[1])
            if pd.isna(lo): lo = s.min()
            if pd.isna(hi): hi = s.max()
            self.num_percentiles_[col] = (float(lo) if lo is not None else None,
                                          float(hi) if hi is not None else None)

        # Imputers
        self.num_imputer_ = SimpleImputer(strategy="median")
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")

        if len(self.numeric_cols_) > 0:
            num_X = pd.DataFrame({c: pd.to_numeric(X[c], errors='coerce') for c in self.numeric_cols_})
            num_arr = self.num_imputer_.fit_transform(num_X)
        else:
            num_arr = np.empty((len(X), 0))

        if len(self.categorical_cols_) > 0:
            cat_X = X[self.categorical_cols_].astype("object")
            cat_arr = self.cat_imputer_.fit_transform(cat_X)
        else:
            cat_arr = np.empty((len(X), 0), dtype=object)

        # Encoder & scaler
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if len(self.categorical_cols_) > 0:
            self.ohe_.fit(pd.DataFrame(cat_arr, columns=self.categorical_cols_))

        self.scaler_ = StandardScaler()
        if len(self.numeric_cols_) > 0:
            self.scaler_.fit(num_arr)

        num_names = self.numeric_cols_
        ohe_names = self.ohe_.get_feature_names_out(self.categorical_cols_).tolist() if len(self.categorical_cols_) else []
        self.output_feature_names_ = num_names + ohe_names

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        assert hasattr(self, "fitted_"), "Preprocessor not fitted"
        X = X.copy()

        # Expand known date columns
        for col in self.date_cols_:
            if col in X.columns:
                parsed = pd.to_datetime(X[col], errors='coerce', infer_datetime_format=True)
            else:
                parsed = pd.Series(pd.NaT, index=X.index)
            X[f"{col}_year"] = parsed.dt.year
            X[f"{col}_month"] = parsed.dt.month
            X[f"{col}_day"] = parsed.dt.day
            X[f"{col}_dow"] = parsed.dt.dayofweek
            X[f"{col}_hour"] = parsed.dt.hour

        X = X.drop(columns=self.date_cols_, errors='ignore')
        X = X.drop(columns=self.constant_cols_, errors='ignore')

        for col in self.numeric_cols_:
            if col not in X.columns:
                X[col] = np.nan
        for col in self.categorical_cols_:
            if col not in X.columns:
                X[col] = np.nan

        # Numeric: coerce and clip
        if len(self.numeric_cols_) > 0:
            num_X = pd.DataFrame({c: pd.to_numeric(X[c], errors='coerce') for c in self.numeric_cols_})
            for col in self.numeric_cols_:
                lo, hi = self.num_percentiles_[col]
                if lo is not None and hi is not None:
                    num_X[col] = num_X[col].clip(lo, hi)
            num_arr = self.num_imputer_.transform(num_X)
            num_arr = self.scaler_.transform(num_arr)
        else:
            num_arr = np.empty((len(X), 0))

        # Categorical: impute + ohe
        if len(self.categorical_cols_) > 0:
            cat_X = X[self.categorical_cols_].astype("object")
            cat_arr = self.cat_imputer_.transform(cat_X)
            cat_df = pd.DataFrame(cat_arr, columns=self.categorical_cols_)
            cat_ohe = self.ohe_.transform(cat_df)
        else:
            cat_ohe = np.empty((len(X), 0))

        features = np.hstack([num_arr, cat_ohe])
        return features

    def get_feature_names_out(self):
        return list(self.output_feature_names_)

# ---------------------------
# Models and param spaces
# ---------------------------

def get_models_and_params(task: str):
    if task == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "SVC": SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
            "KNeighborsClassifier": KNeighborsClassifier(),
        }
        param_spaces = {
            "LogisticRegression": {"model__C": np.logspace(-3, 2, 20)},
            "RandomForestClassifier": {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            "GradientBoostingClassifier": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": np.logspace(-3, 0, 8),
                "model__max_depth": [2, 3, 4, 5],
            },
            "SVC": {
                "model__C": np.logspace(-2, 2, 10),
                "model__gamma": np.logspace(-3, 1, 10),
                "model__kernel": ["rbf"],
            },
            "KNeighborsClassifier": {
                "model__n_neighbors": list(range(3, 31, 2)),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=5000),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "SVR": SVR(),
            "KNeighborsRegressor": KNeighborsRegressor(),
        }
        param_spaces = {
            "Ridge": {"model__alpha": np.logspace(-3, 2, 20)},
            "Lasso": {"model__alpha": np.logspace(-3, 1, 20)},
            "RandomForestRegressor": {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            "GradientBoostingRegressor": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": np.logspace(-3, 0, 8),
                "model__max_depth": [2, 3, 4, 5],
            },
            "SVR": {
                "model__C": np.logspace(-2, 2, 10),
                "model__gamma": np.logspace(-3, 1, 10),
                "model__kernel": ["rbf"],
            },
            "KNeighborsRegressor": {
                "model__n_neighbors": list(range(3, 31, 2)),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        }
    return models, param_spaces

def cross_val_metrics(task: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits=5):
    if task == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"r2": "r2", "neg_mae": "neg_mean_absolute_error", "neg_mse": "neg_mean_squared_error"}

    cv_res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    summary = {}
    for k, vals in cv_res.items():
        if not k.startswith("test_"):
            continue
        metric_name = k.replace("test_", "")
        vals = np.array(vals)
        if task == "regression":
            if metric_name == "neg_mae":
                summary["mae"] = float(-vals.mean())
            elif metric_name == "neg_mse":
                summary["rmse"] = float(np.sqrt(-vals.mean()))
            else:
                summary["r2"] = float(vals.mean())
        else:
            summary[metric_name] = float(vals.mean())
    return summary

# ---------------------------
# Tabular training
# ---------------------------

def train_tabular(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    selected_models: List[str],
    train_mode: str,
    auto_tune: bool,
    n_iter: int,
    cv_splits: int,
    log_fn,
    progress_fn=None,
):
    # Basic cleaning before modeling
    df = df.copy()
    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")
    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        log_fn(f"Dropped {dropped} duplicate rows.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Infer task if "auto"
    if task == "auto":
        task = infer_task_type(y)
        log_fn(f"Inferred task: {task}")

    all_models, param_spaces = get_models_and_params(task)

    # Split strategy
    if train_mode == "50%":
        if task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=RANDOM_STATE)
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    results = []
    bundles = []

    total = len(selected_models)
    for idx, mname in enumerate(selected_models, start=1):
        if progress_fn:
            progress_fn(idx - 1, total)
        log_fn(f"Training {mname} ({idx}/{total}) ...")
        if mname not in all_models:
            log_fn(f"Skipped unknown model: {mname}")
            continue

        model = all_models[mname]
        pipeline = Pipeline(steps=[
            ("prep", TabularPreprocessor()),
            ("model", model),
        ])

        best_est = None
        metrics = {}

        if train_mode == "50%":
            if auto_tune and param_spaces.get(mname):
                log_fn("  Auto-tuning hyperparameters...")
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_spaces[mname],
                    n_iter=n_iter,
                    cv=(StratifiedKFold(cv_splits, shuffle=True, random_state=RANDOM_STATE) if task == "classification"
                        else KFold(cv_splits, shuffle=True, random_state=RANDOM_STATE)),
                    scoring=("accuracy" if task == "classification" else "r2"),
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    refit=True,
                )
                search.fit(X_train, y_train)
                best_est = search.best_estimator_
                log_fn(f"  Best params: {search.best_params_}")
            else:
                best_est = pipeline.fit(X_train, y_train)

            # Evaluate on holdout
            y_pred = best_est.predict(X_test)
            if task == "classification":
                acc = accuracy_score(y_test, y_pred)
                f1m = f1_score(y_test, y_pred, average="macro")
                metrics = {"accuracy": float(acc), "f1_macro": float(f1m)}
            else:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                metrics = {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}

            # Refit on full data for export
            best_est.fit(X, y)
            notes = "Hold-out 50% evaluation, then refit on full data."
        else:
            # 100% mode: CV metrics then refit
            if auto_tune and param_spaces.get(mname):
                log_fn("  Auto-tuning hyperparameters...")
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_spaces[mname],
                    n_iter=n_iter,
                    cv=(StratifiedKFold(cv_splits, shuffle=True, random_state=RANDOM_STATE) if task == "classification"
                        else KFold(cv_splits, shuffle=True, random_state=RANDOM_STATE)),
                    scoring=("accuracy" if task == "classification" else "r2"),
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    refit=True,
                )
                search.fit(X, y)
                best_est = search.best_estimator_
                log_fn(f"  Best params: {search.best_params_}")
                metrics = cross_val_metrics(task, best_est, X, y, n_splits=cv_splits)
            else:
                metrics = cross_val_metrics(task, pipeline, X, y, n_splits=cv_splits)
                best_est = pipeline.fit(X, y)
            notes = f"{cv_splits}-fold cross-validation, refit on full data."

        # Save bundle
        bundle = {
            "type": "tabular",
            "task": task,
            "target": target_col,
            "features": [c for c in df.columns if c != target_col],
            "pipeline": best_est,
            "metrics": metrics,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sklearn_version": get_sklearn_version(),
            "notes": notes,
            "model_name": mname,
        }
        bundles.append(bundle)

        row = {"Model": mname}
        row.update(metrics)
        results.append(row)

        log_fn(f"  Done: {mname} | {json.dumps(metrics)}")

    if progress_fn:
        progress_fn(total, total)
    return results, bundles

# ---------------------------
# Image training (TensorFlow)
# ---------------------------

class ImageTrainer:
    def __init__(self, log_fn):
        self.log = log_fn
        self.model = None
        self.class_names = None
        self.image_size = (224, 224)
        self.history = None
        if tf is None:
            self.log("TensorFlow not installed. Image mode will be disabled.")

    def build_model(self, num_classes: int, base_name: str = "MobileNetV2", image_size=(224, 224)):
        if tf is None:
            raise RuntimeError("TensorFlow not available")
        self.image_size = image_size
        input_shape = image_size + (3,)
        if base_name == "MobileNetV2":
            base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        elif base_name == "EfficientNetB0":
            base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
            preprocess = tf.keras.applications.efficientnet.preprocess_input
        else:
            base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

        base.trainable = False  # start with frozen base
        inputs = tf.keras.Input(shape=input_shape)
        x = preprocess(inputs)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def _ds_from_dir(self, data_dir, subset, val_split, seed, batch_size, image_size):
        return tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="int",
            validation_split=val_split,
            subset=subset,
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

    def train_50_split(self, data_dir, base_model_name, img_size, batch_size, epochs, seed=RANDOM_STATE):
        if tf is None:
            raise RuntimeError("TensorFlow not available")
        self.log("Loading dataset with 50% train / 50% validation...")
        train_ds = self._ds_from_dir(data_dir, "training", 0.5, seed, batch_size, img_size)
        val_ds = self._ds_from_dir(data_dir, "validation", 0.5, seed, batch_size, img_size)

        self.class_names = train_ds.class_names
        num_classes = len(self.class_names)
        self.model = self.build_model(num_classes, base_model_name, img_size)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
        ]
        self.log(f"Training {base_model_name} for up to {epochs} epochs...")
        self.history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)
        self.log("Evaluating on validation set...")
        loss, acc = self.model.evaluate(val_ds, verbose=0)
        self.log(f"Validation accuracy: {acc:.4f}")
        return {"accuracy": float(acc)}

    def _paths_and_labels(self, data_dir):
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        file_paths, labels = [], []
        for idx, cname in enumerate(class_names):
            cdir = os.path.join(data_dir, cname)
            for root, _, files in os.walk(cdir):
                for f in files:
                    fp = os.path.join(root, f)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")):
                        file_paths.append(fp)
                        labels.append(idx)
        return np.array(file_paths), np.array(labels), class_names

    def _ds_from_paths(self, paths, labels, batch_size, image_size, shuffle=False, augment=False):
        def _load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, image_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_STATE)
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
            ])
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def train_kfold(self, data_dir, base_model_name, img_size, batch_size, epochs, cv_splits=5):
        if tf is None:
            raise RuntimeError("TensorFlow not available")
        paths, labels, class_names = self._paths_and_labels(data_dir)
        if len(paths) == 0:
            raise RuntimeError("No images found. Ensure directory has subfolders per class with images.")
        self.class_names = class_names
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_acc = []
        fold = 0
        for train_idx, val_idx in skf.split(paths, labels):
            fold += 1
            self.log(f"Fold {fold}/{cv_splits}: train={len(train_idx)}, val={len(val_idx)}")
            train_ds = self._ds_from_paths(paths[train_idx], labels[train_idx], batch_size, img_size, shuffle=True, augment=True)
            val_ds = self._ds_from_paths(paths[val_idx], labels[val_idx], batch_size, img_size, shuffle=False)

            num_classes = len(class_names)
            model = self.build_model(num_classes, base_model_name, img_size)
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)]
            hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)
            loss, acc = model.evaluate(val_ds, verbose=0)
            fold_acc.append(acc)
            self.log(f"  Fold {fold} acc: {acc:.4f}")

        mean_acc = float(np.mean(fold_acc))
        self.log(f"Mean CV accuracy over {cv_splits} folds: {mean_acc:.4f}")

        # Train final model on all data
        full_ds = self._ds_from_paths(paths, labels, batch_size, img_size, shuffle=True, augment=True)
        num_classes = len(class_names)
        self.model = self.build_model(num_classes, base_model_name, img_size)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=2, restore_best_weights=True)]
        self.model.fit(full_ds, epochs=max(epochs - 1, 1), callbacks=callbacks, verbose=0)
        return {"cv_accuracy": mean_acc}

    def export(self, export_dir):
        if tf is None:
            raise RuntimeError("TensorFlow not available")
        if self.model is None:
            raise RuntimeError("No image model to export.")
        os.makedirs(export_dir, exist_ok=True)
        model_dir = os.path.join(export_dir, "saved_model")
        meta_path = os.path.join(export_dir, "metadata.json")
        tf.keras.models.save_model(self.model, model_dir)
        meta = {
            "type": "image",
            "class_names": self.class_names,
            "image_size": self.image_size,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tf_version": tf.__version__,
            "base": "MobileNetV2",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return export_dir

# ---------------------------
# Tkinter UI
# ---------------------------

class TabularFrame(ttk.Frame):
    def __init__(self, master, log_queue, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.log_queue = log_queue
        self.df = None
        self.file_path = tk.StringVar(value="No file loaded")
        self.target_var = tk.StringVar(value="")
        self.task_var = tk.StringVar(value="auto")  # auto | classification | regression
        self.train_mode_var = tk.StringVar(value="50%")  # 50% or 100%
        self.cv_splits_var = tk.IntVar(value=5)
        self.auto_tune_var = tk.BooleanVar(value=True)
        self.n_iter_var = tk.IntVar(value=15)
        self.models_listbox = None
        self.results_tree = None
        self.trained_bundles = []  # list of dict bundles
        self.progress = None

        self._build_ui()

    def log(self, msg: str):
        self.log_queue.put(msg)

    def _build_ui(self):
        # File load
        file_frame = ttk.Frame(self)
        file_frame.pack(fill="x", padx=8, pady=6)
        ttk.Button(file_frame, text="Load CSV", command=self.load_csv).pack(side="left")
        ttk.Button(file_frame, text="Load sample: Iris", command=lambda: self.load_sample("iris")).pack(side="left", padx=4)
        ttk.Button(file_frame, text="Load sample: California Housing", command=lambda: self.load_sample("housing")).pack(side="left", padx=4)
        ttk.Label(file_frame, textvariable=self.file_path, foreground="#555").pack(side="left", padx=10)

        # Target + task
        target_frame = ttk.Frame(self)
        target_frame.pack(fill="x", padx=8, pady=6)
        ttk.Label(target_frame, text="Target column:").pack(side="left")
        self.target_cb = ttk.Combobox(target_frame, textvariable=self.target_var, state="readonly", width=32, values=[])
        self.target_cb.pack(side="left", padx=6)

        ttk.Label(target_frame, text="Task:").pack(side="left", padx=(20, 2))
        for label, val in [("Auto", "auto"), ("Classification", "classification"), ("Regression", "regression")]:
            ttk.Radiobutton(target_frame, text=label, variable=self.task_var, value=val).pack(side="left")

        # Train config
        cfg_frame = ttk.Frame(self)
        cfg_frame.pack(fill="x", padx=8, pady=6)
        ttk.Label(cfg_frame, text="Train mode:").pack(side="left")
        ttk.Radiobutton(cfg_frame, text="Use 50% for training", variable=self.train_mode_var, value="50%").pack(side="left")
        ttk.Radiobutton(cfg_frame, text="Use 100% (k-fold CV)", variable=self.train_mode_var, value="100%").pack(side="left", padx=(10, 0))

        ttk.Label(cfg_frame, text="CV splits:").pack(side="left", padx=(20, 2))
        ttk.Spinbox(cfg_frame, from_=3, to=10, textvariable=self.cv_splits_var, width=5).pack(side="left")

        ttk.Checkbutton(cfg_frame, text="Auto-tune hyperparameters", variable=self.auto_tune_var).pack(side="left", padx=(20, 4))
        ttk.Label(cfg_frame, text="Iterations:").pack(side="left")
        ttk.Spinbox(cfg_frame, from_=5, to=60, increment=5, textvariable=self.n_iter_var, width=5).pack(side="left")

        # Models selection
        models_frame = ttk.LabelFrame(self, text="Models to try")
        models_frame.pack(fill="x", padx=8, pady=6)
        self.models_listbox = tk.Listbox(models_frame, selectmode="multiple", height=6, exportselection=False)
        self.models_listbox.pack(fill="x", padx=6, pady=4)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=6)
        ttk.Button(btn_frame, text="Train", command=self.start_training).pack(side="left")
        ttk.Button(btn_frame, text="Export selected model", command=self.export_selected).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(btn_frame, mode="determinate")
        self.progress.pack(side="right", fill="x", expand=True, padx=6)

        # Results
        res_frame = ttk.LabelFrame(self, text="Results")
        res_frame.pack(fill="both", expand=True, padx=8, pady=6)
        self.results_tree = ttk.Treeview(res_frame, columns=("Model", "metric1", "metric2", "metric3"), show="headings", height=8)
        for col in ("Model", "metric1", "metric2", "metric3"):
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=160, anchor="w")
        self.results_tree.pack(fill="both", expand=True, padx=6, pady=6)

    def load_csv(self):
        path = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
            return
        self.df = df
        self.file_path.set(path)
        cols = df.columns.tolist()
        self.target_cb["values"] = cols
        if cols:
            self.target_cb.current(0)
        # populate models list with defaults (classification defaults)
        self.refresh_models("classification")

    def load_sample(self, which: str):
        try:
            if which == "iris":
                from sklearn.datasets import load_iris
                data = load_iris(as_frame=True)
                self.df = data.frame.copy()
                self.file_path.set("Sample: Iris")
                self.target_cb["values"] = self.df.columns.tolist()
                self.target_cb.set("target")
                self.refresh_models("classification")
            else:
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing(as_frame=True)
                self.df = data.frame.copy()
                self.file_path.set("Sample: California Housing")
                self.target_cb["values"] = self.df.columns.tolist()
                self.target_cb.set("MedHouseVal")
                self.refresh_models("regression")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample: {e}")

    def refresh_models(self, task_hint="classification"):
        models, _ = get_models_and_params(task_hint)
        self.models_listbox.delete(0, tk.END)
        defaults = ["RandomForestClassifier", "LogisticRegression", "GradientBoostingClassifier"] if task_hint == "classification" else ["RandomForestRegressor", "Ridge", "GradientBoostingRegressor"]
        for m in models.keys():
            self.models_listbox.insert(tk.END, m)
            if m in defaults:
                self.models_listbox.selection_set(tk.END)

    def start_training(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV file first.")
            return
        target = self.target_var.get().strip()
        if target == "" or target not in self.df.columns:
            messagebox.showwarning("Select target", "Choose a valid target column.")
            return

        # Determine task hint to pick default models list if empty
        task_choice = self.task_var.get()
        if task_choice == "auto":
            task_choice = infer_task_type(self.df[target])
        sel_indices = self.models_listbox.curselection()
        if not sel_indices:
            self.refresh_models(task_choice)
            sel_indices = self.models_listbox.curselection()
            if not sel_indices:
                messagebox.showwarning("No models", "Select at least one model.")
                return

        selected_models = [self.models_listbox.get(i) for i in sel_indices]

        # Disable train button during run
        for child in self.winfo_children():
            child.configure(state="disabled")

        def progress_fn(done, total):
            self.progress["maximum"] = total
            self.progress["value"] = done

        def worker():
            try:
                results, bundles = train_tabular(
                    self.df,
                    target_col=target,
                    task=self.task_var.get(),
                    selected_models=selected_models,
                    train_mode=("50%" if self.train_mode_var.get() == "50%" else "100%"),
                    auto_tune=self.auto_tune_var.get(),
                    n_iter=self.n_iter_var.get(),
                    cv_splits=self.cv_splits_var.get(),
                    log_fn=self.log,
                    progress_fn=progress_fn,
                )
                self.trained_bundles.extend(bundles)
                # Update UI table on main thread
                def update_table():
                    for row in self.results_tree.get_children():
                        self.results_tree.delete(row)
                    for r in results:
                        # Normalize metric columns for display
                        keys = list(r.keys())
                        metrics = [k for k in keys if k != "Model"]
                        metric_vals = [f"{r.get(metrics[0], ''):.4f}" if isinstance(r.get(metrics[0], None), float) else r.get(metrics[0], "")] if metrics else [""]
                        metric_vals2 = [f"{r.get(metrics[1], ''):.4f}" if len(metrics) > 1 and isinstance(r.get(metrics[1], None), float) else (r.get(metrics[1], "") if len(metrics) > 1 else "")]
                        metric_vals3 = [f"{r.get(metrics[2], ''):.4f}" if len(metrics) > 2 and isinstance(r.get(metrics[2], None), float) else (r.get(metrics[2], "") if len(metrics) > 2 else "")]
                        self.results_tree.insert("", "end", values=(r["Model"], *(metric_vals + metric_vals2 + metric_vals3)))
                self.after(0, update_table)
            except Exception as e:
                self.log(f"Error: {e}")
                messagebox.showerror("Training error", str(e))
            finally:
                def reenable():
                    for child in self.winfo_children():
                        try:
                            child.configure(state="normal")
                        except tk.TclError:
                            pass
                    self.progress["value"] = 0
                self.after(0, reenable)

        threading.Thread(target=worker, daemon=True).start()

    def export_selected(self):
        sel = self.results_tree.selection()
        if not sel:
            messagebox.showinfo("Select a model", "Select a row in the results to export.")
            return
        model_name = self.results_tree.item(sel[0], "values")[0]
        # Find bundle
        bundle = None
        for b in self.trained_bundles:
            if b.get("model_name") == model_name:
                bundle = b
                break
        if bundle is None:
            messagebox.showerror("Not found", "Could not find model bundle for export.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".joblib",
            filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")],
            initialfile=f"{model_name}.joblib",
        )
        if not path:
            return
        try:
            joblib.dump(bundle, path)
            messagebox.showinfo("Exported", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

class ImageFrame(ttk.Frame):
    def __init__(self, master, log_queue, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.log_queue = log_queue
        self.data_dir_var = tk.StringVar(value="")
        self.base_model_var = tk.StringVar(value="MobileNetV2")
        self.size_var = tk.StringVar(value="224")
        self.batch_var = tk.IntVar(value=32)
        self.epochs_var = tk.IntVar(value=6)
        self.train_mode_var = tk.StringVar(value="50%")  # 50% or 100%
        self.cv_splits_var = tk.IntVar(value=5)
        self.progress = None

        self.trainer = ImageTrainer(self.log)
        self.metrics_last = None

        self._build_ui()

    def log(self, msg: str):
        self.log_queue.put(f"[IMG] {msg}")

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Select image dataset folder", command=self.pick_dir).pack(side="left")
        ttk.Label(top, textvariable=self.data_dir_var, foreground="#555").pack(side="left", padx=8)

        cfg = ttk.LabelFrame(self, text="Training config")
        cfg.pack(fill="x", padx=8, pady=6)
        ttk.Label(cfg, text="Base model:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(cfg, state="readonly", textvariable=self.base_model_var, values=["MobileNetV2", "EfficientNetB0"]).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(cfg, text="Image size:").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Spinbox(cfg, from_=96, to=384, increment=16, textvariable=self.size_var, width=6).grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(cfg, text="Batch:").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        ttk.Spinbox(cfg, from_=8, to=128, increment=8, textvariable=self.batch_var, width=6).grid(row=0, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(cfg, text="Epochs:").grid(row=0, column=6, sticky="w", padx=4, pady=4)
        ttk.Spinbox(cfg, from_=2, to=30, increment=1, textvariable=self.epochs_var, width=6).grid(row=0, column=7, sticky="w", padx=4, pady=4)

        ttk.Label(cfg, text="Train mode:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(cfg, text="Use 50% for training", variable=self.train_mode_var, value="50%").grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(cfg, text="Use 100% (k-fold CV)", variable=self.train_mode_var, value="100%").grid(row=1, column=2, sticky="w", padx=4, pady=4)

        ttk.Label(cfg, text="CV splits:").grid(row=1, column=3, sticky="w", padx=4, pady=4)
        ttk.Spinbox(cfg, from_=3, to=10, textvariable=self.cv_splits_var, width=6).grid(row=1, column=4, sticky="w", padx=4, pady=4)

        btn = ttk.Frame(self)
        btn.pack(fill="x", padx=8, pady=6)
        ttk.Button(btn, text="Train", command=self.start_training).pack(side="left")
        ttk.Button(btn, text="Export model", command=self.export_model).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(btn, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True, padx=6)

    def pick_dir(self):
        path = filedialog.askdirectory(title="Select dataset root folder")
        if path:
            self.data_dir_var.set(path)

    def start_training(self):
        if tf is None:
            messagebox.showerror("TensorFlow missing", "Install TensorFlow to use image training.")
            return
        data_dir = self.data_dir_var.get().strip()
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showwarning("Select folder", "Pick a valid dataset root folder.")
            return
        img_size = (int(self.size_var.get()), int(self.size_var.get()))
        batch = self.batch_var.get()
        epochs = self.epochs_var.get()
        base = self.base_model_var.get()
        mode = self.train_mode_var.get()
        splits = self.cv_splits_var.get()

        # Disable controls
        for child in self.winfo_children():
            child.configure(state="disabled")
        self.progress.start(10)

        def worker():
            try:
                if mode == "50%":
                    metrics = self.trainer.train_50_split(
                        data_dir=data_dir,
                        base_model_name=base,
                        img_size=img_size,
                        batch_size=batch,
                        epochs=epochs,
                    )
                else:
                    metrics = self.trainer.train_kfold(
                        data_dir=data_dir,
                        base_model_name=base,
                        img_size=img_size,
                        batch_size=batch,
                        epochs=max(epochs - 1, 2),
                        cv_splits=splits,
                    )
                self.metrics_last = metrics
                self.log(f"Training complete. Metrics: {metrics}")
                messagebox.showinfo("Done", f"Training complete.\n{metrics}")
            except Exception as e:
                self.log(f"Error: {e}")
                messagebox.showerror("Training error", str(e))
            finally:
                def reenable():
                    for child in self.winfo_children():
                        try:
                            child.configure(state="normal")
                        except tk.TclError:
                            pass
                    self.progress.stop()
                self.after(0, reenable)

        threading.Thread(target=worker, daemon=True).start()

    def export_model(self):
        if self.trainer.model is None:
            messagebox.showinfo("No model", "Train an image model first.")
            return
        path = filedialog.askdirectory(title="Select export folder")
        if not path:
            return
        try:
            out_dir = os.path.join(path, f"image_model_{int(time.time())}")
            self.trainer.export(out_dir)
            messagebox.showinfo("Exported", f"Saved to {out_dir}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quick Model Builder (Tabular + Images)")
        self.geometry("980x720")
        try:
            self.iconbitmap(default="")
        except Exception:
            pass

        self.dataset_type = tk.StringVar(value="tabular")  # tabular | images
        self.log_queue = queue.Queue()

        self._build_ui()
        self._poll_log()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Dataset type:").pack(side="left")
        ttk.Radiobutton(top, text="Tabular Data", variable=self.dataset_type, value="tabular", command=self._switch).pack(side="left")
        ttk.Radiobutton(top, text="Images", variable=self.dataset_type, value="images", command=self._switch).pack(side="left", padx=(6, 0))

        # Main frames
        self.tabular_frame = TabularFrame(self, self.log_queue)
        self.images_frame = ImageFrame(self, self.log_queue)
        self.tabular_frame.pack(fill="both", expand=True)

        # Log area
        log_frame = ttk.LabelFrame(self, text="Logs")
        log_frame.pack(fill="both", expand=False, padx=8, pady=6)
        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _switch(self):
        if self.dataset_type.get() == "tabular":
            self.images_frame.pack_forget()
            self.tabular_frame.pack(fill="both", expand=True)
        else:
            self.tabular_frame.pack_forget()
            self.images_frame.pack(fill="both", expand=True)

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _append_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

if __name__ == "__main__":
    app = App()
    app.mainloop()
