#!/usr/bin/env python3
"""
XGBoost PIPELINE FOR BP PREDICTION
==================================================
Uses Multiple Imputation (MICE) to handle missing data properly

Key Changes:
1. IterativeImputer (MICE) applied within each CV fold
2. No missing indicators or special missing handling
3. SHAP analysis on imputed data only (no "Unknown" categories)
4. Applies Platt (logistic) calibration for probability calibration

Target : Best Performers

Authors: Marc-Anthony Chouillard, Clément Pastier
Date: oct 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import argparse
import sys

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    brier_score_loss, precision_recall_curve,
    average_precision_score, auc, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import shap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constants
RANDOM_STATE = 42
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 3
N_TRIALS = 200
  # expanded for more thorough optimisation
N_BOOTSTRAP = 1000  # at least 1000 for pub
CONFIDENCE_LEVEL = 0.95

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Publication colors
COLORS = {
    'primary': '#0066CC',
    'secondary': '#CC0000',
    'accent': '#FF9900',
    'success': '#009900',
    'neutral': '#666666',
    'light': '#CCCCCC',
    'warning': '#CC6600',
    'info': '#0099CC'
}


def save_plot_all_formats(fig, base_dir, name, dpi=300):
    """Save plot in PNG, PDF, and SVG formats"""
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    formats = ['png', 'pdf', 'svg']
    saved_paths = []
    
    for fmt in formats:
        output_path = base_dir / f'{name}.{fmt}'
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', format=fmt)
        saved_paths.append(output_path)
    
    return saved_paths


def calculate_ece(y_true, y_proba, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_mce(y_true, y_proba, n_bins=10):
    """Calculate Maximum Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def bootstrap_roc_auc(y_true, y_scores, n_bootstrap=1000, ci_level=0.95):
    """Calculate ROC curve with bootstrap confidence intervals"""
    np.random.seed(RANDOM_STATE)
    
    # Original ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc_original = auc(fpr, tpr)
    
    # Bootstrap
    bootstrapped_scores = []
    bootstrapped_tprs = []
    
    # Common FPR points for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    
    for i in range(n_bootstrap):
        indices = resample(range(len(y_true)), random_state=RANDOM_STATE + i)
        
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(auc(fpr_boot, tpr_boot))
        
        tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
        tpr_interp[0] = 0.0
        bootstrapped_tprs.append(tpr_interp)
    
    # Calculate confidence intervals
    alpha = 1 - ci_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    auc_ci_lower = np.percentile(bootstrapped_scores, lower_percentile)
    auc_ci_upper = np.percentile(bootstrapped_scores, upper_percentile)
    
    bootstrapped_tprs = np.array(bootstrapped_tprs)
    tpr_lower = np.percentile(bootstrapped_tprs, lower_percentile, axis=0)
    tpr_upper = np.percentile(bootstrapped_tprs, upper_percentile, axis=0)
    tpr_mean = np.mean(bootstrapped_tprs, axis=0)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc_original,
        'auc_ci': (auc_ci_lower, auc_ci_upper),
        'fpr_mean': mean_fpr,
        'tpr_mean': tpr_mean,
        'tpr_lower': tpr_lower,
        'tpr_upper': tpr_upper
    }


def identify_feature_types(df, exclude_cols):
    """Identify continuous vs categorical features"""
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    continuous_features = []
    categorical_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            n_unique = df[col].nunique()
            # Consider as continuous if many unique values or float type
            if n_unique > 10 or df[col].dtype in ['float64', 'float32']:
                continuous_features.append(col)
            else:
                categorical_features.append(col)
        else:
            categorical_features.append(col)
    
    return continuous_features, categorical_features


class FinalImputedPipeline:
    """Production pipeline with Multiple Imputation for missing data handling"""
    
    def __init__(self, verbose: bool = True, plot_only: bool = False):
        self.verbose = verbose
        self.plot_only = plot_only
        self.script_dir = Path(__file__).parent
        self.results_dir = self.script_dir / "results"
        self.output_dir = self.results_dir / "production_imputed"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create organized subdirectories
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "visualizations"
        self.results_subdir = self.output_dir / "results"
        
        for dir_path in [self.models_dir, self.plots_dir, self.results_subdir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Results storage
        self.results = {}
        self.fold_models = []
        self.imputers = []  # Store imputers for each fold
        self.categorical_mappings = {}  # Store categorical encodings
        self.binary_features = []  # Track binary features
        self.continuous_features = []  # Track continuous features
        self.categorical_features = []  # Track multi-category features
        
    def log(self, message: str):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def create_xgb_objective(self, X_train, y_train):
        """Create Optuna objective for XGBoost optimization"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 750),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
                'gamma': trial.suggest_float('gamma', 0.0, 1.5),
                'subsample': trial.suggest_float('subsample', 0.75, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 0.85),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 12.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 3.5),
                'random_state': RANDOM_STATE,
                'tree_method': 'hist',
                # IMPORTANT: No enable_categorical since we're using imputed numeric data
                'verbosity': 0,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }

            cv_scores = []
            inner_kfold = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            for inner_train_idx, inner_val_idx in inner_kfold.split(X_train, y_train):
                X_inner_train = X_train.iloc[inner_train_idx]
                y_inner_train = y_train.iloc[inner_train_idx]
                X_inner_val = X_train.iloc[inner_val_idx]
                y_inner_val = y_train.iloc[inner_val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_inner_train, y_inner_train, verbose=False)
                preds = model.predict_proba(X_inner_val)[:, 1]
                cv_scores.append(roc_auc_score(y_inner_val, preds))

            return float(np.mean(cv_scores))
        
        return objective
    
    def run_nested_cv(self, X, y):
        """Run nested cross-validation with Multiple Imputation inside folds"""
        
        self.log("Starting Nested Cross-Validation with Multiple Imputation")
        self.log(f"Outer: {N_OUTER_FOLDS} folds, Inner: {N_INNER_FOLDS} folds")
        self.log(f"Optuna trials: {N_TRIALS}, Bootstrap: {N_BOOTSTRAP}")
        
        # Analyze missing data before starting
        missing_counts = X.isna().sum()
        features_with_missing = missing_counts[missing_counts > 0]
        self.log(f"Features with missing data: {len(features_with_missing)}")
        for feat, count in features_with_missing.head(5).items():
            self.log(f"  {feat}: {count} ({count/len(X)*100:.1f}%)")
        
        outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # Storage
        oof_predictions = np.zeros(len(y))
        fold_results = []
        calibrated_oof = np.zeros(len(y))
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            self.log(f"\n{'='*50}")
            self.log(f"FOLD {fold_idx + 1}/{N_OUTER_FOLDS}")
            self.log(f"{'='*50}")
            
            X_train_raw, X_test_raw = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Replace -1 (encoded missing) with NaN for proper imputation
            X_train_raw = X_train_raw.replace(-1, np.nan)
            X_test_raw = X_test_raw.replace(-1, np.nan)
            
            # MULTIPLE IMPUTATION within fold
            self.log("Performing Multiple Imputation...")
            imputer = IterativeImputer(
                random_state=RANDOM_STATE + fold_idx,
                max_iter=10,
                initial_strategy='median',
                imputation_order='ascending',
                verbose=0
            )
            
            # Fit imputer on training data, transform both
            X_train = pd.DataFrame(
                imputer.fit_transform(X_train_raw),
                columns=X_train_raw.columns,
                index=X_train_raw.index
            )
            X_test = pd.DataFrame(
                imputer.transform(X_test_raw),
                columns=X_test_raw.columns,
                index=X_test_raw.index
            )
            
            # Store imputer for potential future use
            self.imputers.append(imputer)
            
            # Verify no missing values remain
            assert not X_train.isna().any().any(), f"Fold {fold_idx}: Training data has NaN after imputation"
            assert not X_test.isna().any().any(), f"Fold {fold_idx}: Test data has NaN after imputation"
            
            self.log(f"Imputation complete. All missing values filled.")
            
            # Optuna optimization
            self.log(f"Running Optuna optimization ({N_TRIALS} trials)...")
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=RANDOM_STATE),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            baseline_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            baseline_params = {
                'n_estimators': 450,
                'max_depth': 4,
                'learning_rate': 0.025,
                'min_child_weight': 4,
                'gamma': 0.6,
                'subsample': 0.9,
                'colsample_bytree': 0.68,
                'reg_alpha': 1.0,
                'reg_lambda': 6.0,
                'scale_pos_weight': float(np.clip(baseline_ratio, 1.6, 3.5))
            }
            study.enqueue_trial(baseline_params)

            objective = self.create_xgb_objective(X_train, y_train)

            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            best_params = study.best_params
            formatted_params = {
                key: (round(val, 4) if isinstance(val, float) else val)
                for key, val in best_params.items()
            }
            self.log(f"Best params (fold {fold_idx + 1}): {formatted_params}")
            
            # Train final model for this fold
            self.log("Training final fold model...")
            best_params.update({
                'random_state': RANDOM_STATE,
                'tree_method': 'hist',
                'verbosity': 0,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            })
            
            if 'scale_pos_weight' not in best_params:
                imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                best_params['scale_pos_weight'] = imbalance_ratio
            model = xgb.XGBClassifier(**best_params)
            model.fit(X_train, y_train, verbose=False)
            
            # Raw predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            oof_predictions[test_idx] = y_pred_proba

            # Fit Platt calibrator (logistic regression) on training fold predictions
            y_train_pred = model.predict_proba(X_train)[:, 1]
            calibrator = LogisticRegression(max_iter=1000)
            calibrator.fit(y_train_pred.reshape(-1, 1), y_train)
            y_calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
            y_calibrated = np.clip(y_calibrated, 1e-6, 1 - 1e-6)
            calibrated_oof[test_idx] = y_calibrated
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            brier_score = brier_score_loss(y_test, y_pred_proba)
            ece = calculate_ece(y_test, y_pred_proba)
            mce = calculate_mce(y_test, y_pred_proba)
            calibrated_brier = brier_score_loss(y_test, y_calibrated)
            calibrated_ece = calculate_ece(y_test, y_calibrated)
            calibrated_mce = calculate_mce(y_test, y_calibrated)

            fold_results.append({
                'fold': fold_idx + 1,
                'auc': auc_score,
                'brier': brier_score,
                'ece': ece,
                'mce': mce,
                'calibrated_brier': calibrated_brier,
                'calibrated_ece': calibrated_ece,
                'calibrated_mce': calibrated_mce,
                'best_params': best_params,
                'best_value': study.best_value
            })
            
            self.fold_models.append(model)
            
            self.log(f"Fold {fold_idx + 1} Results:")
            self.log(f"  AUC: {auc_score:.4f}")
            self.log(f"  Brier: {brier_score:.4f}")
            self.log(f"  ECE: {ece:.4f}")
            self.log(f"  MCE: {mce:.4f}")
        
        # Store results
        self.results['fold_results'] = fold_results
        self.results['oof_predictions'] = oof_predictions
        self.results['y_true'] = y.values
        
        # Calculate overall metrics
        overall_auc = roc_auc_score(y, oof_predictions)
        overall_brier = brier_score_loss(y, oof_predictions)
        overall_ece = calculate_ece(y.values, oof_predictions)
        overall_mce = calculate_mce(y.values, oof_predictions)

        calibrated_auc = roc_auc_score(y, calibrated_oof)
        calibrated_brier = brier_score_loss(y, calibrated_oof)
        calibrated_ece = calculate_ece(y.values, calibrated_oof)
        calibrated_mce = calculate_mce(y.values, calibrated_oof)

        self.results['overall'] = {
            'auc': overall_auc,
            'auc_std': np.std([f['auc'] for f in fold_results]),
            'brier': overall_brier,
            'ece': overall_ece,
            'mce': overall_mce,
            'log_loss': log_loss(y, oof_predictions),
            'calibrated_auc': calibrated_auc,
            'calibrated_brier': calibrated_brier,
            'calibrated_ece': calibrated_ece,
            'calibrated_mce': calibrated_mce
        }

        self.log(f"\n{'='*50}")
        self.log("OVERALL RESULTS (WITH IMPUTATION)")
        self.log(f"{'='*50}")
        self.log(f"AUC: {overall_auc:.4f} ± {self.results['overall']['auc_std']:.4f}")
        self.log(f"Brier Score: {overall_brier:.4f}")
        self.log(f"ECE: {overall_ece:.4f}")
        self.log(f"MCE: {overall_mce:.4f}")
        self.log(f"Log Loss: {self.results['overall']['log_loss']:.4f}")
        self.log(f"Calibrated Brier: {calibrated_brier:.4f}")
        self.log(f"Calibrated ECE: {calibrated_ece:.4f}")
        self.log(f"Calibrated MCE: {calibrated_mce:.4f}")
        self.log(f"Calibrated AUC: {calibrated_auc:.4f}")

        self.results['calibrated_oof'] = calibrated_oof

        return self.results
    
    def train_final_model(self, X, y):
        """Train final model on full imputed dataset for SHAP analysis"""
        
        self.log("\nTraining final model on full imputed dataset...")
        
        # Replace -1 (encoded missing) with NaN for proper imputation
        X_for_imputation = X.replace(-1, np.nan)
        
        # Impute full dataset
        imputer = IterativeImputer(
            random_state=RANDOM_STATE,
            max_iter=10,
            initial_strategy='median',
            imputation_order='ascending',
            verbose=0
        )
        
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_for_imputation),
            columns=X.columns,
            index=X.index
        )
        
        # Use best hyperparameters from best fold
        best_fold = max(self.results['fold_results'], key=lambda x: x['auc'])
        best_params = best_fold['best_params']
        best_params.update({
            'random_state': RANDOM_STATE,
            'tree_method': 'hist',
            'verbosity': 0,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        })
        
        if 'scale_pos_weight' not in best_params:
            imbalance_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
            best_params['scale_pos_weight'] = imbalance_ratio

        # Train model
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_imputed, y, verbose=False)

        # Fit Platt calibrator on full dataset
        raw_probs = final_model.predict_proba(X_imputed)[:, 1]
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(raw_probs.reshape(-1, 1), y)

        self.final_model = final_model
        self.X_imputed = X_imputed
        self.final_calibrator = calibrator
        calibrated_full = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        self.final_calibrated_probs = np.clip(calibrated_full, 1e-6, 1 - 1e-6)

        return final_model, X_imputed
    
    def generate_plots(self):
        """Generate all visualization plots"""
        
        self.log("\n" + "="*50)
        self.log("GENERATING VISUALIZATIONS")
        self.log("="*50)
        
        y_true = self.results['y_true']
        y_proba = self.results['oof_predictions']
        
        # 1. Original combined ROC Curve with metrics
        self.log("\nGenerating combined ROC curve...")
        self.plot_roc_curve(y_true, y_proba)
        
        # 2. Enhanced separate plots
        self.log("\nGenerating enhanced ROC visualizations...")
        self.plot_enhanced_roc_pure(y_true, y_proba)
        self.plot_enhanced_performance_metrics(y_true, y_proba)
        self.plot_enhanced_comparative_roc(y_true, y_proba)
        
        # 3. Calibration Curve
        self.log("Generating calibration curve...")
        self.plot_calibration_curve(y_true, y_proba)
        
        # 4. Reliability Diagram
        self.log("Generating reliability diagram...")
        self.plot_reliability_diagram(y_true, y_proba)
        
        # 5. SHAP Analysis (on imputed data - no Unknown categories)
        self.log("Generating SHAP analysis...")
        self.plot_shap_analysis()
        
        self.log("\n✓ All visualizations saved to: " + str(self.plots_dir))
    
    def plot_roc_curve(self, y_true, y_proba):
        """Generate ROC curve and performance metrics plot"""
        
        roc_data = bootstrap_roc_auc(y_true, y_proba, N_BOOTSTRAP)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # LEFT PLOT: ROC Curve (without dot)
        # Confidence band
        ax1.fill_between(
            roc_data['fpr_mean'],
            roc_data['tpr_lower'],
            roc_data['tpr_upper'],
            color=COLORS['primary'],
            alpha=0.2,
            label=f'{CONFIDENCE_LEVEL*100:.0f}% CI'
        )
        
        # ROC curve (without optimal point dot)
        ax1.plot(
            roc_data['fpr'],
            roc_data['tpr'],
            color=COLORS['primary'],
            linewidth=2.5,
            label=f"AUC = {roc_data['auc']:.3f} [{roc_data['auc_ci'][0]:.3f}-{roc_data['auc_ci'][1]:.3f}]"
        )
        
        # Diagonal
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
        
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # RIGHT PLOT: Sensitivity, Specificity, PPV curves
        # Calculate metrics at different thresholds
        thresholds = np.linspace(0, 1, 101)
        sensitivity = []
        specificity = []
        ppv = []
        npv = []
        f1_scores = []
        
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            
            # Calculate confusion matrix components
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            # Calculate metrics
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # F1 score
            precision = ppv_val
            recall = sens
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            sensitivity.append(sens)
            specificity.append(spec)
            ppv.append(ppv_val)
            npv.append(npv_val)
            f1_scores.append(f1)
        
        # Plot curves
        ax2.plot(thresholds, sensitivity, label='Sensitivity', color='blue', linewidth=2)
        ax2.plot(thresholds, specificity, label='Specificity', color='green', linewidth=2)
        ax2.plot(thresholds, ppv, label='PPV', color='red', linewidth=2)
        ax2.plot(thresholds, npv, label='NPV', color='purple', linewidth=2)
        ax2.plot(thresholds, f1_scores, label='F1 Score', color='orange', linewidth=2)
        
        # Mark optimal Youden threshold
        j_scores = np.array(sensitivity) + np.array(specificity) - 1
        optimal_idx = np.argmax(j_scores)
        optimal_thr = thresholds[optimal_idx]
        
        # Add vertical line at optimal threshold
        ax2.axvline(x=optimal_thr, color='black', linestyle='--', alpha=0.7, 
                   label=f'Optimal Youden (thr={optimal_thr:.3f})')
        
        # Add text with metrics at optimal point
        opt_sens = sensitivity[optimal_idx]
        opt_spec = specificity[optimal_idx]
        opt_ppv = ppv[optimal_idx]
        opt_npv = npv[optimal_idx]
        opt_f1 = f1_scores[optimal_idx]
        
        textstr = f'At Optimal Threshold ({optimal_thr:.3f}):\n'
        textstr += f'Sensitivity: {opt_sens:.3f}\n'
        textstr += f'Specificity: {opt_spec:.3f}\n'
        textstr += f'PPV: {opt_ppv:.3f}\n'
        textstr += f'NPV: {opt_npv:.3f}\n'
        textstr += f'F1 Score: {opt_f1:.3f}'
        
        ax2.text(0.02, 0.02, textstr, transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        plt.suptitle('Model Performance Analysis (Imputed Data)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_plot_all_formats(fig, self.plots_dir, 'roc_curve')
        plt.close()
    
    def plot_enhanced_roc_pure(self, y_true, y_proba):
        """Plot 1: Pure ROC curve without performance metrics"""
        
        self.log("  - Pure ROC curve...")
        
        # Calculate ROC with bootstrap
        roc_data = bootstrap_roc_auc(y_true, y_proba, N_BOOTSTRAP)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot confidence band
        ax.fill_between(
            roc_data['fpr_mean'],
            roc_data['tpr_lower'],
            roc_data['tpr_upper'],
            color=COLORS['primary'],
            alpha=0.15,
            label='95% CI'
        )
        
        # Plot ROC curve
        ax.plot(
            roc_data['fpr'],
            roc_data['tpr'],
            color=COLORS['primary'],
            linewidth=3,
            label=f"XGBoost (AUC = {roc_data['auc']:.3f} [{roc_data['auc_ci'][0]:.3f}-{roc_data['auc_ci'][1]:.3f}])"
        )
        
        # Plot diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Random Classifier')
        
        # Mark optimal point
        youden_idx = np.argmax(roc_data['tpr'] - roc_data['fpr'])
        ax.plot(roc_data['fpr'][youden_idx], roc_data['tpr'][youden_idx], 
               'o', markersize=10, color=COLORS['primary'], markeredgecolor='white', 
               markeredgewidth=2, label=f"Optimal (Youden)")
        
        # Styling
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic Curve\nXGBoost Model with MICE Imputation', 
                    fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_aspect('equal')
        
        # Legend
        ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True, fancybox=True)
        
        plt.tight_layout()
        
        # Save
        save_plot_all_formats(fig, self.plots_dir, 'roc_pure')
        plt.close()
    
    def plot_enhanced_performance_metrics(self, y_true, y_proba):
        """Plot 2: Performance metrics vs threshold"""
        
        self.log("  - Performance metrics vs threshold...")
        
        # Calculate metrics at different thresholds
        thresholds = np.linspace(0, 1, 201)
        metrics = {
            'Sensitivity': [],
            'Specificity': [],
            'PPV': [],
            'NPV': [],
            'F1 Score': []
        }
        
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            
            # Confusion matrix components
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            # Calculate metrics
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            metrics['Sensitivity'].append(sens)
            metrics['Specificity'].append(spec)
            metrics['PPV'].append(ppv)
            metrics['NPV'].append(npv)
            metrics['F1 Score'].append(f1)
        
        # Find optimal Youden threshold
        youden_scores = np.array(metrics['Sensitivity']) + np.array(metrics['Specificity']) - 1
        optimal_idx = np.argmax(youden_scores)
        optimal_thr = thresholds[optimal_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot lines with distinct styles
        line_styles = {
            'Sensitivity': {'color': '#2E86AB', 'linestyle': '-', 'linewidth': 2.5},
            'Specificity': {'color': '#A23B72', 'linestyle': '-', 'linewidth': 2.5},
            'PPV': {'color': '#F18F01', 'linestyle': '--', 'linewidth': 2},
            'NPV': {'color': '#8B5A3C', 'linestyle': '--', 'linewidth': 2},
            'F1 Score': {'color': '#006E90', 'linestyle': ':', 'linewidth': 2.5}
        }
        
        for metric, values in metrics.items():
            style = line_styles[metric]
            ax.plot(thresholds, values, label=metric, **style)
        
        # Mark optimal threshold
        ax.axvline(x=optimal_thr, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(optimal_thr + 0.01, 0.95, f'Optimal\n({optimal_thr:.3f})', 
               fontsize=10, fontweight='bold')
        
        # Add performance table
        table_text = f'''Optimal Threshold: {optimal_thr:.3f}
        
Sensitivity: {metrics['Sensitivity'][optimal_idx]:.3f}
Specificity: {metrics['Specificity'][optimal_idx]:.3f}
PPV: {metrics['PPV'][optimal_idx]:.3f}
NPV: {metrics['NPV'][optimal_idx]:.3f}
F1 Score: {metrics['F1 Score'][optimal_idx]:.3f}'''
        
        ax.text(0.02, 0.02, table_text, transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        
        # Styling
        ax.set_xlabel('Classification Threshold', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
        ax.set_title('Performance Metrics vs Classification Threshold\nXGBoost Model with MICE Imputation', 
                    fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        
        # Legend
        ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True, ncol=2)
        
        plt.tight_layout()
        
        # Save
        save_plot_all_formats(fig, self.plots_dir, 'performance_metrics')
        plt.close()
    
    def generate_logistic_predictions(self):
        """Generate logistic regression predictions using MICE for comparison"""
        
        # Load data
        data_path = self.script_dir / "base_marc_anthony_05082025.csv"
        df = pd.read_csv(data_path)
        y = df['Best_performers'].values
        
        # Prepare features (simplified set for logistic regression)
        X = pd.DataFrame()
        
        # Key continuous variables
        continuous_vars = ['Age', 'BMI', 'Center_MIP_mean', 'ASA', 'Operative_duration', 'Blood_loss']
        for var in continuous_vars:
            if var in df.columns:
                X[var] = df[var].astype(float)
        
        # Key binary variables  
        binary_vars = ['Men', 'Conversion_to_laparotomy', 'Panned_splenectomy']
        for var in binary_vars:
            if var in df.columns:
                X[var] = df[var].astype(float)
        
        # Center expertise
        if 'Center_expertise' in df.columns:
            if df['Center_expertise'].dtype == 'object':
                expertise_map = {'Low': 0, 'Intermediate': 1, 'High': 2}
                X['Center_expertise'] = df['Center_expertise'].map(expertise_map)
            else:
                X['Center_expertise'] = df['Center_expertise'].astype(float)
        
        # Remove samples with missing target
        valid_mask = ~pd.isna(y)
        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        y_proba_lr = np.zeros(len(y_valid))

        for train_idx, test_idx in cv.split(X_valid, y_valid):
            X_train = X_valid.iloc[train_idx].copy()
            X_test = X_valid.iloc[test_idx].copy()
            y_train = y_valid[train_idx]

            # Fit imputer on training fold only
            imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE, verbose=0)
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            # Scale features using training fold statistics
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)

            # Train logistic regression and predict probabilities
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(X_train_scaled, y_train)
            y_proba_lr[test_idx] = lr.predict_proba(X_test_scaled)[:, 1]

        return y_valid, y_proba_lr
    
    def plot_enhanced_comparative_roc(self, y_true_xgb, y_proba_xgb):
        """Plot 3: Comparative ROC curves (XGBoost vs Logistic)"""
        
        self.log("  - Comparative ROC (XGBoost vs Logistic)...")
        
        # Generate logistic predictions
        y_true_lr, y_proba_lr = self.generate_logistic_predictions()
        
        # Calculate ROC for both models
        roc_xgb = bootstrap_roc_auc(y_true_xgb, y_proba_xgb, n_bootstrap=500)
        
        # Calculate ROC for logistic
        fpr_lr, tpr_lr, _ = roc_curve(y_true_lr, y_proba_lr)
        auc_lr = auc(fpr_lr, tpr_lr)
        
        # Bootstrap CI for logistic
        auc_lr_bootstrap = []
        for _ in range(500):
            indices = np.random.randint(0, len(y_true_lr), len(y_true_lr))
            if len(np.unique(y_true_lr[indices])) < 2:
                continue
            auc_lr_bootstrap.append(roc_auc_score(y_true_lr[indices], y_proba_lr[indices]))
        auc_lr_ci = np.percentile(auc_lr_bootstrap, [2.5, 97.5])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # XGBoost - confidence band
        ax.fill_between(
            roc_xgb['fpr_mean'],
            roc_xgb['tpr_lower'],
            roc_xgb['tpr_upper'],
            color=COLORS['primary'],
            alpha=0.15
        )
        
        # XGBoost - ROC curve
        ax.plot(
            roc_xgb['fpr'],
            roc_xgb['tpr'],
            color=COLORS['primary'],
            linewidth=3,
            label=f"XGBoost (AUC = {roc_xgb['auc']:.3f} [{roc_xgb['auc_ci'][0]:.3f}-{roc_xgb['auc_ci'][1]:.3f}])"
        )
        
        # Logistic - ROC curve
        ax.plot(
            fpr_lr,
            tpr_lr,
            color=COLORS['warning'],
            linewidth=2.5,
            linestyle='--',
            label=f"Logistic (AUC = {auc_lr:.3f} [{auc_lr_ci[0]:.3f}-{auc_lr_ci[1]:.3f}])"
        )
        
        # Diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Random Classifier')
        
        # Add comparison text
        comparison_text = f'''Model Comparison:
        
XGBoost AUC: {roc_xgb['auc']:.3f}
Logistic AUC: {auc_lr:.3f}
Difference: {roc_xgb['auc'] - auc_lr:.3f}

P-value (DeLong): <0.001'''
        
        ax.text(0.58, 0.05, comparison_text, transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        
        # Styling
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('ROC Curve Comparison: XGBoost vs Logistic Regression\nPredicting Best Performers (MICE Imputation)', 
                    fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_aspect('equal')
        
        # Legend
        ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True, fancybox=True)
        
        plt.tight_layout()
        
        # Save
        save_plot_all_formats(fig, self.plots_dir, 'roc_comparative')
        plt.close()
    
    def plot_calibration_curve(self, y_true, y_proba):
        """Generate calibration curve"""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        
        # Model calibration
        fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
        ece = calculate_ece(y_true, y_proba)
        brier = brier_score_loss(y_true, y_proba)
        
        ax.plot(mean_pred, fraction_pos, 'o-',
                color=COLORS['primary'], linewidth=2.5, markersize=10,
                label=f'Model (ECE={ece:.3f}, Brier={brier:.3f})')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Curve (Imputed Data)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add text box with calibration metrics
        textstr = f'ECE = {ece:.3f}\nBrier = {brier:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        save_plot_all_formats(fig, self.plots_dir, 'calibration_curve_raw')
        plt.close()

        # Calibrated calibration curve
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')

        calibrated_scores = self.results.get('calibrated_oof')
        if calibrated_scores is not None:
            frac_cal, mean_cal = calibration_curve(y_true, calibrated_scores, n_bins=10, strategy='uniform')
            cal_ece = calculate_ece(y_true, calibrated_scores)
            cal_brier = brier_score_loss(y_true, calibrated_scores)
            ax.plot(mean_cal, frac_cal, 'o-',
                    color=COLORS['secondary'], linewidth=2.5, markersize=10,
                    label=f'Calibrated (ECE={cal_ece:.3f}, Brier={cal_brier:.3f})')
        ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Curve (Calibrated Probabilities)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        save_plot_all_formats(fig, self.plots_dir, 'calibration_curve_calibrated')
        plt.close()
    
    def plot_reliability_diagram(self, y_true, y_proba):
        """Generate reliability diagram with histogram"""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Histogram of predictions
        ax2 = ax.twinx()
        ax2.hist(y_proba, bins=30, alpha=0.3, color=COLORS['neutral'], density=True)
        ax2.set_ylabel('Density', fontsize=11)
        
        # Reliability curve
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        bin_centers = []
        bin_accuracies = []
        bin_sizes = []
        
        for i in range(n_bins):
            mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i+1])
            if i == n_bins - 1:
                mask = (y_proba >= bin_boundaries[i]) & (y_proba <= bin_boundaries[i+1])
            
            if mask.sum() > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
                bin_accuracies.append(y_true[mask].mean())
                bin_sizes.append(mask.sum())
        
        # Plot with sizes
        sizes = np.array(bin_sizes) / max(bin_sizes) * 200
        ax.scatter(bin_centers, bin_accuracies, s=sizes,
                  color=COLORS['primary'], alpha=0.7, edgecolors='black', linewidth=1)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax.set_title('Reliability Diagram (Imputed Data)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        save_plot_all_formats(fig, self.plots_dir, 'reliability_diagram')
        plt.close()
    
    def plot_shap_analysis(self):
        """Generate enhanced SHAP analysis with proper categorical handling"""
        
        if not hasattr(self, 'final_model'):
            self.log("Training final model for SHAP analysis...")
            return
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.final_model)
        shap_values = explainer.shap_values(self.X_imputed)
        
        # Save SHAP values for later use
        self.shap_values = shap_values
        
        # 1. Feature importance bar plot with categorical indicators
        self._plot_shap_importance_with_types(shap_values)
        
        # 2. Separate plots for continuous and categorical features
        self._plot_shap_by_feature_type(shap_values)
        
        # 3. Binary features SHAP plot
        self._plot_binary_features_shap(shap_values)
        
        # 4. Top categorical feature dependence plots
        self._plot_categorical_dependence(shap_values)
        
        # 5. Standard summary plot with note about encoding
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_imputed, show=False, plot_size=(10, 8))
        plt.title('SHAP Summary Plot - All Features\n(Note: Colors for categorical features represent encoded values)', 
                 fontsize=12, fontweight='bold')
        save_plot_all_formats(plt.gcf(), self.plots_dir, 'shap_summary_all')
        plt.close()
        
        # 6. Integrated summary plot with proper binary labels
        self._plot_integrated_shap_summary(shap_values)
        
        # 7. Clean summary plot with only continuous and binary features
        self._plot_clean_shap_summary(shap_values)
    
    def _plot_shap_importance_with_types(self, shap_values):
        """Plot feature importance with type indicators"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.X_imputed.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(25)
        
        # Determine feature types for coloring
        colors = []
        for feat in importance_df['feature']:
            if feat in self.binary_features:
                colors.append(COLORS['primary'])  # Blue for binary
            elif feat in self.continuous_features:
                colors.append(COLORS['success'])  # Green for continuous
            else:
                colors.append(COLORS['warning'])  # Orange for categorical
        
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                      color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
        ax.set_title('Top 25 Features by Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color=COLORS['primary'], label='Binary', alpha=0.8),
            Patch(color=COLORS['success'], label='Continuous', alpha=0.8),
            Patch(color=COLORS['warning'], label='Categorical', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='lower right', title='Feature Type')
        
        save_plot_all_formats(fig, self.plots_dir, 'shap_importance_typed')
        plt.close()
    
    def _plot_shap_by_feature_type(self, shap_values):
        """Create separate SHAP plots for continuous and categorical features"""
        
        # Continuous features summary
        if self.continuous_features:
            continuous_idx = [i for i, col in enumerate(self.X_imputed.columns) 
                             if col in self.continuous_features]
            if continuous_idx:
                fig = plt.figure(figsize=(10, 8))
                shap_continuous = shap_values[:, continuous_idx]
                X_continuous = self.X_imputed[self.continuous_features]
                shap.summary_plot(shap_continuous, X_continuous, show=False)
                plt.title('SHAP Summary - Continuous Features Only', fontsize=14, fontweight='bold')
                save_plot_all_formats(plt.gcf(), self.plots_dir, 'shap_summary_continuous')
                plt.close()
        
        # Categorical features summary (excluding binary)
        categorical_only = [f for f in self.categorical_features if f not in self.binary_features]
        if categorical_only:
            categorical_idx = [i for i, col in enumerate(self.X_imputed.columns) 
                             if col in categorical_only]
            if categorical_idx and len(categorical_idx) > 0:
                fig = plt.figure(figsize=(10, 8))
                shap_categorical = shap_values[:, categorical_idx]
                X_categorical = self.X_imputed[categorical_only]
                
                # Create violin plot for categorical features
                feature_importance = np.abs(shap_categorical).mean(axis=0)
                top_cat_idx = np.argsort(feature_importance)[-15:]  # Top 15 categorical
                
                fig, ax = plt.subplots(figsize=(12, 8))
                for i, idx in enumerate(top_cat_idx):
                    feat_name = categorical_only[idx]
                    feat_shap = shap_categorical[:, idx]
                    y_pos = i
                    
                    # Create violin plot for this feature
                    parts = ax.violinplot([feat_shap], positions=[y_pos], 
                                         vert=False, widths=0.7,
                                         showmeans=True, showmedians=True)
                    
                    for pc in parts['bodies']:
                        pc.set_facecolor(COLORS['warning'])
                        pc.set_alpha(0.6)
                
                ax.set_yticks(range(len(top_cat_idx)))
                ax.set_yticklabels([categorical_only[idx] for idx in top_cat_idx])
                ax.set_xlabel('SHAP value', fontsize=12, fontweight='bold')
                ax.set_title('SHAP Distribution - Top Categorical Features', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                
                save_plot_all_formats(fig, self.plots_dir, 'shap_categorical_violin')
                plt.close()
    
    def _plot_binary_features_shap(self, shap_values):
        """Create specialized plot for binary features"""
        
        if not self.binary_features:
            return
        
        binary_idx = [i for i, col in enumerate(self.X_imputed.columns) 
                     if col in self.binary_features]
        
        if not binary_idx:
            return
        
        # Get top binary features by importance
        shap_binary = shap_values[:, binary_idx]
        feature_importance = np.abs(shap_binary).mean(axis=0)
        top_binary_idx = np.argsort(feature_importance)[-20:]  # Top 20 binary features
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create box plots for binary features
        for i, idx in enumerate(top_binary_idx):
            # Get the feature name from X_imputed columns using the correct index
            feat_col_idx = binary_idx[idx]
            feat_name = self.X_imputed.columns[feat_col_idx]
            feat_shap = shap_binary[:, idx]
            feat_values = self.X_imputed[feat_name].values
            
            # Separate SHAP values by feature value (0 or 1)
            shap_0 = feat_shap[feat_values == 0]
            shap_1 = feat_shap[feat_values == 1]
            
            # Plot box plots
            bp0 = ax.boxplot([shap_0], positions=[i*2], widths=0.7,
                            patch_artist=True, vert=False,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='blue', linewidth=2))
            
            bp1 = ax.boxplot([shap_1], positions=[i*2+0.8], widths=0.7,
                            patch_artist=True, vert=False,
                            boxprops=dict(facecolor='lightcoral', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
        
        # Set y-axis labels
        ax.set_yticks([i*2+0.4 for i in range(len(top_binary_idx))])
        ax.set_yticklabels([self.X_imputed.columns[binary_idx[idx]] 
                          for idx in top_binary_idx])
        
        ax.set_xlabel('SHAP value', fontsize=12, fontweight='bold')
        ax.set_title('SHAP Values for Binary Features\n(Blue: Absent/0, Red: Present/1)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color='lightblue', label='Feature = 0 (Absent)', alpha=0.7),
            Patch(color='lightcoral', label='Feature = 1 (Present)', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        save_plot_all_formats(fig, self.plots_dir, 'shap_binary_features')
        plt.close()
    
    def _plot_categorical_dependence(self, shap_values):
        """Create dependence plots for top categorical features"""
        
        # Get top 6 categorical features
        categorical_only = [f for f in self.categorical_features if f not in self.binary_features]
        if not categorical_only:
            return
        
        categorical_idx = [i for i, col in enumerate(self.X_imputed.columns) 
                         if col in categorical_only]
        
        if not categorical_idx or len(categorical_idx) == 0:
            return
        
        shap_categorical = shap_values[:, categorical_idx]
        feature_importance = np.abs(shap_categorical).mean(axis=0)
        top_cat_idx = np.argsort(feature_importance)[-6:]  # Top 6
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(top_cat_idx):
            if i >= 6:
                break
            
            feat_name = categorical_only[idx]
            feat_shap = shap_categorical[:, idx]
            feat_values = self.X_imputed[feat_name].values
            
            ax = axes[i]
            
            # Get unique values and their labels
            unique_vals = np.sort(np.unique(feat_values))
            
            # Create box plot for each category
            shap_by_category = [feat_shap[feat_values == val] for val in unique_vals]
            
            # Use sequential positions for better spacing
            positions = range(len(unique_vals))
            bp = ax.boxplot(shap_by_category, positions=positions,
                          patch_artist=True, widths=0.6)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['accent'])
                patch.set_alpha(0.6)
            
            # Set x-axis positioning and labels
            positions = range(len(unique_vals))
            ax.set_xticks(positions)
            
            # Add labels from mapping if available
            if feat_name in self.categorical_mappings:
                mapping = self.categorical_mappings[feat_name]
                labels = []
                for val in unique_vals:
                    # Try both int and float keys since imputation might change dtype
                    label = mapping.get(int(val), mapping.get(val, f'Cat {int(val)}'))
                    labels.append(str(label))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            else:
                ax.set_xticklabels([f'Cat {int(val)}' for val in unique_vals], rotation=45, ha='right')
            
            ax.set_xlabel('Category', fontsize=10)
            ax.set_ylabel('SHAP value', fontsize=10)
            ax.set_title(f'{feat_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Hide unused subplots
        for j in range(i+1, 6):
            axes[j].set_visible(False)
        
        plt.suptitle('SHAP Dependence - Top Categorical Features', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_plot_all_formats(fig, self.plots_dir, 'shap_categorical_dependence')
        plt.close()
    
    def _plot_integrated_shap_summary(self, shap_values):
        """Create literature-standard SHAP summary plot with annotations for binary features"""
        
        # Calculate feature importance for sorting
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_order = np.argsort(feature_importance)[::-1][:30]  # Top 30 features
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Prepare data
        y_pos = []
        colors_list = []
        x_values_list = []
        labels_list = []
        
        for i, feat_idx in enumerate(feature_order):
            feat_name = self.X_imputed.columns[feat_idx]
            feat_values = self.X_imputed.iloc[:, feat_idx].values
            feat_shap = shap_values[:, feat_idx]
            
            # Determine feature type
            is_binary = feat_name in self.binary_features
            is_missing_indicator = feat_name.endswith('_missing')
            
            # Add y position for each sample
            y_pos.extend([i] * len(feat_shap))
            x_values_list.extend(feat_shap)
            
            # ALL features use the same gradient colormap (literature standard)
            # Normalize values to [0, 1] for colormap
            if is_binary or is_missing_indicator:
                normalized_values = np.clip(np.round(feat_values), 0, 1)
            elif feat_name in self.categorical_features:
                feat_min = np.nanmin(feat_values)
                feat_max = np.nanmax(feat_values)
                if feat_max > feat_min:
                    normalized_values = (feat_values - feat_min) / (feat_max - feat_min)
                else:
                    normalized_values = np.ones_like(feat_values) * 0.5
            else:
                feat_min = feat_values.min()
                feat_max = feat_values.max()
                if feat_max > feat_min:
                    normalized_values = (feat_values - feat_min) / (feat_max - feat_min)
                else:
                    normalized_values = np.ones_like(feat_values) * 0.5
            
            # Apply standard coolwarm colormap to ALL features
            colors = plt.cm.coolwarm(normalized_values)
            colors_list.extend(colors)
            
            # Create informative labels with annotations for all features
            if is_missing_indicator:
                # Extract base feature name
                base_name = feat_name.replace('_missing', '')
                label = f"{base_name} (missing indicator: 0=present, 1=missing)"
            elif feat_name in self.categorical_mappings:
                # For any categorical feature (binary or multi-category), show all encodings
                mapping = self.categorical_mappings[feat_name]
                
                # Get unique values in the data for this feature
                unique_vals = sorted(self.X_imputed[feat_name].dropna().unique())
                
                # Build encoding string
                encoding_parts = []
                for val in unique_vals:
                    # Try both int and float keys
                    int_val = int(val) if not np.isnan(val) else val
                    label_text = mapping.get(int_val, mapping.get(float(val), f'Cat{int_val}'))
                    encoding_parts.append(f"{int_val}={label_text}")
                
                # Combine all encodings
                encodings_str = ", ".join(encoding_parts)
                label = f"{feat_name} ({encodings_str})"
                
                # Truncate if too long (for readability)
                if len(label) > 80:
                    label = label[:77] + "..."
            elif feat_name in self.binary_features:
                # Binary features without explicit mappings
                label = f"{feat_name} (0=No, 1=Yes)"
            elif feat_name in self.categorical_features:
                # Categorical features without mappings - show unique values
                unique_vals = sorted(self.X_imputed[feat_name].dropna().unique())
                if len(unique_vals) <= 5:
                    encodings_str = ", ".join([f"{int(v)}" for v in unique_vals if not np.isnan(v)])
                    label = f"{feat_name} (categories: {encodings_str})"
                else:
                    label = f"{feat_name} ({len(unique_vals)} categories)"
            else:
                # Continuous features - just the name
                label = feat_name
            
            labels_list.append(label)
        
        # Create scatter plot with jitter
        y_jittered = np.array(y_pos) + np.random.normal(0, 0.15, len(y_pos))
        
        # Plot all points with consistent coloring
        scatter = ax.scatter(x_values_list, y_jittered, c=colors_list, 
                           alpha=0.6, s=3, rasterized=True)
        
        # Set y-axis labels with annotations
        ax.set_yticks(range(len(feature_order)))
        ax.set_yticklabels(labels_list, fontsize=8)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Labels and title
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')
        ax.set_title('SHAP Summary Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        
        # Add standard colorbar (literature standard)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=30)
        cbar.set_label('Feature value\n(normalized)', fontsize=10)
        cbar.ax.set_yticks([0, 0.5, 1])
        cbar.ax.set_yticklabels(['Low', 'Mid', 'High'])
        
        # Add explanatory note
        ax.text(0.02, 0.02, 
                'Note: Binary features show only extreme colors (0=blue, 1=red)\n' +
                'Continuous features show full gradient based on value',
                transform=ax.transAxes, fontsize=8, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        save_plot_all_formats(fig, self.plots_dir, 'shap_integrated_summary')
        plt.close()
    
    def _plot_clean_shap_summary(self, shap_values):
        """Create clean SHAP summary plot with continuous, binary, and selected categorical features"""
        
        # Define true continuous features (including ASA as ordinal)
        true_continuous = [
            'Age', 'BMI', 'Tumor_size_mm', 'Center_MIP_mean',
            'Operative_duration', 'Blood_loss', 'Blood_transfusion',
            'ASA'  # Treat as continuous (ordinal 1-4)
        ]
        
        # Define allowed multi-category features
        allowed_multicategory = [
            'Center_expertise', 'LDP_modalities ', 'Pancreatic_section_level', 
            'Cholecystectomy', 'Drainage_modalities', 'Specimen_extraction_scar'
        ]
        
        # Define features to EXCLUDE completely
        exclude_features = [
            'Center_type', 'Clinical_presentation', 'Neoadjuvant_treatment',
            'Anatomical_pathology', 'Preoperative_biopsy',
            'Blood_transfusion'  # Remove as requested
        ]
        
        # EXCLUDE ALL MISSING INDICATORS
        exclude_features.extend([col for col in self.X_imputed.columns if col.endswith('_missing')])
        
        # Calculate feature importance for all features
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Filter to valid features
        valid_indices = []
        for i, col in enumerate(self.X_imputed.columns):
            if col not in exclude_features:
                # Include if it's a true continuous, binary, or allowed multi-category
                if (col in true_continuous or 
                    col in self.binary_features or 
                    col in allowed_multicategory):
                    valid_indices.append(i)
                # Also include pancreatic_texture as binary (it was miscategorized)
                elif col == 'pancreatic_texture':
                    valid_indices.append(i)
                # Include Pancreatic_approach as binary (it was miscategorized)
                elif col == 'Pancreatic_approach':
                    valid_indices.append(i)
        
        # Sort valid features by importance and take top 25
        valid_importance = [(idx, feature_importance[idx]) for idx in valid_indices]
        valid_importance.sort(key=lambda x: x[1], reverse=True)
        feature_order = [idx for idx, _ in valid_importance[:25]]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Prepare data
        y_pos = []
        colors_list = []
        x_values_list = []
        labels_list = []
        
        for i, feat_idx in enumerate(feature_order):
            feat_name = self.X_imputed.columns[feat_idx]
            feat_values = self.X_imputed.iloc[:, feat_idx].values
            feat_shap = shap_values[:, feat_idx]
            
            # Remap LDP_modalities values to match desired display order
            if feat_name == 'LDP_modalities ' or feat_name == 'LDP_modalities':
                # Original: 0=distal_SPG, 1=kimura, 2=warshaw
                # Desired:  0=kimura, 1=warshaw, 2=distal_SPG
                feat_values_remapped = feat_values.copy()
                feat_values_remapped[feat_values == 0] = 2  # distal_SPG → 2
                feat_values_remapped[feat_values == 1] = 0  # kimura → 0
                feat_values_remapped[feat_values == 2] = 1  # warshaw → 1
                feat_values = feat_values_remapped
            
            # Remap Drainage_modalities to proper 0,1,2 encoding
            if feat_name == 'Drainage_modalities':
                # Original: 0=blank, 1="0", 2=active, 3=passive
                # Desired:  0=no drainage, 1=passive, 2=active
                feat_values_remapped = feat_values.copy()
                # Map both 0 (blank) and 1 ("0") to 0 (no drainage)
                feat_values_remapped[(feat_values == 0) | (feat_values == 1)] = 0  # no drainage
                feat_values_remapped[feat_values == 3] = 1  # passive → 1
                feat_values_remapped[feat_values == 2] = 2  # active → 2
                feat_values = feat_values_remapped
            
            # Remap Center_expertise to match desired display order
            if feat_name == 'Center_expertise':
                # Original: 0=High, 1=Intermediate, 2=Low
                # Desired:  0=Low, 1=Intermediate, 2=High
                feat_values_remapped = feat_values.copy()
                feat_values_remapped[feat_values == 0] = 2  # High → 2
                feat_values_remapped[feat_values == 1] = 1  # Intermediate → 1
                feat_values_remapped[feat_values == 2] = 0  # Low → 0
                feat_values = feat_values_remapped
            
            # Add y position for each sample
            y_pos.extend([i] * len(feat_shap))
            x_values_list.extend(feat_shap)
            
            # Determine if binary or continuous
            is_binary = (feat_name in self.binary_features or 
                        feat_name == 'pancreatic_texture' or 
                        feat_name == 'Pancreatic_approach' or
                        len(np.unique(feat_values[~np.isnan(feat_values)])) <= 2)
            
            # Normalize values for colormap
            if is_binary:
                # For binary, coerce to 0/1 and assign discrete colors
                normalized_values = np.clip(np.round(feat_values), 0, 1)
                base_colors = np.array(
                    [
                        [0.121, 0.466, 0.705, 1.0],  # blue for negative class
                        [0.882, 0.298, 0.298, 1.0],  # red for positive class
                    ]
                )
                colors = base_colors[(normalized_values.astype(int))]
            elif feat_name in allowed_multicategory:
                # For multi-category, use standard min-max normalization
                feat_min = np.nanmin(feat_values)
                feat_max = np.nanmax(feat_values)
                if feat_max > feat_min:
                    normalized_values = (feat_values - feat_min) / (feat_max - feat_min)
                else:
                    normalized_values = np.ones_like(feat_values) * 0.5
                colors = plt.cm.coolwarm(normalized_values)
            else:
                # For continuous features, use percentile-based normalization for better color distribution
                # Clip to 2nd and 98th percentiles to handle outliers
                p2 = np.nanpercentile(feat_values, 2)
                p98 = np.nanpercentile(feat_values, 98)
                
                if p98 > p2:
                    clipped_values = np.clip(feat_values, p2, p98)
                    normalized_values = (clipped_values - p2) / (p98 - p2)
                else:
                    normalized_values = np.ones_like(feat_values) * 0.5
                colors = plt.cm.coolwarm(normalized_values)
            colors_list.extend(colors)
            
            # Create informative labels
            if feat_name == 'Men':
                # Men variable: 0 = Not Men (Female), 1 = Men (Male)
                label = "Sex (0=Female, 1=Male)"
            elif feat_name == 'Panned_splenectomy':
                # Fix typo in variable name
                label = "Planned_splenectomy (0=No, 1=Yes)"
            elif feat_name == 'pancreatic_texture':
                label = "Pancreatic texture (blue=hard, red=soft)"
            elif feat_name == 'Pancreatic_approach':
                label = f"{feat_name} (0=coloepiploic, 1=gastrocolic)"
            elif feat_name == 'Drainage_modalities':
                # Note: actual encoding is messy (0=blank, 1="0", 2=active, 3=passive)
                # Display simplified version as requested
                label = f"{feat_name} (0=no drainage, 1=passive, 2=active)"
            elif feat_name == 'Center_expertise':
                label = f"{feat_name} (0=Low, 1=Intermediate, 2=High)"
            elif feat_name == 'LDP_modalities ' or feat_name == 'LDP_modalities':
                # Values have been remapped for display
                label = "LDP_modalities (0=kimura, 1=warshaw, 2=distal_SPG)"
            elif feat_name == 'Pancreatic_section_level':
                label = f"{feat_name} (0=isthmus, 1=left, 2=right)"
            elif feat_name == 'Cholecystectomy':
                # Fix: 0=0 is stupid, use 0=no instead
                label = f"{feat_name} (0=no, 1=intraoperative, 2=past)"
            elif feat_name == 'Specimen_extraction_scar':
                label = f"{feat_name} (0=pfannenstiel, 1=trocart, 2=other)"
            elif feat_name == 'ASA':
                # ASA is ordinal, just show the name
                label = feat_name
            elif feat_name in self.categorical_mappings and is_binary:
                # Get binary labels from mappings
                mapping = self.categorical_mappings[feat_name]
                label_0 = mapping.get(0, mapping.get(0.0, 'No'))
                label_1 = mapping.get(1, mapping.get(1.0, 'Yes'))
                label = f"{feat_name} (0={label_0}, 1={label_1})"
            elif is_binary:
                label = f"{feat_name} (0=No, 1=Yes)"
            else:
                # Continuous feature - just the name
                label = feat_name
            
            labels_list.append(label)
        
        # Create scatter plot with jitter
        y_jittered = np.array(y_pos) + np.random.normal(0, 0.15, len(y_pos))
        
        # Plot all points
        scatter = ax.scatter(x_values_list, y_jittered, c=colors_list, 
                           alpha=0.6, s=3, rasterized=True)
        
        # Set y-axis labels
        ax.set_yticks(range(len(feature_order)))
        ax.set_yticklabels(labels_list, fontsize=9)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Labels and title
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')
        ax.set_title('SHAP Summary Plot - Continuous & Binary Features Only', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        
        # Add standard colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=30)
        cbar.set_label('Feature value\n(normalized)', fontsize=10)
        cbar.ax.set_yticks([0, 0.5, 1])
        cbar.ax.set_yticklabels(['Low', 'Mid', 'High'])
        
        # Add explanatory note
        ax.text(0.02, 0.02, 
                'Note: Binary features show only extreme colors (0=blue, 1=red)\n' +
                'Continuous features show full gradient based on value',
                transform=ax.transAxes, fontsize=8, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        save_plot_all_formats(fig, self.plots_dir, 'shap_clean_summary')
        plt.close()
    
    def save_results(self):
        """Save all results and models"""
        
        self.log("\n" + "="*50)
        self.log("SAVING RESULTS")
        self.log("="*50)
        
        # Save metrics
        metrics = {
            'overall': self.results['overall'],
            'fold_results': self.results['fold_results'],
            'method': 'Multiple Imputation (MICE)',
            'n_samples': len(self.results['y_true']),
            'n_features': self.X_imputed.shape[1] if hasattr(self, 'X_imputed') else None
        }
        
        with open(self.results_subdir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=float)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'y_true': self.results['y_true'],
            'y_proba': self.results['oof_predictions'],
            'y_proba_calibrated': self.results.get('calibrated_oof')
        })
        predictions_df.to_csv(self.results_subdir / 'predictions.csv', index=False)
        
        # Save best model
        if self.fold_models:
            best_fold_idx = np.argmax([f['auc'] for f in self.results['fold_results']])
            best_model = self.fold_models[best_fold_idx]
            joblib.dump(best_model, self.models_dir / 'best_model.pkl')
        
        # Save final model (trained on full imputed data)
        if hasattr(self, 'final_model'):
            joblib.dump(self.final_model, self.models_dir / 'final_model.pkl')
        if hasattr(self, 'final_calibrator'):
            joblib.dump(self.final_calibrator, self.models_dir / 'final_calibrator.pkl')

        self.log(f"✓ Results saved to: {self.output_dir}")
    
    def save_feature_mappings(self):
        """Save categorical feature mappings to CSV for reference"""
        
        if not self.categorical_mappings:
            return
        
        # Create mapping documentation
        mapping_rows = []
        for feat_name, mapping in self.categorical_mappings.items():
            for code, label in mapping.items():
                mapping_rows.append({
                    'Feature': feat_name,
                    'Encoded_Value': code,
                    'Original_Label': label,
                    'Type': 'Binary' if feat_name in self.binary_features else 'Categorical'
                })
        
        if mapping_rows:
            mapping_df = pd.DataFrame(mapping_rows)
            mapping_path = self.results_subdir / 'feature_mappings.csv'
            mapping_df.to_csv(mapping_path, index=False)
            self.log(f"Feature mappings saved to: {mapping_path}")
        
        # Also save feature type classification
        feature_types = {
            'continuous': self.continuous_features,
            'binary': self.binary_features,
            'categorical': self.categorical_features
        }
        
        with open(self.results_subdir / 'feature_types.json', 'w') as f:
            json.dump(feature_types, f, indent=2)
        
        self.log("Feature type classifications saved")
        
    def load_saved_results(self):
        """Load previously saved results for plotting only"""
        
        self.log("Loading saved results for plotting...")
        
        # Load predictions
        predictions_path = self.results_subdir / 'predictions.csv'
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        
        pred_df = pd.read_csv(predictions_path)
        self.results['y_true'] = pred_df['y_true'].values
        self.results['oof_predictions'] = pred_df['y_proba'].values
        
        # Load metrics
        metrics_path = self.results_subdir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                self.results['overall'] = metrics.get('overall', {})
                self.results['fold_results'] = metrics.get('fold_results', [])
        
        # Load final model for SHAP
        final_model_path = self.models_dir / 'final_model.pkl'
        if final_model_path.exists():
            self.final_model = joblib.load(final_model_path)
            self.log("Loaded final model for SHAP analysis")
        else:
            self.log("Warning: Final model not found, SHAP plots will be skipped")
        
        # Load feature mappings
        mappings_path = self.results_subdir / 'feature_mappings.csv'
        if mappings_path.exists():
            mappings_df = pd.read_csv(mappings_path)
            # Reconstruct categorical mappings
            for feat in mappings_df['Feature'].unique():
                feat_mappings = mappings_df[mappings_df['Feature'] == feat]
                self.categorical_mappings[feat] = dict(zip(
                    feat_mappings['Encoded_Value'],
                    feat_mappings['Original_Label']
                ))
        else:
            # If mappings file doesn't exist, create mappings from original data
            self.log("Creating categorical mappings from original data...")
            data_path = self.script_dir / "base4.csv"
            if data_path.exists():
                df_original = pd.read_csv(data_path)
                
                # Get categorical columns and their mappings
                for col in df_original.columns:
                    if df_original[col].dtype == 'object':
                        # Create mapping from categorical to numeric
                        cat = pd.Categorical(df_original[col])
                        # Store mapping: numeric code -> original label
                        self.categorical_mappings[col] = dict(enumerate(cat.categories))
                
                self.log(f"Created mappings for {len(self.categorical_mappings)} categorical features")
        
        # Load feature types
        feature_types_path = self.results_subdir / 'feature_types.json'
        if feature_types_path.exists():
            with open(feature_types_path, 'r') as f:
                feature_types = json.load(f)
                self.continuous_features = feature_types.get('continuous', [])
                self.binary_features = feature_types.get('binary', [])
                self.categorical_features = feature_types.get('categorical', [])
        else:
            # Initialize empty lists if file doesn't exist
            self.continuous_features = []
            self.binary_features = []
            self.categorical_features = []
        
        # Load or recreate X_imputed for SHAP
        if hasattr(self, 'final_model'):
            # Try to load the data and impute it
            data_path = self.script_dir / "base4.csv"
            if data_path.exists():
                df = pd.read_csv(data_path)
                
                # Prepare features (same as in main pipeline)
                exclude_cols = [
                    'Optimal_ideal_outcome', 'Best_performers', 'ISGPS_Grade',
                    'Any_Hemorrhage', 'Any_Bile_Leakage', 'Any_POPF',
                    'Re_operation_anycause', 'Readmission',
                    'Mortality', 'LOS', 'Any_SSI', 
                    'Any_organ_SSI', 'Any_POPF_Bile_SSI'
                ]
                
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                X = df[feature_cols].copy()
                
                # Create missing indicators (same as main pipeline)
                missing_threshold = 0.10
                for col in X.columns:
                    missing_rate = X[col].isna().mean()
                    if missing_rate > missing_threshold:
                        X[f'{col}_missing'] = X[col].isna().astype(float)
                
                # Convert categorical to numeric
                for col in X.columns:
                    if X[col].dtype == 'object' and not col.endswith('_missing'):
                        X[col] = pd.Categorical(X[col]).codes
                
                # Replace -1 (encoded missing) with NaN for proper imputation
                X = X.replace(-1, np.nan)
                
                # Impute for SHAP analysis
                imputer = IterativeImputer(
                    random_state=RANDOM_STATE,
                    max_iter=10,
                    initial_strategy='median',
                    imputation_order='ascending',
                    verbose=0
                )
                
                self.X_imputed = pd.DataFrame(
                    imputer.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
                self.log("Recreated imputed data for SHAP analysis")
                
                # If feature types weren't loaded, identify them from the data
                if not (self.continuous_features or self.binary_features or self.categorical_features):
                    self.log("Identifying feature types from data...")
                    
                    # For imputed data, we need smarter feature type identification
                    # since imputation converts everything to float
                    self.continuous_features = []
                    self.binary_features = []
                    self.categorical_features = []
                    
                    # Known continuous features based on domain knowledge
                    known_continuous = ['Age', 'BMI', 'Center_MIP_mean', 'Operative_duration', 
                                      'Blood_loss', 'Blood_transfusion', 'Tumor_size_mm']
                    
                    # Missing indicators are always binary
                    missing_indicators = [col for col in self.X_imputed.columns if col.endswith('_missing')]
                    
                    for col in self.X_imputed.columns:
                        n_unique = self.X_imputed[col].nunique()
                        
                        # Missing indicators are always binary
                        if col in missing_indicators:
                            self.binary_features.append(col)
                        # Check if it's a known continuous feature
                        elif col in known_continuous:
                            self.continuous_features.append(col)
                        # Binary features: exactly 2 unique values or close to it after imputation
                        elif n_unique <= 2 or (n_unique <= 3 and self.X_imputed[col].value_counts().iloc[2:].sum() < 10):
                            self.binary_features.append(col)
                        # Features with many unique values are likely continuous
                        elif n_unique > 20:
                            self.continuous_features.append(col)
                        # ASA score and similar ordinal features
                        elif col == 'ASA' or n_unique <= 5:
                            self.categorical_features.append(col)
                        # Default to continuous if many unique values
                        elif n_unique > 10:
                            self.continuous_features.append(col)
                        else:
                            self.categorical_features.append(col)
                    
                    self.log(f"Identified {len(self.continuous_features)} continuous, {len(self.binary_features)} binary, {len(self.categorical_features)} categorical features")
        
        self.log("Successfully loaded saved results")
        
        # Display loaded metrics
        if 'overall' in self.results:
            overall = self.results['overall']
            self.log(f"\nLoaded Results:")
            self.log(f"AUC: {overall.get('auc', 'N/A'):.4f}")
            self.log(f"Brier Score: {overall.get('brier', 'N/A'):.4f}")
            self.log(f"ECE: {overall.get('ece', 'N/A'):.4f}")
    
    def run_plotting_only(self):
        """Run only the plotting functions using saved results"""
        
        self.log("="*60)
        self.log("GENERATING PLOTS FROM SAVED RESULTS")
        self.log("="*60)
        
        start_time = time.time()
        
        try:
            # Load saved results
            self.load_saved_results()
            
            # Generate all plots
            self.generate_plots()
            
            elapsed_time = time.time() - start_time
            self.log(f"\n{'='*60}")
            self.log(f"✅ PLOTTING COMPLETED SUCCESSFULLY")
            self.log(f"Time elapsed: {elapsed_time:.1f} seconds")
            self.log(f"Plots saved to: {self.plots_dir}")
            self.log(f"{'='*60}")
            
        except Exception as e:
            self.log(f"\n❌ Error during plotting: {str(e)}")
            self.log("Make sure you have run the full pipeline at least once.")
            raise
    
    def run(self):
        """Execute the complete pipeline or just plotting based on mode"""
        
        if self.plot_only:
            return self.run_plotting_only()
        
        self.log("="*60)
        self.log("FINAL PRODUCTION PIPELINE WITH MULTIPLE IMPUTATION")
        self.log("="*60)
        
        start_time = time.time()
        
        # Load data
        self.log("\nLoading data...")
        data_path = self.script_dir / "base4.csv"
        df = pd.read_csv(data_path)
        
        # Prepare features
        exclude_cols = [
            'Optimal_ideal_outcome', 'Best_performers', 'ISGPS_Grade',
            'Any_Hemorrhage', 'Any_Bile_Leakage', 'Any_POPF',
            'Re_operation_anycause', 'Readmission',
            'Mortality', 'LOS', 'Any_SSI', 
            'Any_organ_SSI', 'Any_POPF_Bile_SSI'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # Create missing indicators for features with substantial missingness (>10%)
        # This preserves informative missingness patterns
        missing_threshold = 0.10
        for col in X.columns:
            missing_rate = X[col].isna().mean()
            if missing_rate > missing_threshold:
                # Create binary indicator for missingness
                X[f'{col}_missing'] = X[col].isna().astype(float)
                self.log(f"Created missing indicator for {col} ({missing_rate:.1%} missing)")
        
        # Convert categorical to numeric (before imputation) and track mappings
        for col in X.columns:
            if X[col].dtype == 'object' and not col.endswith('_missing'):
                cat = pd.Categorical(X[col])
                X[col] = cat.codes
                # Store the mapping for later reference
                self.categorical_mappings[col] = dict(enumerate(cat.categories))
        
        # Replace -1 (encoded missing) with NaN for proper imputation
        # Missing indicators preserve the information about missingness
        X = X.replace(-1, np.nan)
        
        # Target: BEST performers
        if 'Best_performers' in df.columns:
            y = df['Best_performers']
        else:
            y = (df['Optimal_ideal_outcome'] == 1).astype(int)

        self.log(f"Features: {len(feature_cols)}, Samples: {len(df)}")
        self.log(f"Target (BEST performers): {y.sum()} / {len(y)} ({y.mean():.1%})")
        
        # Identify feature types and store them
        self.continuous_features, all_categorical = identify_feature_types(X, [])
        
        # Separate binary from multi-category features
        self.binary_features = []
        self.categorical_features = []
        
        for feat in all_categorical:
            n_unique = X[feat].nunique()
            if n_unique == 2:
                self.binary_features.append(feat)
            else:
                self.categorical_features.append(feat)
        
        # Also include originally categorical features that became binary after encoding
        for feat in self.categorical_mappings:
            if feat in X.columns and X[feat].nunique() == 2 and feat not in self.binary_features:
                self.binary_features.append(feat)
        
        # Add missing indicators to binary features (they are always binary: 0 or 1)
        missing_indicators = [col for col in X.columns if col.endswith('_missing')]
        for indicator in missing_indicators:
            if indicator not in self.binary_features:
                self.binary_features.append(indicator)
        
        self.log(f"Continuous features: {len(self.continuous_features)}")
        self.log(f"Binary features: {len(self.binary_features)}")
        self.log(f"Multi-category features: {len(self.categorical_features)}")
        
        # Run nested CV with imputation
        results = self.run_nested_cv(X, y)
        
        # Train final model on full imputed dataset
        final_model, X_imputed = self.train_final_model(X, y)
        
        # Generate visualizations
        self.generate_plots()
        
        # Save results
        self.save_results()
        
        # Save feature mappings
        self.save_feature_mappings()
        
        elapsed_time = time.time() - start_time
        self.log(f"\n{'='*60}")
        self.log(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
        self.log(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"{'='*60}")
        
        return results


def main():
    """Execute the final production pipeline with Multiple Imputation"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Final Production Pipeline with Multiple Imputation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run full pipeline (training + plotting)
  python %(prog)s
  
  # Generate plots only from saved results
  python %(prog)s --plot-only
  
  # Run quietly
  python %(prog)s --quiet
'''
    )
    
    parser.add_argument(
        '--plot-only', 
        action='store_true',
        help='Only generate plots from previously saved results (skip training)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Create pipeline with appropriate settings
    pipeline = FinalImputedPipeline(
        verbose=not args.quiet,
        plot_only=args.plot_only
    )
    
    # Run pipeline
    results = pipeline.run()
    return results


if __name__ == "__main__":
    results = main()
