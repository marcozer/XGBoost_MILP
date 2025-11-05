#!/usr/bin/env python3
"""
COMPLETE CASES ANALYSIS - SENSITIVITY ANALYSIS
==============================================
This serves as a sensitivity analysis to show results without any imputation.

Key Features:
Filters to complete cases only (N≈1,225)
No imputation needed : it's a sensitivity analysis

Target: Best Performers

Authors: Marc-Anthony Chouillard, Clément Pastier
Date: sept 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    brier_score_loss, precision_recall_curve,
    average_precision_score, auc, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import shap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constants
RANDOM_STATE = 42
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 3
N_TRIALS = 300  
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Colors
COLORS = {
    'primary': '#0066CC',
    'secondary': '#CC0000',
    'accent': '#FF9900',
    'success': '#009900',
    'neutral': '#666666',
    'light': '#CCCCCC'
}


def save_plot_all_formats(fig, base_dir, name, dpi=300):
    """Save plot in PNG, PDF, and SVG formats"""
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    for fmt in ['png', 'pdf', 'svg']:
        output_path = base_dir / f'{name}.{fmt}'
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', format=fmt)


def calculate_ece(y_true, y_proba, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def bootstrap_roc_auc(y_true, y_scores, n_bootstrap=1000):
    """Bootstrap ROC curve for confidence intervals"""
    np.random.seed(RANDOM_STATE)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc_original = auc(fpr, tpr)
    
    bootstrapped_scores = []
    bootstrapped_tprs = []
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
    
    bootstrapped_tprs = np.array(bootstrapped_tprs)
    tpr_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
    tpr_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
    
    auc_ci_lower = np.percentile(bootstrapped_scores, 2.5)
    auc_ci_upper = np.percentile(bootstrapped_scores, 97.5)
    
    return {
        'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
        'auc': roc_auc_original,
        'auc_ci': (auc_ci_lower, auc_ci_upper),
        'fpr_mean': mean_fpr,
        'tpr_lower': tpr_lower,
        'tpr_upper': tpr_upper
    }


class CompleteCasesPipeline:
    """Pipeline using only complete cases (no missing data)"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.script_dir = Path(__file__).parent
        self.results_dir = self.script_dir.parent / "results"
        self.output_dir = self.results_dir / "production_complete_cases"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Subdirectories
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "visualizations"
        self.results_subdir = self.output_dir / "results"
        
        for dir_path in [self.models_dir, self.plots_dir, self.results_subdir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        self.results = {}
        self.fold_models = []
        
    def log(self, message):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def filter_complete_cases(self, X, y):
        """Filter to complete cases only"""
        # Find complete cases
        complete_mask = ~X.isna().any(axis=1)
        
        # Report statistics
        n_complete = complete_mask.sum()
        n_total = len(X)
        pct_complete = (n_complete / n_total) * 100
        
        self.log(f"Complete cases: {n_complete} / {n_total} ({pct_complete:.1f}%)")
        
        # Features with missing data
        missing_counts = X[~complete_mask].isna().sum()
        features_causing_exclusion = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        self.log("Features causing exclusion:")
        for feat, count in features_causing_exclusion.head(5).items():
            self.log(f"  {feat}: {count} missing values")
        
        # Filter data
        X_complete = X[complete_mask].copy()
        y_complete = y[complete_mask].copy()
        
        return X_complete, y_complete, complete_mask
    
    def create_xgb_objective(self, X_train, y_train, X_val, y_val, sample_weights):
        """Create Optuna objective"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1200),  # Extended from 800
                'max_depth': trial.suggest_int('max_depth', 2, 15),  # Extended from 10
                'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.5, log=True),  # Lower minimum
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),  # Reduced from 50
                'gamma': trial.suggest_float('gamma', 0.0, 3.0),  # Extended from 2.0
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Raised minimum from 0.5
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),  # Lowered minimum from 0.5
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),  # Extended from 5.0
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 15.0),  # Extended from 10.0
                'random_state': RANDOM_STATE,
                'tree_method': 'hist',
                'enable_categorical': True,
                'verbosity': 0
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            return auc_score
        
        return objective
    
    def run_nested_cv(self, X, y):
        """Run nested CV on complete cases"""
        
        self.log("Starting Nested CV on Complete Cases")
        self.log(f"Outer: {N_OUTER_FOLDS} folds, Inner: {N_INNER_FOLDS} folds")
        
        outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        oof_predictions = np.zeros(len(y))
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            self.log(f"\n{'='*50}")
            self.log(f"FOLD {fold_idx + 1}/{N_OUTER_FOLDS}")
            self.log(f"{'='*50}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Sample weights
            pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            sample_weights = np.where(y_train == 1, np.sqrt(pos_weight), 1.0)
            
            # Inner CV
            inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            train_inner_idx, val_inner_idx = next(inner_cv.split(X_train, y_train))
            
            X_train_inner = X_train.iloc[train_inner_idx]
            y_train_inner = y_train.iloc[train_inner_idx]
            X_val_inner = X_train.iloc[val_inner_idx]
            y_val_inner = y_train.iloc[val_inner_idx]
            
            # Optuna optimization
            self.log(f"Running Optuna ({N_TRIALS} trials)...")
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=RANDOM_STATE),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            
            objective = self.create_xgb_objective(
                X_train_inner, y_train_inner,
                X_val_inner, y_val_inner,
                sample_weights[train_inner_idx]
            )
            
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            best_params = study.best_params
            
            # Train final fold model
            best_params.update({
                'random_state': RANDOM_STATE,
                'tree_method': 'hist',
                'enable_categorical': True,
                'verbosity': 0
            })
            
            model = xgb.XGBClassifier(**best_params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            
            # Predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            oof_predictions[test_idx] = y_pred_proba
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            brier_score = brier_score_loss(y_test, y_pred_proba)
            ece = calculate_ece(y_test, y_pred_proba)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'auc': auc_score,
                'brier': brier_score,
                'ece': ece,
                'best_params': best_params,
                'n_train': len(y_train),
                'n_test': len(y_test)
            })
            
            self.fold_models.append(model)
            
            self.log(f"AUC: {auc_score:.4f}, Brier: {brier_score:.4f}, ECE: {ece:.4f}")
        
        # Overall metrics
        overall_auc = roc_auc_score(y, oof_predictions)
        overall_brier = brier_score_loss(y, oof_predictions)
        overall_ece = calculate_ece(y.values, oof_predictions)
        
        self.results = {
            'fold_results': fold_results,
            'oof_predictions': oof_predictions,
            'y_true': y.values,
            'overall': {
                'auc': overall_auc,
                'auc_std': np.std([f['auc'] for f in fold_results]),
                'brier': overall_brier,
                'ece': overall_ece,
                'n_samples': len(y)
            }
        }
        
        self.log(f"\n{'='*50}")
        self.log("OVERALL RESULTS (COMPLETE CASES)")
        self.log(f"AUC: {overall_auc:.4f} ± {self.results['overall']['auc_std']:.4f}")
        self.log(f"Brier: {overall_brier:.4f}, ECE: {overall_ece:.4f}")
        self.log(f"N = {len(y)}")
        
        return self.results
    
    def generate_plots(self):
        """Generate visualization plots"""
        
        self.log("\nGenerating visualizations...")
        
        y_true = self.results['y_true']
        y_proba = self.results['oof_predictions']
        
        # ROC Curve
        self.plot_roc_curve(y_true, y_proba)
        
        # Calibration Curve
        self.plot_calibration_curve(y_true, y_proba)
        
        self.log(f"Plots saved to: {self.plots_dir}")
    
    def plot_roc_curve(self, y_true, y_proba):
        """Generate ROC curve"""
        
        roc_data = bootstrap_roc_auc(y_true, y_proba, N_BOOTSTRAP)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # CI band
        ax.fill_between(
            roc_data['fpr_mean'],
            roc_data['tpr_lower'],
            roc_data['tpr_upper'],
            color=COLORS['primary'], alpha=0.2,
            label=f'95% CI'
        )
        
        # ROC curve
        ax.plot(
            roc_data['fpr'], roc_data['tpr'],
            color=COLORS['primary'], linewidth=2.5,
            label=f"AUC = {roc_data['auc']:.3f} [{roc_data['auc_ci'][0]:.3f}-{roc_data['auc_ci'][1]:.3f}]"
        )
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Optimal point
        j_scores = roc_data['tpr'] - roc_data['fpr']
        optimal_idx = np.argmax(j_scores)
        ax.plot(
            roc_data['fpr'][optimal_idx],
            roc_data['tpr'][optimal_idx],
            'o', color=COLORS['secondary'], markersize=10,
            label=f'Optimal (thr={roc_data["thresholds"][optimal_idx]:.3f})'
        )
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve - Complete Cases Only', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        save_plot_all_formats(fig, self.plots_dir, 'roc_curve')
        plt.close()
    
    def plot_calibration_curve(self, y_true, y_proba):
        """Generate calibration curve"""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        
        fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
        ece = calculate_ece(y_true, y_proba)
        
        ax.plot(mean_pred, fraction_pos, 'o-',
                color=COLORS['primary'], linewidth=2.5, markersize=10,
                label=f'Model (ECE={ece:.3f})')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Curve - Complete Cases Only', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        save_plot_all_formats(fig, self.plots_dir, 'calibration_curve')
        plt.close()
    
    def save_results(self):
        """Save results to disk"""
        
        self.log("\nSaving results...")
        
        # Metrics
        metrics = {
            'overall': self.results['overall'],
            'fold_results': self.results['fold_results'],
            'method': 'Complete Cases Only'
        }
        
        with open(self.results_subdir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=float)
        
        # Predictions
        pred_df = pd.DataFrame({
            'y_true': self.results['y_true'],
            'y_proba': self.results['oof_predictions']
        })
        pred_df.to_csv(self.results_subdir / 'predictions.csv', index=False)
        
        # Best model
        if self.fold_models:
            best_idx = np.argmax([f['auc'] for f in self.results['fold_results']])
            joblib.dump(self.fold_models[best_idx], self.models_dir / 'best_model.pkl')
        
        self.log(f"Results saved to: {self.output_dir}")
    
    def run(self):
        """Execute complete pipeline"""
        
        self.log("="*60)
        self.log("COMPLETE CASES ANALYSIS (SENSITIVITY)")
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
        
        # Convert categorical to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        # Target
        if 'Best_performers' in df.columns:
            y = df['Best_performers']
        else:
            y = (df['Optimal_ideal_outcome'] == 1).astype(int)
        
        self.log(f"Total samples: {len(df)}")
        self.log(f"Features: {len(feature_cols)}")
        
        # Filter to complete cases
        X_complete, y_complete, complete_mask = self.filter_complete_cases(X, y)
        
        self.log(f"Target prevalence: {y_complete.mean():.1%}")
        
        # Save complete case indices for reference
        self.complete_indices = complete_mask
        
        # Run nested CV
        results = self.run_nested_cv(X_complete, y_complete)
        
        # Generate plots
        self.generate_plots()
        
        # Save results
        self.save_results()
        
        elapsed_time = time.time() - start_time
        self.log(f"\n{'='*60}")
        self.log(f"✅ COMPLETE CASES ANALYSIS FINISHED")
        self.log(f"Time: {elapsed_time/60:.1f} minutes")
        self.log(f"Output: {self.output_dir}")
        self.log(f"{'='*60}")
        
        return results


def main():
    """Run complete cases analysis"""
    pipeline = CompleteCasesPipeline(verbose=True)
    results = pipeline.run()
    return results


if __name__ == "__main__":
    results = main()
