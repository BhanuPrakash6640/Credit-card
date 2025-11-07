"""
Model Utilities for Fraud Detection
Includes training, evaluation, and visualization functions
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """Wrapper class for fraud detection model with utilities"""
    
    def __init__(self, model: Optional[Any] = None):
        self.model = model
        self.threshold = 0.5
        self.feature_importance = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **model_params
    ) -> 'FraudDetectionModel':
        """
        Train Random Forest model with optimized parameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            **model_params: Additional model parameters
            
        Returns:
            self
        """
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }
        default_params.update(model_params)
        
        logger.info(f"Training Random Forest with params: {default_params}")
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud labels"""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'f1'
    ) -> float:
        """
        Optimize decision threshold for better recall
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('f1', 'recall', 'precision')
            
        Returns:
            Optimal threshold
        """
        probs = self.predict_proba(X_val)
        
        if metric == 'recall':
            # For high recall, use lower threshold
            precision, recall, thresholds = precision_recall_curve(y_val, probs)
            # Find threshold where recall >= 0.95
            idx = np.where(recall >= 0.95)[0]
            if len(idx) > 0:
                self.threshold = thresholds[idx[0]]
            else:
                self.threshold = 0.3
        else:
            # Optimize F1 score
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                preds = (probs >= threshold).astype(int)
                f1 = f1_score(y_val, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.threshold = best_threshold
        
        logger.info(f"Optimized threshold: {self.threshold:.3f}")
        return self.threshold
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FraudDetectionModel':
        """Load model from disk"""
        data = joblib.load(filepath)
        instance = cls(data['model'])
        instance.threshold = data.get('threshold', 0.5)
        instance.feature_importance = data.get('feature_importance')
        logger.info(f"Model loaded from {filepath}")
        return instance


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Fraud'],
        yticklabels=['Normal', 'Fraud'],
        ax=ax
    )
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label='PR Curve')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PR curve saved to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance
    
    Args:
        feature_importance_df: DataFrame with features and importance
        top_n: Number of top features to plot
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    data = feature_importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=data,
        y='feature',
        x='importance',
        palette='viridis',
        ax=ax
    )
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig
