"""
Enhanced Training Script for Fraud Detection Model
Includes advanced features, SMOTE, threshold tuning, and comprehensive evaluation
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import FraudPreprocessor
from model_utils import (
    FraudDetectionModel,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_fraud_detection_model(
    data_path: str = 'creditcard.csv',
    model_path: str = 'models/rf_fraud_model.joblib',
    assets_path: str = 'assets',
    use_smote: bool = True,
    optimize_threshold: bool = True
):
    """
    Complete training pipeline for fraud detection model
    
    Args:
        data_path: Path to training data CSV
        model_path: Path to save trained model
        assets_path: Path to save plots and assets
        use_smote: Whether to use SMOTE for balancing
        optimize_threshold: Whether to optimize decision threshold
    """
    logger.info("=" * 50)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 50)
    
    # Create directories
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(assets_path).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and preprocess data
    logger.info("\n[1/6] Loading and preprocessing data...")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please ensure 'creditcard.csv' is in the project directory")
        logger.info("You can download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return
    
    preprocessor = FraudPreprocessor()
    X, y = preprocessor.prepare_features(df, target_col='Class', fit=True)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Step 2: Split data
    logger.info("\n[2/6] Splitting data into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 3: Apply SMOTE
    if use_smote:
        logger.info("\n[3/6] Applying SMOTE for class balancing...")
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE - Train size: {len(X_train_res)}")
        logger.info(f"Class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")
    else:
        logger.info("\n[3/6] Skipping SMOTE...")
        X_train_res, y_train_res = X_train, y_train
    
    # Step 4: Train model
    logger.info("\n[4/6] Training Random Forest model...")
    model = FraudDetectionModel()
    model.train(X_train_res, y_train_res)
    
    # Step 5: Optimize threshold
    if optimize_threshold:
        logger.info("\n[5/6] Optimizing decision threshold...")
        threshold = model.optimize_threshold(X_val, y_val, metric='f1')
        logger.info(f"Optimal threshold: {threshold:.3f}")
    else:
        logger.info("\n[5/6] Using default threshold: 0.5")
    
    # Step 6: Evaluate on test set
    logger.info("\n[6/6] Evaluating on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n" + "=" * 50)
    print("TEST SET PERFORMANCE")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("=" * 50)
    
    # Generate predictions for visualization
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Create visualizations
    logger.info("\nGenerating visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(assets_path, 'confusion_matrix.png')
    )
    
    # ROC Curve
    plot_roc_curve(
        y_test, y_proba,
        save_path=os.path.join(assets_path, 'roc_curve.png')
    )
    
    # Precision-Recall Curve
    plot_precision_recall_curve(
        y_test, y_proba,
        save_path=os.path.join(assets_path, 'precision_recall_curve.png')
    )
    
    # Feature Importance
    plot_feature_importance(
        model.get_feature_importance(top_n=20),
        save_path=os.path.join(assets_path, 'feature_importance.png')
    )
    
    logger.info(f"All plots saved to {assets_path}/")
    
    # Save model and preprocessor
    logger.info("\nSaving model and preprocessor...")
    model.save(model_path)
    
    import joblib
    preprocessor_path = model_path.replace('.joblib', '_preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(assets_path, 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save feature importance
    feature_imp_path = os.path.join(assets_path, 'feature_importance.csv')
    model.get_feature_importance().to_csv(feature_imp_path, index=False)
    logger.info(f"Feature importance saved to {feature_imp_path}")
    
    print("\n✅ Training completed successfully!")
    print(f"📊 Model saved to: {model_path}")
    print(f"📈 Assets saved to: {assets_path}/")
    
    return model, preprocessor, metrics


if __name__ == "__main__":
    train_fraud_detection_model()
