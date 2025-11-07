"""
Production-Ready Streamlit Dashboard for Credit Card Fraud Detection
Modern, interactive UI with explainability, metrics, and professional visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import joblib
import logging
from typing import Optional, Dict
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_utils import FraudDetectionModel
from preprocess import FraudPreprocessor
from ui_components import (
    apply_custom_css, show_metric_card, show_alert_banner,
    plot_fraud_distribution, plot_fraud_pie_chart, 
    plot_feature_importance_bar, plot_amount_distribution,
    plot_time_series, show_transaction_details, create_download_button,
    create_metrics_dashboard, show_fraud_alert_modal, create_searchable_table
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_css()


@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor"""
    model_path = 'models/rf_fraud_model.joblib'
    preprocessor_path = 'models/rf_fraud_model_preprocessor.joblib'
    
    # Fallback to old paths
    if not os.path.exists(model_path):
        model_path = 'rf_fraud_model.joblib'
    if not os.path.exists(preprocessor_path):
        preprocessor_path = 'rf_fraud_model_preprocessor.joblib'
    
    try:
        model = FraudDetectionModel.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Model and preprocessor loaded successfully")
        return model, preprocessor
    except Exception as e:
        logger.warning(f"Model not found: {e}")
        return None, None


def generate_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate sample transaction data for demo"""
    np.random.seed(42)
    
    # Generate mostly normal transactions
    normal_count = int(n_samples * 0.95)
    fraud_count = n_samples - normal_count
    
    data = []
    
    # Normal transactions
    for i in range(normal_count):
        row = {
            'Time': np.random.randint(0, 172800),
            'Amount': np.random.exponential(50),
            **{f'V{j}': np.random.randn() for j in range(1, 29)}
        }
        data.append(row)
    
    # Fraud transactions (with different patterns)
    for i in range(fraud_count):
        row = {
            'Time': np.random.randint(0, 172800),
            'Amount': np.random.exponential(200),  # Higher amounts
            **{f'V{j}': np.random.randn() * 2 for j in range(1, 29)}  # Different distribution
        }
        data.append(row)
    
    return pd.DataFrame(data)


def explain_prediction(transaction: pd.Series, model: FraudDetectionModel) -> Dict:
    """Generate explanation for fraud prediction"""
    explanation = {
        'text': '',
        'top_features': {}
    }
    
    # Get feature importance
    if model.feature_importance is not None:
        top_features = model.feature_importance.head(5)
        
        # Build explanation text
        prob = transaction.get('fraud_probability', 0)
        if prob > 0.7:
            risk_level = "HIGH RISK"
            color = "🔴"
        elif prob > 0.4:
            risk_level = "MEDIUM RISK"
            color = "🟡"
        else:
            risk_level = "LOW RISK"
            color = "🟢"
        
        explanation['text'] = f"""
        {color} **{risk_level}** - Fraud Probability: {prob*100:.1f}%
        
        This transaction was flagged based on the following key factors:
        """
        
        # Add top contributing features
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            if feature in transaction.index:
                value = transaction[feature]
                explanation['top_features'][feature] = importance
                explanation['text'] += f"\n- **{feature}**: {value:.3f} (importance: {importance:.3f})"
    
    return explanation


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-shield-green.png", width=100)
        st.title("🛡️ Fraud Detection AI")
        st.markdown("---")
        
        st.markdown("""
        ### About
        Advanced ML-powered fraud detection system using Random Forest with:
        - 🎯 High-precision predictions
        - 🔍 Real-time analysis
        - 📊 Explainable AI
        - 🚀 Production-ready
        """)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home", "📊 Dashboard", "🔍 Explainability", "📈 Model Metrics", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        use_sample = st.button("📂 Load Sample Data", use_container_width=True)
    
    # Load model (optional - app works without it)
    model, preprocessor = load_model_and_preprocessor()
    
    if model is None or preprocessor is None:
        st.warning("⚠️ Model not found - Running in DEMO mode with sample predictions")
        with st.expander("ℹ️ How to train the model"):
            st.code("python src/train.py", language="bash")
            st.info("You can still explore the app with sample data!")
    
    # Main content based on page selection
    if "Home" in page or "Dashboard" in page:
        show_dashboard_page(model, preprocessor, use_sample)
    elif "Explainability" in page:
        show_explainability_page(model, preprocessor, use_sample)
    elif "Model Metrics" in page:
        show_metrics_page(model)
    elif "About" in page:
        show_about_page()


def show_dashboard_page(model: Optional[FraudDetectionModel], preprocessor: Optional[FraudPreprocessor], use_sample: bool):
    """Main dashboard page"""
    
    # Header
    st.title("🚨 Credit Card Fraud Detection Dashboard")
    st.markdown("Upload transaction data to detect fraudulent activities in real-time")
    
    # File upload section
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Transaction Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with transaction data"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Use Sample Data", use_container_width=True) or use_sample:
            uploaded_file = "sample"
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file == "sample":
                with st.spinner("Generating sample data..."):
                    df = generate_sample_data(500)
                show_alert_banner("ℹ️ Using sample data for demonstration", "info")
            else:
                df = pd.read_csv(uploaded_file)
                show_alert_banner(f"✅ Loaded {len(df)} transactions successfully", "success")
            
            # Preprocess and predict
            with st.spinner("🔄 Analyzing transactions..."):
                if model is not None and preprocessor is not None:
                    # Use real model
                    X, _ = preprocessor.prepare_features(df, fit=False)
                    probs = model.predict_proba(X)
                    preds = model.predict(X)
                else:
                    # Demo mode - generate random predictions
                    np.random.seed(42)
                    probs = np.random.beta(2, 10, size=len(df))  # Most values low, some high
                    preds = (probs > 0.5).astype(int)
                
                df['fraud_probability'] = probs
                df['predicted_fraud'] = preds
            
            # Summary metrics
            fraud_count = preds.sum()
            normal_count = len(preds) - fraud_count
            total_count = len(df)
            
            st.markdown("---")
            
            # Alert banner for fraud detection
            show_fraud_alert_modal(fraud_count)
            
            # Key metrics
            st.markdown("### 📊 Detection Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", f"{total_count:,}")
            with col2:
                st.metric("🔴 Fraud Detected", f"{fraud_count:,}", 
                         delta=f"{(fraud_count/total_count)*100:.1f}%")
            with col3:
                st.metric("🟢 Normal", f"{normal_count:,}")
            with col4:
                avg_prob = df['fraud_probability'].mean()
                st.metric("Avg Risk Score", f"{avg_prob*100:.1f}%")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "🚨 Flagged Transactions",
                "📊 Visualizations", 
                "📋 All Transactions",
                "💾 Export"
            ])
            
            with tab1:
                st.markdown("### 🚨 High-Risk Transactions")
                flagged = df[df['predicted_fraud'] == 1].sort_values(
                    'fraud_probability', ascending=False
                )
                
                if len(flagged) > 0:
                    # Show top flagged transactions
                    st.markdown(f"**Found {len(flagged)} suspicious transactions**")
                    
                    # Display table
                    display_cols = ['fraud_probability', 'Amount'] if 'Amount' in df.columns else ['fraud_probability']
                    display_cols += [col for col in df.columns if col.startswith('V')][:5]
                    
                    st.dataframe(
                        flagged[display_cols].head(50),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Alert simulation
                    st.markdown("### 📧 Automated Alerts")
                    if st.button("🔔 Simulate Alert Notification"):
                        st.success(f"✅ Email alert sent for {len(flagged)} flagged transactions")
                        st.info("📱 SMS notification sent to fraud team")
                        st.balloons()
                else:
                    st.success("✅ No fraudulent transactions detected!")
            
            with tab2:
                st.markdown("### 📊 Visual Analytics")
                
                # Charts in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fraud distribution
                    fig1 = plot_fraud_distribution(df)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Amount distribution (if available)
                    if 'Amount' in df.columns:
                        fig3 = plot_amount_distribution(df)
                        if fig3:
                            st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig2 = plot_fraud_pie_chart(fraud_count, normal_count)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Time series (if available)
                    if 'Time' in df.columns:
                        fig4 = plot_time_series(df)
                        if fig4:
                            st.plotly_chart(fig4, use_container_width=True)
            
            with tab3:
                st.markdown("### 📋 All Transactions")
                
                # Filter options
                col1, col2 = st.columns([1, 3])
                with col1:
                    filter_option = st.selectbox(
                        "Filter by",
                        ["All", "Fraud Only", "Normal Only"]
                    )
                
                # Apply filter
                if filter_option == "Fraud Only":
                    display_df = df[df['predicted_fraud'] == 1]
                elif filter_option == "Normal Only":
                    display_df = df[df['predicted_fraud'] == 0]
                else:
                    display_df = df
                
                st.dataframe(display_df, use_container_width=True, height=400)
            
            with tab4:
                st.markdown("### 💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Flagged Transactions")
                    flagged_df = df[df['predicted_fraud'] == 1]
                    create_download_button(
                        flagged_df,
                        'flagged_transactions.csv',
                        'Download Flagged Transactions'
                    )
                
                with col2:
                    st.markdown("#### Complete Results")
                    create_download_button(
                        df,
                        'all_predictions.csv',
                        'Download All Predictions'
                    )
                
                # Generate PDF report button (placeholder)
                st.markdown("---")
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    st.info("📄 PDF report generation feature coming soon!")
        
        except Exception as e:
            st.error(f"❌ Error processing data: {e}")
            logger.error(f"Error in dashboard: {e}", exc_info=True)


def show_explainability_page(model: Optional[FraudDetectionModel], preprocessor: Optional[FraudPreprocessor], use_sample: bool):
    """Explainability and feature importance page"""
    
    st.title("🔍 Model Explainability")
    st.markdown("Understand what drives fraud predictions")
    
    if model is None:
        st.warning("⚠️ Model not loaded - Train the model to see explainability features")
        st.info("Run: `python src/train.py` to train the model")
        return
    
    # Feature importance
    st.markdown("### 📊 Global Feature Importance")
    st.markdown("Features that contribute most to fraud detection across all transactions")
    
    if model.feature_importance is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            top_n = st.slider("Number of features to display", 5, 30, 15)
            fig = plot_feature_importance_bar(model.feature_importance, top_n)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Top 10 Features")
            st.dataframe(
                model.feature_importance.head(10)[['feature', 'importance']],
                hide_index=True,
                use_container_width=True
            )
    
    # Individual prediction explanation
    st.markdown("---")
    st.markdown("### 🎯 Individual Transaction Explanation")
    
    uploaded_file = st.file_uploader(
        "Upload a transaction file for detailed explanation",
        type=['csv'],
        key="explain_upload"
    )
    
    if st.button("🎲 Use Sample Transaction") or use_sample:
        sample_df = generate_sample_data(10)
        uploaded_file = "sample_explain"
        st.session_state['explain_df'] = sample_df
    
    if uploaded_file is not None:
        try:
            if uploaded_file == "sample_explain":
                df = st.session_state.get('explain_df', generate_sample_data(10))
            else:
                df = pd.read_csv(uploaded_file)
            
            # Process data
            X, _ = preprocessor.prepare_features(df, fit=False)
            probs = model.predict_proba(X)
            df['fraud_probability'] = probs
            
            # Select transaction
            st.markdown("#### Select a transaction to explain")
            transaction_idx = st.selectbox(
                "Transaction Index",
                range(len(df)),
                format_func=lambda x: f"Transaction {x} (Risk: {df.iloc[x]['fraud_probability']*100:.1f}%)"
            )
            
            transaction = df.iloc[transaction_idx]
            
            # Show explanation
            explanation = explain_prediction(transaction, model)
            show_transaction_details(transaction, explanation)
            
        except Exception as e:
            st.error(f"Error generating explanation: {e}")


def show_metrics_page(model: Optional[FraudDetectionModel]):
    """Model performance metrics page"""
    
    st.title("📈 Model Performance Metrics")
    st.markdown("Comprehensive evaluation of the fraud detection model")
    
    if model is None:
        st.warning("⚠️ Model not loaded - Train the model to see metrics")
        st.info("Run: `python src/train.py` to train the model")
        return
    
    # Load saved metrics if available
    metrics_path = 'assets/model_metrics.csv'
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.iloc[0].to_dict()
        
        create_metrics_dashboard(metrics)
        
        # Detailed metrics
        st.markdown("---")
        st.markdown("### 📊 Detailed Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Classification Metrics")
            st.dataframe(
                pd.DataFrame([metrics]).T.rename(columns={0: 'Value'}),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Model Information")
            st.info(f"""
            **Model Type:** Random Forest Classifier
            **Decision Threshold:** {model.threshold:.3f}
            **Features:** {len(model.feature_importance) if model.feature_importance is not None else 'N/A'}
            """)
    else:
        st.warning("⚠️ No saved metrics found. Train the model to see performance metrics.")
    
    # Display plots if available
    st.markdown("---")
    st.markdown("### 📊 Performance Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('assets/confusion_matrix.png'):
            st.image('assets/confusion_matrix.png', caption='Confusion Matrix')
        
        if os.path.exists('assets/precision_recall_curve.png'):
            st.image('assets/precision_recall_curve.png', caption='Precision-Recall Curve')
    
    with col2:
        if os.path.exists('assets/roc_curve.png'):
            st.image('assets/roc_curve.png', caption='ROC Curve')
        
        if os.path.exists('assets/feature_importance.png'):
            st.image('assets/feature_importance.png', caption='Feature Importance')


def show_about_page():
    """About page with project information"""
    
    st.title("ℹ️ About Fraud Detection AI")
    
    st.markdown("""
    ## 🛡️ Advanced Credit Card Fraud Detection System
    
    This is a production-ready machine learning system designed to detect fraudulent credit card transactions
    with high accuracy and reliability.
    
    ### 🎯 Key Features
    
    - **High Accuracy**: Random Forest model with advanced feature engineering
    - **Real-time Detection**: Process transactions instantly
    - **Explainable AI**: Understand why transactions are flagged
    - **Interactive Dashboard**: Beautiful, modern UI with real-time visualizations
    - **Production Ready**: Docker support, logging, error handling
    - **Scalable**: Designed for enterprise deployment
    
    ### 🔬 Technical Stack
    
    - **ML Framework**: scikit-learn, imbalanced-learn
    - **UI**: Streamlit with Plotly visualizations
    - **Data Processing**: pandas, numpy
    - **Model**: Random Forest with SMOTE balancing
    
    ### 📊 Model Performance
    
    Our model achieves:
    - High recall to catch most fraud cases
    - Balanced precision to minimize false alarms
    - Optimized threshold for business requirements
    
    ### 🚀 Deployment Options
    
    - **Streamlit Cloud**: One-click deployment
    - **Docker**: Containerized deployment
    - **Render/Railway**: Cloud platform deployment
    - **HuggingFace Spaces**: ML-focused hosting
    
    ### 👥 Team
    
    Built with ❤️ for hackathon excellence
    
    ### 📞 Support
    
    For questions or issues, please contact the development team.
    """)


if __name__ == "__main__":
    main()
