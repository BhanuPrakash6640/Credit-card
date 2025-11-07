"""
UI Components for Streamlit Dashboard
Reusable components for metrics, charts, and visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional
import time


def show_metric_card(label: str, value: str, delta: Optional[str] = None, icon: str = "📊"):
    """Display a metric card with icon"""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"<h1 style='text-align: center; margin: 0;'>{icon}</h1>", unsafe_allow_html=True)
    with col2:
        st.metric(label=label, value=value, delta=delta)


def show_alert_banner(message: str, alert_type: str = "warning"):
    """Display an alert banner"""
    colors = {
        "success": "#d4edda",
        "warning": "#fff3cd",
        "danger": "#f8d7da",
        "info": "#d1ecf1"
    }
    border_colors = {
        "success": "#c3e6cb",
        "warning": "#ffeeba",
        "danger": "#f5c6cb",
        "info": "#bee5eb"
    }
    
    color = colors.get(alert_type, colors["info"])
    border = border_colors.get(alert_type, border_colors["info"])
    
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            border: 1px solid {border};
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def plot_fraud_distribution(df: pd.DataFrame, proba_col: str = 'fraud_probability'):
    """Create interactive fraud probability distribution plot"""
    fig = px.histogram(
        df,
        x=proba_col,
        nbins=50,
        title='Distribution of Fraud Probability',
        labels={proba_col: 'Fraud Probability', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#FF6B6B']
    )
    fig.update_layout(
        xaxis_title="Fraud Probability",
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    return fig


def plot_fraud_pie_chart(fraud_count: int, normal_count: int):
    """Create pie chart for fraud vs normal transactions"""
    fig = go.Figure(data=[go.Pie(
        labels=['Normal', 'Fraud'],
        values=[normal_count, fraud_count],
        hole=0.4,
        marker_colors=['#4ECDC4', '#FF6B6B']
    )])
    fig.update_layout(
        title='Fraud vs Normal Transactions',
        height=400,
        annotations=[dict(text='Transactions', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig


def plot_feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 15):
    """Create interactive feature importance bar chart"""
    data = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    fig = px.bar(
        data,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig


def plot_amount_distribution(df: pd.DataFrame, fraud_col: str = 'predicted_fraud'):
    """Plot amount distribution for fraud vs normal"""
    if 'Amount' not in df.columns:
        return None
    
    fig = px.box(
        df,
        x=fraud_col,
        y='Amount',
        color=fraud_col,
        title='Transaction Amount Distribution by Fraud Status',
        labels={fraud_col: 'Fraud Status', 'Amount': 'Transaction Amount'},
        color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
    )
    fig.update_layout(height=400)
    return fig


def plot_time_series(df: pd.DataFrame, fraud_col: str = 'predicted_fraud'):
    """Plot fraud detection over time"""
    if 'Time' not in df.columns:
        return None
    
    # Create hourly aggregation
    df_copy = df.copy()
    df_copy['hour'] = (df_copy['Time'] / 3600) % 24
    hourly = df_copy.groupby('hour')[fraud_col].agg(['sum', 'count']).reset_index()
    hourly['fraud_rate'] = (hourly['sum'] / hourly['count']) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly['hour'],
        y=hourly['fraud_rate'],
        mode='lines+markers',
        name='Fraud Rate',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Fraud Detection Rate by Hour of Day',
        xaxis_title='Hour of Day',
        yaxis_title='Fraud Rate (%)',
        height=400,
        hovermode='x'
    )
    return fig


def show_transaction_details(transaction: pd.Series, explanation: Optional[Dict] = None):
    """Display detailed transaction information"""
    st.subheader("🔍 Transaction Details")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'Amount' in transaction.index:
            st.metric("Amount", f"${transaction['Amount']:.2f}")
    with col2:
        if 'fraud_probability' in transaction.index:
            st.metric("Fraud Probability", f"{transaction['fraud_probability']*100:.1f}%")
    with col3:
        if 'Time' in transaction.index:
            hour = (transaction['Time'] / 3600) % 24
            st.metric("Hour of Day", f"{int(hour)}:00")
    
    # Explanation
    if explanation:
        st.markdown("### 🎯 Why This Was Flagged")
        st.markdown(explanation.get('text', 'No explanation available'))
        
        if 'top_features' in explanation:
            st.markdown("**Top Contributing Features:**")
            for feat, contrib in explanation['top_features'].items():
                st.write(f"- {feat}: {contrib:.3f}")


def create_download_button(df: pd.DataFrame, filename: str, label: str):
    """Create a styled download button"""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ {label}",
        data=csv,
        file_name=filename,
        mime='text/csv',
        use_container_width=True
    )


def show_loading_animation(message: str = "Processing..."):
    """Show loading animation"""
    with st.spinner(message):
        time.sleep(0.5)


def show_success_animation(message: str):
    """Show success message with animation"""
    st.success(message)
    st.balloons()


def create_metrics_dashboard(metrics: Dict[str, float]):
    """Create a metrics dashboard with cards"""
    st.markdown("### 📊 Model Performance Metrics")
    
    cols = st.columns(5)
    metric_list = [
        ("Accuracy", metrics.get('accuracy', 0), "🎯"),
        ("Precision", metrics.get('precision', 0), "🔍"),
        ("Recall", metrics.get('recall', 0), "✅"),
        ("F1 Score", metrics.get('f1', 0), "⚖️"),
        ("AUC-ROC", metrics.get('auc_roc', 0), "📈")
    ]
    
    for i, (label, value, icon) in enumerate(metric_list):
        with cols[i]:
            st.metric(
                label=f"{icon} {label}",
                value=f"{value:.3f}"
            )


def show_fraud_alert_modal(fraud_count: int):
    """Show fraud alert in a modal-style display"""
    if fraud_count > 0:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                <h2 style="margin: 0; color: white;">🚨 FRAUD ALERT</h2>
                <h1 style="margin: 10px 0; color: white;">{fraud_count}</h1>
                <p style="margin: 0; color: white;">Suspicious transactions detected</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def create_searchable_table(df: pd.DataFrame, key: str = "table"):
    """Create a searchable and sortable dataframe"""
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        hide_index=True
    )


def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #f0f2f6;
            border-radius: 5px 5px 0px 0px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4ECDC4;
            color: white;
        }
        h1 {
            color: #2C3E50;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
