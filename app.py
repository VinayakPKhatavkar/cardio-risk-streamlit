import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# Define Application Configuration and layout
st.set_page_config(page_title="Cardiovascular Predictive Core", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f1f5f9; }
    h1 { color: #0f172a; font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 800; }
    
    .stMetric { 
        background-color: #1e293b !important; /* Dark background to make white text pop */
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
    }
    </style>
""", unsafe_allow_html=True)


st.title("🫀 Heart Disease Prediction Dashboard Based on Cardio Vascular Data")
st.caption("Multi-model performance metrics running evaluations on the John Smith Kaggle repository dataset.")

# Left Sidebar Control Panels
st.sidebar.header("📁 User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload evaluation slice ('test_data.csv')", type=["csv"])

st.sidebar.header("🤖 Choose Prediction Algorithm To Calculate:")
model_option = st.sidebar.selectbox(
    "",
    ("Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbor Classifier", "Naive Bayes Classifier", "Ensemble Random Forest"),
    label_visibility="collapsed"
)

# Mapping above drop down values to serialized object strings
model_mapping = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree Classifier": "decision_tree",
    "K-Nearest Neighbor Classifier": "knn",
    "Naive Bayes Classifier": "naive_bayes",
    "Ensemble Random Forest": "random_forest"
}

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        ground_truth_df = pd.read_csv("model/hidden_ground_truth.csv")
        y_true = ground_truth_df['target']
        
        st.success(f"Processing link established. Evaluated {input_data.shape} diagnostic rows.")

        # Display Metrices to user.
        ui_tab1, ui_tab2, ui_tab3 = st.tabs([
            f"📊 Performance Metrics Summary", 
            f"🔲 Analytical Confusion Matrix",
            f"📋 Raw Uploaded Data Matrix Preview"
        ])
        
        model_key = model_mapping[model_option]
        scaler = joblib.load("model/scaler.pkl")
        loaded_model = joblib.load(f"model/{model_key}_model.pkl")
        
        # transform and calculate
        X_scaled = scaler.transform(input_data)
        y_preds = loaded_model.predict(X_scaled)
        y_probs = loaded_model.predict_proba(X_scaled)[:, 1] if hasattr(loaded_model, "predict_proba") else y_preds
        
        with ui_tab1:
            st.subheader(f"✨ Statistical Scores Tracker ({model_option})")
            
            acc = accuracy_score(y_true, y_preds)
            auc = roc_auc_score(y_true, y_probs)
            prec = precision_score(y_true, y_preds, zero_division=0)
            rec = recall_score(y_true, y_preds, zero_division=0)
            f1 = f1_score(y_true, y_preds, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_preds)
            
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row1_col1.metric("Accuracy Score", f"{acc:.4f}")
            row1_col2.metric("ROC AUC Area Score", f"{auc:.4f}")
            row1_col3.metric("F1 Performance Vector", f"{f1:.4f}")
            
            row2_col1, row2_col2, row2_col3 = st.columns(3)
            row2_col1.metric("Precision Index Score", f"{prec:.4f}")
            row2_col2.metric("Recall Index (Sensitivity)", f"{rec:.4f}")
            row2_col3.metric("Matthews Correlation (MCC)", f"{mcc:.4f}")
            
            st.subheader("Console Output classification report")
            st.text_area("Classification Report Matrix Summary Details:", classification_report(y_true, y_preds), height=180)
            
        with ui_tab2:
            st.subheader(f"Confusion Matrix Heatmap Breakdown ({model_option})")
            cm = confusion_matrix(y_true, y_preds)
            fig, ax = plt.subplots(figsize=(4.8, 3.5))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='mako',
                xticklabels=['Healthy (0)', 'Disease Risk (1)'],
                yticklabels=['Healthy (0)', 'Disease Risk (1)'],
                cbar=False, ax=ax
            )
            plt.ylabel('Ground Truth Records')
            plt.xlabel('System Predicted Outcomes')
            plt.tight_layout()
            st.pyplot(fig)

        with ui_tab3:
            st.subheader(f"Extracted Test Observations Vectors ({model_option})")
            st.dataframe(input_data.head(55), width='stretch')
            
    except Exception as e:
        st.error(f"Exception Occoured: {str(e)}. Ensure uploaded files match original configurations.")
else:
    st.info("💡 Appropriate File is not uploaded. Use Upload button or Drag and drop your compiled 'test_data.csv' array into the file portal container.")