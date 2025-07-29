import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load model and encoders
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="Employee Salary Prediction", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stRadio > div {
        flex-direction: row;
    }
    .stSelectbox div[role='listbox'] {
        overflow-y: auto;
        max-height: 200px;
    }
    .css-1kyxreq {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Employee Salary Prediction App")

# Helper: preprocess input
def preprocess_input(df):
    df = df.copy()
    for col in label_encoders:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    if 'fnlwgt' in df.columns:
        df.drop('fnlwgt', axis=1, inplace=True)
    if 'income' in df.columns:
        df.drop('income', axis=1, inplace=True)
    scaled_data = scaler.transform(df)
    return scaled_data

option = st.radio("Choose Input Method:", ['🧍 Predict One', '📁 Upload CSV for Batch'])

if option == '🧍 Predict One':
    st.subheader("Enter Employee Details")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 17, 75, 30)
            workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
            education = st.selectbox("Education", label_encoders['education'].classes_)
            educational_num = st.slider("Education Number", 5, 16, 10)
            marital_status = st.selectbox("Marital Status", label_encoders['marital-status'].classes_)
            occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
        with col2:
            relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)
            race = st.selectbox("Race", label_encoders['race'].classes_)
            gender = st.selectbox("Gender", label_encoders['gender'].classes_)
            capital_gain = st.number_input("Capital Gain", min_value=0)
            capital_loss = st.number_input("Capital Loss", min_value=0)
            hours_per_week = st.slider("Hours per Week", 1, 100, 40)
            native_country = st.selectbox("Native Country", label_encoders['native-country'].classes_)

        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        input_dict = {
            "age": [age],
            "workclass": [workclass],
            "education": [education],
            "educational-num": [educational_num],
            "marital-status": [marital_status],
            "occupation": [occupation],
            "relationship": [relationship],
            "race": [race],
            "gender": [gender],
            "capital-gain": [capital_gain],
            "capital-loss": [capital_loss],
            "hours-per-week": [hours_per_week],
            "native-country": [native_country]
        }

        input_df = pd.DataFrame(input_dict)
        processed = preprocess_input(input_df)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0]
        result = label_encoders['income'].inverse_transform([prediction])[0]

        st.success(f"🔍 Predicted Salary Category: **{result}**")
        st.write(f"📊 Probability of >50K: **{probability[1]:.2f}**")

        st.subheader("🔎 Explanation with SHAP")
        shap_values = explainer.shap_values(processed)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0][0],
            base_values=explainer.expected_value[0],
            data=input_df.iloc[0],
            feature_names=input_df.columns), max_display=10)
        st.pyplot(fig)

elif option == '📁 Upload CSV for Batch':
    st.subheader("Upload CSV File with Employee Records")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("📄 Uploaded Data Preview:", df.head())

        try:
            processed = preprocess_input(df)
            predictions = model.predict(processed)
            probabilities = model.predict_proba(processed)[:, 1]
            df['Predicted Income'] = label_encoders['income'].inverse_transform(predictions)
            df['Probability >50K'] = probabilities
            st.success("✅ Predictions completed!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download Results CSV", data=csv, file_name="salary_predictions.csv", mime="text/csv")

            if 'income' in df.columns:
                y_true = df['income'].map(lambda x: label_encoders['income'].transform([x])[0])
                fpr, tpr, _ = roc_curve(y_true, probabilities)
                roc_auc = auc(fpr, tpr)

                st.subheader("📈 ROC Curve")
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in processing: {e}")
