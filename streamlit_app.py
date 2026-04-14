import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load('final_churn_model.pkl')

# Page config
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# Dark theme
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stApp {background-color: #0E1117;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Customer Churn Prediction Dashboard")
st.markdown("### End-to-End ML System with Explainability")

# Tabs
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📂 Bulk Prediction"])

# =========================
# 🔮 SINGLE PREDICTION
# =========================
with tab1:

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
        Partner = st.selectbox("Partner", ["Yes","No"])
        Dependents = st.selectbox("Dependents", ["Yes","No"])

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes","No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])

    with col3:
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"
        ])

    col4, col5 = st.columns(2)

    with col4:
        tenure = st.slider("Tenure", 0, 72, 12)
        MonthlyCharges = st.slider("Monthly Charges", 0, 150, 70)

    with col5:
        TotalCharges = st.slider("Total Charges", 0, 10000, 2000)
        OnlineBackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    input_df = pd.DataFrame({
        'gender':[gender],
        'SeniorCitizen':[SeniorCitizen],
        'Partner':[Partner],
        'Dependents':[Dependents],
        'tenure':[tenure],
        'PhoneService':[PhoneService],
        'MultipleLines':[MultipleLines],
        'InternetService':[InternetService],
        'OnlineSecurity':[OnlineSecurity],
        'OnlineBackup':[OnlineBackup],
        'DeviceProtection':[DeviceProtection],
        'TechSupport':[TechSupport],
        'StreamingTV':[StreamingTV],
        'StreamingMovies':[StreamingMovies],
        'Contract':[Contract],
        'PaperlessBilling':[PaperlessBilling],
        'PaymentMethod':[PaymentMethod],
        'MonthlyCharges':[MonthlyCharges],
        'TotalCharges':[TotalCharges]
    })

    if st.button("🚀 Predict"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️ Customer likely to CHURN ({prob:.2%})")
        else:
            st.success(f"✅ Customer will STAY ({prob:.2%})")

        st.progress(int(prob * 100))

# =========================
# 📊 SHAP EXPLANATION (FINAL PRO VERSION)
# =========================

st.subheader("📊 SHAP Explanation")

try:
    # -------------------------
    # 🔹 INPUT TRANSFORM
    # -------------------------
    X_input = model.named_steps['prep'].transform(input_df)
    feature_names = model.named_steps['prep'].get_feature_names_out()

    X_input_df = pd.DataFrame(X_input, columns=feature_names)

    # -------------------------
    # 🔹 BACKGROUND DATA
    # -------------------------
    data = pd.read_csv("Telco-Customer-Churn.csv")

    data = data.drop("customerID", axis=1)
    data['TotalCharges'] = data['TotalCharges'].replace(" ", "0")
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(0)
    #data['TotalCharges'].fillna(0, inplace=True)
    data['Churn'] = data['Churn'].map({'Yes':1,'No':0})

    X_bg = data.drop("Churn", axis=1).sample(100, random_state=42)

    X_bg_transformed = model.named_steps['prep'].transform(X_bg)
    X_bg_df = pd.DataFrame(X_bg_transformed, columns=feature_names)

    # -------------------------
    # 🔹 SHAP EXPLAINER
    # -------------------------
    explainer = shap.LinearExplainer(
        model.named_steps['model'],
        X_bg_df
    )

    shap_values = explainer(X_input_df)

    # -------------------------
    # 🔍 1. WATERFALL
    # -------------------------
    st.markdown("### 🔍 Individual Explanation")

    fig1, ax1 = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig1)

    # -------------------------
    # 📋 2. TOP 10 FEATURE TABLE
    # -------------------------
    st.markdown("### 📋 Top 10 Feature Contributions")

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0]
    })

    contrib_df = contrib_df[abs(contrib_df["SHAP Value"]) > 1e-4]
    contrib_df = contrib_df.sort_values(by="SHAP Value", key=abs, ascending=False).head(10)

    # 🔥 Color coding
    def color_shap(val):
        if val > 0:
            return 'color: green; font-weight: bold'
        elif val < 0:
            return 'color: red; font-weight: bold'
        else:
            return ''
    #st.dataframe(contrib_df.style.applymap(color_shap, subset=["SHAP Value"]))
    st.dataframe(contrib_df.style.map(color_shap, subset=["SHAP Value"]))

    # -------------------------
    # 🌍 3. GLOBAL SUMMARY
    # -------------------------
    st.markdown("### 🌍 Global Feature Importance")

    shap_values_bg = explainer(X_bg_df)

    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values_bg, X_bg_df, show=False)
    st.pyplot(fig2)

    # -------------------------
    # 📥 4. DOWNLOAD REPORT
    # -------------------------
    st.markdown("### 📥 Download SHAP Report")

    download_df = contrib_df.copy()
    download_df["Impact"] = download_df["SHAP Value"].apply(
        lambda x: "Increase Churn" if x > 0 else "Decrease Churn"
    )

    csv = download_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📄 Download SHAP Report",
        data=csv,
        file_name="shap_report.csv",
        mime="text/csv"
    )

except Exception as e:
    st.warning(f"SHAP error: {str(e)}")

# =========================
# 📂 BULK PREDICTION
# =========================
with tab2:

    st.subheader("📂 Upload CSV for Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.write("Preview Data:", df.head())

        # =========================
        # 🔥 VALIDATION CHECK
        # =========================
        required_cols = model.feature_names_in_

        missing_cols = set(required_cols) - set(df.columns)

        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()

        # =========================
        # 🔥 PREPROCESSING
        # =========================

        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        # Fix TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].replace(" ", "0")
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(0)
            #df['TotalCharges'].fillna(0, inplace=True)

        # Fix SeniorCitizen
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

        # Strip spaces
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()

        # =========================
        # ✅ PREDICTION
        # =========================

        try:
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:,1]

            df['Prediction'] = predictions
            df['Churn_Probability'] = probabilities

            st.success("✅ Prediction Successful!")
            st.write(df.head())

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "predictions.csv")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")