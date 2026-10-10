import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load trained model
MODEL_PATH = Path(__file__).parent / "loan_model.pkl"

st.title("🏦 Loan Approval Prediction")
st.write(
    "Enter the applicant details below to predict "
    "the loan approval status using Machine Learning."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found! Please run "
        "'python train_model.py' first."
    )
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
except Exception as error:
    st.error(f"Error loading model: {error}")
    st.stop()

# Applicant information form
st.subheader("Applicant Information")

with st.form("loan_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

    with col2:
        applicant_income = st.number_input(
            "Applicant Income (monthly)",
            min_value=0,
            value=5000,
            step=500
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income (monthly)",
            min_value=0,
            value=0,
            step=500
        )

        loan_amount = st.number_input(
            "Loan Amount (in thousands)",
            min_value=1,
            value=150,
            step=10
        )

        loan_term = st.selectbox(
            "Loan Term (months)",
            [120, 180, 240, 300, 360],
            index=4
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda value: (
                "Good / Available (1)"
                if value == 1.0
                else "Poor / Not available (0)"
            )
        )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

    submitted = st.form_submit_button(
        "Predict Loan Status",
        use_container_width=True
    )

# Prediction
if submitted:

    applicant = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    try:
        # Match the training feature order
        applicant = applicant[features]

        # Predict loan status
        prediction = model.predict(applicant)[0]

        st.divider()
        st.subheader("Prediction Result")

        # Display estimated approval probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(applicant)[0]
            classes = list(model.classes_)

            if "Y" in classes:
                approval_probability = float(
                    probabilities[classes.index("Y")]
                )

                st.metric(
                    "Estimated Approval Probability",
                    f"{approval_probability:.2%}"
                )

                st.progress(approval_probability)

        # Display result
        if prediction == "Y":
            st.success(
                "✅ The model predicts: Likely Loan Approval"
            )

        else:
            st.error(
                "❌ The model predicts: Likely Loan Rejection"
            )

        # Show applicant details
        with st.expander("View Applicant Details"):
            st.dataframe(
                applicant,
                use_container_width=True
            )

        st.warning(
            "This is an educational demonstration using "
            "synthetic data. Predictions are not based on "
            "real bank decisions and must not be used to "
            "make actual lending decisions."
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")