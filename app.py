import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_data.csv"
MODEL_PATH = ROOT / "student_model.pkl"
METRICS_PATH = ROOT / "model_metrics.json"

FEATURES = [
    "request_type", "priority", "submission_channel",
    "assigned_staff_workload", "processing_days", "missing_documents"
]
CATEGORICAL = ["request_type", "priority", "submission_channel"]
NUMERIC = ["assigned_staff_workload", "processing_days", "missing_documents"]
TARGET = "sla_breach"

st.set_page_config(page_title="Student Service Request Monitoring", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

def validate_dataset(frame):
    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if frame.empty:
        return False, "The uploaded dataset has no rows."
    return True, "Dataset schema is valid."

def home():
    st.title("🎓 Student Service Request Monitoring")
    st.write("A decision-support prototype for monitoring student service requests and identifying possible SLA breaches.")
    st.warning("Demo limitation: the included dataset and model are synthetic examples. Predictions must be reviewed by authorized staff and must never automatically deny student services.")
    a,b,c = st.columns(3)
    data = pd.read_csv(DATA_PATH)
    a.metric("Sample requests", f"{len(data):,}")
    b.metric("SLA breach rate", f"{data[TARGET].mean():.1%}")
    c.metric("Features", len(FEATURES))
    st.markdown("Use the sidebar to explore the dataset, analyze request patterns, test a prediction, and review the system workflow.")

def dataset_page():
    st.title("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload a student service request CSV", type=["csv"])
    if uploaded:
        try:
            frame = pd.read_csv(uploaded)
            ok, message = validate_dataset(frame)
            (st.success if ok else st.error)(message)
            st.dataframe(frame.head(30), use_container_width=True)
            st.write(f"Rows: **{len(frame)}** · Columns: **{len(frame.columns)}**")
            st.subheader("Data quality summary")
            x,y = st.columns(2)
            x.metric("Missing cells", int(frame.isna().sum().sum()))
            y.metric("Duplicate rows", int(frame.duplicated().sum()))
            st.dataframe(pd.DataFrame({"Missing values":frame.isna().sum(), "Data type":frame.dtypes.astype(str)}), use_container_width=True)
            if not ok:
                st.info("Use the required schema shown in README.md.")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    else:
        st.caption("No upload yet. Showing the bundled synthetic demonstration data.")
        frame = pd.read_csv(DATA_PATH)
        st.dataframe(frame.head(20), use_container_width=True)

def eda_page():
    st.title("Exploratory Data Analysis")
    frame = pd.read_csv(DATA_PATH)
    st.subheader("Descriptive statistics")
    st.dataframe(frame.describe(include="all").T, use_container_width=True)
    st.subheader("Interactive charts")
    c1,c2 = st.columns(2)
    with c1:
        counts = frame[TARGET].map({0:"No breach",1:"SLA breach"}).value_counts().rename_axis("Outcome").reset_index(name="Requests")
        st.plotly_chart(px.pie(counts, names="Outcome", values="Requests", title="SLA outcome distribution"), use_container_width=True)
    with c2:
        by_type = frame.groupby("request_type", as_index=False)[TARGET].mean()
        st.plotly_chart(px.bar(by_type, x="request_type", y=TARGET, title="SLA breach rate by request type"), use_container_width=True)
    by_priority = frame.groupby("priority", as_index=False)[TARGET].mean()
    st.plotly_chart(px.bar(by_priority, x="priority", y=TARGET, title="SLA breach rate by priority"), use_container_width=True)
    st.caption("Charts describe the synthetic bundled dataset, not verified institutional performance.")

def prediction_page():
    st.title("Predictive Modeling")
    st.write("Enter request details. The model estimates whether the request may breach its SLA.")
    model = load_model()
    with st.form("prediction_form"):
        c1,c2,c3 = st.columns(3)
        request_type = c1.selectbox("Request type", ["Enrollment","Document Request","Scholarship","ID Replacement","Grade Concern","Other"])
        priority = c2.selectbox("Priority", ["Low","Normal","High","Urgent"], index=1)
        channel = c3.selectbox("Submission channel", ["Online","Walk-in","Email"])
        c4,c5,c6 = st.columns(3)
        workload = c4.number_input("Assigned staff workload (open requests)", min_value=0, max_value=200, value=20)
        days = c5.number_input("Processing days so far", min_value=0.0, max_value=90.0, value=2.0, step=0.5)
        missing_docs = c6.selectbox("Missing documents?", ["No","Yes"])
        submitted = st.form_submit_button("Estimate SLA risk")
    if submitted:
        row = pd.DataFrame([{
            "request_type":request_type, "priority":priority, "submission_channel":channel,
            "assigned_staff_workload":workload, "processing_days":days,
            "missing_documents":1 if missing_docs=="Yes" else 0
        }], columns=FEATURES)
        prediction = int(model.predict(row)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row)[0][1])
        st.subheader("Prediction result")
        st.metric("Estimated outcome", "Possible SLA breach" if prediction else "No breach predicted")
        if probability is not None:
            st.metric("Estimated breach probability", f"{probability:.1%}")
        if prediction:
            st.warning("Suggested human follow-up: authorized staff should review the request, check missing documents, workload, and applicable service deadlines.")
        else:
            st.info("Suggested human follow-up: continue normal monitoring and verify the request against service policy.")
        st.caption("This is a model estimate, not a final decision or guarantee. A staff member must review the case.")

def dashboard_page():
    st.title("Decision-Support Dashboard")
    frame = pd.read_csv(DATA_PATH)
    total = len(frame)
    breaches = int(frame[TARGET].sum())
    k1,k2,k3 = st.columns(3)
    k1.metric("Total requests", total)
    k2.metric("Potential SLA breaches", breaches)
    k3.metric("Observed breach share", f"{breaches/max(total,1):.1%}")
    st.subheader("Review queue (demonstration records)")
    queue = frame[frame[TARGET] == 1].copy().sort_values(["priority","processing_days"], ascending=[True,False])
    st.dataframe(queue.head(25), use_container_width=True)
    st.info("The review queue is based on synthetic demonstration labels. In production, authorized staff should verify records and document final decisions.")

def system_design():
    st.title("System Design and SAD Integration")
    st.subheader("Actors")
    st.markdown("- **Student Services Staff:** upload and review request records, inspect risk estimates, and record follow-up.\n- **System Administrator:** maintain application access, model files, and configuration.\n- **Student:** submits service requests through the institution's authorized channel (outside this prototype).")
    st.subheader("Functional requirements")
    for item in [
        "The system shall accept CSV uploads and validate required columns.",
        "The system shall summarize missing values and duplicate records.",
        "The system shall display descriptive statistics and interactive charts.",
        "The system shall load a saved model and estimate SLA-breach risk for a new request.",
        "The system shall present a review queue and human follow-up guidance."
    ]: st.markdown(f"- {item}")
    st.subheader("Non-functional requirements")
    for item in ["Usability: provide clear navigation, labels, and validation messages.",
                 "Security: use authorized access and avoid publishing identifiable student records.",
                 "Reliability: handle invalid files and show model limitations."]: st.markdown(f"- {item}")
    st.subheader("Decision workflow")
    st.code("CSV upload → schema validation → data analysis → saved model inference → risk flag → authorized staff review → documented human decision", language="text")
    st.caption("Predictions support staff decisions; they do not automatically reject requests or deny student services.")

menu = st.sidebar.selectbox("Module", ["Home","Dataset","Data Analysis","Prediction","Dashboard","System Design"])
if menu == "Home": home()
elif menu == "Dataset": dataset_page()
elif menu == "Data Analysis": eda_page()
elif menu == "Prediction": prediction_page()
elif menu == "Dashboard": dashboard_page()
else: system_design()
