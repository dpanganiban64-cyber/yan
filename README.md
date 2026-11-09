# Student Service Request Monitoring (Section B)

A Streamlit decision-support prototype that estimates possible Service-Level Agreement (SLA) breaches in student service requests.

## Important note about the supplied files
The accompanying laboratory handout says to reuse the **actual target, features, dataset, trained model, and findings from your midterm work**. Those original midterm files were not included with the uploaded handout. Therefore, this ZIP contains a runnable **synthetic demonstration dataset and model** so you can test the application. Replace them with your authorized midterm data/model before submission. Do not present the synthetic metrics as your actual midterm results.

## Files
- `train_model.py` — trains and evaluates a scikit-learn pipeline.
- `app.py` — Streamlit app with Home, Dataset, Data Analysis, Prediction, Dashboard, and System Design modules.
- `requirements.txt` — Python dependencies.
- `student_data.csv` — synthetic demonstration data.
- `student_model.pkl` — saved preprocessing + Random Forest pipeline.
- `model_metrics.json` — evaluation metrics from the bundled synthetic data.
- `.gitignore` — excludes local environments and sensitive/local files.

## Dataset schema
Target: `sla_breach` (0 = no breach, 1 = SLA breach)

Features:
- `request_type` (categorical)
- `priority` (categorical)
- `submission_channel` (categorical)
- `assigned_staff_workload` (numeric)
- `processing_days` (numeric)
- `missing_documents` (0 = no, 1 = yes)

The demo target is synthetically generated for demonstration. It is not evidence of real institutional SLA performance.

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Train the model (optional; the ZIP already includes a trained model):

   ```bash
   python train_model.py
   ```

6. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

## Replace with your midterm output
1. Back up the included synthetic files.
2. Replace `student_data.csv` with your authorized midterm dataset.
3. Update `TARGET`, `CATEGORICAL`, `NUMERIC`, and `FEATURES` in `train_model.py` to exactly match your midterm target and features.
4. Update the prediction input widgets in `app.py` to collect those exact feature names and types.
5. Run `python train_model.py` to regenerate `student_model.pkl` and `model_metrics.json`.
6. Test predictions against the same cases used in your midterm notebook. Check for target leakage and ensure test labels/future-only fields are not used as inference inputs.

## Deploy to Streamlit Community Cloud
Push the project to GitHub, then create a Streamlit app using `app.py` as the entry point. Do not commit passwords or sensitive student records. Use synthetic or properly authorized data for public deployment.

## Human review and limitations
Predictions are decision-support outputs only. Authorized staff must verify records, consider context, and record the final decision. The model must never automatically reject a request or deny student services.
