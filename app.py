"""
Streamlit dashboard for the CICIDS2017 Network Intrusion Detection project.

Run with:
    streamlit run app.py

Requires that `outputs/models/` contains the artifacts produced by
`notebooks/02_modeling_and_evaluation.ipynb`:
    - binary_random_forest.joblib
    - binary_scaler.joblib
    - multiclass_random_forest.joblib
    - multiclass_scaler.joblib
    - multiclass_label_encoder.joblib
    - feature_names.txt

And that `data/processed/cicids2017_clean.parquet` exists (from notebook 01).
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

MODEL_DIR = Path(__file__).resolve().parent / "outputs" / "models"
DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "cicids2017_clean.parquet"

st.set_page_config(page_title="CICIDS2017 — Intrusion Detection", layout="wide")
st.title("Network Intrusion Detection — CICIDS2017")
st.caption(
    "Random Forest classifiers trained on CICIDS2017 flow features to "
    "detect malicious network traffic and identify the attack category."
)


@st.cache_resource
def load_artifacts():
    binary_model = joblib.load(MODEL_DIR / "binary_random_forest.joblib")
    binary_scaler = joblib.load(MODEL_DIR / "binary_scaler.joblib")
    mc_model = joblib.load(MODEL_DIR / "multiclass_random_forest.joblib")
    mc_scaler = joblib.load(MODEL_DIR / "multiclass_scaler.joblib")
    mc_encoder = joblib.load(MODEL_DIR / "multiclass_label_encoder.joblib")
    with open(MODEL_DIR / "feature_names.txt") as f:
        feature_names = f.read().splitlines()
    return binary_model, binary_scaler, mc_model, mc_scaler, mc_encoder, feature_names


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


missing = []
if not DATA_PATH.exists():
    missing.append(str(DATA_PATH))
required_models = [
    "binary_random_forest.joblib", "binary_scaler.joblib",
    "multiclass_random_forest.joblib", "multiclass_scaler.joblib",
    "multiclass_label_encoder.joblib", "feature_names.txt",
]
for m in required_models:
    if not (MODEL_DIR / m).exists():
        missing.append(str(MODEL_DIR / m))

if missing:
    st.warning(
        "Required artifacts not found. Run `notebooks/01_eda_and_preprocessing.ipynb` "
        "and `notebooks/02_modeling_and_evaluation.ipynb` first to generate them.\n\n"
        "Missing:\n" + "\n".join(f"- {m}" for m in missing)
    )
    st.stop()

binary_model, binary_scaler, mc_model, mc_scaler, mc_encoder, feature_names = load_artifacts()
df = load_data()

tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Random Sample Inspector", "Manual Prediction"])

with tab1:
    st.subheader("Class distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Binary (BENIGN vs ATTACK)**")
        st.bar_chart(df["label_binary"].value_counts())
    with col2:
        st.write("**Multiclass (attack category)**")
        st.bar_chart(df["label_multiclass"].value_counts())

    st.subheader("Dataset summary")
    st.write(f"Total flows: {len(df):,}")
    st.write(f"Features: {len(feature_names)}")
    st.dataframe(df[feature_names].describe().T, use_container_width=True)

with tab2:
    st.subheader("Pick a random flow and see what the models predict")
    n = st.slider("Number of random flows", 1, 20, 5)

    if st.button("Sample new flows"):
        st.session_state["sample_idx"] = df.sample(n).index.tolist()

    if "sample_idx" not in st.session_state:
        st.session_state["sample_idx"] = df.sample(n).index.tolist()

    sample = df.loc[st.session_state["sample_idx"]]
    X_sample = sample[feature_names]

    X_bin_scaled = pd.DataFrame(
        binary_scaler.transform(X_sample), columns=feature_names, index=X_sample.index
    )
    X_mc_scaled = pd.DataFrame(
        mc_scaler.transform(X_sample), columns=feature_names, index=X_sample.index
    )

    bin_pred = binary_model.predict(X_bin_scaled)
    bin_proba = binary_model.predict_proba(X_bin_scaled)
    # The saved multiclass Random Forest was trained directly on string
    # labels (see notebook 02, section B.1), so its predictions are
    # already human-readable -- no need for mc_encoder here.
    mc_pred = mc_model.predict(X_mc_scaled)

    display_df = pd.DataFrame({
        "true_label": sample["Label"].values,
        "predicted_binary": bin_pred,
        "binary_attack_confidence": bin_proba[:, list(binary_model.classes_).index("ATTACK")].round(3),
        "predicted_attack_type": mc_pred,
    }, index=sample.index)

    st.dataframe(display_df, use_container_width=True)

with tab3:
    st.subheader("Manually adjust feature values and predict")
    st.caption(
        "Sliders are pre-filled with a random row's values. Adjust a few "
        "and see how the prediction changes."
    )

    if "manual_row" not in st.session_state:
        st.session_state["manual_row"] = df.sample(1)[feature_names].iloc[0]

    if st.button("Load a new random row"):
        st.session_state["manual_row"] = df.sample(1)[feature_names].iloc[0]

    base_row = st.session_state["manual_row"]

    # Only expose a curated subset of features as sliders -- 77 sliders
    # would be unusable. The rest are held fixed at the sampled row's values.
    interesting_features = [
        f for f in [
            "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count",
            "ACK Flag Count", "Destination Port",
        ] if f in feature_names
    ]

    cols = st.columns(2)
    edited = base_row.copy()
    for i, feat in enumerate(interesting_features):
        col = cols[i % 2]
        val = float(base_row[feat])
        lo, hi = float(df[feat].quantile(0.01)), float(df[feat].quantile(0.99))
        if lo == hi:
            lo, hi = val - 1, val + 1
        edited[feat] = col.slider(feat, min_value=lo, max_value=hi, value=val)

    X_manual = pd.DataFrame([edited])[feature_names]

    X_manual_bin = pd.DataFrame(
        binary_scaler.transform(X_manual), columns=feature_names
    )
    X_manual_mc = pd.DataFrame(
        mc_scaler.transform(X_manual), columns=feature_names
    )

    bin_pred = binary_model.predict(X_manual_bin)[0]
    bin_proba = binary_model.predict_proba(X_manual_bin)[0]
    mc_pred = mc_model.predict(X_manual_mc)[0]

    attack_conf = bin_proba[list(binary_model.classes_).index("ATTACK")]

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Binary prediction", bin_pred)
    c2.metric("Attack confidence", f"{attack_conf:.1%}")
    c3.metric("Predicted attack type", mc_pred)
