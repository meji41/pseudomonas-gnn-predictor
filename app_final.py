import shutil
import tempfile
from pathlib import Path

import streamlit as st

from predict_morgan_gnn import (
    create_prediction_graphs,
    load_model,
    predict_from_graphs,
)
st.set_page_config(
    page_title="Pseudomonas Antimicrobial Predictor",
    page_icon="🧬",
    layout="centered",
)

PROJECT_DIR = Path(__file__).parent
BACKEND_DIR = PROJECT_DIR

MODEL_PATH = BACKEND_DIR / "final_model.pt"
TRAIN_SCRIPT = BACKEND_DIR / "train_baseline_gnns.py"
FPPOOL_PATH = PROJECT_DIR / "FPPool"
PREPARE_SCRIPT = PROJECT_DIR / "prepare_fppool_graphs.py"
FIXED_MORGAN_SCRIPT = PROJECT_DIR / "make_fixed_morgan_graphs.py"

@st.cache_resource
def get_cached_model():
    model, device, model_info = load_model(
        model_path=MODEL_PATH,
        train_script_path=TRAIN_SCRIPT,
    )
    return model, device, model_info

def predict_with_final_model(smiles: str):
    model, device, model_info = get_cached_model()

    temp_dir = tempfile.mkdtemp(prefix="streamlit_prediction_")

    try:
        graphs_path, skipped_path = create_prediction_graphs(
            smiles_list=[smiles],
            prepare_script=PREPARE_SCRIPT,
            fixed_morgan_script=FIXED_MORGAN_SCRIPT,
            fppool_path=FPPOOL_PATH,
            radius=2,
            n_bits=2048,
            temp_dir=temp_dir,
        )

        results = predict_from_graphs(
            model=model,
            device=device,
            graphs_path=graphs_path,
            threshold=0.8,
            batch_size=64,
        )

        if len(results) == 0:
            raise ValueError("No prediction result was returned.")

        return results.iloc[0].to_dict()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


st.sidebar.title("🧬 Living In Silico")
st.sidebar.write("**Project:** Pseudomonas antimicrobial activity prediction")
st.sidebar.write("**Model:** Final GIN + MorganFP")
st.sidebar.write("**Threshold:** 0.80")
st.sidebar.write("**Input:** Molecular SMILES")
st.sidebar.write("**Output:** Prediction, probability, confidence")

st.title("Pseudomonas Antimicrobial Activity Predictor")

st.write(
    "Enter a molecular SMILES string to predict antimicrobial activity using "
    "Zhangbin's finalized GIN + MorganFP model."
)

st.subheader("SMILES Input")

examples = {
    "Zhangbin example": "CC1=CC(=CC=C1)C2=NN3C(=C2)N=C(C=C3NCCCN4C=CN=C4)C",
    "Ethanol": "CCO",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Benzene": "c1ccccc1",
    "Custom": "",
}

choice = st.selectbox("Choose an example or enter your own:", list(examples.keys()))

smiles = st.text_input(
    "Enter SMILES",
    value=examples[choice],
    placeholder="Example: CCO",
)

if st.button("Predict"):
    if not smiles.strip():
        st.warning("Please enter a SMILES string.")
    else:
        with st.spinner("Running final GIN + MorganFP prediction..."):
            try:
                result = predict_with_final_model(smiles)

                prediction = result["prediction"]
                probability_active = float(result["probability_active"])
                threshold = float(result["threshold"])
                confidence = float(result["confidence"])
                canonical_smiles = result["canonical_smiles"]

                st.divider()
                st.subheader("Prediction Results")

                if prediction == "Active":
                    st.success("Prediction: ACTIVE")
                else:
                    st.error("Prediction: INACTIVE")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                with col2:
                    st.metric("Threshold used", f"{threshold:.2f}")

                st.write("**Canonical SMILES**")
                st.code(canonical_smiles)

                st.write("**Probability Active**")
                st.progress(probability_active)
                st.write(f"{probability_active * 100:.2f}%")

                st.caption(
                    "These values represent model confidence scores from the trained GIN + MorganFP model, "
                    "not experimentally validated biological probabilities."
                )

            except Exception as e:
                error_message = str(e)

                if (
                    "SMILES Parse Error" in error_message
                    or "Failed parsing SMILES" in error_message
                    or "No valid graphs were created" in error_message
                    or "skipped_fppool_errors.csv" in error_message
                ):
                    st.error(
                        "The entered SMILES is not a valid molecular structure. Please check it and try again."
                    )
                else:
                    st.error("Prediction failed.")
                    st.exception(e)