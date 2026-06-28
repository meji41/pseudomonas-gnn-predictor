import json
import subprocess
import sys
from pathlib import Path
import tempfile
import streamlit as st

st.set_page_config(
    page_title="Pseudomonas Antimicrobial Predictor",
    page_icon="🧬",
    layout="centered",
)

PROJECT_DIR = Path(__file__).parent
BACKEND_DIR = PROJECT_DIR
PREDICT_SCRIPT = BACKEND_DIR / "predict_morgan_gnn.py"
MODEL_PATH = BACKEND_DIR / "final_model.pt"
FPPOOL_PATH = PROJECT_DIR / "FPPool"
PREPARE_SCRIPT = PROJECT_DIR / "prepare_fppool_graphs.py"
OUTPUT_DIR = BACKEND_DIR / "prediction_results"


def predict_with_final_model(smiles: str):
    with tempfile.TemporaryDirectory() as temp_output_dir, tempfile.TemporaryDirectory() as temp_graph_dir:
        command = [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--model-path",
            str(MODEL_PATH),
            "--smiles",
            smiles,
            "--fppool-path",
            str(FPPOOL_PATH),
            "--prepare-script",
            str(PREPARE_SCRIPT),
            "--output-dir",
            temp_output_dir,
            "--temp-dir",
            temp_graph_dir,
            "--save-predictions",
        ]

        result = subprocess.run(
            command,
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        json_path = Path(temp_output_dir) / "smiles_predictions.json"

        with open(json_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        return predictions[0]


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