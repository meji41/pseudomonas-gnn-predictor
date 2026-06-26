import streamlit as st
import torch

from predict import load_model, predict_smiles

st.set_page_config(page_title="Pseudomonas Antimicrobial Predictor")

st.title("Pseudomonas Antimicrobial Predictor")

st.write(
    "This tool uses a trained GCN model to predict antimicrobial activity from molecular SMILES strings."
)

@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("best_model.pt", device=device)
    return model, device

model, device = get_model()

smiles = st.text_input(
    "Enter SMILES",
    value="CC(=O)Oc1ccccc1C(=O)O"
)

if st.button("Predict"):
    try:
        result = predict_smiles(smiles, model, device=device)

        st.success(f"Prediction: {result['prediction']}")
        st.write(f"Active Probability: {result['active_probability']}")
        st.write(f"Inactive Probability: {result['inactive_probability']}")

    except ValueError:
        st.error("Invalid SMILES input. Please enter a valid molecular SMILES string.")

    except Exception:
        st.error("Prediction failed. Please check the input and try again.")