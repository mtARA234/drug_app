
import streamlit as st
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBClassifier, XGBRegressor

# ===============================
# CONFIG
# ===============================
st.set_page_config(layout="wide")
st.title("🧪 AI Drug Discovery Platform for antidepressants")

# ===============================
# MODEL FOLDER (relative to repo)
# ===============================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ===============================
# FINGERPRINT + DESCRIPTORS
# ===============================
morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_descriptors(mol):
    from rdkit.Chem import Descriptors
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol)
    ]

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.array(morgan.GetFingerprint(mol))
    desc = np.array(get_descriptors(mol))
    return np.concatenate([fp, desc])

# ===============================
# LOAD HYBRID IC50 MODELS
# ===============================
targets = ["SERT","DAT","D2","D3","D4","5HT1A","5HT6","5HT7"]
hybrid_models = {}

for name in targets:
    try:
        reg = XGBRegressor()
        reg.load_model(os.path.join(MODEL_DIR, f"{name}_reg.json"))
        clf = XGBClassifier()
        clf.load_model(os.path.join(MODEL_DIR, f"{name}_clf.json"))
        hybrid_models[name] = (reg, clf)
    except Exception as e:
        print(f"Skipping {name}: {e}")

# ===============================
# LOAD TOXICITY & COMPATIBILITY MODELS
# ===============================
tox_model = XGBClassifier()
tox_model.load_model(os.path.join(MODEL_DIR,"tox_model.json"))

compat_model = XGBClassifier()
compat_model.load_model(os.path.join(MODEL_DIR,"compatibility_xgb.json"))

# ===============================
# EXCIPIENTS
# ===============================
excipients = {
    "Lactose": "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](CO)O[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O",
    "PEG": "OCCO",
    "PVP": "C=CC(=O)N1CCCC1",
    "HPMC": "COC1=CC=CC=C1O",
    "Ethanol": "CCO",
    "Sodium Benzoate": "C1=CC=C(C=C1)C(=O)[O-].[Na+]",
    "Sucrose": "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](CO)O[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O",
    "Glycerol": "C(C(CO)O)O",
    "Propylene Glycol": "CC(CO)O",
    "Mannitol": "C(C(C(C(C(CO)O)O)O)O)O",
    "Starch": "C(C1C(C(C(C(O1)O)O)O)O)O",  # simplified fragment
    "Citric Acid": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
    "Polysorbate 80": "CC(C)OC(=O)CCCCCCCCCCCCC",
    # Add more excipients here as needed
}

# ===============================
# PREDICTION FUNCTIONS
# ===============================
def predict_ic50(smiles):
    features = smiles_to_features(smiles)
    if features is None:
        return None
    features = features.reshape(1,-1)
    results = {}
    for name, (reg, clf) in hybrid_models.items():
        prob = clf.predict_proba(features)[0][1]
        if prob > 0.5:
            pic50 = reg.predict(features)[0]
            ic50 = 10**(-pic50)*1e9
            results[name] = {"Active": True,"Confidence": round(prob,2),"pIC50": round(pic50,2),"IC50_nM": round(ic50,2)}
        else:
            results[name] = {"Active": False,"Confidence": round(prob,2)}
    return results

def predict_toxicity(smiles):
    features = smiles_to_features(smiles)
    if features is None:
        return "Invalid"
    pred = tox_model.predict(features.reshape(1,-1))[0]
    return "High" if pred==1 else "Low"

def predict_compatibility(drug_smiles, excipient_smiles):
    features1 = smiles_to_features(drug_smiles)
    features2 = smiles_to_features(excipient_smiles)
    if features1 is None or features2 is None:
        return "Invalid",0
    features = np.concatenate([features1,features2]).reshape(1,-1)
    prob = compat_model.predict_proba(features)[0][1]
    return ("Compatible" if prob>0.5 else "Incompatible"), prob

def find_best_excipient(smiles):
    best, best_score = None, -1
    for name, s in excipients.items():
        label, prob = predict_compatibility(smiles,s)
        if prob>best_score:
            best_score, best = prob, name
    return best, best_score

# ===============================
# STREAMLIT UI
# ===============================
smiles = st.text_input("Enter Drug SMILES")
excipient = st.selectbox("Select Excipient", list(excipients.keys()))

if st.button("Run Prediction"):
    st.subheader("📊 Results")

    ic50_results = predict_ic50(smiles)
    tox = predict_toxicity(smiles)
    comp, prob = predict_compatibility(smiles, excipients[excipient])
    best, score = find_best_excipient(smiles)

    # IC50
    st.write("### IC50 Predictions")
    if ic50_results:
        for t,res in ic50_results.items():
            if res["Active"]:
                st.success(f"{t}: Active | pIC50={res['pIC50']} | IC50={res['IC50_nM']} nM | Confidence={res['Confidence']}")
            else:
                st.error(f"{t}: Inactive | Confidence={res['Confidence']}")

    # Toxicity
    st.write("### Toxicity")
    st.success(tox)

    # Compatibility
    st.write("### Excipient Compatibility")
    st.write(f"{comp} (Confidence: {prob:.2f})")

    # Best excipient
    st.write("### Suggested Best Excipient")
    st.success(f"{best} (Score: {score:.2f})")
    