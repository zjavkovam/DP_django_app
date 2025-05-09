import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import joblib
from collections import Counter
import re
import numpy as np
import sklearn
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp
import json

# Load the original feature names (used during training)
FEATURE_COLUMNS = [desc[0] for desc in Descriptors.descList]  # Get descriptor names

XGB_PATH = os.path.join(os.path.dirname(__file__), "models", "xgb_model.pkl")
RF_PATH = os.path.join(os.path.dirname(__file__), "models", "rf_model.pkl")
KNN_PATH = os.path.join(os.path.dirname(__file__), "models", "knn_model.pkl")
SVM_PATH = os.path.join(os.path.dirname(__file__), "models", "svm_model.pkl")
DT_PATH = os.path.join(os.path.dirname(__file__), "models", "dt_model.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "models", "meta_model.pkl")

feature_names = os.path.join(os.path.dirname(__file__), "models", "feature_sets.json")

important_hyperparams_dict = {
    "XGBoost": [
        "n_estimators", "learning_rate", "max_depth", "subsample",
        "colsample_bytree", "gamma", "min_child_weight", "lambda",
        "alpha", "objective"
    ],
    "Random Forest": [
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "bootstrap", "criterion", "oob_score",
        "random_state", "class_weight"
    ],
    "Decision Tree": [
        "criterion", "splitter", "max_depth", "min_samples_split",
        "min_samples_leaf", "max_features", "random_state"
    ],
    "KNN": [
        "n_neighbors", "weights", "algorithm", "leaf_size",
        "p", "metric"
    ],
    "SVM": [
        "C", "kernel", "degree", "gamma", "coef0",
        "shrinking", "probability", "tol"
    ],
    "Neural Network": [
        "hidden_layer_sizes", "activation", "solver", "alpha",
        "batch_size", "learning_rate", "learning_rate_init",
        "max_iter", "momentum", "nesterovs_momentum"
    ]
}



def get_model_parameters(model):
    # Get the model parameters (weights and biases)
    model_parameters = {}
    
    for name, param in model.named_parameters():
        model_parameters[name] = param.detach().cpu().numpy()  # Detach tensor and convert to numpy for easy access

    return model_parameters



def detect_input_type(input_str):
    cas_pattern = r"^\d{2,7}-\d{2}-\d{1}$"
    smiles_pattern = r"[A-Za-z0-9@+\-=#\$%\(\)\[\]\\/]"
    
    if re.match(cas_pattern, input_str):
        return "CAS Number"
    elif re.search(smiles_pattern, input_str) and "-" not in input_str:
        return "SMILES"
    else:
        return "Unknown format"


def process_input(input):
    if detect_input_type(input) == "CAS Number":
        input = cirpy.resolve(input, 'smiles')
    elif detect_input_type(input) == "Unknown format":
        raise ValueError("Invalid Input")
    
    mol = Chem.MolFromSmiles(input)
    if mol is None:
        raise ValueError("Invalid Input")

    # Compute descriptors and create a DataFrame with one row
    descriptor_values = [func(mol) for _, func in Descriptors.descList]
    feature_df = pd.DataFrame([descriptor_values], columns=FEATURE_COLUMNS)

    return [input, feature_df]  # Returns a DataFrame


def make_prediction(features, selected_model):
    model_parameters = {}
    shap_plot = ""


    # Load models
    xgb_model = joblib.load(XGB_PATH)
    dt_model = joblib.load(DT_PATH)
    rf_model = joblib.load(RF_PATH)
    svm_model = joblib.load(SVM_PATH)
    knn_model = joblib.load(KNN_PATH)
    meta_model = joblib.load(META_PATH)

    # Load feature lists
    with open(feature_names) as f:
        feature_dict = json.load(f)

    missing = [f for f in feature_dict['xgb_features'] if f not in features.columns]
    if missing:
        raise ValueError(f"Missing features for XGB model: {missing}")


    # Prepare inputs
    stacking_inputs = np.column_stack([
        xgb_model.predict(features[feature_dict['xgb_features']]),
        dt_model.predict(features[feature_dict['dt_features']]),
        rf_model.predict(features[feature_dict['rf_features']]),
        svm_model.predict(features[feature_dict['svm_features']]),
        knn_model.predict(features[feature_dict['knn_features']])
    ])

    # Predict with meta-model
    final_predictions = meta_model.predict(stacking_inputs)
    final_predictions = knn_model.predict(features[feature_dict['knn_features']])

    """
        important_hyperparams = important_hyperparams_dict[selected_model]
        # Get model parameters
        model_parameters = {param: model.get_params()[param] for param in important_hyperparams if param in model.get_params()}
        #Convert parameters to the desired format
        model_parameters = [{"name": key, "value": str(value)} for key, value in model_parameters.items()]
    """

    return {
        "prediction": final_predictions[0],
        "trust_score": 85,
        "roc_curve": f"/static/{"stacking_aucroc.png"}",
        "shap_plot": f"/static/{"ensambleshap.png"}" ,
        "model_parameters": model_parameters,
    }



def sanitize_filename(smiles):
    """Convert SMILES into a safe filename by removing special characters."""
    return re.sub(r'[^\w\-]', '_', smiles)  # Replace non-alphanumeric characters with "_"

def get_molecule_img(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  

    Chem.rdDepictor.Compute2DCoords(mol)

    # Define the base directory (project root)
    base_dir = os.path.dirname(os.path.dirname(__file__))  
    img_dir = os.path.join(base_dir, "static")  
    os.makedirs(img_dir, exist_ok=True)  # Ensure the directory exists

        # Use a cleaned version of the SMILES string as the filename
    img_filename = f"{sanitize_filename(smiles)}.png"  
    img_path = os.path.join(img_dir, img_filename)

    Draw.MolToFile(mol, img_path, size=(500, 300))

    return f"/static/{img_filename}"  # Return the relative path for Django

import pubchempy as pcp

def get_chemical_info(smiles):
    # Search PubChem for compounds based on the SMILES string
    compounds = pcp.get_compounds(smiles, 'smiles')
    
    if not compounds:
        raise ValueError("No compounds found for the given SMILES.")
    
    compound = compounds[0]  # Select the first compound result

    # Extracting chemical information
    chemical_name = compound.iupac_name if compound.iupac_name else "N/A"
    molecular_formula = compound.molecular_formula
    synonyms = compound.synonyms if compound.synonyms else []

    # Finding CAS Number in synonyms (not always available)
    cas_number = "Not Available"
    for synonym in synonyms:
        if "-" in synonym and synonym[0].isdigit():  # CAS numbers are numeric with dashes
            cas_number = synonym
            break  # Take the first CAS number found

    # Return dictionary with results
    chemical_info = {
        'CAS Number': cas_number,
        'Chemical Name': chemical_name,
        'Molecular Formula': molecular_formula
    }

    # Print results
    print(f"CAS Number: {chemical_info['CAS Number']}")
    print(f"Chemical Name: {chemical_info['Chemical Name']}")
    print(f"Molecular Formula: {chemical_info['Molecular Formula']}")

    return chemical_info


