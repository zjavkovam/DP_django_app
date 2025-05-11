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
from sklearn.preprocessing import MinMaxScaler
import cirpy

# Load the original feature names (used during training)
FEATURE_COLUMNS = [desc[0] for desc in Descriptors.descList]  # Get descriptor names

XGB_PATH = os.path.join(os.path.dirname(__file__), "models", "xgb_model.pkl")
RF_PATH = os.path.join(os.path.dirname(__file__), "models", "rf_model.pkl")
KNN_PATH = os.path.join(os.path.dirname(__file__), "models", "knn_model.pkl")
SVM_PATH = os.path.join(os.path.dirname(__file__), "models", "svm_model.pkl")
DT_PATH = os.path.join(os.path.dirname(__file__), "models", "dt_model.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "models", "meta_model.pkl")
VT_PATH = os.path.join(os.path.dirname(__file__), "models", "soft_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")


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
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    descriptor_values = [func(mol) for _, func in Descriptors.descList]
    feature_df = pd.DataFrame([descriptor_values], columns=descriptor_names)

    columns_to_filter = ['MaxAbsEStateIndex',
 'MaxEStateIndex',
 'MinAbsEStateIndex',
 'MinEStateIndex',
 'qed',
 'SPS',
 'MolWt',
 'HeavyAtomMolWt',
 'ExactMolWt',
 'NumValenceElectrons',
 'NumRadicalElectrons',
 'MaxPartialCharge',
 'MinPartialCharge',
 'MaxAbsPartialCharge',
 'MinAbsPartialCharge',
 'FpDensityMorgan1',
 'FpDensityMorgan2',
 'FpDensityMorgan3',
 'AvgIpc',
 'BalabanJ',
 'BertzCT',
 'Chi0',
 'Chi0n',
 'Chi0v',
 'Chi1',
 'Chi1n',
 'Chi1v',
 'Chi2n',
 'Chi2v',
 'Chi3n',
 'Chi3v',
 'Chi4n',
 'Chi4v',
 'HallKierAlpha',
 'Ipc',
 'Kappa1',
 'Kappa2',
 'Kappa3',
 'LabuteASA',
 'PEOE_VSA1',
 'PEOE_VSA10',
 'PEOE_VSA11',
 'PEOE_VSA12',
 'PEOE_VSA13',
 'PEOE_VSA14',
 'PEOE_VSA2',
 'PEOE_VSA3',
 'PEOE_VSA4',
 'PEOE_VSA5',
 'PEOE_VSA6',
 'PEOE_VSA7',
 'PEOE_VSA8',
 'PEOE_VSA9',
 'SMR_VSA1',
 'SMR_VSA10',
 'SMR_VSA2',
 'SMR_VSA3',
 'SMR_VSA4',
 'SMR_VSA5',
 'SMR_VSA6',
 'SMR_VSA7',
 'SMR_VSA9',
 'SlogP_VSA1',
 'SlogP_VSA10',
 'SlogP_VSA11',
 'SlogP_VSA12',
 'SlogP_VSA2',
 'SlogP_VSA3',
 'SlogP_VSA4',
 'SlogP_VSA5',
 'SlogP_VSA6',
 'SlogP_VSA7',
 'SlogP_VSA8',
 'TPSA',
 'EState_VSA1',
 'EState_VSA10',
 'EState_VSA11',
 'EState_VSA2',
 'EState_VSA3',
 'EState_VSA4',
 'EState_VSA5',
 'EState_VSA6',
 'EState_VSA7',
 'EState_VSA8',
 'EState_VSA9',
 'VSA_EState1',
 'VSA_EState10',
 'VSA_EState2',
 'VSA_EState3',
 'VSA_EState4',
 'VSA_EState5',
 'VSA_EState6',
 'VSA_EState7',
 'VSA_EState8',
 'VSA_EState9',
 'FractionCSP3',
 'HeavyAtomCount',
 'NHOHCount',
 'NOCount',
 'NumAliphaticCarbocycles',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAromaticCarbocycles',
 'NumAromaticHeterocycles',
 'NumAromaticRings',
 'NumHAcceptors',
 'NumHDonors',
 'NumHeteroatoms',
 'NumRotatableBonds',
 'NumSaturatedCarbocycles',
 'NumSaturatedHeterocycles',
 'NumSaturatedRings',
 'RingCount',
 'MolLogP',
 'MolMR',
 'fr_Al_COO',
 'fr_Al_OH',
 'fr_Al_OH_noTert',
 'fr_ArN',
 'fr_Ar_COO',
 'fr_Ar_N',
 'fr_Ar_NH',
 'fr_Ar_OH',
 'fr_COO',
 'fr_COO2',
 'fr_C_O',
 'fr_C_O_noCOO',
 'fr_C_S',
 'fr_HOCCN',
 'fr_Imine',
 'fr_NH0',
 'fr_NH1',
 'fr_NH2',
 'fr_N_O',
 'fr_Ndealkylation1',
 'fr_Ndealkylation2',
 'fr_Nhpyrrole',
 'fr_SH',
 'fr_aldehyde',
 'fr_alkyl_carbamate',
 'fr_alkyl_halide',
 'fr_allylic_oxid',
 'fr_amide',
 'fr_amidine',
 'fr_aniline',
 'fr_aryl_methyl',
 'fr_azide',
 'fr_azo',
 'fr_barbitur',
 'fr_benzene',
 'fr_bicyclic',
 'fr_diazo',
 'fr_epoxide',
 'fr_ester',
 'fr_ether',
 'fr_furan',
 'fr_guanido',
 'fr_halogen',
 'fr_hdrzine',
 'fr_hdrzone',
 'fr_imidazole',
 'fr_imide',
 'fr_isocyan',
 'fr_ketone',
 'fr_ketone_Topliss',
 'fr_lactam',
 'fr_lactone',
 'fr_methoxy',
 'fr_morpholine',
 'fr_nitrile',
 'fr_nitro',
 'fr_nitro_arom',
 'fr_nitro_arom_nonortho',
 'fr_oxazole',
 'fr_oxime',
 'fr_para_hydroxylation',
 'fr_phenol',
 'fr_phenol_noOrthoHbond',
 'fr_phos_acid',
 'fr_phos_ester',
 'fr_piperdine',
 'fr_piperzine',
 'fr_priamide',
 'fr_pyridine',
 'fr_quatN',
 'fr_sulfide',
 'fr_sulfonamd',
 'fr_sulfone',
 'fr_term_acetylene',
 'fr_tetrazole',
 'fr_thiazole',
 'fr_thiocyan',
 'fr_thiophene',
 'fr_unbrch_alkane',
 'fr_urea']
    feature_df = feature_df[columns_to_filter]

    scaler = joblib.load(SCALER_PATH)  # Load trained scaler
    scaled_input = scaler.transform(feature_df)
    scaled_df = pd.DataFrame(scaled_input, columns=feature_df.columns)
    print(scaled_df)


    return [input, scaled_df]  # Returns a DataFrame


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
    vt_model = joblib.load(VT_PATH)

    # Load feature lists
    with open(feature_names) as f:
        feature_dict = json.load(f)

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
    #select_features = ['MaxEStateIndex', 'qed', 'AvgIpc', 'BertzCT', 'Chi4n', 'PEOE_VSA1', 'PEOE_VSA11', 'PEOE_VSA3', 'PEOE_VSA6', 'PEOE_VSA9', 'SMR_VSA6', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'VSA_EState2', 'VSA_EState4', 'VSA_EState6', 'NumHeteroatoms', 'fr_NH1', 'fr_NH2', 'fr_amide']

    #final_predictions = vt_model.predict(features[select_features])
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


