from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from scipy.spatial.distance import cdist
import numpy as np
import glob
import gzip
import bz2
import os
import _pickle as cPickle
import joblib
import sklearn
import logging
import pickle
import cloudpickle
import pandas as pd
from io import StringIO
import csv
from tqdm import tqdm
import logging
import warnings

import io
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warn(*args, **kwargs):
    pass
warnings.warn = warn

root = os.path.dirname(os.path.abspath(__file__))
MODELPATH = os.path.join(root, '../..', 'checkpoints/')

MODEL_DICT = {
    'Hepatic Stability': ['Dataset_01B_hepatic-stability_15min_imbalanced-morgan_RF.pgz',
                          'Dataset_01C_hepatic-stability_30min_imbalanced-morgan_RF.pgz',
                          'Dataset_01D_hepatic-stability_60min_imbalanced-morgan_RF.pgz'],
    'Microsomal Half-life Sub-cellular': ['Dataset_02A_microsomal-half-life-subcellular_imbalanced-morgan_RF.pgz'],
    'Microsomal Half-life Tissue': ['Dataset_02B_microsomal-half-life_30-min_binary_unbalanced_morgan_RF.pgz'],
    'Renal Clearance': ['dataset_03_renal-clearance_0.1-threshold_balanced-morgan_RF.pgz',
                        'dataset_03_renal-clearance_0.5-threshold_imbalanced-morgan_RF.pgz',
                        'dataset_03_renal-clearance_1.0-threshold_balanced-morgan_RF.pgz'],
    'BBB Permeability': ['dataset_04_bbb-permeability_balanced-morgan_RF.pgz'],
    'CNS Activity': ['dataset_04_cns-activity_1464-compounds_imbalanced-morgan_RF.pgz'],
    'CACO2': ['Dataset_05A_CACO2_binary_unbalanced_morgan_RF.pgz'],
    'Plasma Protein Binding': ['Dataset_06_plasma-protein-binding_binary_unbalanced_morgan_RF.pgz'],
    'Plasma Half-life': ['Dataset_08_plasma_half_life_12_hr_balanced-morgan_RF.pgz',
                         'Dataset_08_plasma_half_life_1_hr_balanced-morgan_RF.pgz',
                         'Dataset_08_plasma_half_life_6_hr_imbalanced-morgan_RF.pgz'],
    'Microsomal Intrinsic Clearance': ['Dataset_09_microsomal-intrinsic-clearance_12uL-min-mg-threshold-imbalanced-morgan_RF.pgz'],
    'Oral Bioavailability': ['dataset_10_oral_bioavailability_0.5_threshold_imbalanced-morgan_RF.pgz',
                             'dataset_10_oral_bioavailability_0.8_balanced-morgan_RF.pgz']
}

MODEL_DICT_INVERT = {v: key for key, val in MODEL_DICT.items() for v in val}

CLASSIFICATION_DICT = {
    'Hepatic Stability': {
        1: "Hepatic stability <= 50% at 15 minutes",
        2: "Hepatic stability <= 50% between 15 and 30 minutes",
        3: "Hepatic stability <= 50% between 30 and 60 minutes",
        4: "Hepatic stability > 50% at 60 minutes"
    },
    'Microsomal Half-life Sub-cellular': {
        0: "Sub-cellular Hepatic Half-life > 30 minutes",
        1: "Sub-cellular Hepatic Half-life <= 30 minutes"
    },
    'Microsomal Half-life Tissue': {
        0: "Tissue Hepatic Half-life > 30 minutes",
        1: "Tissue Hepatic Half-life <= 30 minutes"
    },
    'Renal Clearance': {
        1: "Renal clearance below 0.10 ml/min/kg",
        2: "Renal clearance between 0.10 and 0.50 ml/min/kg",
        3: "Renal clearance between 0.50 and 1.00 ml/min/kg",
        4: "Renal clearance above 1.00 ml/min/kg"
    },
    'BBB Permeability': {
        0: "Does not permeate blood brain barrier",
        1: "Does permeate blood brain barrier"
    },
    'CNS Activity': {
        0: "Does not exhibit central nervous system activity",
        1: "Does exhibit central nervous system activity"
    },
    'CACO2': {
        0: "Does not permeate Caco-2",
        1: "Does permeate Caco-2"
    },
    'Plasma Protein Binding': {
        0: "Plasma protein binder",
        1: "Weak/non plasma protein binder"
    },
    'Plasma Half-life': {
        1: "Half-life below 1 hour",
        2: "Half-life between 1 and 6 hours",
        3: "Half-life between 6 and 12 hours",
        4: "Half-life above 12 hours"
    },
    'Microsomal Intrinsic Clearance': {
        0: "Microsomal intrinsic clearance < 12 uL/min/mg",
        1: "Microsomal intrinsic clearance >= 12 uL/min/mg"
    },
    'Oral Bioavailability': {
        1: "Less than 0.5 F",
        2: "Between 0.5 and 0.8 F",
        3: "Above 0.8 F"
    }
}


AD_DICT = {
    True: "Inside",
    False: "Outside"
}

def run_prediction(model, model_data, smiles, calculate_ad=True):
    fp = np.zeros((2048, 1))
    _fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=3, nBits=2048)
    DataStructs.ConvertToNumpyArray(_fp, fp)

    if hasattr(model, 'predict_proba'):
        pred_proba = model.predict_proba(fp.reshape(1, -1))[:, 1]
    else:
        pred_proba = None

    pred = 1 if pred_proba is not None and pred_proba > 0.5 else 0

    if pred == 0 and pred_proba is not None:
        pred_proba = 1 - pred_proba

    if calculate_ad:
        ad = model_data["D_cutoff"] > np.min(cdist(model_data['Descriptors'].to_numpy(), fp.reshape(1, -1)))
        return pred, pred_proba, ad
    return pred, pred_proba, None
    

def get_prob_map(model, smiles):
    def get_fp(mol, idx):
        fps = np.zeros((2048, 1))
        _fps = SimilarityMaps.GetMorganFingerprint(mol, idx, radius=3, nBits=2048)
        DataStructs.ConvertToNumpyArray(_fps, fps)
        return fps

    def get_proba(fps):
        return float(model.predict_proba(fps.reshape(1, -1))[:, 1])

    mol = Chem.MolFromSmiles(smiles)
    fig, _ = SimilarityMaps.GetSimilarityMapForModel(mol, get_fp, get_proba)
    imgdata = io.StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    plt.savefig(imgdata, format="svg", bbox_inches="tight")

    return imgdata.getvalue()


def multiclass_ranking(ordered_preds):
    idx = 0
    one_detected = False
    for i, o in enumerate(ordered_preds):
        if int(o) == 1:
            if not one_detected:
                idx = i+1
                one_detected = True
        if int(o) == 0:
            if one_detected:
                idx = 0
                return idx
    return idx if idx != 0 else len(ordered_preds)+1


def load_model_and_data(model_endpoint, model_data_endpoint):
  with gzip.open(model_endpoint, 'rb') as f:
      model = joblib.load(f)

  with bz2.BZ2File(model_data_endpoint, 'rb') as f:
      model_data = joblib.load(f)

  if 'node' in model_data and 'missing_go_to_left' not in model_data['node'].dtype.names:
      expected_dtype = [('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'),
                        ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'),
                        ('weighted_n_node_samples', '<f8'), ('missing_go_to_left', 'u1')]
      model_data['node'] = np.array(model_data['node'], dtype=expected_dtype)

  return model, model_data
    
def test_pickled_model_data(files):
    for file_path in files:
        print(f"Testing file: {file_path}")
        try:
            with bz2.open(file_path, 'rb') as f:
                data = pickle.load(f)
            print("Successfully loaded the pickled model data.")
        except Exception as e:
            print(f"An error occurred while loading the pickled model data: {e}")

def main(smiles, calculate_ad=True, make_prop_img=False, **kwargs):
    def default(key, d):
        if key in d.keys():
            return d[key]
        else:
            return False

    models = sorted([f for f in glob.glob(os.path.join(MODELPATH, "*.pgz"))], key=lambda x: x.split("_")[1])
    models_data = sorted([f for f in glob.glob(os.path.join(MODELPATH, "*.pbz2"))], key=lambda x: x.split("_")[1])

    values = {}
    
    for model_endpoint, model_data_endpoint in zip(models, models_data):
        model_basename = os.path.basename(model_endpoint)
        model_key = MODEL_DICT_INVERT.get(model_basename)
        if model_key is None:
            print(f"Model endpoint key not found: {model_basename}")
            continue 

        if not default(model_key, kwargs):
            continue

        model, model_data = load_model_and_data(model_endpoint, model_data_endpoint)
        pred, pred_proba, ad = run_prediction(model, model_data, smiles, calculate_ad=calculate_ad)
        svg_str = ""
        
        if make_prop_img:
            svg_str = get_prob_map(model, smiles)
        print(model_basename)

        pred_proba_str = "N/A" 
        if pred_proba is not None:
            pred_proba_str = str(round(float(pred_proba) * 100, 2)) + "%"

        ad_value = AD_DICT.get(ad, "Unknown AD status") 

        values.setdefault(model_key, []).append([int(pred), pred_proba_str, ad_value, svg_str])


    processed_results = []
    for key, val in values.items():
        if key in ['Hepatic Stability', 'Renal Clearance', 'Plasma Half-life', 'Oral Bioavailability']:
            new_pred = multiclass_ranking([_[0] for _ in val])
            if new_pred == 0:
                processed_results.append([key, "Inconsistent result: no prediction", "Very unconfident", "NA", ""])
            else:
                if new_pred in [1, 2]:
                    p = 0
                else:
                    p = new_pred - 2
                processed_results.append([key, CLASSIFICATION_DICT[key][new_pred], val[p][1], val[p][2], val[p][3]])
        else:
            processed_results.append([key, CLASSIFICATION_DICT[key][val[0][0]], val[0][1], val[0][2], val[0][3]])

    return processed_results


from tqdm import tqdm
import csv
from io import StringIO
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm


def write_csv_file(smiles_list, calculate_ad=False):
    headers = list(MODEL_DICT.keys())

    if calculate_ad:
        headers = headers + [_ + "_AD" for _ in headers]

    output = [["SMILES"] + headers] 

    for smiles in tqdm(smiles_list):
        molecule = MolFromSmiles(smiles)

        row = [''] * (len(headers) + 1) 

        if molecule is None:
            row[0] = f"(invalid){smiles}"
            output.append(row)
            continue

        data = main(smiles, calculate_ad=calculate_ad, **MODEL_DICT)

        for model_name, pred, pred_proba, ad, _ in data:
            try:
                pred_proba = float(pred_proba[:-1]) / 100  
                if pred_proba < 0 or pred_proba > 1:
                    pred_proba = None
            except ValueError:
                pred_proba = None
            
            index = headers.index(model_name)
            row[index + 1] = pred_proba
            
            if calculate_ad:
                row.append(ad)

        row[0] = smiles
        output.append(row)

    return output

if __name__ == "__main__":
    import logging
    import argparse
    import csv
    from io import StringIO
    from rdkit.Chem import MolFromSmiles
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Script is running...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True, help="Location of the CSV file containing SMILES")
    parser.add_argument("--outfile", type=str, default="phakin_output.csv", help="Output CSV file path")
    parser.add_argument("--smiles_col", type=str, default="SMILES", help="Column name containing SMILES of interest")
    parser.add_argument("--ad", action="store_true", help="Calculate the AD")
    args = parser.parse_args()

    smiles_list = []
    with open(args.infile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            smiles_list.append(row[args.smiles_col])

    try:
        output = write_csv_file(smiles_list, calculate_ad=args.ad)
        with open(args.outfile, 'w') as outfile:
            outfile.write(output)
        logger.info("CSV file generation complete.")
    except Exception as e:
        logger.exception("An error occurred during CSV file generation.")
        raise e