# imports
import os
import csv
import sys
import gzip
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import _pickle as cPickle

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
PATH_TO_CHECKPOINTS = os.path.join(root, "..", "..", 'checkpoints')

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

HEADER_TO_MODEL = {
    'hs_15min': 'Dataset_01B_hepatic-stability_15min_imbalanced-morgan_RF.pgz',
    'hs_30min': 'Dataset_01C_hepatic-stability_30min_imbalanced-morgan_RF.pgz',
    'hs_60min': 'Dataset_01D_hepatic-stability_60min_imbalanced-morgan_RF.pgz',
    'mhl_subcellular': 'Dataset_02A_microsomal-half-life-subcellular_imbalanced-morgan_RF.pgz',
    'mhl_tissue': 'Dataset_02B_microsomal-half-life_30-min_binary_unbalanced_morgan_RF.pgz',
    'rc_01': 'dataset_03_renal-clearance_0.1-threshold_balanced-morgan_RF.pgz',
    'rc_05': 'dataset_03_renal-clearance_0.5-threshold_imbalanced-morgan_RF.pgz',
    'rc_1': 'dataset_03_renal-clearance_1.0-threshold_balanced-morgan_RF.pgz',
    'bbb_permeability': 'dataset_04_bbb-permeability_balanced-morgan_RF.pgz',
    'cns_activity': 'dataset_04_cns-activity_1464-compounds_imbalanced-morgan_RF.pgz',
    'caco2_permeability': 'Dataset_05A_CACO2_binary_unbalanced_morgan_RF.pgz',
    'ppb': 'Dataset_06_plasma-protein-binding_binary_unbalanced_morgan_RF.pgz',
    'phl_1': 'Dataset_08_plasma_half_life_1_hr_balanced-morgan_RF.pgz',
    'phl_6': 'Dataset_08_plasma_half_life_6_hr_imbalanced-morgan_RF.pgz',
    'phl_12': 'Dataset_08_plasma_half_life_12_hr_balanced-morgan_RF.pgz',
    'miclearence': 'Dataset_09_microsomal-intrinsic-clearance_12uL-min-mg-threshold-imbalanced-morgan_RF.pgz',
    'ob_05': 'dataset_10_oral_bioavailability_0.5_threshold_imbalanced-morgan_RF.pgz',
    'ob_08': 'dataset_10_oral_bioavailability_0.8_balanced-morgan_RF.pgz'}


HEADERS = ['hs_15min', 'hs_30min', 'hs_60min', 'mhl_subcellular', 'mhl_tissue', 
            'rc_01', 'rc_05', 'rc_1', 'bbb_permeability', 'cns_activity', 'caco2_permeability', 
            'ppb', 'phl_1', 'phl_6', 'phl_12', 'miclearence', 'ob_05', 'ob_08']
df = pd.DataFrame()

# Featurize comopunds
fps = []
for smiles in smiles_list:
    arr = np.zeros((2048,), dtype=int)
    mol = Chem.MolFromSmiles(smiles)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    fps.append(arr)
fps = np.array(fps)
features = [f"Bit_{i}" for i in range(len(arr))]
fps = pd.DataFrame(fps, columns=features)

for header in HEADERS:

    # Load model
    with gzip.open(open(os.path.join(PATH_TO_CHECKPOINTS, HEADER_TO_MODEL[header]), 'rb')) as f:
        model = cPickle.load(f)

    # Make predictions
    preds = model.predict_proba(fps)

    # Save results
    df[header] = preds[:, 1]


#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(df)
assert input_len == output_len


# write output in a .csv file
df.to_csv(output_file, sep=",", index=False)