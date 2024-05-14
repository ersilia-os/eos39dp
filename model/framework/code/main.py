import os
import csv
import sys
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

from phakinpro import write_csv_file

input_file = sys.argv[1]
output_file = sys.argv[2]

root = os.path.dirname(os.path.abspath(__file__))


def my_model(smiles_list):
    return [MolWt(Chem.MolFromSmiles(smi)) for smi in smiles_list]

with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)
    smiles_list = [r[0] for r in reader]
outputs = write_csv_file(smiles_list, calculate_ad=False)
print(outputs)
input_len = len(smiles_list)
output_len = len(outputs) - 1
assert input_len == output_len

with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["value"])
    for o in outputs:
        writer.writerow([o])