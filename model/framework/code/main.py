import csv
import sys
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from phakinpro import write_csv_file

input_file = sys.argv[1]
output_file = sys.argv[2]

def my_model(smiles_list):
    return [MolWt(Chem.MolFromSmiles(smi)) for smi in smiles_list]

with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)
    smiles_list = [r[0] for r in reader]

outputs = write_csv_file(smiles_list, calculate_ad=False)

# Debug print the output before writing to file
print("Generated Outputs:")
print(outputs)

# Writing the structured output to CSV
with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    for row in outputs:
        writer.writerow(row)
