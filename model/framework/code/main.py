# imports
import os
import csv
import sys
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import argparse
from phakinpro import write_csv_file

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))


# my model
#def my_model(smiles_list):
    #return [MolWt(Chem.MolFromSmiles(smi)) for smi in smiles_list]

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]
print(smiles_list)
# run model

"""
parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, required=True, help="Location of the CSV file containing SMILES")
parser.add_argument("--outfile", type=str, default="phakin_output.csv", help="Output CSV file path")
parser.add_argument("--smiles_col", type=str, default="SMILES", help="Column name containing SMILES of interest")
parser.add_argument("--ad", action="store_true", help="Calculate the AD")
args = parser.parse_args()
"""

# Read SMILES from input CSV
"""
smiles_list = []
with open(args.infile, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        smiles_list.append(row[args.smiles_col])
"""
#output = write_csv_file(smiles_list, calculate_ad=args.ad)

outputs = write_csv_file(smiles_list, calculate_ad=False)
print(outputs)
#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
print(input_len)
print(output_len)
assert input_len == output_len

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["value"])  # header
    for o in outputs:
        writer.writerow([o])
