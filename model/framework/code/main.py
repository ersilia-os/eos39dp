# imports
import os
import csv
import sys
from io import StringIO

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(root)
from phakinpro import write_csv_file


# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run model
outputs = write_csv_file(smiles_list)

csv_file_like = StringIO(outputs)

reader = csv.reader(csv_file_like)

# Write the content to a new CSV file
with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    header = next(reader)
    writer.writerow(header[1:])
    for row in reader:
        writer.writerow(row[1:])
