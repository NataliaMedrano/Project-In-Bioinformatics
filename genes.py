import pandas as pd

annot_file = "GPL570.annot"
df = pd.read_csv(annot_file, sep="\t", comment="#", skiprows=27)

drd2_data = df[df["Gene symbol"] == "COMT"]
print(drd2_data)