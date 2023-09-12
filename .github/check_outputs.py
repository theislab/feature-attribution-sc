from pathlib import Path
from rich import print

import pandas as pd
import os

# standardized dataframes
data_path = Path(f"{os.getcwd()}/outputs/").resolve()


task1_df = pd.read_csv(f"{data_path}/baselines/task1_random.csv")  # this contains control
task2_df = pd.read_csv(f"{data_path}/baselines/task2_random.csv")  # this contains unlabled


def make_features_unique(df):
    """De-duplicates features in `df` in the same way as `anndata`."""
    import anndata as ad

    dfa = ad.AnnData(df.set_index("gene_symbols").T)
    dfa.var_names_make_unique()
    df["gene_symbols"] = dfa.var_names


catch = {}
for i, file in enumerate(os.walk(data_path)):
    if i == 0:
        continue

    for attrib in file[2]:
        corrected = False
        path = file[0] + "/" + attrib

        if "gini" in path:  # ignore gini calculations
            continue

        df = pd.read_csv(path)

        ### check for feature column ###
        if "gene_symbols" not in df.columns:
            catch[path] = f"No column named 'gene_symbols' found."

        ### check that features are labeled ###
        if df.columns[0] != "gene_symbols":
            # correct if it's just misnamed
            if type(df[[df.columns[0]]].values[0][0]) is str:  # assume genes if str
                df.columns = ["gene_symbols"] + list(df.columns[1:])
                corrected = True

            else:
                raise ValueError("Please provide a column named `gene_symbols` containing gene names.")

        ### check for unique features and attempt to fix ###
        if len(set(df["gene_symbols"])) != df.shape[0]:
            make_features_unique(df)
            corrected = True

        ### check that columns are labeled, number of columns, and values of features ###
        if "task1.5" in attrib:
            # check columns

            # check features
            print("replogle")
            pass

        elif "task1" in attrib:
            # checking columns
            diff = set(df.columns[1:]) - set(task1_df.columns[1:])
            if abs(len(diff)) > 0:
                # no problem if output was for latent dimensions
                if "latent" in attrib:
                    pass

                else:
                    catch[path] = f"Categories are not the same. Saw \n{diff}"

            # checking features
            diff = set(df["gene_symbols"].values) - set(task1_df["gene_symbols"].values)
            if len(diff) > 0:
                catch[path] = f"Contains genes not found in the original adata!! {diff}"

        elif "task2" in attrib:
            # checking columns - should be at least a subset of all `scanvi_label`
            diff = set(df.columns[1:]) - set(task2_df.columns[1:])
            if abs(len(diff)) > 0:
                catch[path] = f"Categories are not the same. Expected only {task2_df.columns[1:]} but saw \n{diff}"

            # checking features
            if set(df["gene_symbols"].values) != set(task2_df["gene_symbols"].values):
                catch[path] = "Gene symbols do not match!!"

        elif "task3" in attrib:
            pass

        else:
            raise ValueError(f"No task identified in {attrib}!!")

        # save auto-fixes
        if corrected:
            df.set_index("gene_symbols").to_csv(path)

for path, error in catch.items():
    print(path, error)

if len(catch) != 0:
    raise ValueError("Please fix the issues in output formatting detailed above!")
