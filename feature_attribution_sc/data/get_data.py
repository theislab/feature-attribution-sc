# from Sergei's https://github.com/theislab/feature-attribution-sc/blob/main/notebooks/train_scgen.ipynb

import scvi
import scanpy as sc
import scgen
import os
import pandas as pd
import numpy as np


def get_scgen_data(batch_size, data_path="/home/icb/yuge.ji/projects/feature-attribution-sc"):
    """
    Receive the adata object given a path to the corresponding .h5ad file
    :param batch_size: batch size of the SCGEN model
    :param data_path: PATH where the .h5ad file is located
    :return:
    """
    adata = sc.read(f"{data_path}/datasets/scgen_norman19.h5ad")
    models = {}
    for file in os.listdir(f"{data_path}/models"):
        if "scgen" in file:
            print("loading", file)
            models["_".join(file.split("_")[1:])] = scgen.SCGEN.load(f"{data_path}/models/{file}", adata=adata)

    indices = np.random.choice(adata.n_obs, size=100, replace=False)
    # for model in models:
    #     dataloaders = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
    return None, adata


def get_hlca_data(batch_size):
    """
    Receive the PyTorch dataloader and adata object of the HLCA
    :param batch_size: batch size
    :return: dataloader, adata
    """
    hlca_path = "/lustre/groups/ml01/workspace/hlca_lisa.sikkema_malte.luecken/HLCA_reproducibility/data/HLCA_core_h5ads/HLCA_v1_integration/HLCA_v1_scANVI_input.h5ad"
    adata = sc.read(hlca_path)
    model = scvi.model.SCANVI.load(
        "/home/icb/yuge.ji/projects/HLCA_reproducibility/notebooks/3_atlas_extension/scanvi_model/", adata
    )

    # parse marker gene dictionary from csv for ground truth
    marker_df = pd.read_csv(
        "/home/icb/yuge.ji/projects/HLCA_reproducibility/notebooks/3_atlas_extension/markergenes.csv", index_col=0
    )

    marker_dict = {}
    for i in range(0, marker_df.shape[1], 3):
        ct = marker_df.columns[i].split("_")[0]
        l = list(
            marker_df[marker_df[f"{ct}_marker_for"].isin([ct, f"{ct} (poss. lowly expressed, non-unique)"])][
                f"{ct}_marker"
            ].values
        )
        # some markers are skipped due to uncertainty in annotation
        if len(l) > 0:
            marker_dict[ct] = l

    indices = np.random.choice(adata.n_obs, size=100, replace=False)
    dataloader = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
    return dataloader, adata
