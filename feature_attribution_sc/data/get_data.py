# from Sergei's https://github.com/theislab/feature-attribution-sc/blob/main/notebooks/train_scgen.ipynb

import scvi
import scanpy as sc
import pandas as pd
import numpy as np


def get_hlca_data(batch_size,
                  hlca_path='/lustre/groups/ml01/workspace/hlca_lisa.sikkema_malte.luecken/HLCA_reproducibility/'
                            'data/HLCA_core_h5ads/HLCA_v1_integration/HLCA_v1_scANVI_input.h5ad',
                  scanvi_model_path='/lustre/groups/ml01/workspace/hlca_lisa.sikkema_malte.luecken/'
                                    'HLCA_reproducibility/notebooks/3_atlas_extension/scanvi_model/',
                  marker_gene_path='/lustre/groups/ml01/workspace/hlca_lisa.sikkema_malte.luecken/'
                                   'HLCA_reproducibility/notebooks/3_atlas_extension/markergenes.csv', ):
    """
    Receive the PyTorch dataloader and adata object of the HLCA
    :param hlca_path: path to the h5ad file
    :param marker_gene_path: path to the csv file containing the marker genes
    :param scanvi_model_path: path to the scanvi model
    :param batch_size: batch size
    :return: dataloader, adata
    """
    adata = sc.read(hlca_path)
    model = scvi.model.SCANVI.load(scanvi_model_path, adata)

    # parse marker gene dictionary from csv for ground truth
    marker_df = pd.read_csv(marker_gene_path, index_col=0)

    marker_dict = {}
    for i in range(0, marker_df.shape[1], 3):
        ct = marker_df.columns[i].split('_')[0]
        marker_list = list(marker_df[
                     marker_df[f'{ct}_marker_for'].isin([ct,
                                                         f'{ct} (poss. lowly expressed, non-unique)'])
                 ][f'{ct}_marker'].values)
        # some markers are skipped due to uncertainty in annotation
        if len(marker_list) > 0:
            marker_dict[ct] = marker_list

    indices = np.random.choice(adata.n_obs, size=100, replace=False)
    dataloader = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
    return dataloader, adata
