import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scgen
from feature_attribution_sc.explainers.mask import mask, generate_rankings
from feature_attribution_sc.models.scgen_models import SCGENCustom

data_path = '/home/icb/yuge.ji/projects/feature-attribution-sc'

feature_importance_files = [
     f'{data_path}/outputs/baselines/task1_random.csv',
#     f'{data_path}/outputs/differential_expression/task1_DE_control.csv',  # these are way slower for some reason???
#    f'{data_path}/outputs/expected_gradients/task1_absolute_expected_grads_v2.csv'
]

thresholds = list(range(10, 100, 10))
adata = sc.read(f'{data_path}/datasets/2301_scgen_norman19.h5ad')

for feature_importance in feature_importance_files:
    attrib_df = pd.read_csv(feature_importance)
    attrib_key = feature_importance.split("/")[-2]

    # reverse order for DE
    if 'differential_expression' in feature_importance:
        attrib_df[attrib_df.columns[1:]] *= -1

    # todo: should include a check for adata.var_names vs attrib_df.gene_symbols

    n_features = attrib_df.shape[0]
    if n_features < adata.shape[1]:
        raise ValueError(
            f"Attributions only calculated for {n_features} genes but adata has {adata.shape[1]}")
    elif n_features > adata.shape[1]:
        print(f'Only using attributions for {adata.shape[1]} genes.')
        attrib_df = attrib_df.set_index('gene_symbols').loc[adata.var_names].reset_index()

    for threshold in thresholds:
        print('training at threshold =', threshold)
        SCGENCustom.setup_anndata(adata)
        save_str = f'{data_path}/models/scgen_norman19_ROAR_{attrib_key}_{threshold}'
        model = SCGENCustom(adata, feature_importance=attrib_df, threshold=threshold/100, n_hidden=400, n_latent=30)
        model.train(max_epochs=50, batch_size=32)
        model.save(save_str, overwrite=True)
