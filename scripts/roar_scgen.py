import sys
if len(sys.argv) != 3:
    print("Usage: python roar_scgen.py <path/to/filestr> <iteration>")
    sys.exit(1)
feature_importance = sys.argv[1]
iteration = sys.argv[2]

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scgen
from feature_attribution_sc.explainers.mask import mask, generate_rankings, validate_rankings
from feature_attribution_sc.models.scgen_models import SCGENCustom


# dynamically generate absolute save path assuming dir structure
data_path = "/".join(feature_importance.split("/")[:-3])
print('Saving to', data_path)

thresholds = list(range(10, 100, 10)) + [99]
adata = sc.read(f'{data_path}/datasets/2301_scgen_norman19.h5ad')

# model settings
parameters = {
    'n_hidden': 400,
    'n_latent': 30,
    'max_epochs': 50,
    'batch_size': 32
}

#for feature_importance in feature_importance_files:
attrib_df = pd.read_csv(feature_importance)
attrib_key = feature_importance.split("/")[-2]

# reverse order for DE
if 'differential_expression' in feature_importance:
    attrib_df[attrib_df.columns[1:]] *= -1

# todo: should include a check for adata.var_names vs attrib_df.gene_symbols
attrib_df = validate_rankings(attrib_df, adata)

for threshold in thresholds:
    print(f'training {attrib_key} at threshold =', threshold)
    SCGENCustom.setup_anndata(adata, batch_key='perturbation_name')
    save_str = f'{data_path}/models/ROAR/scgen_norman19_ROAR_{attrib_key}_{iteration}_{threshold}'
    model = SCGENCustom(adata, feature_importance=attrib_df, threshold=threshold/100, n_hidden=parameters['n_hidden'], n_latent=parameters['n_latent'], feature_importance_str=feature_importance)
    model.train(max_epochs=parameters['max_epochs'], batch_size=parameters['batch_size'])
    model.save(save_str, overwrite=True)
