import argparse

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import scvi
from feature_attribution_sc.explainers.mask import mask, generate_rankings

sc.set_figure_params(dpi=100, frameon=False, color_map='Reds', facecolor=None)

def generate_masked_inputs(attrib_df, thresholds, batch, class_label='labels'):
    """Returns the masked inputs given rankings from an attribution file.
    
    Masking can take a while to iterate over all tensors.
    
    Params
    ------
    attrib_df : pd.DataFrame
        Formatted dataframe of feature attributions, where the index contains feature names
        and the columns the classes.
    thresholds : list[int]
        List of percentages of features to remove.
    batch : dict[torch.Tensor]
        scVI batch dictionary containing tensors.
    class_label : str
        Batch label which contains the class information used for masking, matching the columns in attrib_df.
        
    Returns a dictionary of the masked datasets and a list of tuples of the sparsity per threshold.
    """
    rankings, gene_indices = generate_rankings(attrib_df)
    
    incremental_sparsity = []
    masked_inputs = {}
    for threshold in thresholds:
        t = float(threshold)/100
        print(t)
        masked_inpt = mask(
            batch['X'], batch[class_label], attrib_df, rankings, gene_indices, threshold=t)

        # store input for later use
        masked_inputs[f'masked_{threshold}'] = masked_inpt

        # record the sparsity for plotting. only applicable if we're masking with zeroes
        sparsity_after = np.count_nonzero(masked_inputs[f'masked_{threshold}']) / (masked_inpt.shape[0] * masked_inpt.shape[1])
        incremental_sparsity.append((t, sparsity_after))
        
    return masked_inputs, incremental_sparsity


### Add arguments ###
parser = argparse.ArgumentParser()

# Data (data_label, AnnData/cell_line embedding, gRNA_embedding)
parser.add_argument("--csv", type=str, required=True, help='Path to file with feature importance attributions.')
parser.add_argument("--task", type=int, required=True, help='One of the prediction tasks: 1(perturbations), 2(celltype), or 3(multimodal).')
parser.add_argument("--skip_zero", action="store_true", required=False, help='Whether to skip the zero calculation or not to save computation time.')
parser.add_argument("--model_file", type=str, required=False, help='A specific model file to use instead of the default.')
parser.add_argument("--thresholds", type=int, nargs='+', required=False, help='A list of thresholds')

args = parser.parse_args()
task = args.task
skip_zero = args.skip_zero
model_file = args.model_file
thresholds = list(args.thresholds)
if thresholds is None:
    thresholds = list(range(0, 100, 10)) + [99]
feature_importance = args.csv
method = feature_importance.split("/")[-2]
attrib_df = pd.read_csv(feature_importance)

### Load data ###
if task == 1:
    adata = sc.read(f'../datasets/2301_scgen_norman19.h5ad')
    scgen.SCGEN.setup_anndata(adata, batch_key='perturbation_name')
    model = scgen.SCGEN(adata)
    if model_file is None: model_file = 'scgen_norman19_model0_random_v2'
    model = scgen.SCGEN.load(f'../models/{model_file}', adata=adata)
elif task == 2:
    adata = sc.read('../datasets/hlca.h5ad')
    model = scvi.model.SCANVI.load('../models/scanvi_model/', adata)
else:
        raise ValueError('Task must be one of 1, 2, 3.')

batch_size = adata.shape[0]
scdl = model._make_data_loader(adata=adata, indices=list(range(adata.shape[0])), batch_size=batch_size)
batch = next(scdl.__iter__())

### Create masks ###

# reverse order for DE
if 'differential_expression' in feature_importance:
    attrib_df[attrib_df.columns[1:]] *= -1

# TODO: placeholder for unlabeled for HLCA
if task == 1 and 'unlabeled' not in attrib_df.columns:
    attrib_df['unlabeled'] = 0

n_features = attrib_df.shape[0]
if n_features < adata.shape[1]:
    raise ValueError(
        f"Attributions only calculated for {n_features} genes but adata has {adata.shape[1]}")
elif n_features > adata.shape[1]:
    print(f'Only using attributions for {adata.shape[1]} genes.')
    attrib_df = attrib_df.set_index('gene_symbols').loc[adata.var_names].reset_index().rename({'index':'gene_symbols'}, axis=1)

attribution_masks, incremental_sparsity = \
    generate_masked_inputs(attrib_df, thresholds, batch)

### plot sparsity ###
x, y = zip(*incremental_sparsity)
plt.scatter(x, y, label=feature_importance.split('/')[-1].split('.')[0])
plt.ylabel('frac non-zero')
plt.xlabel('frac top important features masked')
plt.legend(bbox_to_anchor=(1.01, 1.05))
plt.savefig(f'sparsity_task{task}_{method}.png', bbox_inches='tight')
pd.DataFrame(incremental_sparsity).to_csv(f'sparsity_task{task}_{method}.csv')

### Forward pass through model ###
for threshold in thresholds:
    if threshold == 0 and skip_zero:
        continue

    obs_key = f'{method}_masked_{threshold}_pred'

    if obs_key in adata.obs.columns:
        continue

    if task == 1:
        new_batch = batch.copy()
        new_batch['X'] = attribution_masks[f'masked_{threshold}']

        adata.layers[obs_key] = model.module.forward(new_batch, compute_loss=False)[1]['px'].detach().numpy()

    elif task == 2:
        adata.X = attribution_masks[f'masked_{threshold}']
        X = model.get_latent_representation(adata)

        ref_emb = ad.AnnData(X, obs=adata.obs)
        print('calculating neighbors on ref embedding for', obs_key)
        sc.pp.neighbors(ref_emb, n_neighbors=30)

        print('label transfer')
        _ = perform_label_transfer(
            ref_emb=ref_emb, query_emb=ref_emb, cell_type_column="scanvi_label"
        )

        if _ != -1:
            adata.obs[obs_key] = ref_emb.obs['transf_scanvi_label'].copy()


if task == 1:
    adata.write(f'masking_res_task{task}_{method}.h5ad')  # careful, this might become huge
if task == 2:
    adata.obs.to_csv(f'masking_res_task{task}_{method}.csv')
