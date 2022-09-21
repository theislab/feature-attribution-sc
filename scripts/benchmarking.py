import pandas as pd
from sklearn.metrics import roc_auc_score

def hlca_markers():
    """Parse marker gene dictionary from csv for ground truth.
    """
    marker_df = pd.read_csv('/home/icb/yuge.ji/projects/HLCA_reproducibility/notebooks/3_atlas_extension/markergenes.csv', index_col=0)
    hlca_hvgs = pd.read_csv('datasets/hlca_hvgs.csv').gene_symbols.values

    marker_dict = {}
    for i in range(0, marker_df.shape[1], 3):
        ct = marker_df.columns[i].split('_')[0]
        query_res = marker_df[[c for c in marker_df.columns if ct in c]][[f'{ct}_marker', f'{ct}_marker_for']].dropna().values
        for value, key in query_res:
            k = (key+' ').split('(')[0][:-1]  # TODO: major hack for now, fix later
            marker_dict.setdefault(k, []).append(value)
    
    # remove markers if not in adata due to HVG subsetting. Remaining: 105/150
    # this causes some cell types to have no genes
    to_remove = []
    for ct, l in marker_dict.items():
        l = list(set(l) & set(hlca_hvgs))
        if len(l) > 0:
            marker_dict[ct] = l
        else:
            to_remove.append(ct)
    # cell types with no genes
    for ct in to_remove:
        del marker_dict[ct]
    return marker_dict

def roc_auc_hlca(ranking):
    """Calculate roc_auc for hlca given a list of rankings.
    """
    marker_dict = hlca_markers()
    y_true = ranking[list(set(ranking.columns) & set(marker_dict.keys()))].copy()
    y_true[:] = 0
    for ct, genes in marker_dict.items():
        
        if ct in y_true.columns:
            y_true.loc[genes, ct] = 1

    return roc_auc_score(y_true.values, mean_df[y_true.columns].values)
