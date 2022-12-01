import pandas as pd
from sklearn.metrics import roc_auc_score


def hlca_markers():
    """Parse marker gene dictionary from csv for ground truth.

    Returns
    -------
    A dictionary of form {cell_type: [marker genes]}
    """
    df = pd.read_csv("/home/icb/yuge.ji/projects/feature-attribution-sc/datasets/hlca_marker_dict.csv")
    hlca_hvgs = pd.read_csv("datasets/hlca_hvgs.csv").gene_symbols.values

    marker_dict = {c: list(df[c].dropna().values) for c in df.columns}
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

    Params
    ------
    ranking : pd.DataFrame
    """
    marker_dict = hlca_markers()
    y_true = ranking[list(set(ranking.columns) & set(marker_dict.keys()))].copy()
    y_true[:] = 0
    for ct, genes in marker_dict.items():

        if ct in y_true.columns:
            y_true.loc[genes, ct] = 1

    return roc_auc_score(y_true.values, ranking[y_true.columns].values)


def roc_auc_crispr(ranking):
    """
    Calculate roc_auc for a CRISPR dataset where the source of ground
    truth is the sgRNA of the condition.

    Params
    ------
    ranking : pd.DataFrame
    """
    y_true = ranking[list(set(ranking.columns) & set(ranking.index))].copy()
    y_true[:] = 0
    for pert in ranking.columns:
        if pert in y_true.index:
            y_true.loc[pert, pert] = 1
    return roc_auc_score(y_true.values, ranking[y_true.columns].values)
