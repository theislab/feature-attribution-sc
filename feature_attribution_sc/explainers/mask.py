import pandas as pd
import torch


def apply_mask(x, y, threshold=0.0, feature_importance="random"):
    """Apply the mask of a feature importance metrics

    :param x: batch of input data
    :param y: batch of corresponding cell types
    :param feature_importance: feature importance method
    :param threshold: threshold of the feature importance
    :return: masked batch of input data
    """
    if feature_importance == "random":  # don't apply feature importance, if subset, use random
        mask = torch.distributions.Bernoulli(probs=1.0 - threshold).sample(x.size())
        # upscale inputs to compensate for masking
        masked_outputs = y * mask  # scale possible with 1. / (1. - threshold)
        return masked_outputs
    else:
        return x


def produce_mask(
    x, y, CSV_PATH="/home/icb/till.richter/git/feature-attribution-sc/outputs/random/task1_random.csv", threshold=0.5
):
    """Given a batch of data and their corresponding labels, apply a mask that is unique for each label

    :param x: data batch, e.g., perturbations x genes, cells x genes
    :param y: labels, e.g., perturbations, cells
    :param CSV_PATH: Path to the csv file, that contains the ranking of feature importance maps
    :param threshold: threshold of this experiment -> amount of features set to 0
    :return: masked output
    """
    importance = pd.read_csv(CSV_PATH)
    importance_alpha = importance.sort_values("gene_symbols")

    # rank each column in ascending order
    ranks = importance_alpha.rank(method="first")

    # from the ranking and the threshold, build a 0-1 mask for each perturbation
    num_ones = int(threshold * len(ranks))
    mask = pd.DataFrame(ranks >= num_ones).astype(int)

    # given data (all genes and batch of perturbations) and labels (name of perturbation), apply corresponding mask
    for pert in range(len(y)):
        mask_i = mask[y[pert]].tolist()
        x[:, pert] *= mask_i

    return x
