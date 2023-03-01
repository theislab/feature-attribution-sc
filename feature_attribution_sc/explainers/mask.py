import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple


def generate_rankings(df: pd.DataFrame) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, int]]:
    """
    Generate a dictionary that maps each perturbation to a list of tuples where each tuple consists of
    a gene symbol and its ranking for the perturbation.
    :param df: DataFrame with gene symbols as rows and perturbation names as columns with rankings as values
    :return: Dictionary that maps perturbation names to lists of (gene symbol, ranking) tuples
    """
    rankings = {}
    gene_indices = {}
    for i, gene in enumerate(df["gene_symbols"]):
        gene_indices[gene] = i
    for col in df.columns[1:]:
        rankings[col] = list(zip(df['gene_symbols'], df[col]))
    rankings[col].sort(key=lambda x: x[1])
    return rankings, gene_indices


def mask(data: torch.tensor,
         labels: torch.tensor,
         df: pd.DataFrame,
         rankings: Dict[str, List[Tuple[str, int]]],
         gene_indices: Dict[str, int],
         threshold: float, ) -> torch.tensor:
    """
    Mask a batch of data based on ranking data and a threshold.
    This time we know that the genes in the ranking data are the same as the genes in the data points
    N = batch size, e.g., 2048
    M = number of genes, e.g., 19018
    :param data: Batch of data to be masked, N x M array
    :param labels: Batch of labels, N x 1 array
    :param df: DataFrame with gene symbols as rows and perturbation names as columns with rankings as values
    :param rankings: Ranking data, Dict[label, ranking data]
    :param gene_indices: Dictionary that maps gene names to indices in the data array
    :param threshold: Masking threshold
    :return: Masked data, N x M tensor
    """
    # Initialize masked data as a copy of the input data
    masked_data = np.copy(data)
    # Loop over the rows of data_point and label
    for i in range(len(data)):
        data_point = data[i]
        label = labels[i]
        # Get the perturbation corresponding to the current label
        perturbation = df.columns.tolist()[int(label + 1)]
        # Get the ranking data for the current perturbation
        ranking = rankings[perturbation]
        # Find the top threshold percent of the ranking data
        top_ranking_data = ranking[:int(len(ranking) * threshold)]
        # Initialize mask to 1 for all genes
        mask = np.ones(data_point.shape[0])
        # Set the mask to zero for the top threshold percent of the ranking data
        for gene, _ in top_ranking_data:
            index = gene_indices[gene]
            mask[index] = 0
        # Multiply the data by the mask to zero out the specified values
        masked_data[i] = data_point * mask
    return torch.tensor(masked_data)
