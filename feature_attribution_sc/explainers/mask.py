import torch
import scvi
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple


def generate_rankings(df: pd.DataFrame) -> Dict[str, List[Tuple[str, int]]]:
    """
    Generate a dictionary that maps each perturbation to a list of tuples where each tuple consists of
    a gene symbol and its ranking for the perturbation.
    :param df: DataFrame with gene symbols as rows and perturbation names as columns with rankings as values
    :return: Dictionary that maps perturbation names to lists of (gene symbol, ranking) tuples
    """
    rankings = {}
    gene_indices = {}
    for i, gene in enumerate(df['gene_symbols']):
        gene_indices[gene] = i
    for col in df.columns[1:]:
        rankings[col] = list(zip(df['gene_symbols'], df[col]))
        rankings[col].sort(key=lambda x: x[1])
    return rankings, gene_indices


def mask(data: np.ndarray, labels: np.ndarray, rankings: Dict[str, List[Tuple[str, int]]], gene_indices: Dict[str, int], threshold: float) -> np.ndarray:
    """
    Mask a batch of data based on ranking data and a threshold.
    :param data: Batch of data to be masked, N x M array
    :param labels: Batch of labels, N x 1 array
    :param rankings: Ranking data, Dict[label, ranking data]
    :param gene_indices: Dictionary that maps gene names to indices in the data array
    :param threshold: Masking threshold
    :return: Masked data, N x M array
    """
    # Transpose the data array to have shape M x N
    data = data.T
    # Initialize masked data as a copy of the input data
    masked_data = np.copy(data)
    # Loop through the data points in the batch
    for i, (data_point, label) in enumerate(zip(data, labels)):
        # Get the ranking data for the current label
        ranking = rankings[label]
        # Find the top threshold percent of the ranking data
        top_ranking_data = ranking[:int(len(ranking) * threshold)]
        # Set the mask to zero for the top threshold percent of the ranking data
        mask = np.ones_like(data_point)
        for gene, _ in top_ranking_data:
            index = gene_indices[gene]
            mask[index] = 0
        # Multiply the data by the mask to zero out the specified values
        masked_data[i] = data_point * mask
    # Transpose the masked data back to have shape N x M
    return masked_data




'''
def mask(data: np.ndarray, pert_names: List[str], ranking: List[Tuple[str, int]], label: str, threshold: float) -> np.ndarray:
    """
    Mask data based on ranking data and a threshold.
    :param data: Data to be masked, 1 x P array
    :param pert_names: Perturbation names, 1 x P array
    :param ranking: Ranking data, 1 x P array
    :param label: Current label
    :param threshold: Masking threshold
    :return: Masked data
    """
    # Find the top threshold percent of the ranking data
    top_ranking_data = ranking[:int(len(ranking) * threshold)]
    # Set the mask to zero for the top threshold percent of the ranking data
    mask = np.ones_like(data)
    print('Mask: ', mask, mask.shape)
    print('Data: ', data, data.shape)
    print('Pert names: ', pert_names, len(pert_names))
    print('Rop ranking data: ', top_ranking_data, len(top_ranking_data))
    print('Is in: ', np.isin(pert_names, [x[0] for x in top_ranking_data]))
    mask[:, np.isin(pert_names, [x[0] for x in top_ranking_data])] = 0
    # Multiply the data by the mask to zero out the specified values
    return data * mask


def apply_mask(x: np.ndarray, y: np.ndarray, pert_names: List[str], rankings: Dict[str, List[Tuple[str, int]]], threshold: float) -> List[np.ndarray]:
    """
    Mask a batch of data based on ranking data and a threshold.
    :param x: Batch of data to be masked, N x P array
    :param y: Batch of labels, N x 1 array
    :param pert_names: Perturbation names, 1 x P array
    :param rankings: Ranking data, Dict[label, ranking data]
    :param threshold: Masking threshold
    :return: List of masked data
    """
    masked_x = []
    # Loop through the data points in x and y
    for i, (data, label) in enumerate(zip(x, y)):
        # Mask the data point based on the ranking data for the corresponding label
        masked_x.append(mask(data=data, pert_names=pert_names, ranking=rankings[label], label=label, threshold=threshold))
    return masked_x







import torch
import scvi
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Callable, Dict, List, Optional


# Check if all items of list a are in list b
def check_all_in_list(a, b):
    for item in a:
        if item not in b:
            return False
    return True


def generate_rankings(df: pd.DataFrame) -> defaultdict:
    """
    Generate a dictionary of rankings data from a pandas DataFrame.
        This basically avoids reading a .csv file from disk at each iteration over a batch.
    :param df: .csv file of a feature importance ranking
    :return: dictionary of rankings
    """
    # Initialize a defaultdict to store the ranking data
    rankings = defaultdict(list)

    # Get the column names from the .csv file
    column_names = df.columns[1:]

    # Iterate over the rows of the .csv file
    for i, row in df.iterrows():
        # Get the gene name for this row
        gene_name = row[0]

        # Iterate over the columns of data
        for column_name in column_names:
            # Get the ranking for this column
            ranking = row[column_name]

            # Add the ranking to the list for this column
            rankings[column_name].append((gene_name, ranking))

    return rankings


def mask(data: np.ndarray, ranking: List[Tuple[str, int]], threshold: float) -> np.ndarray:
    """
    Mask data based on ranking data and a threshold.
    :param data: Data to be masked, 1 x P array
    :param ranking: Ranking data, list of tuples (gene_name, ranking)
    :param threshold: Threshold of the mask
    :return: Masked data
    """
    # Sort the ranking data by ranking in descending order
    ranking.sort(key=lambda x: x[1], reverse=True)

    # Take the top threshold percent of the ranking data
    threshold_index = int(len(ranking) * threshold)
    top_ranking_data = ranking[:threshold_index]

    # Mask the data if its gene name is in the top threshold percent of ranking data
    if data.gene_name in [x[0] for x in top_ranking_data]:
        data[:] = 0
    return data


def apply_mask(x: np.ndarray, y: List[str], pert_names: List[str], rankings: defaultdict, threshold: float) -> List[np.ndarray]:
    """
    Apply a mask to a batch of data based on ranking data for specified labels and a threshold.
    :param x: Batch of input data, genes x perturbations
    :param y: Batch of corresponding perturbation names
    :param rankings: Dict of ranking for each perturbation, generated based on a csv
    :param threshold: Threshold of the mask
    :return:
    """
    # Check that all strings in y are in rankings
    if not all(label in rankings for label in y):
        raise ValueError('One or more labels are not present in the rankings data.')

    # Initialize a list to store the masked data
    masked_x = []

    # Loop through the data points in x and y
    for i, (data, label) in enumerate(zip(x, y)):
        # Mask the data point based on the ranking data for the corresponding label
        masked_x.append(mask(data=data, pert_names=pert_names, rankings=rankings, labels=label, threshold=threshold))

    return masked_x
    
    
    
    
    I think there is a problem with the logic. This is your current code:

def generate_rankings(df: pd.DataFrame) -> Dict[str, List[Tuple[str, int]]]:
    """
    Generate a dictionary that maps each perturbation to a list of tuples where each tuple consists of
    a gene symbol and its ranking for the perturbation.
    :param df: DataFrame with gene symbols as rows and perturbation names as columns with rankings as values
    :return: Dictionary that maps perturbation names to lists of (gene symbol, ranking) tuples
    """
    rankings = {}
    for col in df.columns[1:]:
        rankings[col] = list(zip(df['gene_symbols'], df[col]))
        rankings[col].sort(key=lambda x: x[1])
    return rankings


def mask(data: np.ndarray, pert_names: List[str], ranking: List[Tuple[str, int]], label: str, threshold: float) -> np.ndarray:
    """
    Mask data based on ranking data and a threshold.
    :param data: Data to be masked, 1 x P array
    :param pert_names: Perturbation names, 1 x P array
    :param ranking: Ranking data, 1 x P array
    :param label: Current label
    :param threshold: Masking threshold
    :return: Masked data
    """
    # Find the top threshold percent of the ranking data
    top_ranking_data = ranking[:int(len(ranking) * threshold)]
    # Set the mask to zero for the top threshold percent of the ranking data
    mask = np.ones_like(data)
    mask[:, np.isin(pert_names, [x[0] for x in top_ranking_data])] = 0
    # Multiply the data by the mask to zero out the specified values
    return data * mask


def apply_mask(x: np.ndarray, y: np.ndarray, pert_names: List[str], rankings: Dict[str, List[Tuple[str, int]]], threshold: float) -> List[np.ndarray]:
    """
    Mask a batch of data based on ranking data and a threshold.
    :param x: Batch of data to be masked, N x P array
    :param y: Batch of labels, N x 1 array
    :param pert_names: Perturbation names, 1 x P array
    :param rankings: Ranking data, Dict[label, ranking data]
    :param threshold: Masking threshold
    :return: List of masked data
    """
    masked_x = []
    # Loop through the data points in x and y
    for i, (data, label) in enumerate(zip(x, y)):
        # Mask the data point based on the ranking data for the corresponding label
        masked_x.append(mask(data=data, pert_names=pert_names, ranking=rankings[label], label=label, threshold=threshold))
    return masked_x

I've added some prints in the mask function to make the problem clearer:
    print('Mask: ', mask, mask.shape)
    print('Data: ', data, data.shape)
    print('Pert names: ', pert_names, len(pert_names))
    print('Rop ranking data: ', top_ranking_data, len(top_ranking_data))
    print('Is in: ', np.isin(pert_names, [x[0] for x in top_ranking_data]))
For the toy example this resulted in
Mask:  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] (106,)
Data:  [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548  0.64589411
 0.43758721 0.891773   0.96366276 0.38344152 0.79172504 0.52889492
 0.56804456 0.92559664 0.07103606 0.0871293  0.0202184  0.83261985
 0.77815675 0.87001215 0.97861834 0.79915856 0.46147936 0.78052918
 0.11827443 0.63992102 0.14335329 0.94466892 0.52184832 0.41466194
 0.26455561 0.77423369 0.45615033 0.56843395 0.0187898  0.6176355
 0.61209572 0.616934   0.94374808 0.6818203  0.3595079  0.43703195
 0.6976312  0.06022547 0.66676672 0.67063787 0.21038256 0.1289263
 0.31542835 0.36371077 0.57019677 0.43860151 0.98837384 0.10204481
 0.20887676 0.16130952 0.65310833 0.2532916  0.46631077 0.24442559
 0.15896958 0.11037514 0.65632959 0.13818295 0.19658236 0.36872517
 0.82099323 0.09710128 0.83794491 0.09609841 0.97645947 0.4686512
 0.97676109 0.60484552 0.73926358 0.03918779 0.28280696 0.12019656
 0.2961402  0.11872772 0.31798318 0.41426299 0.0641475  0.69247212
 0.56660145 0.26538949 0.52324805 0.09394051 0.5759465  0.9292962
 0.31856895 0.66741038 0.13179786 0.7163272  0.28940609 0.18319136
 0.58651293 0.02010755 0.82894003 0.00469548 0.67781654 0.27000797
 0.73519402 0.96218855 0.24875314 0.57615733] (106,)
Pert names:  ['AHR' 'ARID1A' 'ARRDC3' 'ATL1' 'BAK1' 'BCL2L11' 'BCORL1' 'BPGM' 'C3orf72'
 'C19orf26' 'CBFA2T3' 'CBL' 'CDKN1A' 'CDKN1B' 'CDKN1C' 'CEBPA' 'CEBPB'
 'CEBPE' 'CELF2' 'CITED1' 'CKS1B' 'CLDN6' 'CNN1' 'CNNM4' 'COL1A1' 'COL2A1'
 'CSRNP1' 'DLX2' 'DUSP9' 'EGR1' 'ELMSAN1' 'ETS2' 'FEV' 'FOSB' 'FOXA1'
 'FOXA3' 'FOXF1' 'FOXL2' 'FOXO4' 'GLB1L2' 'HES7' 'HK2' 'HNF4A' 'HOXA13'
 'HOXB9' 'HOXC13' 'IER5L' 'IGDCC3' 'IKZF3' 'IRF1' 'ISL2' 'JUN' 'KIAA1804'
 'KIF2C' 'KIF18B' 'KLF1' 'KMT2A' 'LHX1' 'LYL1' 'MAML2' 'MAP2K3' 'MAP2K6'
 'MAP4K3' 'MAP4K5' 'MAP7D1' 'MAPK1' 'MEIS1' 'MIDN' 'NCL' 'NIT1' 'OSR2'
 'PLK4' 'POU3F2' 'PRDM1' 'PRTG' 'PTPN1' 'PTPN9' 'PTPN12' 'PTPN13' 'RHOXF2'
 'RREB1' 'RUNX1T1' 'S1PR2' 'SAMD1' 'SET' 'SGK1' 'SLC4A1' 'SLC6A9'
 'SLC38A2' 'SNAI1' 'SPI1' 'STIL' 'TBX2' 'TBX3' 'TGFBR2' 'TMSB4X' 'TP73'
 'TSC22D1' 'UBASH3A' 'UBASH3B' 'ZBTB1' 'ZBTB10' 'ZBTB25' 'ZC3HAV1'
 'ZNF318' 'control'] 106
Rop ranking data:  [('RP11-206L10.9', 1094), ('RP5-857K21.4', 1462), ('LINC00115', 1911), ('FO538757.2', 5670), ('RP11-34P13.3', 7589)] 5
Is in:  [False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False]

You see that in `top_ranking_data`, we have the 5 of 10 genes we want to mask - this is correct. The remaining code erroneously tries to mask 50% of the perturbations. The logic should rather be as follows:
1. You get a batch (x, y)
2. For each element i, x_i should have the 10 genes and y_i should indicate the perturbation, that is a str
3. The mask() function should then mask 50% of the 10 genes in x_i according to the ranking that is found from the generate_rankings() functions and specified with the perturbation of y_i
Can you update the functions for that purpose?
    
    
    
    
    
    
    
    
    
    
    
    
'''