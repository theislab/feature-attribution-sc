import os
import pandas as pd
import seaborn as sb
import scanpy as sc
import matplotlib.pyplot as plt 
import numpy as np


class AttrAnalysis():
    def __init__(self, data_path: str):
        self.df = self.read_data(data_path)
        
    def read_data(self, data_path) -> pd.DataFrame:
        """
        Params
        ------
        data_path: str
            a path to the csv data file that contains the attribution scores.
            the general shape is advised to be (genes, cell_types)

        Returns
        -------
        df : pd.DataFrame
            a data frame that was loaded from the data_path
        """
        df = pd.read_csv(data_path, index_col = 0)
        genes = df.shape[0]
        cell_types = df.shape[1]
        print(f"The loaded data has {genes} genes and {cell_types} cell types")
        return df
    
    
    def get_attr_sign_count(self) -> pd.DataFrame:
        """
        a method for counting the attribution score count per column in the input data

        Returns
        -------
        df : pd.DataFrame
            dataframe of shape: (columns, 3), with columns: [negative, zero, positive] per column of the input data
        """
        df = self.df
        columns_names = df.columns.to_list()
        attribution_count_df = pd.DataFrame(index=columns_names, columns=["negative","zero","postive"])
        for idx, name in enumerate(columns_names):
            attribution_count_df.iloc[idx,:] = [(df[name] < 0).sum(), (df[name] == 0).sum(), (df[name] > 0).sum()]
        
        return attribution_count_df
    
    def attr_sign_heat_map(self):
        """
        a method for creating a heat map for the attribution score counts, where the the a cell in the map
        is colored if either has postive or negative value.
        """
        map_mask = (self.df>0) | (self.df<0)
        return sb.heatmap(self.df, mask = map_mask, cbar = False, cmap = 'autumn')
    
    def get_attr_sum(self) -> pd.DataFrame:
        """
        a method for getting the total sum of the positive and negative attribution score per column

        Returns
        -------
        df : pd.DataFrame
            dataframe of shape (columns, 2), with columns: [positive_sum, negative_sum] per column of the input data
        """
        df = self.df
        columns_names = df.columns.to_list()
        attribution_sum_df = pd.DataFrame(index=columns_names,columns=["negative_sum","positive_sum"], dtype='float64')

        for idx, name in enumerate(columns_names):
            attribution_sum_df.iloc[idx,:] = [df[df[name]<0][name].sum(), self.df[self.df[name]>0][name].sum()]


        attribution_sum_df = attribution_sum_df.sort_values(by = 'positive_sum')
        attribution_sum_df = attribution_sum_df.reset_index()
        attribution_sum_df = attribution_sum_df.rename(columns = {'index':'cell_type'})
        return attribution_sum_df
    
    def attr_sum_barchart(self):
        """
        a method for creating a barchart for the total sum of the positive and negative attribution scores.
        """
        attribution_sum_df = self.get_attr_sum()
        x = pd.melt(attribution_sum_df,id_vars = attribution_sum_df.columns[0] ,value_vars = attribution_sum_df.columns[1:] ,var_name="source", value_name="value_numbers")
        sb.barplot(x = x.cell_type, y = x.value_numbers, hue = x.source)
        plt.xticks(rotation = 90);

        
    def get_highest_attr_genes(self, highest: int=20) -> pd.DataFrame:
        """
        a method for getting the highest rows values per column.
        Can be used to get the genes with highest attribution score per cell if the input data is [rows: genes, columns: cell_types]


        Params
        ------
        highest: int (default: 20)
            the number of rows to be considered in the output dataframe.

        Returns
        -------
        df : pd.DataFrame
            dataframe of shape (highest, columns), where the rows are the rows of the highest values per columns.
        """
        df = self.df
        cell_types_imp_genes = {}
        for col in df.columns:
            index_sorted = df[df[col]>0][col].sort_values(ascending = False).index
            cell_types_imp_genes[col] = index_sorted[:highest]
        cell_types_imp_genes_df = pd.DataFrame.from_dict(cell_types_imp_genes)
        return cell_types_imp_genes_df
