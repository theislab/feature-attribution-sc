import pandas as pd
import seaborn as sb
import scanpy as sc
import matplotlib.pyplot as plt 
import numpy as np


class AttrAnalysis():
    def __init__(self, data_path: str):
        self.df = self.read_data(data_path)
        
    def read_data(self, data_path) -> pd.DataFrame:
        df = pd.read_csv(data_path, index_col = 0)
        genes = df.shape[0]
        cell_types = df.shape[1]
        print(f"The loaded data has {genes} genes and {cell_types} cell types")
        return df
    
    
    def get_attr_sign_count(self):
        df = self.df
        cell_type_names = df.columns.to_list()
        attribution_count_df = pd.DataFrame(index=cell_type_names,columns=["negative","zero","postive"])
        i = 0
        for name in cell_type_names:
            attribution_count_df.iloc[i,:] = [(df[name] < 0).sum(), (df[name] == 0).sum(), (df[name] > 0).sum()]
            i+=1
        return attribution_count_df
    
    def attr_sign_heat_map(self):
        sb.heatmap(self.df, mask = (self.df>0) | (self.df<0), cbar = False, cmap = 'autumn')
    
    def get_attr_sum(self):
        df = self.df
        cell_type_names = df.columns.to_list()
        attribution_sum_df = pd.DataFrame(index=cell_type_names,columns=["negative_sum","positive_sum"], dtype='float64')

        i = 0
        for name in cell_type_names:
            attribution_sum_df.iloc[i,:] = [df[df[name]<0][name].sum(), self.df[self.df[name]>0][name].sum()]
            i+=1


        attribution_sum_df = attribution_sum_df.sort_values(by = 'positive_sum')
        attribution_sum_df = attribution_sum_df.reset_index()
        attribution_sum_df = attribution_sum_df.rename(columns = {'index':'cell_type'})
        return attribution_sum_df
    
    def attr_sum_barchart(self):
        attribution_sum_df = self.get_attr_sum()
        x = pd.melt(attribution_sum_df,id_vars = attribution_sum_df.columns[0] ,value_vars = attribution_sum_df.columns[1:] ,var_name="source", value_name="value_numbers")
        sb.barplot(x = x.cell_type, y = x.value_numbers, hue = x.source)
        plt.xticks(rotation = 90);

        
    def get_highest_attr_genes(self, highest: int=20):
        df = self.df
        cell_types_imp_genes = {}
        for col in df.columns:
            index_sorted = df[df[col]>0][col].sort_values(ascending = False).index
            cell_types_imp_genes[col] = index_sorted[:highest]
        cell_types_imp_genes_df = pd.DataFrame.from_dict(cell_types_imp_genes)
        return cell_types_imp_genes_df
