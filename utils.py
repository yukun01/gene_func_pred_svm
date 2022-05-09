"""
some utils
"""

import pandas as pd
import numpy as np


class Utils:
    def __init__(self, gene_data):
        self.gene_data = gene_data
        self.gene_data_dataframe = None
        self.average = None
        self.std = None
        self.z_score = None
        self._convert_to_dataframe()

    def _convert_to_dataframe(self):
        gene_data_dataframe = {}
        for gene_id in self.gene_data:
            if not self.gene_data[gene_id]['gene_expression_profile']:
                continue
            gene_data_dataframe[gene_id] = self.gene_data[gene_id]['gene_expression_profile']
        gene_data_dataframe = pd.DataFrame(gene_data_dataframe, dtype=np.float64).transpose()
        self.gene_data_dataframe = gene_data_dataframe

    def calculate_average(self):
        self.average = self.gene_data_dataframe.mean()
        return self.gene_data_dataframe.mean()

    def calculate_standard_deviation(self):
        self.std = self.gene_data_dataframe.std(ddof=1)
        return self.gene_data_dataframe.std(ddof=1)

    def calculate_z_score(self):
        self.z_score = \
            (self.gene_data_dataframe - self.gene_data_dataframe.mean()) / self.gene_data_dataframe.std(ddof=1)
        return self.z_score


if __name__ == '__main__':
    gene_data = np.load('./file/gene_data.npy', allow_pickle=True).item()

    zcore = Utils(gene_data)
    zcore.calculate_average()
    zcore.calculate_standard_deviation()
    zcore.calculate_z_score()
    zcore.gene_data_dataframe.to_csv('./gene_data.csv')
    for i in range(1, 12):
        print(i)
        print(zcore.z_score[zcore.z_score >= 3].count(axis=1)[zcore.z_score[zcore.z_score >= 3].count(axis=1) == i].index)
        print('')
