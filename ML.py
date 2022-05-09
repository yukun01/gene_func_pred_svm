"""
Machine learning
"""

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
import joblib

import copy
import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import Utils


class FeatureEncoding:
    def __init__(self, gene_data):
        self.gene_data = gene_data
        self.gene_data_dataframe = Utils(self.gene_data).gene_data_dataframe
        self.feature = None

        self._initialize_feature()

    def _initialize_feature(self):
        self.feature = Utils(self.gene_data).gene_data_dataframe
        self.feature.reset_index(inplace=True)
        self.feature.rename(columns={'index': 'geneID'}, inplace=True)

    def z_score(self):
        """
        calculate the number of tissues which Z-score >= 3 of each gene
        """
        z_score = Utils(self.gene_data).calculate_z_score()
        tissue_nums = z_score.shape[1]
        z_score_count = pd.DataFrame()

        for i in range(tissue_nums + 1):
            genes = list(z_score[z_score >= 3].count(axis=1)[z_score[z_score >= 3].count(axis=1) == i].index)

            z_score_count_temp = pd.DataFrame(
                dict(zip(list(range(len(genes))), [{'geneID': j, 'z_score_count': i} for j in genes]))
            ).transpose()
            z_score_count = pd.concat([z_score_count, z_score_count_temp], ignore_index=True)
        self.feature = pd.merge(self.feature, z_score_count, on=['geneID'])


class MLVisualize:
    def __init__(self,
                 model,
                 X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.DataFrame,
                 y_test: pd.DataFrame,
                 y_pred,
                 savedir: str,
                 show: bool = False):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.savedir = savedir
        self.show = show
        self.X = pd.concat([X_train, X_test], ignore_index=True)
        self.y = pd.concat([y_train, y_test], ignore_index=True)

    def roc(self):
        skplt.metrics.plot_roc(self.y_test.astype(int), self.y_pred)
        plt.savefig(f'{self.savedir}/roc.pdf')
        if self.show:
            plt.show()

    def pr_curve(self):
        skplt.metrics.plot_precision_recall_curve(self.y_test.astype(int), self.y_pred, cmap='nipy_spectral')
        plt.savefig(f'{self.savedir}/pr_curve.pdf')
        if self.show:
            plt.show()

    def ks_statistic(self):
        """
        only for binary classification
        """
        y_pred_all = self.model.predict_proba(self.X)
        skplt.metrics.plot_ks_statistic(self.y, y_pred_all)
        plt.savefig(f'{self.savedir}/ks_statistic.pdf')
        if self.show:
            plt.show()

    def confusion_matrix(self):
        predictions = cross_val_predict(self.model, self.X, self.y.astype(int))
        skplt.metrics.plot_confusion_matrix(self.y.astype(int), predictions, normalize=True)
        plt.savefig(f'{self.savedir}/confusion_matrix.pdf')
        if self.show:
            plt.show()

    def learning_curve(self):
        """
        this function may cost a lot of time
        """
        skplt.estimators.plot_learning_curve(self.model, self.X, self.y.astype(int))
        plt.savefig(f'{self.savedir}/learning_curve.pdf')
        if self.show:
            plt.show()

    def lift_curve(self):
        """
        only for binary classification
        """
        skplt.metrics.plot_lift_curve(self.y_test.astype(int), self.y_pred)
        plt.savefig(f'{self.savedir}/lift_curve.pdf')
        if self.show:
            plt.show()

    def cumulative_gain(self):
        """
        only for binary classification
        """
        skplt.metrics.plot_cumulative_gain(self.y_test.astype(int), self.y_pred)
        plt.savefig(f'{self.savedir}/cumulative_gain.pdf')
        if self.show:
            plt.show()

    def feature_importance(self):
        model = RandomForestClassifier()
        model.fit(self.X, self.y)
        skplt.estimators.plot_feature_importances(model)
        plt.savefig(f'{self.savedir}/feature_importance.pdf')
        if self.show:
            plt.show()


class SVM:
    def __init__(self,
                 feature: pd.DataFrame,
                 mark: pd.DataFrame,
                 pca: bool = False,
                 n_components: int = None,
                 draw_pca_explained_variance_ratio: bool = False,
                 scale: bool = False):
        self.feature = feature
        self.mark = mark
        self.pca = pca
        self.n_components = n_components
        self.draw_pca_explained_variance_ratio = draw_pca_explained_variance_ratio
        self.scale = scale

        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.y_pred = None
        self.clf = None

        self._preprocessing()

    def _preprocessing(self):
        gene_id = self.feature['geneID']
        self.feature.set_index('geneID', inplace=True)

        if self.scale:
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.feature)
            self.feature = scaler.transform(self.feature)

        if self.pca:
            pca = PCA(n_components=self.n_components)
            pca.fit(self.feature)
            original_data = copy.deepcopy(self.feature)
            self.feature = pca.transform(self.feature)
            explained_variance_ratio = pca.explained_variance_ratio_
            print(explained_variance_ratio)

            if self.draw_pca_explained_variance_ratio:
                plt.figure()
                plt.bar([f'PC{i}' for i in range(1, len(explained_variance_ratio)+1)], explained_variance_ratio)
                plt.show()

        self.feature = pd.DataFrame(self.feature)
        self.feature['geneID'] = gene_id

        self.feature = pd.merge(self.feature, self.mark, on=['geneID'])
        self.feature.set_index('geneID', inplace=True)

        self.mark = self.feature['mark']
        del self.feature['mark']

        if self.pca and self.draw_pca_explained_variance_ratio:
            skplt.decomposition.plot_pca_2d_projection(pca, original_data, self.mark.astype(int))
            plt.show()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature, self.mark, test_size=0.5)

    def ml(self):
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(self.X_train, self.y_train.astype(int))
        self.clf = clf
        self.y_pred = clf.predict_proba(self.X_test)

    def visualize(self, savedir):
        visualize = MLVisualize(self.clf,
                                self.X_train,
                                self.X_test,
                                self.y_train,
                                self.y_test,
                                self.y_pred,
                                savedir=savedir)
        visualize.roc()
        visualize.pr_curve()
        visualize.confusion_matrix()
        # visualize.learning_curve()
        visualize.feature_importance()


if __name__ == '__main__':
    gene_data = np.load('./file/gene_data.npy', allow_pickle=True).item()

    feature = FeatureEncoding(gene_data)
    feature.z_score()
    test_feature = feature.feature
    mark = pd.read_table('./file/mark.txt')
    print(test_feature)
    svm = SVM(test_feature,
              mark,
              scale=True,
              pca=True,
              n_components=5,
              draw_pca_explained_variance_ratio=False)
    svm.ml()
    svm.visualize('./res/improve')
