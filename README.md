# gene_func_pred_svm
本模型运用基因注释区分转录因子和功能基因，再运用SVM进行普通基因、重要基因和关键基因的分类。分类器对于1类基因，即普通基因，有着较好的预测效果；而对于2类基因和3类基因，即重要基因和关键基因，的预测仍有所不足。这很大程度上是因为2类和3类基因“同源”且3类基因过少导致。若要进一步改进分类器，需要更大的数据集和更加科学的分类标签。