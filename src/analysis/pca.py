import pandas as pd
from sklearn import datasets

# read in RNA seq data
df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')


df = df.drop('Gene', axis=1)
X = df.to_numpy() # X is an ndarray of just features (18864 genes x 134 samples)
X = X.transpose()
X.shape

# filter out samples to just have certain timepoints? or only reducing the number of genes?
# paper does PCA with 4154 genes and 558 methylation sites
