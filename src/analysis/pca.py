import pandas as pd
from sklearn import datasets
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# read in RNA seq data
df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')



# prepare feature data
df = df.drop('Gene', axis=1)
X = df.to_numpy() 
X = X.transpose() # X is an ndarray of just features (134 samples x 18864 genes)
X.shape

# filter out samples to just have certain timepoints? or only reducing the number of genes?
# paper does PCA with 4154 genes and 558 methylation sites- filter first?



# data scaling
x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)
print(x_scaled)



# run PCA
pca = PCA(n_components=0.8) # n_components is the number of top components to keep, or if <1 is the percent of variance we want explained
pca_features = pca.fit_transform(x_scaled)
print('Shape before PCA: ', x_scaled.shape)
print('Shape after PCA: ', pca_features.shape)
 
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC'+str(i+1) for i in range(0, pca_features.shape[1])]
    )

pca_df.head()




# plot explained variance of each PC
variance = pca.explained_variance_ # list with the explained variance of each PC
#sns.set() # set seaborn theme for plotting
plt.bar(range(1,len(variance)+1), variance)
plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Feature Explained Variance')
plt.show()

variance_pct = pca.explained_variance_ratio_ # list with the explained variance of each PC
#sns.set() # set seaborn theme for plotting
plt.bar(range(1,len(variance_pct)+1), variance_pct*100)
plt.xlabel('PCA Feature')
plt.ylabel('Percent of variance explained')
plt.title('Feature Explained Variance')
plt.show()




# add targets to the PCA dataframe
annotation_df_samples_to_keep = df.columns
allergy_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_allergy_status']
allergy_status_df = allergy_status_df.reindex(columns = annotation_df_samples_to_keep)
 
pca_df['target'] = allergy_status_df.values[0] #check the order of the labels


# plot the first 2 PCs
sns.set()
sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df, 
    hue='target', 
    fit_reg=False, 
    legend=True
    )
plt.title('2D PCA Graph')
plt.show()


# plot first 3 features
