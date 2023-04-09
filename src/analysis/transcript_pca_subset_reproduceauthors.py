# PCA analysis on a subset of the RNAseq data
# as of 4/5, are using the 4154 genes identified by the authors


import pandas as pd
from sklearn import datasets
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('..\\..\\src\\util\\')
from helper_functions import filter_samples

# read in RNA seq data
#os.chdir("C:\\Users\\myrad\\20440-project")
df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')


##### unique to the subset version of the script #####
# read in the subset of genes identified by the authors
gene_subset = pd.read_csv('..\\..\\data\\raw\\supplementary_data\\4154_diff_expressed_genes_supp1.csv', header=1)
gene_subset.head()

# filter the RNAseq data
df = df[df['Gene'].isin(gene_subset['transcript ID'])]
#####

df = filter_samples(df, annotation_df, 'Sample_characteristics_ch1_age_yrs', 1)

display(annotation_df)

# prepare feature data
df = df.drop('Gene', axis=1)
X = df.to_numpy() 
X = X.transpose() # X is an ndarray of just features (134 samples x 4154 genes)
X.shape


# data scaling
x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)
print(x_scaled)



# run PCA
pca = PCA(n_components=11) # n_components is the number of top components to keep, or if <1 is the percent of variance we want explained
pca_features = pca.fit_transform(x_scaled)
print('Shape before PCA: ', x_scaled.shape)
print('Shape after PCA: ', pca_features.shape)
 
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC'+str(i+1) for i in range(0, pca_features.shape[1])]
    )

pca_df.head()

path_to_save_figures = '..\\..\\fig\\supp_fig\\PCA\\reproducing_authors_PCA\\' # for github

#try:
#    os.chdir(path_to_save_figures)
#except:
#    os.mkdir(path_to_save_figures)
print(path_to_save_figures + "PCA_11components_explainedvariance")

# plot explained variance of each PC
variance = pca.explained_variance_ # list with the explained variance of each PC
#sns.set() # set seaborn theme for plotting
plt.bar(range(1,len(variance)+1), variance)
plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Feature Explained Variance')
plt.savefig(path_to_save_figures + "PCA_11components_explainedvariance.png")
plt.show()

variance_pct = pca.explained_variance_ratio_ # list with the explained variance of each PC
#sns.set() # set seaborn theme for plotting
plt.bar(range(1,len(variance_pct)+1), variance_pct*100)
plt.xlabel('PCA Feature')
plt.ylabel('Percent of variance explained')
plt.title('Feature Explained Variance')
plt.savefig(path_to_save_figures + "PCA_11components_explainedvariancepct.png")
plt.show()



# add allergy status to the PCA dataframe
annotation_df_samples_to_keep = df.columns
allergy_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_allergy_status']
allergy_status_df = allergy_status_df.reindex(columns = annotation_df_samples_to_keep)
 
pca_df['allergy status'] = allergy_status_df.values[0] #check the order of the labels

target_names = {
    'control':0,
    'allergic':1, 
    'resolved':1
}
#pca_df['target_numerical'] = pca_df['allergy status'].map(target_names)

# add a column for activation_status values to pca_df:
annotation_df_samples_to_keep = df.columns
activation_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_activation_status']
activation_status_df = activation_status_df.reindex(columns = annotation_df_samples_to_keep)
pca_df['activation status'] = activation_status_df.values[0] #check the order of the labels



pca_df.to_pickle('..\\..\\data\\results\\reproducing_authors_pca_df_11components.pkl')

pca_df
# plot the first 2 PCs
sns.set()
lm = sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df, 
    hue='allergy status',
    col='activation status',
    fit_reg=False, 
    legend=True
    )
#plt.title('2D PCA Graph')
axes = lm.axes
#axes[0,0].set_xlim(-65,65)
#axes[0,1].set_ylim(-20,30)

plt.savefig(path_to_save_figures + "PCA1_vs_PCA2_colorbyallergystatus")
plt.show()

# plot first 3 features
# plt.style.use('default')
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['target_numerical'], cmap='viridis')
# plt.title(f'3D Scatter of RNAseq Data')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# plt.savefig(path_to_save_figures + "PCA1_vs_PCA2_vs_PCA3_colorbyallergystatus")
# plt.show()

sns.pairplot(pca_df, hue='target')
plt.savefig(path_to_save_figures + "all_PCA_comparisons_colorbyallergystatus")
plt.show()

