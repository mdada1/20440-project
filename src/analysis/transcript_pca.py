# PCA analysis on all RNAseq data
# filter to only include later time point (2/4 yrs)
# also filter to only use activated cells- do both ways and pick result later

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
#os.chdir("C:\\Users\\myrad\\20440-project\\src\\analysis")

df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')

df = filter_samples(df, annotation_df, 'Sample_characteristics_ch1_age_yrs', (2,4))
df = filter_samples(df, annotation_df, 'Sample_characteristics_ch1_activation_status', 1)
df

#display(annotation_df)
# prepare feature data
df = df.drop('Gene', axis=1)
X = df.to_numpy() 
X = X.transpose() # X is an ndarray of just features (50 or 26 samples x 18864 genes)
X.shape


# data scaling
x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)
print(x_scaled)



# run PCA
pca = PCA(n_components=0.99999999999999) # n_components is the number of top components to keep, or if <1 is the percent of variance we want explained
pca_features = pca.fit_transform(x_scaled)
print('Shape before PCA: ', x_scaled.shape)
print('Shape after PCA: ', pca_features.shape)
 
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC'+str(i+1) for i in range(0, pca_features.shape[1])]
    )

pca_df.head()

path_to_save_figures = '..\\..\\fig\\supp_fig\\PCA\\our_analysis-transcriptPCA\\' # for github
name_of_PCA_run = 'activatedonly_allcomponents_saveactstatus' #_activatedonly'


print(path_to_save_figures + "PCA_" + name_of_PCA_run + "_explainedvariance")

# plot explained variance of each PC
variance = pca.explained_variance_ # list with the explained variance of each PC
#sns.set() # set seaborn theme for plotting
plt.bar(range(1,len(variance)+1), variance)
plt.xlabel('PCA Feature')
plt.ylabel('Explained Variance')
plt.title('Explained Variance of Top PCA Components')
plt.savefig(path_to_save_figures + "PCA_" + name_of_PCA_run + "_explainedvariance.png")
plt.show()

variance_pct = pca.explained_variance_ratio_ # list with the explained variance of each PC
#sns.set() # set seaborn theme for plotting
plt.bar(range(1,len(variance_pct)+1), variance_pct*100)
plt.xlabel('PCA Feature')
plt.ylabel('Percent of Variance Explained')
plt.title('Explained Variance of Top PCA Components')
plt.savefig(path_to_save_figures + "PCA_" + name_of_PCA_run + "_explainedvariancepct.png")
plt.show()



# add allergy status to the PCA dataframe
annotation_df_samples_to_keep = df.columns
allergy_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_allergy_status']
allergy_status_df = allergy_status_df.reindex(columns = annotation_df_samples_to_keep)

# add activation status to the PCA dataframe
annotation_df_samples_to_keep = df.columns
activation_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_activation_status']
activation_status_df = activation_status_df.reindex(columns = annotation_df_samples_to_keep)
 
pca_df['allergy status'] = allergy_status_df.values[0] #check the order of the labels
pca_df['activation status'] = activation_status_df.values[0] #check the order of the labels

target_names = {
    'control':0,
    'allergic':1, 
    'resolved':2
}
pca_df['allergy_status_numerical'] = pca_df['allergy status'].map(target_names)

pca_df.to_pickle("..\\..\\data\\results\\pca_df_" + name_of_PCA_run + ".pkl")


# plot the first 2 PCs
sns.set()
sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df, 
    hue='activation status', 
    fit_reg=False, 
    legend=True
    )
#plt.title('2D PCA Graph')
plt.savefig(path_to_save_figures + name_of_PCA_run + "-PCA1_vs_PCA2_colorbyallergystatus")
plt.show()


# same plot as above but with legend underneath
sns.set()
p = sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df, 
    hue='allergy status', 
    fit_reg=False, 
)
p.fig.legend(loc='lower center', ncol=2, title='Allergy Status', bbox_to_anchor=(0.475, -0.1))




# plot first 3 features
# plt.style.use('default')
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['allergy_status_numerical'], cmap='viridis')
# plt.title(f'3D Scatter of RNAseq Data')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# plt.savefig(path_to_save_figures + "PCA1_vs_PCA2_vs_PCA3_colorbyallergystatus")
# plt.show()

sns.pairplot(pca_df, hue='allergy status')
plt.savefig(path_to_save_figures + name_of_PCA_run + "-all_PCA_comparisons_colorbyallergystatus")
plt.show()


