# tSNE on the principal components obtained from filtered (activated cells + 2/4 yrs) transcript data
import pandas as pd
from sklearn import datasets
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('..\\..\\src\\util\\')
from helper_functions import filter_samples
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

path_to_save_figures = '..\\..\\fig\\supp_fig\\tSNE\\' # for github
name_of_run = 'tSNE_allgenes_activatedonly' #_activatedonly'

df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')

df = filter_samples(df, annotation_df, 'Sample_characteristics_ch1_age_yrs', (2,4))
#df = filter_samples(df, annotation_df, 'Sample_characteristics_ch1_activation_status', 1)
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


tsne_features = TSNE(n_components=2).fit_transform(x_scaled)

tsne_df = pd.DataFrame(tsne_features, columns=["X","Y"])

print('Shape before tSNE: ', x_scaled.shape)
print('Shape after tSNE: ', tsne_df.shape)


# add allergy status to the PCA dataframe
annotation_df_samples_to_keep = df.columns
allergy_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_allergy_status']
allergy_status_df = allergy_status_df.reindex(columns = annotation_df_samples_to_keep)

# add activation status to the PCA dataframe
annotation_df_samples_to_keep = df.columns
activation_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_activation_status']
activation_status_df = activation_status_df.reindex(columns = annotation_df_samples_to_keep)
 
tsne_df['allergy status'] = allergy_status_df.values[0] #check the order of the labels
tsne_df['activation status'] = activation_status_df.values[0] #check the order of the labels

target_names = {
    'control':0,
    'allergic':1, 
    'resolved':2
}
tsne_df['allergy_status_numerical'] = tsne_df['allergy status'].map(target_names)


tsne_df.to_pickle('..\\..\\data\\results\\' + name_of_run + '.pkl')

sns.set()
sns.lmplot(
    x='X', 
    y='Y', 
    data=tsne_df,
    hue='allergy status', 
    fit_reg=False, 
    legend=True
    )
#plt.title('2D PCA Graph')
plt.savefig(path_to_save_figures + name_of_run +'colorbyallergy', dpi=600)
plt.show()

sns.lmplot(
    x='X', 
    y='Y', 
    data=tsne_df,
    hue='activation status', 
    fit_reg=False,
    legend=True
    )
#plt.title('2D PCA Graph')
plt.savefig(path_to_save_figures + name_of_run, dpi=600)

plt.show()
