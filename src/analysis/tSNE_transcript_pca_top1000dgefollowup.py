# tSNE on the principal components obtained from filtered (activated cells + 2/4 yrs) transcript data
import pandas as pd
from sklearn import datasets
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

path_to_save_figures = '..\\..\\fig\\supp_fig\\tSNE\\' # for github
name_of_run = 'activatedonly_top1000DGE_allcomponents' #_activatedonly'


# import df containing all principal components (generated in transcript_pca.py)
pca_df = pd.read_pickle('..\\..\\data\\results\\pca_df_activatedonly_top1000DGE_allcomponents.pkl')
pca_df.head(100)

# run tSNE on the PCs
pca_array = pca_df.drop(['allergy status', 'activation status', 'allergy_status_numerical'], axis=1, inplace=False).to_numpy()

tsne_features = TSNE(n_components=2).fit_transform(pca_array)

tsne_df = pd.DataFrame(tsne_features, columns=["X","Y"])
tsne_df['allergy status'] = pca_df['allergy status']

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
plt.savefig(path_to_save_figures + name_of_run)
plt.show()
