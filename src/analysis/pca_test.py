# trying to do PCA following a tutorial first: https://www.jcchouinard.com/pca-with-python/
import pandas as pd
from sklearn import datasets
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# read in data
iris = datasets.load_iris()
 
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}
 
df = pd.DataFrame(
    iris.data, 
    columns=iris.feature_names
    )
 
df['target'] = iris.target
df['target_names'] = df['target'].map(target_names)
df.head(10)

X = iris.data
y = iris.target
print(X)



# data scaling
x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)
print(x_scaled)



# run PCA
pca = PCA(n_components=4) # n_components is the number of top components to keep, or if <1 is the percent of variance we want explained
pca_features = pca.fit_transform(x_scaled)
print('Shape before PCA: ', x_scaled.shape)
print('Shape after PCA: ', pca_features.shape)
 
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC1', 'PC2', 'PC3', 'PC4'])

pca_df.head()



# add targets to the PCA dataframe
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}
 
pca_df['target'] = y
pca_df['target'] = pca_df['target'].map(target_names)
 


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
plt.bar(range(1,len(variance_pct)+1), variance_pct)
plt.xlabel('PCA Feature')
plt.ylabel('Percent of variance explained')
plt.title('Feature Explained Variance')
plt.show()



# repeat and plot just top two components (~95% of the variance)
# Reduce from 4 to 2 features with PCA
pca = PCA(n_components=0.95)
# Fit and transform data
pca_features = pca.fit_transform(x_scaled)
# Create dataframe
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC1', 'PC2'])
# map target names to PCA features   
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}
pca_df['target'] = y
pca_df['target'] = pca_df['target'].map(target_names)
pca_df.head()
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