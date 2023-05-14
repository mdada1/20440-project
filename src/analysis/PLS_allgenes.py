# PLS-DA analysis on all RNAseq data
# filter to only include later time point (2/4 yrs)
# also filter to only use activated cells?

import pandas as pd
import numpy as np
from sklearn import datasets
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append('..\\..\\src\\util\\')
from helper_functions import filter_samples
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# read in RNA seq data
#os.chdir("C:\\Users\\myrad\\20440-project\\src\\analysis")


### PLS FOR DIMENSIONALITY REDUCTION- ALL POINTS

df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')

df = filter_samples(df, annotation_df, 'Sample_characteristics_ch1_age_yrs', (2,4))
df = filter_samples(df, annotation_df, 'Sample_characteristics_ch1_activation_status', 1)
df


# add allergy status to the PCA dataframe
annotation_df_samples_to_keep = df.columns
allergy_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_allergy_status']
allergy_status_df = allergy_status_df.reindex(columns = annotation_df_samples_to_keep)

# add activation status to the PCA dataframe
annotation_df_samples_to_keep = df.columns
activation_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_activation_status']
activation_status_df = activation_status_df.reindex(columns = annotation_df_samples_to_keep)
 
label_df = pd.DataFrame()
label_df['allergy status'] = allergy_status_df.values[0] #check the order of the labels
label_df['activation status'] = activation_status_df.values[0] #check the order of the labels

target_names = {
    'control':2,
    'allergic':0, 
    'resolved':1
}
label_df['allergy_status_numerical'] = label_df['allergy status'].map(target_names)
label_df = label_df.drop(0)

#display(annotation_df)
# prepare feature data
df = df.drop('Gene', axis=1)
X = df.to_numpy() 
X = X.transpose() # X is an ndarray of just features (50 or 26 samples x 18864 genes)
X.shape


# data scaling
x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)
print(x_scaled)

y = label_df['allergy_status_numerical'].to_numpy()

# run PLS
plsr = PLSRegression(n_components=2, scale=False) 
plsr.fit(x_scaled, y) 


plsr_df = pd.DataFrame(plsr.x_scores_)
plsr_df.index=label_df['allergy status']
plsr_df.columns = ['Latent Variable 1', 'Latent Variable 2']
plsr_df = plsr_df.reset_index()

# get variance explained
variance_in_x = np.var(plsr.x_scores_, axis = 0) 
fractions_of_explained_variance = variance_in_x / np.sum(variance_in_x)


path_to_save_figures = '..\\..\\fig\\supp_fig\\PLS\\' # for github
name_of_run = 'PLS_allgenes_activatedonly' 

sns.set()
plt.figure(figsize=(8, 10))
ax = sns.lmplot(
    x='Latent Variable 1', 
    y='Latent Variable 2', 
    data=plsr_df, 
    hue='allergy status', 
    fit_reg=False, 
    #legend=True
    )
ax.fig.legend(loc='lower center', ncol=2, title='Allergy Status', bbox_to_anchor=(0.475, -0.1))
#p.set(xlabel='LV1 (30.4% Variance)', ylabel='LV2 (20.1% Variance)')
ax.set(xlabel=f'LV1 ({fractions_of_explained_variance[0]:.1%} Variance)'
                    , ylabel=f'LV2 ({fractions_of_explained_variance[1]:.1%} Variance)')

plt.savefig(path_to_save_figures + name_of_run + "_colorbyallergystatus", dpi=600)
plt.show()

plsr_df.to_pickle("..\\..\\data\\results\\pls_df_" + name_of_run + ".pkl")


### PERFORM CLASSIFICATION AND CROSS-VALIDATION 


def pls_da(X_train, y_train, X_test):

    plsr = PLSRegression(n_components=2, scale=False)
    plsr.fit(X_train, y_train)

    y_pred = plsr.predict(X_test)[:,0]
    pred = (plsr.predict(X_test)[:,0] > 0.5).astype('uint8')
   # print(y_pred, pred)
    return pred, y_pred



# determine best number of folds

avg_accuracies = []
avg_precisions = []
avg_recalls = []
avg_f1s = []
num_folds = range(2, 10)

for i in range(2, 10):

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    cval = StratifiedKFold(n_splits=i, shuffle=True, random_state=19)
    for train, test in cval.split(x_scaled, y):
        
        y_pred, y_pred_class = pls_da(x_scaled[train,:], y[train], x_scaled[test,:])
        
        accuracies.append(accuracy_score(y[test], y_pred))
        precisions.append(precision_score(y[test], y_pred))
        recalls.append(recall_score(y[test], y_pred))
        f1s.append(f1_score(y[test], y_pred))
    
    #print("Average accuracy: ", np.array(accuracies).mean())
    avg_accuracies.append(np.array(accuracies).mean())
    avg_precisions.append(np.array(precisions).mean())
    avg_recalls.append(np.array(recalls).mean())
    avg_f1s.append(np.array(f1s).mean())

plt.plot(num_folds, avg_accuracies, label='Avg. Accuracy')
plt.plot(num_folds, avg_precisions, label='Avg. Precision')
plt.plot(num_folds, avg_recalls, label='Avg. Recall')
plt.plot(num_folds, avg_f1s, label='Avg. F1 Score')
plt.xlabel('Number of Folds in K-Fold Cross-Validation')
plt.ylabel('Performance Metric')
plt.legend(loc='right', ncol=1, title='Performance Metric', bbox_to_anchor=(1.4, 0.5))


### USE 3-FOLD CROSS VALIDATION
accuracies = []
precisions = []
recalls = []
f1s = []

cval = StratifiedKFold(n_splits=3, shuffle=True, random_state=19)
for train, test in cval.split(x_scaled, y):
    y_pred_indices = test
    y_pred, y_pred_raw = pls_da(x_scaled[train,:], y[train], x_scaled[test,:])
    
    accuracies.append(accuracy_score(y[test], y_pred))
    precisions.append(precision_score(y[test], y_pred))
    recalls.append(recall_score(y[test], y_pred))
    f1s.append(f1_score(y[test], y_pred))

print("Average accuracy: ", np.array(accuracies).mean())
print("Average precision: ", np.array(precisions).mean())
print("Average recall: ", np.array(recalls).mean())
print("Average F1 score: ", np.array(f1s).mean())


from matplotlib.patches import Circle

plt.figure(figsize=(8, 10))
ax = sns.scatterplot(
    x='Latent Variable 1',
    y='Latent Variable 2',
    data=plsr_df,
    hue='allergy status',
    legend=False
)

# Calculate the distance from each cluster center to the decision boundary
distances = np.abs(y_pred_raw - 0.5) / np.linalg.norm(plsr.x_scores_[y_pred_indices], axis=1)

# Define a scaling factor for the radius to control the circle size
scaling_factor = 0.5  # Adjust this value to control the circle size

# Plot circles representing clusters with radius based on the distance to the decision boundary
for i, (_, cluster_center) in enumerate(zip(y_pred, plsr.x_scores_)):
    radius = scaling_factor * distances[i]
    circle = Circle(cluster_center, radius=radius, alpha=0.5, fc='none', ec='black')
    ax.add_patch(circle)
    ax.annotate(f'Cluster {i+1}', cluster_center, textcoords="offset points", xytext=(-10, 10), ha='center')

# Set axis labels and title
ax.set_xlabel(f'LV1 ({fractions_of_explained_variance[0]:.1%} Variance)')
ax.set_ylabel(f'LV2 ({fractions_of_explained_variance[1]:.1%} Variance)')
plt.title('PLS-DA Clustering')

# Show the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower center', ncol=2, title='Allergy Status', bbox_to_anchor=(0.475, -0.1))

plt.show()



from sklearn.inspection import DecisionBoundaryDisplay
disp = DecisionBoundaryDisplay.from_estimator(plsr, X, response_method="predict", alpha=0.3)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y)
plt.show()