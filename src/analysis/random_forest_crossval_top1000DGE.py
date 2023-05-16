# random forest classifier on top 1000 genes from DGE
# includes k-fold cross validation
# https://www.datacamp.com/tutorial/random-forests-classifier-python

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sys.path.append('..\\..\\src\\util\\')
from helper_functions import filter_samples

### READ IN AND PROCESS DATA
df_raw = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')
top1000dge = pd.read_pickle('..\\..\\data\\results\\differential_gene_expression\\top1000from1000dge_followup.pkl')

df_raw = df_raw[df_raw['Gene'].isin(top1000dge['Gene'])]

df_raw = filter_samples(df_raw, annotation_df, 'Sample_characteristics_ch1_age_yrs', (2,4))
df_raw = filter_samples(df_raw, annotation_df, 'Sample_characteristics_ch1_activation_status', 1)
df = df_raw.T
headers = df.iloc[0]
df  = pd.DataFrame(df.values[1:], columns=headers)
df.index.names = ['Sample']
df

# add allergy status to the dataframe
annotation_df_samples_to_keep = df_raw.columns
allergy_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_allergy_status']
allergy_status_df = allergy_status_df.reindex(columns = annotation_df_samples_to_keep)
allergy_status_df = allergy_status_df.drop('Gene', axis=1)

# add activation status to the PCA dataframe
# annotation_df_samples_to_keep = df.columns
# activation_status_df = annotation_df.loc[annotation_df['Sample_title'] == 'Sample_characteristics_ch1_activation_status']
# activation_status_df = activation_status_df.reindex(columns = annotation_df_samples_to_keep)
 
df['allergy status'] = allergy_status_df.values[0] #check the order of the labels
#df['activation status'] = activation_status_df.values[0] #check the order of the labels

target_names = {
    'control':2,
    'allergic':0, 
    'resolved':1
}
df['allergy_status_numerical'] = df['allergy status'].map(target_names)
df = df.drop('allergy status', axis=1)
df



### SPLIT INTO TRAINING AND TEST DATA 
X = df.drop('allergy_status_numerical', axis=1)
#X = X.to_numpy()
y = df['allergy_status_numerical']
#y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



### TRAIN THE MODEL
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)



### HYPERPARAMETER TUNING 
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)



### REDO MODEL WITH BEST HYPERPARAMETERS- K FOLD CROSS VALIDATION


# # Run classifier with cross-validation and plot ROC curves
# cv = StratifiedKFold(n_splits=4)

# classifier = RandomForestClassifier(max_depth=rand_search.best_params_['max_depth'], 
#                                     n_estimators=rand_search.best_params_['n_estimators'])





num_repeats = 10
num_folds = 4

estimators = []
accuracies = []
precisions = []
recalls = []
aucs = []

#OOBs = []

probas = []
y_true = []
importance_dfs = []

fprs = []
tprs = []
aucs2 = []

skf = StratifiedKFold(n_splits=num_folds)

try:
    X_df = X
    y_df = y
    X = X.to_numpy()
    y = y.to_numpy()
except AttributeError:
    pass

feature_names = X_df.columns  # Store the feature names
X_tests = []
y_tests = []

for i in range(num_repeats):
    print('repeat:', i)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_tests.append(X_test)
        y_tests.append(y_test)
        print('test length', len(y_test))

        my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                      ('model', RandomForestClassifier(max_depth=rand_search.best_params_['max_depth'],
                                                                       n_estimators=rand_search.best_params_[
                                                                           'n_estimators']
                                                                       ))
                                      ])

        y_pred_proba = cross_val_predict(my_pipeline, X_test, y_test,
                                         cv=num_folds,
                                         method='predict_proba')[:, 1]

        probas.extend(y_pred_proba)
        y_true.extend(y_test)

        scores = cross_validate(my_pipeline, X_train, y_train,
                                cv=num_folds,
                                scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
                                return_estimator=True)
        
        estimators.extend(scores['estimator'])
        accuracies.extend(scores['test_accuracy'])
        precisions.extend(scores['test_precision'])
        recalls.extend(scores['test_recall'])
        aucs.extend(scores['test_roc_auc'])

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs2.append(auc_score)

        # Save feature importances
        for estimator in estimators:
            # Create a series containing feature importances from the model and feature names from the training data
            feature_importances = pd.Series(estimator[1].feature_importances_, index=feature_names).sort_values(ascending=False)
            importance_dfs.append(feature_importances)

print("Average Accuracy:", np.mean(accuracies))
print(np.std(accuracies, ddof=1))
print("Average Precision:", np.mean(precisions))
print(np.std(precisions))
print("Average Recall:", np.mean(recalls))
print(np.std(recalls))


##################


from sklearn.metrics import plot_roc_curve
sns.set_theme(style='white')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

X_tests2 = np.repeat(X_tests,num_folds)
y_tests2 = np.repeat(y_tests, num_folds)

fig, ax = plt.subplots()
for i, estimator in enumerate(estimators):
    viz = plot_roc_curve(estimator, X_tests2[i], y_tests2[i],
                        name='_',
                        alpha=0, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.4f)' % (mean_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
    title="ROC Curve")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
plt.savefig('..\\..\\fig\\supp_fig\\random_forest\\RFcrossval_top1000DGE.png', dpi=600)
plt.show()


# convert the list of series to a DataFrame
avg_importance_df = pd.concat(importance_dfs, axis=1, keys=range(len(importance_dfs)))

# group the DataFrame by the gene index and calculate the mean of each group
avg_importance_df = avg_importance_df.mean(axis=1)
avg_importance_df = avg_importance_df.sort_values(ascending=False)

pd.to_pickle(avg_importance_df, '..\\..\\data\\results\\avgfeatureimportance_RFtop1000DGE_4folds10repeats.pkl')

# Plot a simple bar chart
avg_importance_df.head(100).plot.bar()
plt.title('Top 100 Features')
plt.show()

avg_importance_df.head(30).plot.bar()
plt.title('Top 30 Features')
plt.show()

avg_importance_df.head(10).plot.bar()
plt.title('Top 10 Features')
plt.show()