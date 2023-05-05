# random forest classifier on original RNAseq data (all transcripts)
# includes k-fold cross validation
# https://www.datacamp.com/tutorial/random-forests-classifier-python

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn import datasets
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('..\\..\\src\\util\\')
from helper_functions import filter_samples

### READ IN AND PROCESS DATA
df_raw = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')

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



### SPLIT INTO TRAINING AND TEST DATA (REPLACE WITH KFOLD CROSS VAL)
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



### HYPERPARAMETER TUNING ???
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
#OOBs = []

for i in range(0, num_repeats):


    # https://wandb.ai/wandb_fc/kaggle_tutorials/reports/Using-K-Fold-Cross-Validation-To-Improve-Your-Machine-Learning-Models--VmlldzoyMTY0MjM2#the-final-word
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                ('model', RandomForestClassifier(max_depth=rand_search.best_params_['max_depth'], 
                                                                n_estimators=rand_search.best_params_['n_estimators']
                                                                ))
                                ])


    scores = cross_validate(my_pipeline, X, y,
                                cv=num_folds,
                                scoring=['accuracy','precision','recall'],
                                return_estimator=True)

    # print("Average Accuracy:", scores['test_accuracy'].mean())
    # print("Average Precision:", scores['test_precision'].mean())
    # print("Average Recall:", scores['test_recall'].mean())

    for estimator in scores['estimator']:
        # Generate predictions with the best model
        y_pred = estimator.predict(X_test)
        #estimator.oob_score_

        # Create the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        #ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        #plt.show()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # print("Accuracy:", accuracy)
        # print("Precision:", precision)
        # print("Recall:", recall)

        #rf_disp = RocCurveDisplay.from_estimator(estimator, X_test, y_test)
    # plt.show()


        # ### FEATURE IMPORTANCE
        # # Create a series containing feature importances from the model and feature names from the training data
        # feature_importances = pd.Series(estimator.feature_importances_, index=X_train.columns).sort_values(ascending=False)

        # # Plot a simple bar chart
        # feature_importances.head(100).plot.bar()
        # plt.title('Top 100 Features')
        # plt.show()

        # feature_importances.head(30).plot.bar()
        # plt.title('Top 30 Features')
        # plt.show()

        # feature_importances.head(10).plot.bar()
        # plt.title('Top 10 Features')
        # plt.show()

    estimators.extend(scores['estimator'])
    accuracies.extend(scores['test_accuracy'])
    precisions.extend(scores['test_precision'])
    recalls.extend(scores['test_recall'])
   # OOBs.extend(

print("Average Accuracy:", np.array(accuracies).mean())
print("Average Precision:", np.array(precisions).mean())
print("Average Recall:", np.array(recalls).mean())


sns.set_theme(style='white')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for estimator in estimators:
    viz = plot_roc_curve(estimator, X_test, y_test,
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
plt.show()