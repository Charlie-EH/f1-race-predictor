#standard library
import subprocess
from pathlib import Path

#third party
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import fastf1 as f1

# SKlearn
from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import (roc_auc_score, root_mean_squared_error, confusion_matrix,
                              precision_recall_curve,roc_curve,precision_score,recall_score, f1_score)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

#local
from extract_enrich import print_to_excel

############################# function










#load in enriched dataset
csv_path = "/home/edward_c_holder/ml/ml_proj_2/data/output.csv"
#pandas adds an index to a csv, but this csv is already a df to drop it
full_dataset = pd.read_csv("/home/edward_c_holder/ml/ml_proj_2/data/full_dataset_enriched.csv", index_col=0)

#some rows in grid position came out as NaN or 0, fill with last place for simplicity
full_dataset["GridPosition"] = full_dataset["GridPosition"].replace(0, 20)
full_dataset["GridPosition"] = full_dataset["GridPosition"].fillna(20)
full_dataset["Position"] = full_dataset["Position"].replace(0, 20)
full_dataset["Position"] = full_dataset["Position"].fillna(20)

#drop second index col, not sure where that came from lol
full_dataset = full_dataset.drop(columns=["Unnamed: 0"])

#add a column of combined attributes (to be explored later)
full_dataset["QualiAnomaly"] = full_dataset["GridPosition"]-full_dataset["DriverRollingAvgFinish"]
full_dataset["DriverForm"] = full_dataset["DriverRollingAvgFinish"]-full_dataset["ConstructorRollingAverage"]
full_dataset["PackCloseness"] = full_dataset["QualiGapToPole"]/full_dataset["GridPosition"]
#print_to_excel(csv_path,full_dataset)

############## exploratory data analysis ###############

#gives a short description of data
#can see columns of floats, integers and objects
#print(full_dataset.info())

#for text attributes this shows number of occurances
#print(full_dataset["Status"].value_counts())
#describe method gets a sum of numerical attributes
#print(full_dataset.describe())

#full_dataset.hist(bins=50, figsize=(20,15))
#plt.show()

#set aside a test set geron does train_test_split, for f1 we will just take out the 2025 season:
train_set = full_dataset[full_dataset["Season"] != 2025].copy()
test_set = full_dataset[full_dataset["Season"] == 2025].copy()

#having split out a test set, set aside and ignore for now
#time to visualise the training set
#no net to replicate geron's nice map graph, straight to correlations

corr_matrix = train_set.corr(numeric_only=True)
#correlaiton matrix used to show how much each value correlates with "Position"

#this can also be visualised
#use most promising attribues from corr matrix to plot scatter matrix for better understanding
attributes = ["Position","Points","GridPosition","ConstructorRollingAverage","DriverRollingAvgFinish","ChampionshipStanding","DriverTrackHistory"]
#scatter_matrix(train_set[attributes],figsize=(12,8))
#repeate for podium and poitns for curiousity sake, maybe this is useful later

#train_set.plot(kind="scatter",  x="GridPosition",y="Position",alpha=0.1,grid=True)

#Prepare data for machine learning
#separate out labels
#also we can eliminate teamname, abbreviation and location because all useful info is given from these
# in the new collumns we made! no need to do any encoding as geron does
#for x we want to eliminate all columns we're trying to predict or ones that just aren't useful for prediction
X_train = train_set.drop(columns=["Position","Podium","Points","Status","TeamName","Abbreviation","Location","Season","Round"])

#taking one label for each task regresison and classification
y_train_classifier = train_set["Podium"].copy()
y_train_regressor = train_set["Position"].copy()

#repeat for test set
X_test = test_set.drop(columns=["Position","Podium","Points","Status","TeamName","Abbreviation","Location","Season","Round"])

#taking one label for each task regresison and classification
y_test_classifier = test_set["Podium"].copy()
y_test_regressor = test_set["Position"].copy()

#to take care of NaN data, specifically found in round one of 2022, where there is no prior data to predict off, 
# use of KNNimputer will predict based on the other available preictors looking at other outcomes in the training data
imputer = KNNImputer(n_neighbors=5)
#fit only on data from train set, to stop data leakage
#no need to specify values as it already takes care of only NaN values
imputer.fit(X_train)

#transform data from test set too!
#this line is going to return a numpy array not a dataframe and prevent print_to_excel working
X_train_imputed = pd.DataFrame(imputer.transform(X_train),columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test),columns=X_test.columns)

#scale parameters, ml models cant handle big varience in dimensions
std_scaler = StandardScaler()
X_train_scaled = pd.DataFrame(std_scaler.fit_transform(X_train_imputed),columns=X_train.columns)
X_test_scaled = pd.DataFrame(std_scaler.transform(X_test_imputed),columns=X_test.columns)

########### Select and Train Model #############
#before moving to test set, try different models on the training set
#sample 5 random indices for testing model
np.random.seed(42)


lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled,y_train_regressor)
f1_lin_reg_predicts = lin_reg.predict(X_train_scaled)

sample_indices = np.random.randint(0,len(f1_lin_reg_predicts),5)

#check performance on first 5 rows
# print("predictions:",f1_lin_reg_predicts[sample_indices])
# print("lables:",y_train_regressor[sample_indices])

lin_reg_rmse = root_mean_squared_error(y_train_regressor,f1_lin_reg_predicts)
print("Lin_reg RMSE error:",lin_reg_rmse)

#try decision tree regressor which should be more powreful
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_scaled,y_train_regressor)
f1_tree_reg_predicts = tree_reg.predict(X_train_scaled)
# tree_rmse = root_mean_squared_error(y_train_regressor,f1_tree_reg_predicts)
# print("Tree RMSE:",tree_rmse)

############ training with transformation pipelines #################
#pipeline will streamline the preprocessing stage turning the imputation and scaling steps into 
# one single obeject

f1_pipeline = Pipeline([
    ("impute", KNNImputer(n_neighbors=5)),
    ("standardize",StandardScaler())
])

#pipeline takes list of name estimator pairs (2-tuples), provided they're all transofrmers save the last one
#note they don't have to be named:
#the following code should work too
#f1_pipeline = make_pipeline(KNNImputer(n_neighbors=5),StandardScaler())


######## cross validation ############
tree_rmse = cross_val_score(tree_reg, X_train_scaled,y_train_regressor,
                            scoring="neg_root_mean_squared_error",cv=8,n_jobs=-1)
#print(pd.Series(tree_rmse).describe())
#at thsi stage, looks like lin_reg is generalisng better

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train_scaled,y_train_regressor)
forest_rmse = cross_val_score(forest_reg, X_train_scaled,y_train_regressor,
                            scoring="neg_root_mean_squared_error",cv=8,n_jobs=-1)
print("RandomForest scores:",pd.Series(forest_rmse).describe())
#random forest generalizing similarly to lin_reg now


#try more models before fine tuning for now i'll move on for learning purposes
##########fine tuning with grid search ############

full_pipeline = Pipeline([
  ("impute_and_scale",f1_pipeline),
  ("random_forest",RandomForestRegressor(random_state=42))
])

param_grid = [
    {"impute_and_scale__impute__n_neighbors":[3,5,7],
     "random_forest__n_estimators":[50,100,200],
     "random_forest__max_depth":[None,10,20],
     "random_forest__max_features":[4,6,8]}
]

grid_search = GridSearchCV(full_pipeline,param_grid,cv=3,
                           scoring="neg_root_mean_squared_error",n_jobs=-1)
grid_search.fit(X_train,y_train_regressor)
print("BestParams:",grid_search.best_params_)
print("BestEstimator:",grid_search.best_estimator_)
#training set is automatically retrained on best params if refit=True, the defult
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score",ascending=False,inplace=True)
print("Resultsofgridserch:\n",cv_res.head())
#none better than standard linear regressor


############## binary podium clasifier ###############

#begin with SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train_scaled,y_train_classifier)

sgd_sample = X_train_scaled[(train_set["Season"] == 2024) & (train_set["Round"] == 5)]
sgd_sample_labels = y_train_classifier[(train_set["Season"] == 2024) & (train_set["Round"] == 5)]

# print(sgd_clf.predict(sgd_sample))
# print(sgd_sample_labels.values)

#take scoring f1 for imbalanced classes
#reminder that cv is fold number
# print(cross_val_score(sgd_clf, X_train_scaled,y_train_classifier,cv=3,scoring="f1",n_jobs=-1))

#confusion matrix evaluation

y_train_predict = cross_val_predict(sgd_clf,X_train_scaled,y_train_classifier,cv=3,n_jobs=-1)


cm_sgd = confusion_matrix(y_train_classifier,y_train_predict)
print(cm_sgd)

#plot precision recall curve
#square brakets cut off artifact created by sklearn and make all arrays the same size
#y_sgd_scores = cross_val_predict(sgd_clf,X_train_scaled,y_train_classifier,cv=3,method="decision_function",n_jobs=-1)
#precisions_sgd,recalls_sgd,thresholds_sgd = precision_recall_curve(y_train_predict,y_sgd_scores)
# plt.plot(thresholds_sgd,precisions_sgd[:-1],"b--",label="Precision",linewidth=2)
# plt.plot(thresholds_sgd,recalls_sgd[:-1],"g-",label="Recall",linewidth=2)
# #plt.vlines(threshold,0,)
# plt.show()

# fpr,tpr,thresholds = roc_curve(y_train_classifier,y_sgd_scores)
# plt.plot(fpr,tpr,linewidth=2,label="ROC")
# plt.show()

# print("ROC AUC:",roc_auc_score(y_train_classifier,y_sgd_scores))

forest_clf = RandomForestClassifier(random_state=42, class_weight="balanced")
forest_clf_default = RandomForestClassifier(random_state=42)
y_prob_forest = cross_val_predict(forest_clf,X_train_scaled,y_train_classifier,cv=3,method="predict_proba",n_jobs=-1)
y_pred_forest_default = cross_val_predict(forest_clf_default,X_train_scaled,y_train_classifier,cv=3,n_jobs=-1)
y_score_forest = y_prob_forest[:,1]
y_pred_custom = (y_score_forest >= 0.3).astype(int)
# y_scores_forest = y_prob_forest[:,1]
# precisions_forest,recalls_forest,thresholds_forest = precision_recall_curve(
#     y_train_classifier,y_scores_forest
# )
# plt.plot(recalls_forest,precisions_forest,"b-",linewidth=2,label="RandomForest")
# plt.plot(recalls_sgd,precisions_sgd,"--",linewidth=2,label="SGD")
# plt.show()
#print(roc_auc_score(y_train_classifier,y_scores_forest))

# cm_forest = confusion_matrix(y_train_classifier,y_pred_custom)
# print(cm_forest)

#on inspection it seems that my best predictors are RFC std and RFC set to balanced class weight,
# and with a probabiliy threshold of 0.3 (from the default 0.5)
# compare f1 score for both methods
print("default f1_score:",f1_score(y_train_classifier,y_pred_forest_default))
print("balenced_0.3_f1_score:",f1_score(y_train_classifier,y_pred_custom))