# standard library
import subprocess
from pathlib import Path

# third party
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

# local
from extract_enrich import print_to_excel

# ------------------------- constants --------------------------------

# load in enriched dataset
CSV_PATH = "/home/edward_c_holder/ml/Geron_Part_1/ml_proj_2/data/output.csv"
ENRICHED_DATA_PATH = "/home/edward_c_holder/ml/Geron_Part_1/ml_proj_2/data/full_dataset_enriched.csv"

# -------------------------functions----------------------------------- 

def race_instance_sampler(year,round,train_set,predictors,all_labels):
        sample = predictors[(train_set["Season"] == year) & (train_set["Round"] == round)].copy()
        labels = all_labels[(train_set["Season"] == year) & (train_set["Round"] == round)].copy()
        return sample, labels

# ----------------------------core--------------------------------------

# wrapping code like this prevents it automatically running if called in another script
if __name__ =="__main__":

    # pandas adds an index to a csv, but this csv is already a dataframe so dropped
    full_dataset = pd.read_csv(ENRICHED_DATA_PATH, index_col=0)

    # some rows in grid position came out as NaN or 0, fill with last place for simplicity
    full_dataset["GridPosition"] = full_dataset["GridPosition"].replace(0, 23)
    full_dataset["GridPosition"] = full_dataset["GridPosition"].fillna(23)
    full_dataset["Position"] = full_dataset["Position"].replace(0, 23)
    full_dataset["Position"] = full_dataset["Position"].fillna(23)

    # drop duplicat index col from csv import
    full_dataset = full_dataset.drop(columns=["Unnamed: 0"])

    # engineered features from exisitng predictors
    full_dataset["QualiAnomaly"] = full_dataset["GridPosition"]-full_dataset["DriverRollingAvgFinish"]
    full_dataset["DriverForm"] = full_dataset["DriverRollingAvgFinish"]-full_dataset["ConstructorRollingAverage"]
    full_dataset["PackCloseness"] = full_dataset["QualiGapToPole"]/full_dataset["GridPosition"]

    # set aside a test set.
    # Geron uses train_test_split, for this project just take out the 2025 season:
    train_set = full_dataset[full_dataset["Season"] != 2025].copy()
    test_set = full_dataset[full_dataset["Season"] == 2025].copy()

    # ----------------- exploratory data analysis -------------------

    # Uncomment blocks to visually explore dataset
    # results were used to infrom feature selection and model choice

    # gives a short description of data:
    # print(full_dataset.info())

    # for text attributes this shows number of occurances
    # print(full_dataset["Status"].value_counts())
    # describe method gets a sum of numerical attributes
    # print(full_dataset.describe())

    # gives a general overview of feature distribution
    # full_dataset.hist(bins=50, figsize=(20,15))
    # plt.show()

    # Used to show how much each value correlates with "Position"
    # corr_matrix = train_set.corr(numeric_only=True)
    # print(corr_matrix)

    # Plot scatter matrix for better visualization
    # attributes = ["Position","Points","GridPosition","ConstructorRollingAverage","DriverRollingAvgFinish","ChampionshipStanding","DriverTrackHistory"]
    # scatter_matrix(train_set[attributes],figsize=(12,8))

    # ------------------data preparation -----------------------------

    # separate out labels
    # eliminated teamname, abbreviation and location because all useful info from these
    # is captured in engineered columns
    # for feature set X_train, eliminate all target or useless columns
    X_train = train_set.drop(columns=["Position","Podium","Points","Status","TeamName","Abbreviation","Location","Season","Round"])
    X_test = test_set.drop(columns=["Position","Podium","Points","Status","TeamName","Abbreviation","Location","Season","Round"]) #repeat for test set

    # taking one label for each task, regresison and classification
    y_train_classifier = train_set["Podium"].copy() #binary
    y_train_regressor = train_set["Position"].copy()
    y_test_classifier = test_set["Podium"].copy()
    y_test_regressor = test_set["Position"].copy()

    # to take care of NaN data, specifically found in round one of 2022, where there is no prior data to predict off, 
    # use of KNNimputer will predict based on the other available preictors looking at other outcomes in the training data
    imputer = KNNImputer(n_neighbors=5)
    # fit only on data from train set, to stop data leakage
    # no need to specify values as it already takes care of only NaN values
    imputer.fit(X_train)

    # note the following steps could be done using pipeline but, since this is a learning exercise I cose to explicate first
    # transform data from test set too!
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imputed  = pd.DataFrame(imputer.transform(X_test),  columns=X_test.columns,  index=X_test.index)


    # scale parameters, ml models cant handle big varience in dimensions
    std_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(std_scaler.fit_transform(X_train_imputed),columns=X_train.columns,index=X_train.index)
    X_test_scaled = pd.DataFrame(std_scaler.transform(X_test_imputed),columns=X_test.columns,index=X_test.index)

    # ------------------ regression modelling ---------------------------

    # sample 5 random indices for testing model
    np.random.seed(42)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled,y_train_regressor)
    f1_lin_reg_predicts = lin_reg.predict(X_train_scaled)

    sample_indices = np.random.randint(0,len(f1_lin_reg_predicts),5)

    # check performance on first 5 rows
    print("for five random instances:")
    print("\nLinearRegression predictions:\n\n",f1_lin_reg_predicts[sample_indices])
    print("\nLinearRegression lables:\n\n",y_train_regressor.iloc[sample_indices])

    # calculate rmse for liner regression
    lin_reg_rmse = root_mean_squared_error(y_train_regressor,f1_lin_reg_predicts)
    print("\nLinearRegression RMSE:\n\n",lin_reg_rmse)

    # evaluate DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_train_scaled,y_train_regressor)
    f1_tree_reg_predicts = tree_reg.predict(X_train_scaled)
    tree_rmse = root_mean_squared_error(y_train_regressor,f1_tree_reg_predicts)
    print("DecisionTreeRegressor RMSE:\n",tree_rmse)

    # ----------------- training with transformation pipelines -------------------

    # pipeline will streamline the preprocessing stage turning the imputation and scaling steps into 
    # one single obeject

    f1_pipeline = Pipeline([
        ("impute", KNNImputer(n_neighbors=5)),
        ("standardize",StandardScaler())
    ])

    # pipeline takes list of name estimator pairs (2-tuples), provided they're all transformers save the last one
    # note they don't have to be named:
    # the following code should work too
    # f1_pipeline = make_pipeline(KNNImputer(n_neighbors=5),StandardScaler())


    # ------------------------ cross validation ------------------------

    # cross validation evaluates the models k times (cv=k) on a different feature each time
    tree_rmse = cross_val_score(tree_reg, X_train_scaled,y_train_regressor,
                                scoring="neg_root_mean_squared_error",cv=8,n_jobs=-1)
    print("\nRMSE from cross-validation on DecisionTreeRegressor:\n\n",pd.Series(tree_rmse).describe())

    # it's likely at this stage, with DecisionTreeRegressor performing worse than LinearRegressor,
    # that the risk is overfitting rather than underfitting, which RFR would be able to improve
    # this check is therefore likely unnecessary but included for learning purposes
    forest_reg = RandomForestRegressor(random_state=42)
    forest_reg.fit(X_train_scaled,y_train_regressor)
    forest_rmse = cross_val_score(forest_reg, X_train_scaled,y_train_regressor,
                                scoring="neg_root_mean_squared_error",cv=8,n_jobs=-1)
    print("\nRandomForestRegressor scores:\n\n",pd.Series(forest_rmse).describe())

    # -----------------------fine tuning with grid search---------------------

    # Grid search is a useful fine tuning tool which makes use of cross validation on a choice of parameters
    # and ouptust the best combination
    full_pipeline = Pipeline([
    ("impute_and_scale",f1_pipeline), # here f1 pipeline is nested within the full_pipeline
    ("random_forest",RandomForestRegressor(random_state=42))
    ])

    # create a tuple of parameters to be evaluated
    param_grid = [
        {"impute_and_scale__impute__n_neighbors":[3,5,7], # double__underscores take down layers of nested pipelines
        "random_forest__n_estimators":[50,100,200],
        "random_forest__max_depth":[None,10,20],
        "random_forest__max_features":[4,6,8]}
    ]

    # run the grid search
    grid_search = GridSearchCV(full_pipeline,param_grid,cv=3,
                            scoring="neg_root_mean_squared_error",n_jobs=-1)
    grid_search.fit(X_train,y_train_regressor)
    print("\n Grid_search for RandomForestRegressor produces the following:\n\n")
    print("BestParams:\n\n",grid_search.best_params_)
    print("\nBestEstimator:\n\n",grid_search.best_estimator_)
    # training set is automatically retrained on best params if refit=True, the defult
    cv_res = pd.DataFrame(grid_search.cv_results_)
    cv_res.sort_values(by="mean_test_score",ascending=False,inplace=True)
    print("\nResultsofgridserch:\n",cv_res.head())
    # none perform better than standard linear regressor

    # -----------------------binary podium clasifier--------------------

    # begin with SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train_scaled,y_train_classifier)

    # test a particular race:
    sgd_sample, sgd_sample_labels = race_instance_sampler(2023,5,train_set,X_train_scaled,y_train_classifier)

    print("\n SGDClassifier prediction:\n\n",sgd_clf.predict(sgd_sample))
    print("\n True Result:\n\n",sgd_sample_labels.values)

    # take scoring=f1 for imbalanced classes
    print("\ncross-validation score using 'f1' measure:\n\n",cross_val_score(sgd_clf, X_train_scaled,y_train_classifier,cv=3,scoring="f1",n_jobs=-1))

    #confusion matrix evaluation
    y_train_predict = cross_val_predict(sgd_clf,X_train_scaled,y_train_classifier,cv=3,n_jobs=-1)
    cm_sgd = confusion_matrix(y_train_classifier,y_train_predict)
    print("\nConfusion matrix for SGDClassifier:\n\n",cm_sgd)

    # plot precision recall curve to find thresholds for prediction
    # square brakets cut off artifact created by sklearn and make all arrays the same size
    y_sgd_scores = cross_val_predict(sgd_clf,X_train_scaled,y_train_classifier,cv=3,method="decision_function",n_jobs=-1)
    precisions_sgd,recalls_sgd,thresholds_sgd = precision_recall_curve(y_train_classifier,y_sgd_scores)
    plt.figure()
    plt.plot(thresholds_sgd,precisions_sgd[:-1],"b--",label="Precision",linewidth=2)
    plt.plot(thresholds_sgd,recalls_sgd[:-1],"g-",label="Recall",linewidth=2)
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("./data/precision_recall_sgd.png")
    # steep drop off at threshold indicative of poor metric as a result of the skewed dataset

    # ROC also a common metric, but again, not useful with highly skewed data
    fpr,tpr,thresholds = roc_curve(y_train_classifier,y_sgd_scores)
    plt.figure()
    plt.plot(fpr,tpr,linewidth=2,label="ROC")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.legend()
    plt.savefig("./data/roc_sgd.png")
    
    # checking area under curve for Reciever operating characteristic a good alternative but
    # again not highly useful for skewed data
    print("ROC AUC:",roc_auc_score(y_train_classifier,y_sgd_scores))

    # check performance of RandomForestClassifier, ensemble method:
    forest_clf = RandomForestClassifier(random_state=42, class_weight="balanced")
    forest_clf_default = RandomForestClassifier(random_state=42) # removed class weight setting
    y_prob_forest = cross_val_predict(forest_clf,X_train_scaled,y_train_classifier,cv=3,method="predict_proba",n_jobs=-1)
    y_pred_forest_default = cross_val_predict(forest_clf_default,X_train_scaled,y_train_classifier,cv=3,n_jobs=-1)
    y_score_forest = y_prob_forest[:,1]
    y_pred_custom = (y_score_forest >= 0.3).astype(int) #ensure a binary result above a new threshold of 0.3 confidence rather than 0.5 default
    
    precisions_forest,recalls_forest,thresholds_forest = precision_recall_curve(
        y_train_classifier,y_score_forest
    )
    plt.figure()
    plt.plot(recalls_forest,precisions_forest,"b-",linewidth=2,label="RandomForest")
    plt.plot(recalls_sgd,precisions_sgd,"--",linewidth=2,label="SGD")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("./data/precision_recall_rf_vs_sgd.png")
    print("\nRandomForestClassifier roc/auc score:\n\n",roc_auc_score(y_train_classifier,y_score_forest))

    cm_forest = confusion_matrix(y_train_classifier,y_pred_custom)
    print("\nRandomForestClassifier confusion matrix:\n\n",cm_forest)

    # on inspection it seems that my best predictors are RFC std and RFC set to balanced class weight,
    # and with a probabiliy threshold of 0.3 (from the default 0.5)
    # compare f1 score for both methods
    print("default f1_score:",f1_score(y_train_classifier,y_pred_forest_default))
    print("balenced_0.3_f1_score:",f1_score(y_train_classifier,y_pred_custom))
    # the best classifier seems to be balenced rfc with a threshold set to 0.3
    # this configuration produces the highest f1 score, i.e maximising recall and precision

    # -------------------- test final model ----------------------------

    # refit best models on full training set before evaluating on test set
    # regressor: LinearRegression was best performer (~4.3 RMSE on cross validation)
    lin_reg_final = LinearRegression()
    lin_reg_final.fit(X_train_scaled, y_train_regressor)
    y_test_reg_predictions = lin_reg_final.predict(X_test_scaled)

    test_rmse = root_mean_squared_error(y_test_regressor, y_test_reg_predictions)
    print("\n==================== FINAL TEST SET RESULTS ====================")
    print("\nRegressor (LinearRegression)")
    print("Test RMSE:", test_rmse)

    # classifier: RFC balanced with 0.3 threshold was best performer (F1 ~0.65)
    forest_clf_final = RandomForestClassifier(random_state=42, class_weight="balanced")
    forest_clf_final.fit(X_train_scaled, y_train_classifier)
    y_test_prob = forest_clf_final.predict_proba(X_test_scaled)[:, 1]
    y_test_clf_predictions = (y_test_prob >= 0.3).astype(int)

    # Baseline test, Does model predict more than just grid position?
    grid = test_set["GridPosition"]

    baseline_podium = (grid <= 3).astype(int)

    print("\nClassifier (RandomForestClassifier, balanced, threshold=0.3)")
    print("Confusion Matrix:\n", confusion_matrix(y_test_classifier, y_test_clf_predictions))
    print("F1 Score:", f1_score(y_test_classifier, y_test_clf_predictions))
    print("Precision:", precision_score(y_test_classifier, y_test_clf_predictions))
    print("Recall:", recall_score(y_test_classifier, y_test_clf_predictions))
    print("\nBaseline regressor (finish = grid position)")
    print("Test RMSE:", root_mean_squared_error(y_test_regressor, grid))
    print("\nBaseline classifier (podium = started top 3)")
    print("F1:", f1_score(y_test_classifier, baseline_podium))
    print("Precision:", precision_score(y_test_classifier, baseline_podium))
    print("Recall:", recall_score(y_test_classifier, baseline_podium))
    print("\n================================================================")

    plt.show()