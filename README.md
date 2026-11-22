# F1 Race Predictor

A self-test of what I learned from Part I of Aurélien Géron's *Hands-On Machine Learning*,
and a coding refresher after some time away.

Using race and qualifying data from [FastF1](https://github.com/theOehrly/Fast-F1), it predicts:
- **finishing position** (regression: LinearRegression, DecisionTree, RandomForest)
- **podium finish** (classification: SGD, RandomForest)

Trained on the 2022–2024 seasons and tested on 2025.

## Plots

**Precision and recall vs threshold (SGD)**
![Precision and recall vs threshold](data/precision_recall_sgd.png)
How precision and recall trade off as the decision threshold moves. This plot is used to pick a threshold.

**ROC curve (SGD)**
![ROC curve](data/roc_sgd.png)
AUC ≈ 0.89. It looks strong, but ROC flatters imbalanced data, and only about 15% of results are podiums.

**Precision–recall: RandomForest vs SGD**
![RandomForest vs SGD](data/precision_recall_rf_vs_sgd.png)
RandomForest stays above SGD for most recall values, which is why it's the final classifier.

## Run
    python extract_enrich.py        # download and build the dataset
    python classical_f1_predictor.py
