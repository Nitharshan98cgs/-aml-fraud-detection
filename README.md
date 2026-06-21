# Aml-Fraud-Detection
AML Transaction Fraud Detection

A transaction-monitoring model that flags suspicious payments in a large, highly imbalanced dataset — built to mirror the operational reality of financial-crime detection, where genuine cases are rare and the cost of false positives is high.

The operational problem

In real transaction monitoring, the challenge isn't catching fraud in isolation — it's catching it without burying investigators in false alerts. The vast majority of monitoring alerts in practice turn out to be false positives, and every one consumes analyst time. A model that flags everything "catches fraud" on paper but is operationally useless.

This project treats that trade-off as the central question. Fraud here is just 0.13% of transactions, so overall accuracy is meaningless — the focus is alert precision (how many alerts are genuine) and detection coverage (how much fraud is caught).

Approach


Scope the data — fraud occurs only in TRANSFER and CASH_OUT payments; filter to those (6.36M → 2.77M transactions).
Feature engineering — derive "balance reconciliation" signals that flag when account balances don't move as they should, reflecting how funds are moved and extracted.
Handle rarity — 70/30 train/test split, with SMOTE applied to the training data only, keeping the evaluation set realistically rare.
Model — Logistic Regression, Random Forest, and XGBoost on identical data.
Evaluate operationally — precision, recall, F1, and the raw false-alert and missed-case counts that translate directly into investigator workload and risk exposure.


Results

Held-out test set (831,123 transactions; fraud class):

ModelDetection (recall)Alert precisionF1Cases missedFalse alertsLogistic Regression90.0%4.7%0.08924745,326Random Forest99.6%97.7%0.987958XGBoost99.6%91.9%0.95611216

Random Forest gave the best operational profile — near-complete detection (only 9 cases missed) with a very low false-alert load (58 across ~829,000 clean payments).

The Logistic Regression result is the cautionary one: 90% detection looks acceptable until you see the 45,326 false alerts behind it. At 4.7% precision, roughly 19 of every 20 alerts would be a wasted investigation — the exact failure mode that makes real monitoring systems slow and expensive.

A note on model risk

PaySim is synthetic, and its fraud pattern is close to deterministic, so the tree models separate it almost perfectly. Real transaction data is far noisier and would not yield results this clean. The value here is the methodology — correct handling of class imbalance, evaluation on the metrics that govern operational cost, and clear visibility into the precision–recall trade-off — not the headline scores.

How to run


Download the dataset from Kaggle: PaySim — Synthetic Financial Datasets for Fraud Detection
Place it in the project folder and rename it to paysim.csv.
Install dependencies:


   pip install pandas scikit-learn imbalanced-learn xgboost


Run:


   python fraud_detection.py

Tech stack

Python · pandas · scikit-learn · imbalanced-learn (SMOTE) · XGBoost
