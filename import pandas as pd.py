import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\AML-3\paysim.csv.csv")   # match your real filename

print(df.shape)
print(df.head())

pd.set_option("display.max_columns", None)
print(df.head())

#%%
# 1a.How rare is fraud?
print(df["isFraud"].value_counts())
print(df["isFraud"].value_counts(normalize=True) * 100)
#%%

# 1b.Where does fraud happen?
print(pd.crosstab(df["type"], df["isFraud"]))

#%%

#Fraud only exists in TRANSFER and CASH_OUT. The other three types have exactly zero. (4,116 + 4,097 = 8,213 fraud cases total, ~0.13% of all rows.)
# Step 2.Clean & Engineer
#2a. Keep only the transaction types where fraud can occur

df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

print(df.shape)
print(df["isFraud"].value_counts(normalize=True) * 100)

#2b. Drop unhelpful columns and convert type to a number

df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])

df["type"] = df["type"].map({"TRANSFER": 0, "CASH_OUT": 1})

print(df.head())

#2c.Build "balance error" features
df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

print(df.head())
print(df.shape)

#%%
#3.Prepare for Modelling
# 3a.Separate the inputs from the answer
feature_names = df.drop(columns=["isFraud"]).columns.tolist()
X = df.drop(columns=["isFraud"])   # the inputs the model learns from
y = df["isFraud"]                  # the answer we want it to predict

#3b — Split into a learning set and a test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

print("Train:", X_train.shape, " Test:", X_test.shape)

#3c. — Put all features on the same scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # learn the scale from training data only
X_test  = scaler.transform(X_test)        # apply that same scale to the test data

#3d — Balance the training set with SMOTE
from imblearn.over_sampling import SMOTE
import pandas as pd

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

print("Train before SMOTE:", pd.Series(y_train).value_counts().to_dict())
print("Train after SMOTE: ", pd.Series(y_train_bal).value_counts().to_dict())

#%%
#Step 4 — Train the Three Models
#4a. — Train all three
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_bal, y_train_bal)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)

xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss")
xgb.fit(X_train_bal, y_train_bal)


#5— Test and Compare
#5a. Score every model on the unseen test set
from sklearn.metrics import classification_report, confusion_matrix

for name, model in [("Logistic Regression", logreg),
                    ("Random Forest", rf),
                    ("XGBoost", xgb)]:
    preds = model.predict(X_test)
    print("=" * 55)
    print(name)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, digits=4))


# ===== EXPORT FOR POWER BI =====
# make a Power-BI-friendly copy with readable transaction-type names
df_bi = df.copy()
df_bi["type_name"] = df_bi["type"].map({0: "TRANSFER", 1: "CASH_OUT"})
df_bi.to_csv(r"C:\AML-3\paysim_clean.csv", index=False)

# tiny table of the model results
import pandas as pd
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Recall": [0.900, 0.996, 0.996],
    "Precision": [0.047, 0.977, 0.919],
    "F1": [0.089, 0.987, 0.956],
    "Frauds_Missed": [247, 9, 11],
    "False_Alarms": [45326, 58, 216],
})
results.to_csv(r"C:\AML-3\model_results.csv", index=False)
print("Exported paysim_clean.csv and model_results.csv")
