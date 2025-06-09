# Starter Python Notebook for Multi-Step Regression + Classification Project

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, roc_auc_score, r2_score, mean_squared_error

# 2. Load Data
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')  # Example: IBM dataset

# 3. Preprocessing
# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop(['Attrition'], axis=1)
y = df['Attrition']  # 1 = Yes, 0 = No

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Attrition Prediction (Classification)
# Model Training
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predictions and Metrics
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print('F1 Score:', f1_score(y_test, y_pred))
print('AUC-ROC:', roc_auc_score(y_test, y_proba))

# 5. Simulate Future Salaries
# Simple fixed increment simulation
df['FutureSalary'] = df['MonthlyIncome'] * 1.08

# 6. Identify "Likely to Stay" Employees
P_leave = clf.predict_proba(X)[:, 1]
P_stay = 1 - P_leave
likely_to_stay_idx = np.where(P_stay > 0.6)[0]

X_stay = X.iloc[likely_to_stay_idx]
y_salary = df.loc[likely_to_stay_idx, 'FutureSalary']

# 7. Salary Prediction (Regression)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_stay, y_salary, test_size=0.2, random_state=42)

regressor = RandomForestRegressor()
regressor.fit(X_train_reg, y_train_reg)

salary_pred = regressor.predict(X_test_reg)

print('R2 Score:', r2_score(y_test_reg, salary_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test_reg, salary_pred)))

# 8. Estimate Expected Salary Loss
expected_loss = P_leave * df['FutureSalary']
df['ExpectedLoss'] = expected_loss

total_expected_loss = df['ExpectedLoss'].sum()
print('Total Expected Financial Loss: $', total_expected_loss)

