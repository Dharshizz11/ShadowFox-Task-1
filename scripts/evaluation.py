import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:/Users/Dharshini/OneDrive/Desktop/Shadowfox/data/HousingData.csv")

df.fillna(df.mean(), inplace=True)

X = df.drop(columns=['MEDV'])
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)

with open("../models/linear_regression_model.pkl", "rb") as f:
    lr = pickle.load(f)

with open("../models/decision_tree_model.pkl", "rb") as f:
    dt = pickle.load(f)

y_pred_lr = lr.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

with open("evaluation_results.txt", "w") as f:
    f.write(f"Linear Regression - MSE: {mse_lr}, R² Score: {r2_lr}\n")
    f.write(f"Decision Tree - MSE: {mse_dt}, R² Score: {r2_dt}\n")

print("Evaluation completed! Check evaluation_results.txt.")