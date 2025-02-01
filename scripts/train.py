import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("C:/Users/Dharshini/OneDrive/Desktop/Shadowfox/data/HousingData.csv")

df.fillna(df.mean(), inplace=True)
print(df.isnull().sum())

X = df.drop(columns=['MEDV'])
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

try:
    with open("../models/linear_regression_model.pkl", "wb") as f:
        pickle.dump(lr, f)

    with open("../models/decision_tree_model.pkl", "wb") as f:
        pickle.dump(dt, f)

    with open("../models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Models trained and saved successfully!")

except FileNotFoundError:
    print("Error: 'models' directory not found. Please create it manually in the project folder.")





