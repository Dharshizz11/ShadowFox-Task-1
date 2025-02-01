import pickle
import numpy as np
import pandas as pd

with open("../models/linear_regression_model.pkl", "rb") as f:
    lr = pickle.load(f)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

new_data = np.array([[0.1, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]])  

new_data_df = pd.DataFrame(new_data, columns=feature_names)

new_data_scaled = scaler.transform(new_data_df)

predicted_price = lr.predict(new_data_scaled)
print(f"Predicted House Price: {predicted_price[0]}")