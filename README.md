# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  initialize weights
2. predict output
3. compute error
4. update weights


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: F.SENASH AYISHA
RegisterNumber:25018487
Ex No: 4 import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample dataset
# Features: [house_size_sqft, number_of_rooms]
X = np.array([
    [800, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2000, 4],
    [2200, 5],
    [2500, 5]
])

# Targets: [house_price, number_of_occupants]
y = np.array([
    [150000, 2],
    [200000, 3],
    [240000, 3],
    [300000, 4],
    [350000, 4],
    [400000, 5],
    [450000, 6]
])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SGD Regressor model (wrapped for multi-output prediction)
model = MultiOutputRegressor(SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01))

# Train the model
model.fit(X_train, y_train)

# Test prediction
prediction = model.predict(X_test)

print("Test Input (Scaled):")
print(X_test)

print("\nPredicted [House Price, Occupants]:")
print(prediction)

print("\nActual Values:")
print(y_test)

# Predict for a new house
new_house = np.array([[2100, 4]])
new_house_scaled = scaler.transform(new_house)

new_prediction = model.predict(new_house_scaled)

print("\nPrediction for new house (2100 sqft, 4 rooms):")
print("Estimated Price:", new_prediction[0][0])
print("Estimated Occupants:", round(new_prediction[0][1])) 
*/
```

## Output:
<img width="775" height="107" alt="image" src="https://github.com/user-attachments/assets/7e575fc4-b542-4395-9a72-f0eda43250a3" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
