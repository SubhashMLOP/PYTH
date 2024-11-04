# creatin my new git project for ml

import numpy as np
#example coefficient for the linear model(y=ax+b)
a,b = 2,1

coefficients = np.array([a,b])

x_data = np.array([[1,1],[2,1],[3,1]])
y_data = np.array([3,5,7]) # Actual values

y_pred = x_data @ coefficients

# calculate the loss (mean squared error)

loss = np.mean((y_data - y_pred) ** 2)

print("Predictes y:", y_pred)
print("loss (MSE):" , loss)