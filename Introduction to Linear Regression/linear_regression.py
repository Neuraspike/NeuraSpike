# import the necessary packages
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Defining the size of the house and converting
# our array into an vector. (rank 1 array)
x = np.array([1000, 1750, 2200])
x = x.reshape(-1, 1)

# Defining our target variable which is
# the price of these houses
y = np.array([950000, 2250000, 2400000])

# instantiate the Linear Regression class and
# training the model with available data-points
classifier = LinearRegression()
classifier.fit(x, y)

# assigning the intercept coefficient to theta0
# and regression coefficient the slope of a line to theta1
theta0 = classifier.intercept_
theta1 = classifier.coef_

# The equation of a straight line
y_pred_skl = theta0 + (x * theta1)

# Visualization the line that best fits our dataset
plt.figure(figsize=(14, 8))
plt.scatter(x, y, s=200, marker='*', color='blue', cmap='RdBu')
plt.plot(x, y_pred_skl, c='r')
plt.xlabel('area(sqr ft)')
plt.ylabel('price($)')
plt.title('Linear Regression Sk-learn')
plt.grid()

# Make some new predictions given a sample it has never seen
# before using the .predict() function and passing in the
# size of the house
x_test = [1800]
prediction = classifier.predict([x_test])

print("House price for size {}(srft) \
               is ${:.2f}".format(x_test[0], prediction[0]))

plt.scatter(x_test[0], prediction[0], s=300, marker='*',
            color='green', label='test data')
plt.legend()
plt.show()