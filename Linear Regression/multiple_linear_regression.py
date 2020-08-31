# import the necessary packages
import numpy as np
from sklearn.linear_model import LinearRegression

# the size of the house, no of bedrooms,
# and age of the home (years)
X = np.array([
    [1000, 3.0, 12],
    [1750, 4.0, 10],
    [2200, 5.0, 5]
])

# The price of these houses
y = np.array([950000, 2250000, 2400000])

# instantiate the Linear Regression class and
# training the model with available data-points
classifier = LinearRegression()
classifier.fit(X, y)

# Make some new predictions given a sample it has never seen
# before using the .predict() function and passing in the
# size of the house
x_test = np.array([1800, 4.0, 7])
prediction = classifier.predict([x_test])

print("House price with {} srft, {} bedrooms and {} years"
      " old is ${:.2f}".format(x_test[0], x_test[1],
                               x_test[2], prediction[0]))
