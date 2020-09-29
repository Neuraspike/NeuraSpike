# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

from mlxtend.evaluate import bias_variance_decomp

# preparing the dataset into inputs (feature matrix) and outputs (target vector)
data = fetch_california_housing()  # instantiate the model
X = data.data  # feature matrix
y = data.target  # target vector

# split the data into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# define the model
model = LinearRegression()

# estimating the bias and variance
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, X_train,
                                                            y_train, X_test,
                                                            y_test,
                                                            loss='mse',
                                                            num_rounds=50,
                                                            random_seed=20)

# summary of the results
print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)
