# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='darkgrid')


def gradient_descent(X, y, alpha=0.01, epochs=30):
    """
    :param x: feature matrix
    :param y: target vector
    :param alpha: learning rate (default:0.01)
    :param epochs: maximum number of iterations of the
           linear regression algorithm for a single run (default=30)
    :return: weights, list of the cost function changing overtime
    """

    m = np.shape(X)[0]  # total number of samples
    n = np.shape(X)[1]  # total number of features

    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    W = np.random.randn(n + 1, 1)

    # stores the updates on the cost function (loss function)
    cost_history_list = []

    # iterate until the maximum number of epochs
    for current_iteration in np.arange(epochs):  # begin the process

        # compute the forward pass of the network
        y_estimated = X.dot(W)

        # calculate the difference between the actual and predicted value
        errors = y - y_estimated

        # calculate the cost (Mean squared error - MSE)
        cost = (1 / 2 * m) * np.sum(errors ** 2)

        # Now we need to compute the backward pass of the network,
        # where we will update our weights by taking the second
        # derivative of our loss function
        gradient = -(1 / m) * X.T.dot(errors)

        # Now we have to update our weights
        W = W - alpha * gradient

        # Let's print out the cost to see how these values
        # changes after every 100th iteration
        if current_iteration % 100 == 0:
            print(f"cost:{cost} \t iteration: {current_iteration}")

        # keep track the cost as it changes in each iteration
        cost_history_list.append(cost)

    return W, cost_history_list


def main():
    # Link to the data
    url_link = "https://git.io/JU4VS"

    # retrieves content from the webserver
    auto_dataset = pd.read_csv(url_link)

    # selected the feature variables we are interested in
    features_index = ["length", "width", "curb-weight", "engine-size", "bore",
                      "horsepower", "city-mpg", "highway-mpg", "wheel-base"]

    # specify our feature matrix
    X = auto_dataset[features_index].values

    # the price of the car given the features
    label_index = ["price"]
    y = auto_dataset[label_index].values

    # perform data standardization
    X = (X - X.mean()) / X.std()

    # calls the gradient descent method
    weight, cost_history_list = gradient_descent(X, y, alpha=0.1, epochs=1000)

    # visualize how our cost decreases over time
    plt.plot(np.arange(len(cost_history_list)), cost_history_list)
    plt.xlabel("Number of iterations (Epochs)")
    plt.ylabel("Cost function  J(Θ)")
    plt.title("Gradient Descent")
    plt.show()


if __name__ == '__main__':
    main()
