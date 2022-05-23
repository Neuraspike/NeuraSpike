# import the necessary packages
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style='darkgrid')


def next_batch(features, labels, batch_size):
    """
    :param features: feature matrix
    :param labels: target vector
    :param batch_size:  size of mini-batch 
    :return: a list of the current batched features and labels
    """
 
    # iterate though a mini batch of both our features and labels
    for data in range(0, np.shape(features)[0], batch_size):
        
        # append the current batched features and labels in a list
        yield (features[data: data+batch_size], labels[data: data+batch_size])


def stochastic_gradient_descent(X, y, alpha=0.01, epochs=100, batch_size=1):
    """
    :param X: feature vector
    :param y: target vector
    :param alpha: learning rate (default=0.01)
    :param epochs: maximum number of iterations of the linear regression 
                         algorithm for a single run (default=100)
    :param batch_size: the portion of the mini-batch (default=1)
    :return: weights, list of cost function changing overtime
    """

    m = np.shape(X)[0]  # total number of samples
    n = np.shape(X)[1]  # total number of features

    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    W = np.random.randn(n + 1, )

    # stores the updates on the lost/cost function
    cost_history_list = []

    # iterate until the maximum number of epochs
    for current_iteration in range(epochs):  # begin the process

        # save the total lost/cost during each batch
        batch_epoch_loss_list = []

        for (X_batch, y_batch) in next_batch(X, y, batch_size):
            # current size of the feature batch
            batch_m = np.shape(X_batch)[0]

            # compute the dot product between our
            # feature 'X_batch' and weight 'W'
            y_estimated = X_batch.dot(W)

            # calculate the difference between the actual
            # and estimated value
            error = y_estimated - y_batch

            # get the cost (Mean squared error -MSE)
            cost = (1 / 2 * m) * np.sum(error ** 2)

            batch_epoch_loss_list.append(cost)  # save it to a list

            # Update our gradient by the dot product between the
            # transpose of 'X_batch' and our error divided by the
            # few number of samples
            gradient = (1 / batch_m) * X_batch.T.dot(error)

            # Now we have to update our weights
            W = W - alpha * gradient

        # Let's print out the cost to see how these values
        # changes after every each iteration
        print(f"cost:{np.average(batch_epoch_loss_list)} \t"
              f" iteration: {current_iteration}")

        # keep track of the cost
        cost_history_list.append(np.average(batch_epoch_loss_list))

    # return both our weight and list of cost function changing overtime
    return W, cost_history_list


def main():
    rng = np.random.RandomState(10)
    X = 10 * rng.rand(1000, 5)  # feature matrix
    y = 0.9 + np.dot(X, [2.2, 4., -4, 1, 2])  # target vector

    # calls the stochastic gradient descent method
    weight, cost_history_list = stochastic_gradient_descent(X, y,
                                                            alpha=0.001,
                                                            epochs=10,
                                                            batch_size=32)

    # visualize how our cost decreases over time
    plt.plot(np.arange(len(cost_history_list)), cost_history_list)
    plt.xlabel("Number of iterations (Epochs)")
    plt.ylabel("Cost function  J(Θ)")
    plt.title("Stochastic Gradient Descent")
    plt.show()


if __name__ == '__main__':
    main()
