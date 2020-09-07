current_guess = 5  # we randomly start at x=5
alpha = 0.15  # the learning rate
total_iteration = 30  # total number of time we will run the algorithm
current_iteration = 0  # keep track of the current iteration
precision = 0.0001  # determines the stop condition of the step-wise descent
height = float('inf')  # set the height as maximum


# the derivative of our function (x^2 - 4x + 2)
def derivative(x):
    """
    :param x: the initial starting point (numerical value)
    :return: the derivative of x based on the input value (x)
    """
    return 2 * x - 4


# check if the difference between our previous guess
# and current guess is small and also if we haven't
# reached the total number of iterations defined.

while height > precision and current_iteration < total_iteration:
    previous_guess = current_guess  # keep track of our previous guess

    # perform gradient descent
    current_guess = previous_guess - alpha * derivative(current_guess)

    # increment the counter once the process is complete
    current_iteration = current_iteration + 1

    # keep track of the difference between our previous and current guess
    height = abs(current_guess - previous_guess)

    print(f"Epoch: {current_iteration}/{total_iteration}\t"
          f" x: {current_guess:.4f}\theight {height:.4f}")
