import numpy as np
import matplotlib.pyplot as plt



def softmax(z):

    z_stabilize = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stabilize)

    probabilities = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return probabilities


def cross_entropy_loss_func(y_true, y_pred):
    # get the number of samples that we have
    samples = y_true.shape[0]
    # create a zero matrix with the same shape of our prediction
    y_true_one_hot = np.zeros_like(y_pred)
    # Insert to the right place 1 instead of 0 for one hot encode
    # the np.arange(samples) = create for as the row positions
    # the y_true give for as the column position
    y_true_one_hot[np.arange(samples), y_true] = 1

    # calc the sum of loss function
    loss_val = np.sum(y_true_one_hot * np.log(y_pred + 1e-15)) / samples

    return loss_val

def gradient_descent(matrix, y, weights, learning_rate):
    length = len(y)
    # clac the score for each data in the matrix
    scores = matrix.dot(weights)
    # send the matrix to softmax to evaluate the y_pred
    probs = softmax(scores)
    y_true_one_hot = np.zeros_like(probs)
    y_true_one_hot[np.arange(length), y] = 1

    derivative_weight = np.dot(matrix.T, (probs - y_true_one_hot)) / length
    weights -= derivative_weight * learning_rate
    loss = cross_entropy_loss_func(y, probs)
    return weights, loss


def train_softmax_regression(X, y, learning_rate=0.1, epochs=1000):
    weights = np.random.randn(X.shape[1], len(np.unique(y))) * 0.01
    loss_history = []  # To store loss for each epoch
    for epoch in range(epochs):
        weights, loss = gradient_descent(X, y, weights, learning_rate)
        loss_history.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    return weights, loss_history

def predict(X, weights):
    scores = np.dot(X, weights)  # Compute raw scores (logits)
    probs = softmax(scores)      # Get probabilities via softmax
    return np.argmax(probs, axis=1)  # Predict class with highest probability



# Generate a toy dataset for demonstration
np.random.seed(42)
X = np.random.randn(150, 2)  # 150 samples, 2 features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

y = np.random.choice([0, 1, 2], size=150)  # 3 classes

# Train the model
learning_rate = 0.01
epochs = 5000
weights, loss_history = train_softmax_regression(X, y, learning_rate, epochs)


# Plot the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid()
plt.show()

# Visualize predictions
y_pred = predict(X, weights)
accuracy = np.mean(y_pred == y)  # Compute accuracy
print(f"Accuracy: {accuracy:.2%}")
y_true = np.array([0, 0, 2])

# Predicted probabilities (from Softmax)
y_pred = np.array([[0.7, 0.2, 0.1],
                   [0.9, 0.8, 0.1],
                   [0.2, 0.3, 0.5]])

# Compute the loss
loss = cross_entropy_loss_func(y_true, y_pred)

print("\nCross-Entropy Loss:")
print(loss)


