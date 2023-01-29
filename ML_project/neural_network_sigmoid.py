import numpy as np
import tensorflow as tf

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Binary classification problem: image is 6 or not
y_train = (y_train == 6).astype(int)
y_test = (y_test == 6).astype(int)

# Define the model
inputs = np.random.randn(784, 60000)
weights_1 = np.random.randn(64, 784)
weights_2 = np.random.randn(64, 64)
weights_3 = np.random.randn(1, 64)
bias_1 = np.zeros((64, 1))
bias_2 = np.zeros((64, 1))
bias_3 = np.zeros((1, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


# Train the model
for epoch in range(5):
    # Forward pass
    hidden_layer_1 = sigmoid(np.dot(weights_1, inputs) + bias_1)
    hidden_layer_2 = sigmoid(np.dot(weights_2, hidden_layer_1) + bias_2)
    logits = np.dot(weights_3, hidden_layer_2) + bias_3
    probs = softmax(logits)

    # Compute the loss
    labels = y_train.reshape(1, -1)
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    # Backward pass
    d_logits = probs - labels
    d_weights_3 = np.dot(d_logits, hidden_layer_2.T)
    d_bias_3 = np.sum(d_logits, axis=1, keepdims=True)
    d_hidden_layer_2 = np.dot(weights_3.T, d_logits) * \
        hidden_layer_2 * (1 - hidden_layer_2)
    d_weights_2 = np.dot(d_hidden_layer_2, hidden_layer_1.T)
    d_bias_2 = np.sum(d_hidden_layer_2, axis=1, keepdims=True)
    d_hidden_layer_1 = np.dot(
        weights_2.T, d_hidden_layer_2) * hidden_layer_1 * (1 - hidden_layer_1)

    d_weights_1 = np.dot(d_hidden_layer_1, inputs.T)
    d_bias_1 = np.sum(d_hidden_layer_1, axis=1, keepdims=True)

    # Update the weights and biases
    learning_rate = 0.001
    weights_3 -= learning_rate * d_weights_3
    bias_3 -= learning_rate * d_bias_3
    weights_2 -= learning_rate * d_weights_2
    bias_2 -= learning_rate * d_bias_2
    weights_1 -= learning_rate * d_weights_1
    bias_1 -= learning_rate * d_bias_1

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Evaluate the model
hidden_layer_1 = sigmoid(np.dot(weights_1, x_test.T) + bias_1)
hidden_layer_2 = sigmoid(np.dot(weights_2, hidden_layer_1) + bias_2)
logits = np.dot(weights_3, hidden_layer_2) + bias_3
probs = softmax(logits)
predictions = (probs > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy}')
