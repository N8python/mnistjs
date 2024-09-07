import numpy as np
import struct
import time
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(y):
    return 1 - y**2

class LinearLayer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(output_size, input_size) / np.sqrt(input_size)
        self.bias = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, input) + self.bias
        self.activated = self.activation(self.output)
        return self.activated

    def backward(self, output_error, learning_rate):
        if self.activation == sigmoid:
            d_activated = output_error * sigmoid_derivative(self.activated)
        else:
            d_activated = output_error * tanh_derivative(self.activated)
        
        d_weights = np.dot(d_activated, self.input.T)
        d_bias = np.mean(d_activated, axis=1, keepdims=True)
        d_input = np.dot(self.weights.T, d_activated)
        
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        return d_input

class MLP:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(LinearLayer(layers[i], layers[i+1], 
                                           sigmoid if i == len(layers) - 2 else tanh))

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, error, learning_rate):
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

def load_mnist_data(data_path, labels_path, num_samples):
    # Load raw image data
    with open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Reshape data to (num_samples, 784)
    data = data[:num_samples * 784].reshape(num_samples, 784)
    data = data / 255.0  # Normalize

    # Load raw label data
    with open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    labels = labels[:num_samples]

    if len(data) != len(labels):
        raise ValueError(f"Data and label counts do not match. Data: {len(data)}, Labels: {len(labels)}")

    dataset = []
    for i in range(len(data)):
        input = data[i].reshape(-1, 1)
        output = np.zeros((10, 1))
        output[labels[i]] = 1
        dataset.append((input, output))

    return dataset

def create_progress_bar(total, length=30):
    def update(current, max, epoch, loss, accuracy):
        filled = int(length * current / max)
        bar = '█' * filled + '-' * (length - filled)
        percent = round(100 * current / max)
        print(f'\rEpoch {epoch}: [{bar}] {percent}% | {current}/{max} iters, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}', end='')
    return update

def train(network, epochs=10, learning_rate=1.0, batch_size=128):
    learning_rate /= batch_size
    training_set = load_mnist_data('mnist_train_data.bin', 'mnist_train_labels.bin', 60000)
    test_set = load_mnist_data('mnist_test_data.bin', 'mnist_test_labels.bin', 10000)

    for epoch in range(epochs):
        total_error = 0
        accuracy = 0
        np.random.shuffle(training_set)

        update_progress_bar = create_progress_bar(len(training_set))
        total_batches = len(training_set) // batch_size
        start_time = time.time()

        for batch_idx in range(0, len(training_set), batch_size):
            batch = training_set[batch_idx:batch_idx + batch_size]
            inputs = np.hstack([data[0] for data in batch])
            targets = np.hstack([data[1] for data in batch])

            outputs = network.forward(inputs)
            errors = outputs - targets

            network.backward(errors, learning_rate)

            batch_error = np.mean(np.sum(errors**2, axis=0))
            batch_acc = np.mean(np.argmax(outputs, axis=0) == np.argmax(targets, axis=0))

            total_error += batch_error * len(batch)
            accuracy += batch_acc * len(batch)

            update_progress_bar(batch_idx // batch_size, total_batches, epoch + 1, batch_error, batch_acc)

        end_time = time.time()
        avg_error = total_error / len(training_set)
        avg_accuracy = accuracy / len(training_set)
        update_progress_bar(total_batches, total_batches, epoch + 1, avg_error, avg_accuracy)
        
        test_accuracy = test(network, test_set)
        print(f', Test accuracy: {test_accuracy:.3f}, Time taken: {end_time - start_time:.2f}s, Images/sec: {len(training_set) // (end_time - start_time):.0f}')

def test(network, dataset):
    correct = 0
    for data in dataset:
        input, target = data
        output = network.forward(input)
        if np.argmax(output) == np.argmax(target):
            correct += 1
    return correct / len(dataset)

def main():
    network = MLP([784, 64, 10])
    train(network, 10)

if __name__ == "__main__":
    main()