import pandas as pd
import numpy as np
import sys
import random

#Read training data
train_dataset = pd.read_csv(sys.argv[1]).as_matrix()
train_labels = train_dataset[:, 0]
train_dataset = np.delete(train_dataset, 0, 1)
print("Training data", train_dataset.shape)

final_train_dataset = []
for count in range(len(train_dataset)):
	new_x = train_dataset[count]/255.0
	new_x = new_x.reshape((len(new_x), 1))
	new_y = np.zeros(10)
	new_y[train_labels[count]] = 1
	new_y = new_y.reshape((10, 1))
	final_train_dataset.append((new_x, new_y))

print(final_train_dataset[0][1].shape)
#Read test data
test_dataset = pd.read_csv(sys.argv[2]).as_matrix()
print("Test data", test_dataset.shape)

final_test_dataset = []
for count in range(len(test_dataset)):
	new_elem = test_dataset[count]/255.0
	new_elem = new_elem.reshape((len(new_elem), 1))
	final_test_dataset.append(new_elem)

#print(final_test_dataset[0])

sizes = [len(final_train_dataset[0][0]), 30, 10]
numlayers = len(sizes)

biases = [np.random.randn(y, 1) for y in sizes[1:]]
#print(biases)

weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
#print(weights)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def forward(a):
	global biases
	global weights
	for b, w in zip(biases, weights):
		a = sigmoid(np.dot(w, a)+b)
	return a

def cost_derivative(output_activations, y):
	return (output_activations-y)

def backprop(x, y):
	#print(x.shape)
	#print(y.shape)
	global biases
	global weights
	global numlayers
	nabla_b = [np.zeros(b.shape) for b in biases]
	nabla_w = [np.zeros(w.shape) for w in weights]
	activation = x
	activations = [x]
	zs = []
	#print(activation.shape)
	for b, w in zip(biases, weights):
		z = np.dot(w, activation) +b
		zs.append(z)
		activation = sigmoid(z)
		#print(activation.shape)
		activations.append(activation)
	delta = cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
	#print(len(activations))
	#print(delta.shape)
	#print(activations[-2].transpose().shape)
	nabla_b[-1] = delta
	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
	for l in xrange(2, numlayers):
		z = zs[-l]
		sp = sigmoid_prime(z)
		#print(delta.shape)
		#print(weights[-l+1].transpose().shape)
		delta = np.dot(weights[-l+1].transpose(), delta)*sp
		nabla_b[-l] = delta
		#print(delta.shape)
		#print(activations[-l-1].transpose().shape)
		nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	return (nabla_b, nabla_w)

def update_mini_batch(mini_batch, eta):
	global biases
	global weights
	nabla_b = [np.zeros(b.shape) for b in biases]
	nabla_w = [np.zeros(w.shape) for w in weights]
	for x, y in mini_batch:
		delta_nabla_b, delta_nabla_w = backprop(x, y)
		nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
		nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
	weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(weights, nabla_w)]
	biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(biases, nabla_b)]

def SGD(training_data, epochs, mini_batch_size, eta):
	n = len(training_data)
	for j in xrange(epochs):
		print(j)
		random.shuffle(training_data)
		mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
		for mini_batch in mini_batches:
			update_mini_batch(mini_batch, eta)


#input = np.array([1, 1])
#print(forward(input))
print(len(final_train_dataset[0][0]))
SGD(final_train_dataset, 100, 100, 3.0)

predictions = [np.argmax(forward(x)) for x in final_test_dataset]

#predictions = np.zeros(test_dataset.shape[0], dtype=np.int64)

dataFrame = pd.DataFrame(predictions, columns=["Label"])
dataFrame.index += 1
dataFrame.to_csv("output.csv", index_label="ImageId")
print("Generated output.csv")