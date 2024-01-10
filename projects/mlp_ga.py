import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits

# Load the dataset
data = load_digits()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MLP classifier
mlp = MLPClassifier(solver='liblinear', activation='relu')

# Initialize the genetic algorithm optimizer
population_size = 50
max_iterations = 100
crossover_rate = 0.5
mutation_rate = 0.05

# Create an instance of the MLP classifier with random weights and biases
initial_individual = np.array([mlp.coef_, mlp.intercept_])

# Evaluate the fitness of the initial individual
fitness = mlp.fit(X_train, y_train)
fitness = fitness.score(X_test, y_test)

# Create a genetic algorithm optimizer
ga = GeneticAlgorithm(population_size, max_iterations, crossover_rate, mutation_rate)

# Run the genetic algorithm to optimize the MLP parameters
opt_individual = ga.run(initial_individual, mlp.fit)

# Plot the results
plt.plot(np.c_[X_train, X_test], y_train, 'o', label='Training data', alpha=0.5)
plt.plot(np.c_[X_train, X_test], y_test, 'o', label='Testing data', alpha=0.5)
plt.plot(X_train, y_train, 'b', label='Fitted MLP')
plt.legend()
plt.show()
