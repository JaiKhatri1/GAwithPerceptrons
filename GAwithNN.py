import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_bias(self):
        return self.bias

    def set_bias(self, bias):
        self.bias = bias


class GeneticAlgorithm:
    def __init__(self, population_size, input_size):
        self.population_size = population_size
        self.input_size = input_size
        self.population = [Perceptron(input_size) for _ in range(population_size)]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(0, self.input_size)
        child_weights = np.concatenate((parent1.get_weights()[:crossover_point],
                                        parent2.get_weights()[crossover_point:]))
        child_bias = (parent1.get_bias() + parent2.get_bias()) / 2
        child = Perceptron(self.input_size)
        child.set_weights(child_weights)
        child.set_bias(child_bias)
        return child

    def mutate(self, perceptron, mutation_rate):
        weights = perceptron.get_weights()
        bias = perceptron.get_bias()
        for i in range(len(weights)):
            if np.random.rand() < mutation_rate:
                weights[i] += np.random.normal(scale=0.1)
        if np.random.rand() < mutation_rate:
            bias += np.random.normal(scale=0.1)
        perceptron.set_weights(weights)
        perceptron.set_bias(bias)

    def evolve(self, fitness_scores, mutation_rate):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = np.random.choice(self.population, size=2, p=fitness_scores/np.sum(fitness_scores), replace=False)
            child = self.crossover(parent1, parent2)
            self.mutate(child, mutation_rate)
            new_population.append(child)
        self.population = new_population

    def get_best_perceptron(self, inputs, targets):
        best_perceptron = None
        best_accuracy = 0
        for perceptron in self.population:
            accuracy = sum(perceptron.predict(inputs[i]) == targets[i] for i in range(len(inputs))) / len(inputs)
            if accuracy > best_accuracy:
                best_perceptron = perceptron
                best_accuracy = accuracy
        return best_perceptron, best_accuracy

def plot_decision_boundary(perceptron):
    xx, yy = np.meshgrid(np.arange(-0.5, 1.5, 0.01),
                         np.arange(-0.5, 1.5, 0.01))
    Z = np.array([perceptron.predict([x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(data[:, 0], data[:, 1], c=targets, cmap=plt.cm.binary)
    plt.title('Decision Boundary of Best-Performing Perceptron')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.show()

# # Experiemnt 1
# population_size = 200
# input_size = 2
# mutation_rate = 0.7
# generations = 100

'''Linearly separable problems'''
# data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# targets = np.array([0, 1, 1, 1])

# data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# targets = np.array([0, 1, 1, 1])

# data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# targets = np.array([1, 1, 1, 0])

# data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.2, 0.8]])
# targets = np.array([1, 1, 1, 0, 1, 1])

# data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.2, 0.8]])
# targets = np.array([1, 0, 1, 0, 1, 1])

# Experiemnt 2
population_size = 150
input_size = 2
mutation_rate = 0.8
generations = 200

# -------------------
'''XOR problem (Non-Linearly separable problem)'''
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])




ga = GeneticAlgorithm(population_size, input_size)

best_accuracies = []

for generation in range(generations):
    fitness_scores = []
    for perceptron in ga.population:
        accuracy = sum(perceptron.predict(data[i]) == targets[i] for i in range(len(data))) / len(data)
        fitness_scores.append(accuracy)
    best_perceptron, best_accuracy = ga.get_best_perceptron(data, targets)
    best_accuracies.append(best_accuracy)
    print(f"Generation {generation + 1}: Best Accuracy - {best_accuracy}")
    if best_accuracy == 1.0:  # If best accuracy is 100%, end the program
        print("Best accuracy reached 100% in generation", generation + 1)
        plt.plot(range(1, generation + 2), best_accuracies)
        plt.title('Evolution of Best Accuracy Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Accuracy')
        plt.show()
        plot_decision_boundary(best_perceptron)
        exit()  # End the program

    ga.evolve(np.array(fitness_scores), mutation_rate)

best_perceptron, best_accuracy = ga.get_best_perceptron(data, targets)
print("Final Best Accuracy:", best_accuracy)
print("Best Weights:", best_perceptron.get_weights())
print("Best Bias:", best_perceptron.get_bias())

plt.plot(range(1, generations + 1), best_accuracies)
plt.title('Evolution of Best Accuracy Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Accuracy')
plt.show()

plot_decision_boundary(best_perceptron)
