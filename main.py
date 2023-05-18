import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class GeneticAlgorithm:
    def __init__(self, population_size, num_generations):
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = []

    def generate_individual(self):
        hidden_layer_sizes = [random.randint(5, 20) for _ in range(random.randint(1, 3))]
        activation = random.choice(['relu', 'tanh', 'logistic'])
        solver = random.choice(['adam', 'sgd'])
        learning_rate = random.choice(['constant', 'adaptive'])
        max_iter = random.randint(100, 500)

        return {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'learning_rate': learning_rate,
            'max_iter': max_iter
        }

    def generate_population(self):
        for _ in range(self.population_size):
            individual = self.generate_individual()
            self.population.append(individual)

    def evaluate_individual(self, individual):
        algorithm = MLPClassifier(
            hidden_layer_sizes=individual['hidden_layer_sizes'],
            activation=individual['activation'],
            solver=individual['solver'],
            learning_rate=individual['learning_rate'],
            max_iter=individual['max_iter']
        )

        algorithm.fit(X_train, y_train)
        y_pred = algorithm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def select_parents(self, num_parents):
        fitness_scores = [self.evaluate_individual(individual) for individual in self.population]
        selected_indices = np.argsort(fitness_scores)[-num_parents:]
        parents = [self.population[idx] for idx in selected_indices]

        return parents

    def crossover(self, parents):
        offspring = []

        for _ in range(self.population_size):
            parent1, parent2 = random.choices(parents, k=2)
            child = {}

            for key in parent1:
                if random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]

            offspring.append(child)

        return offspring

    def mutate(self, offspring):
        for individual in offspring:
            if random.random() < 0.1:
                individual['hidden_layer_sizes'] = [random.randint(5, 20) for _ in range(random.randint(1, 3))]
            if random.random() < 0.1:
                individual['activation'] = random.choice(['relu', 'tanh', 'logistic'])
            if random.random() < 0.1:
                individual['solver'] = random.choice(['adam', 'sgd'])
            if random.random() < 0.1:
                individual['learning_rate'] = random.choice(['constant', 'adaptive'])
            if random.random() < 0.1:
                individual['max_iter'] = random.randint(100, 500)

    def evolve(self):
        self.generate_population()

        for generation in range(self.num_generations):
            parents = self.select_parents(num_parents=10)
            offspring = self.crossover(parents)
            self.mutate(offspring)

            self.population = parents + offspring

        best_individual = max(self.population, key=self.evaluate_individual)
        best_accuracy = self.evaluate_individual(best_individual)

        return best_individual, best_accuracy


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

ga = GeneticAlgorithm(population_size=50, num_generations=100)
best_individual, best_accuracy = ga.evolve()

print("Best Individual:")
print(best_individual)
print("Best Accuracy:", best_accuracy)
