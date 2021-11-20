import random
import math
import numpy as np
import statistics
from report import plot_graphs

class OptmizationEvolutionaryStrategy:
    population = []
    interval = 0
    n = 0

    def __init__(self, interval: float, n: int) -> None:
        self.interval = interval
        self.n = n
        self.min_mutation_step = 1
        self.learing_rate = 1/math.sqrt(self.n)

    def fitness(self, params: 'list[float]') -> float:
        raise NotImplementedError

    def generate_population(self, size: int) -> None:
        self.population = []
        for i in range(size):
            params = np.random.uniform(-self.interval, self.interval, self.n).tolist()
            mutation_step = self.min_mutation_step
            chromosome = params+[mutation_step]
            self.population.append(chromosome)
    
    # Uniform selection of two parents
    def select_parents(self) -> None:
        population_size = len(self.population)
        parents = (
            self.population[int(random.uniform(0, population_size))],
            self.population[int(random.uniform(0, population_size))]
        )

        return parents

    # Discrete crossover
    def crossover(self, parents: 'tuple[list[float], list[float]]') -> 'list[float]':
        child = []
        for i in range(self.n+1):
            gene = random.choice([parents[0][i], parents[1][i]])
            child.append(gene)
        
        return child
    
    # Uncorrelated mutation with one step size
    def mutate(self, chromosome: 'list[float]') -> 'list[float]':
        mutation_step = chromosome[-1]
        params = chromosome[:-1]

        mutated_mutation_step = mutation_step*math.exp(random.gauss(0, self.learing_rate))

        # Avoid too little steps
        if mutated_mutation_step < self.min_mutation_step:
            mutated_mutation_step = self.min_mutation_step

        mutated_params = [x+random.gauss(0, mutated_mutation_step) for x in params]
        mutated_chromosome = mutated_params+[mutated_mutation_step]
        
        return mutated_chromosome
    
    # (μ, λ) selection
    def survival_selection(self, children: 'list[list[float]]', μ: int) -> None:
        children.sort(key=self.fitness)
        self.population.sort(key=self.fitness)
        self.population[-μ:] = children[:μ]
    
    def best_solution(self):
        self.population.sort(key=self.fitness)
        best_solution = self.fitness(self.population[0])

        return self.population[0], best_solution

    def fitness_statistics(self):
        fitness_population = []
        for individuo in self.population:
            fitness_population.append(self.fitness(individuo))
        return statistics.mean(fitness_population),statistics.pvariance(fitness_population),statistics.pstdev(fitness_population)

class AckleyEvolutionaryStrategy(OptmizationEvolutionaryStrategy):
    def __init__(self, interval: float, n: int) -> None:
        super().__init__(interval, n)

    def fitness(self, params: 'list[float]') -> float:
        c1, c2, c3, n = 20, 0.2, 2*math.pi, self.n
        return (
            -c1*math.exp(-c2*math.sqrt((1/n)*sum([x**2 for x in params])))
            -math.exp((1/n)*sum([math.cos(c3*x) for x in params]))
            +c1+1
        )

def evolution():
    population_size = 1000
    interval = 15
    n = 30
    max_generation = 200000
    μ = 30
    λ = 200
    optimal_solution = 0

    best_fitness_by_generation = []
    statistics_fitness_by_generation = []

    ackley_evolutionary_strategy = AckleyEvolutionaryStrategy(interval, n)
    ackley_evolutionary_strategy.generate_population(population_size)

    best_chromosome, best_solution = ackley_evolutionary_strategy.best_solution()
    statistics_fitness_by_generation.append(ackley_evolutionary_strategy.fitness_statistics())
    best_fitness_by_generation.append(best_solution)
    generation = 0
    children = []
    while best_solution != optimal_solution and generation < max_generation:
        parents = ackley_evolutionary_strategy.select_parents()
        child = ackley_evolutionary_strategy.crossover(parents)
        child = ackley_evolutionary_strategy.mutate(child)
        children.append(child)

        if len(children) == λ:
            ackley_evolutionary_strategy.survival_selection(children, μ)
            children = []

        generation += 1
        last_best_solution = best_solution
        best_chromosome, best_solution = ackley_evolutionary_strategy.best_solution()
        statistics_fitness_by_generation.append(ackley_evolutionary_strategy.fitness_statistics())
        best_fitness_by_generation.append(best_solution)

        if best_solution < last_best_solution:
            print(f"A new promising chromosome was found in generation {generation} with fitness {best_solution}")
            print(f"Chromosome: {best_chromosome}")
            print()

    print(f"Search completed in {generation} generations")
    print(f"The best chromosome found has fitness of {best_solution}")
    print(f"Chromosome: {best_chromosome}")

    plot_graphs(best_fitness_by_generation,statistics_fitness_by_generation)

if __name__ == '__main__':
    evolution()