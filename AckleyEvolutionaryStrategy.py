import random
import math

class AckleyEvolutionaryStrategy:
    population = []
    interval = 0
    n = 0

    def __init__(self, interval: float, n: int) -> None:
        self.interval = interval
        self.n = n
        self.tal = 1/math.sqrt(n)

    def ackley(self, object: list(float)) -> float:
        c1, c2, c3, n = 20, 0.2, 2*math.pi, self.n
        return (
            -c1*math.exp(-c2*math.sqrt((1/n)*sum([x**2 for x in object])))
            -math.exp((1/n)*sum([math.cos(c3*x) for x in object]))
            +c1+1
        )

    def generate_population(self, size: int) -> None:
        self.population = []
        for i in range(size):
            object = random.sample(range(-self.interval, self.interval), self.n)
            mutation_step = 0
            chromosome = object+[mutation_step]
            self.population.append(chromosome)
    
    # Uniform selection of two parents
    def select_parents(self) -> None:
        population_size = len(self.population)
        parents = (
            self.population[random.uniform(0, population_size)],
            self.population[random.uniform(0, population_size)]
        )

        return parents

    # Discrete crossover
    def crossover(self, parents: tuple(list(float), list(float))) -> list(float):
        child = []
        for i in range(self.n+1):
            gene = random.choice([parents[0][i], parents[1][i]])
            child.append(gene)
        
        return child
    
    # Uncorrelated mutation with one step size
    def mutate(self, chromosome: list(float)) -> list(float):
        mutation_step = chromosome[-1]
        object = chromosome[:-1]
        mutated_mutation_step = mutation_step*math.exp(random.gauss(0, self.tal))
        mutated_object = [x+random.gauss(0, mutated_mutation_step) for x in object]
        mutated_chromosome = mutated_object+mutated_mutation_step
        
        return mutated_chromosome
    
    # (μ, λ) selection
    def survival_selection(self, children: list(list(float)), λ: int) -> None:
        children.sort(key=self.ackley)
        self.population.sort(key=self.ackley)
        self.population[-λ:] = children[:λ]