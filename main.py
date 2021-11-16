import random
import math
import numpy as np
import matplotlib.pyplot as plt

class AckleyFunction:
    _population = []

    def __init__(self) -> None:
        pass
    
    def solution(self):
       #add implementation

    def rank(self, sample=None):
        #add implementation
    
    def calculate_fitness(self, chromosome):
       #add implementation

    def select_random_parents(self, population):
      #add implementation

    def parent_selection(self, population):
      #add implementation
    
    def cut_and_crossfill(self, parents):
      #add implementation

    def mutate(self, child):
       #add implementation
  
    def crossfill(self,child, parent,cut_point):
        #add implementation

    def survivors_selection(self,children):
        #add implementation

    def generate_population(self, size):
        #add implementation
        
def main():
    #ackley_func_opt = AckleyFunction()

    #ackley_func_opt.generate_population(100)
    #solution = ackley_func_opt.solution()
    #population_fitness = ackley_func_opt.rank()
    #count = 0
    while solution == None and count < 10000:
        #parents = ackley_func_opt.parent_selection(population_fitness)
        #children = ackley_func_opt.cut_and_crossfill(parents)
        #ackley_func_opt.survivors_selection(children)
        #population_fitness = ackley_func_opt.rank()
        #solution = ackley_func_opt.solution()
        #count+=1
    total_converged = len(list(filter(lambda x : x[1] == 1, population_fitness)))
    return (count, total_converged, calculate_mean(population_fitness,1), calculate_std(population_fitness,1), max(list(map(lambda x : x[1], population_fitness))))

def calculate_mean(generations, pos):
    return np.mean(list(map(lambda x : x[pos], generations)))

def calculate_std(generations, pos):
    return np.std(list(map(lambda x : x[pos], generations)))

def plotFig(generations, pos,name, xlabel, ylabel):
    iterations = list(map(lambda x : x[pos], generations))
    fig = plt.figure()
    plt.plot(iterations)
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    fig.savefig(name, dpi=fig.dpi)

if __name__ == '__main__':
    avaliacao = []
    for i in range(30):
        avaliacao.append(main())

    converged_by_sample = list(filter(lambda x : x[1] != 0, avaliacao))

    print("Quantidade de convergências: ", len(converged_by_sample))
    
    print('Média de iterações que o algoritmo convergiu: ', calculate_mean(converged_by_sample, 0), ' Desvio Padrão das iterações que o algoritmo convergiu :', calculate_std(converged_by_sample, 0))
    
    print('Número de indivíduos que convergiram por execução:')
    for i, a in enumerate(avaliacao):
        print(f"Iteração {i}: {a[1]}")
    
    print('Fitness médio da população em cada uma das execuções:')
    for i, a in enumerate(avaliacao):
        print(f"Iteração {i}: {a[2]}")
    
    plotFig(avaliacao, 0, 'Gráfico de convergência com a média de iterações por execução', 'Execução', 'Média de iterações')

    plotFig(avaliacao, 4, 'Gráfico de convergência com o melhor indivíduo por execução', 'Execução', 'Melhor indivídio')
    
    print('Media Fitness: ', calculate_mean(avaliacao, 2), ' Desvio Padrão Fitness:', calculate_std(avaliacao, 2))