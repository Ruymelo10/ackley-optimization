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

    def square_sum(self, vetor):
        sum = 0
        for x in vetor:
            sum += x*x
        return sum

    def cos_sum(self, vetor):
        sum = 0
        for x in vetor:
            sum += math.cos(math.pi*2*x)
        return sum

    def calculate_fitness(self):
        result = []
        for x,desvio in self._population:
            exp1 = (-20.0)* math.exp( (-0.2) *math.sqrt(self.square_sum(x)/30.0) )
            exp2 = -math.exp(self.cos_sum(x)/30.0)
            exp =  exp2 + math.exp(1) +exp1 + 20.0 
            result.append((x,desvio,exp))
        return result
    
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
        while size > 0:
            gene = []
            for i in range(30):
                gene.append(min((random.random() * 31), 30) - 15)
            self._population.append((gene, np.random.normal(0, 1)))
            size -= 1
        
def main():
    ackley_func_opt = AckleyFunction()

    ackley_func_opt.generate_population(10)
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

    print("Quantidade de converg??ncias: ", len(converged_by_sample))
    
    print('M??dia de itera????es que o algoritmo convergiu: ', calculate_mean(converged_by_sample, 0), ' Desvio Padr??o das itera????es que o algoritmo convergiu :', calculate_std(converged_by_sample, 0))
    
    print('N??mero de indiv??duos que convergiram por execu????o:')
    for i, a in enumerate(avaliacao):
        print(f"Itera????o {i}: {a[1]}")
    
    print('Fitness m??dio da popula????o em cada uma das execu????es:')
    for i, a in enumerate(avaliacao):
        print(f"Itera????o {i}: {a[2]}")
    
    plotFig(avaliacao, 0, 'Gr??fico de converg??ncia com a m??dia de itera????es por execu????o', 'Execu????o', 'M??dia de itera????es')

    plotFig(avaliacao, 4, 'Gr??fico de converg??ncia com o melhor indiv??duo por execu????o', 'Execu????o', 'Melhor indiv??dio')
    
    print('Media Fitness: ', calculate_mean(avaliacao, 2), ' Desvio Padr??o Fitness:', calculate_std(avaliacao, 2))