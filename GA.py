import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import argparse
import matplotlib.colors as mcolors

# Funções objetivo
def peaks(x):
    x, y = x
    return 3*(1 - x)**2 * np.exp(-x**2 - (y + 1)**2) - \
           10*(x/5 - x**3 - y**5)*np.exp(-x**2 - y**2) - \
           1/3 * np.exp(-(x + 1)**2 - y**2)

def ackley(x):
    x = np.array(x)
    first_sum = np.sum(x**2)
    second_sum = np.sum(np.cos(2 * np.pi * x))
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(first_sum / n)) - np.exp(second_sum / n) + 20 + np.e

def rastrigin(x):
    x = np.array(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

OBJECTIVE_FUNCTIONS = {
    "peaks": (peaks, 2, np.array([[-3, 3], [-3, 3]])),
    "ackley": (ackley, 2, np.array([[-35, 35], [-35, 35]])),
    "rastrigin": (rastrigin, 2, np.array([[-5.12, 5.12], [-5.12, 5.12]])),
}


def bit2num(bits, value_range):
    decimal = int("".join(str(b) for b in bits), 2)
    max_val = 2**len(bits) - 1
    return value_range[0] + (value_range[1] - value_range[0]) * decimal / max_val

def decode(individual, bit_length, search_space):
    return [bit2num(individual[i * bit_length:(i + 1) * bit_length], search_space[i])
            for i in range(len(search_space))]

def create_latin_square(n, interleave=True):
    latin_square = np.empty((n, n), dtype=int)
    latin_square[:, 0] = np.arange(1, n + 1)

    if interleave:
        shifts = [int(((0.5 - (i % 2)) / 0.5) * np.ceil((i + 1) / 2)) for i in range(1, n)]
    else:
        shifts = list(range(n - 1, 0, -1))

    for col in range(1, n):
        latin_square[:, col] = np.roll(latin_square[:, 0], shifts[col - 1])

    return latin_square

def initialize_population(population_size, total_bits, search_space=None, bits_per_variable=None, init_method="random"):
    if init_method == "latin_square":
        num_vars = len(search_space)
        latin = create_latin_square(population_size)
        population = []

        for row in latin:
            individual = []
            for i, val in enumerate(row[:num_vars]):
                bounds = search_space[i]
                val_scaled = bounds[0] + (bounds[1] - bounds[0]) * (val - 1) / (population_size - 1)

                max_int = 2 ** bits_per_variable - 1
                scaled = int(round((val_scaled - bounds[0]) * max_int / (bounds[1] - bounds[0])))
                bin_str = f"{scaled:0{bits_per_variable}b}"
                individual.extend([int(b) for b in bin_str])
            population.append(individual)

        return population
    else:
        return [np.random.randint(0, 2, total_bits).tolist() for _ in range(population_size)]

def evaluate_population(population, bit_length, search_space, objective_function, maximize):
    values = [objective_function(decode(ind, bit_length, search_space)) for ind in population]
    return [-v if maximize else v for v in values] # Faz o codigo funcionar para maximizacao (corrige o select)

def select(population, fitness):
    min_fit = min(fitness)
    if min_fit < 0:
        fitness = [f - min_fit + 1e-6 for f in fitness]  # desloca para que todos sejam positivos
    total_fit = sum(fitness)
    probs = [f / total_fit for f in fitness]
    return population[np.random.choice(len(population), p=probs)]

def crossover(p1, p2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1, p2

def mutate(individual, mutation_rate):
    return [bit if random.random() > mutation_rate else 1 - bit for bit in individual]

def plot_3d_points_with_surface(generational_points, objective_name):
    func, _, bounds = OBJECTIVE_FUNCTIONS[objective_name]
    x_range = np.linspace(bounds[0][0], bounds[0][1], 60)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 60)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[func([x, y]) for x in x_range] for y in y_range])

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f9f9f9')  # cinza bem claro

    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=cm.jet, antialiased=False, edgecolor='none', linewidth=0.1, alpha=0.3)
    #ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='lightsteelblue', linewidth=0.5, alpha=0.6)
    #ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)-5, cmap='Blues', linewidths=0.5)

    markers = ['o', '^', 's', 'X', 'P', '*', 'D', 'v', '<', '>']
    legend_handles = []

    for gen_idx, (all_points, best_point) in enumerate(generational_points):
        if gen_idx % 10 == 0 or gen_idx+1 == len(generational_points):
            m = markers[(gen_idx // 5) % len(markers)]
            all_but_best = [pt for pt in all_points if pt != best_point]
            if all_but_best:
                Xp, Yp, Zp = zip(*all_but_best[0:20])
                sc1 = ax.scatter(Xp, Yp, Zp, c='gray', edgecolor='black', marker=m, s=40, alpha=1, label=f'Indivíduos da geração {gen_idx+1}')
            sc2 = ax.scatter(*best_point, c='red', marker=m, s=220, edgecolor='black', linewidth=0.6)
            legend_handles.append(sc1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    ax.set_title(f'Evolução na Superfície - {objective_name}')
    ax.legend(handles=legend_handles)
    plt.show()

def plot_2d(best_scores, objective_name):
    plt.plot(range(1, len(best_scores)+1), best_scores, label="Fitness obtido", color='blue')
    plt.xlabel("Gerações")
    plt.ylabel("Melhor Fitness")
    plt.title(f"Evolução do GA - {objective_name}")
    plt.legend()
    plt.grid()
    plt.show()


def run_ga(objective_name, generations, population_size, crossover_rate, mutation_rate, elitism, bit_length, maximize, init_method="latin_cube"):
    objective_function, num_variables, search_space = OBJECTIVE_FUNCTIONS[objective_name]
    total_bits = bit_length * num_variables
    population = initialize_population(
        population_size, 
        total_bits,
        search_space=search_space,
        bits_per_variable=bit_length,
        init_method=init_method
    )
    elitism_count = int(population_size * elitism)

    best_scores = []
    generational_points = []
    unchanged_generations = 0
    last_best_val = None

    for gen in range(generations):
        fitness = evaluate_population(population, bit_length, search_space, objective_function, maximize)
        sorted_indices = np.argsort(fitness)
        elite_individuals = [population[i] for i in sorted_indices[:elitism_count]]
        new_population = elite_individuals.copy()

        gen_points = [(decode(ind, bit_length, search_space)[0],
                       decode(ind, bit_length, search_space)[1],
                       objective_function(decode(ind, bit_length, search_space)))
                      for ind in population]

        best_index = np.argmin(fitness)
        best_ind = decode(population[best_index], bit_length, search_space)
        best_val = objective_function(best_ind)
        best_point = (*best_ind, best_val)
        generational_points.append((gen_points, best_point))

        print(f"Geração {gen+1} | Melhor fitness: {best_val:.4f} | X: {best_ind[0]:.4f}, Y: {best_ind[1]:.4f}, Z: {best_val:.4f}")

        if best_val == last_best_val:
            unchanged_generations += 1
        else:
            unchanged_generations = 0
            last_best_val = best_val

        if unchanged_generations >= 20:
            print("Parando execução: melhor indivíduo não mudou por 20 gerações.")
            break

        while len(new_population) < population_size:
            parent1 = select(population, fitness)
            parent2 = select(population, fitness)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population[:population_size]
        best_scores.append(best_val if not maximize else -best_val)

    #plot_2d(best_scores, objective_name)
    plot_3d_points_with_surface(generational_points, objective_name)
    return best_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algoritmo Genético CLI")
    parser.add_argument("-o", "--objective", choices=OBJECTIVE_FUNCTIONS.keys(), default="rastrigin", help="Função objetivo")
    parser.add_argument("-g", "--generations", type=int, default=40, help="Número de gerações")
    parser.add_argument("-p", "--population", type=int, default=25, help="Tamanho da população")
    parser.add_argument("-c", "--crossover", type=float, default=0.6, help="Taxa de cruzamento")
    parser.add_argument("-m", "--mutation", type=float, default=0.03, help="Taxa de mutação")
    parser.add_argument("-e", "--elitism", type=float, default=0.55, help="Taxa de Elitismo")
    parser.add_argument("-b", "--bits", type=int, default=10, help="Tamanho do cromossomo em bits por variável")
    parser.add_argument("-f", "--function", type=str, default="max", help="Executa GA para maximizar a função objetivo")
    parser.add_argument("-i", "--init", type=str, default="latin_cube", help="Metodo de init da primeira população")
    args = parser.parse_args()

    run_ga(
        objective_name = args.objective,
        generations = args.generations,
        population_size = args.population,
        crossover_rate =args.crossover,
        mutation_rate = args.mutation,
        elitism = args.elitism,
        bit_length = args.bits,
        maximize = args.function == "max",
        init_method = args.init
    )
