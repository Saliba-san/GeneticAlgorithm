import random
import statistics
import csv
from GA import run_ga, OBJECTIVE_FUNCTIONS

def evaluate_hyperparameters(objective, generations, population, crossover, mutation, elitism, bits, maximize=True, runs=5):
    results = []
    gen_counts = []
    for _ in range(runs):
        try:
            result = run_ga(
                objective, generations, population, crossover, mutation, elitism, bits, maximize
            )
            if isinstance(result, list) and result:
                results.append(result[-1])
                gen_counts.append(len(result))
        except:
            continue

    if not results:
        return float('inf'), float('inf'), float('inf')

    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0
    gen_avg = statistics.mean(gen_counts)

    score = avg + (std * 0.5) + (gen_avg * 0.1)
    return score, avg, std

def meta_ga(objective, population_size=15, generations=30, export_csv=True):
    bounds = {
        'generations': (10, 50),
        'population': (10, 100),
        'crossover': (0.6, 0.8),
        'mutation': (0.01, 0.05),
        'elitism': (0.55, 0.75),
        'bits': (10, 35),
    }

    def random_individual():
        return {
            'generations': random.randint(*bounds['generations']),
            'population': random.randint(*bounds['population']),
            'crossover': round(random.uniform(*bounds['crossover']), 4),
            'mutation': round(random.uniform(*bounds['mutation']), 4),
            'elitism': round(random.uniform(*bounds['elitism']), 4),
            'bits': random.randint(*bounds['bits']),
        }

    def mutate(ind):
        key = random.choice(list(bounds.keys()))
        if isinstance(bounds[key][0], int):
            ind[key] = random.randint(*bounds[key])
        else:
            ind[key] = round(random.uniform(*bounds[key]), 4)
        return ind

    def crossover(p1, p2):
        child = {}
        for k in bounds:
            child[k] = p1[k] if random.random() < 0.5 else p2[k]
        return child

    population = [random_individual() for _ in range(population_size)]
    all_results = []

    for gen in range(generations):
        print(f"\nMeta-Geração {gen+1}/{generations}")
        scored = []
        for ind in population:
            score, avg_fit, std = evaluate_hyperparameters(objective, **ind)
            scored.append((score, avg_fit, std, ind))
            print(f"Score: {score:.4f} -> {ind}")

        scored.sort(key=lambda x: x[0])
        population = [entry[3] for entry in scored[:population_size//2]]
        all_results.extend(scored)

        while len(population) < population_size:
            p1, p2 = random.sample(population[:5], 2)
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
            population.append(child)

    best = min(all_results, key=lambda x: x[0])
    print(f"\nMelhor configuração encontrada para {objective}:")
    print(f"Score final: {best[0]:.4f}")
    print("Parâmetros:", best[3])

    if export_csv:
        with open(f"meta_results_{objective}.csv", mode='w', newline='') as csvfile:
            fieldnames = list(bounds.keys()) + ["score", "avg_fitness", "std_dev"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for score, avg_fit, std, ind in all_results:
                row = ind.copy()
                row.update({"score": score, "avg_fitness": avg_fit, "std_dev": std})
                writer.writerow(row)

    return best

def meta_ga_all():
    for objective in OBJECTIVE_FUNCTIONS.keys():
        meta_ga(objective)

meta_ga_all()
