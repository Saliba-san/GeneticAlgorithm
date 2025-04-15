from GA import run_ga, OBJECTIVE_FUNCTIONS
import itertools
import statistics
import csv

param_space = {
    'generations': [10, 30, 50],
    'population': [10, 50, 100],
    'crossover': [0.6, 0.7, 0.8],
    'mutation': [0.01, 0.03, 0.05],
    'elitism': [0.55, 0.65, 0.75],
    'bits': [10, 20, 35],
    'maximize': [True],
}

keys, values = zip(*param_space.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for objective in OBJECTIVE_FUNCTIONS.keys():
    if objective == "rastrigin":
        continue
    print(f"\n===== Otimizando parâmetros para: {objective} =====")
    best_result = float('inf')
    best_params = None
    best_generation_count = None
    best_stddev = float('inf')
    
    csv_filename = f"results_{objective}.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        fieldnames = list(param_space.keys()) + ["avg_fitness", "std_dev", "avg_generations"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, params in enumerate(param_combinations):
            print(f"\n--- Execução {i+1}/{len(param_combinations)} ---")
            print("Parâmetros:", params)

            try:
                all_runs = []
                gen_counts = []
                for _ in range(5):  # 5 execuções por configuração
                    result = run_ga(
                        objective,
                        params['generations'],
                        params['population'],
                        params['crossover'],
                        params['mutation'],
                        params['elitism'],
                        params['bits'],
                        params['maximize']
                    )

                    if isinstance(result, list) and len(result) > 0:
                        all_runs.append(result[-1])
                        gen_counts.append(len(result))

                if all_runs:
                    avg_result = statistics.mean(all_runs)
                    std_dev = statistics.stdev(all_runs) if len(all_runs) > 1 else 0
                    avg_gen = statistics.mean(gen_counts)

                    row = params.copy()
                    row.update({
                        "avg_fitness": avg_result,
                        "std_dev": std_dev,
                        "avg_generations": avg_gen
                    })
                    writer.writerow(row)

                    print(f"Média fitness: {avg_result:.4f}, Desvio padrão: {std_dev:.4f}, Gerações médias: {avg_gen:.1f}")

                    if avg_result < best_result or (avg_result == best_result and std_dev < best_stddev):
                        best_result = avg_result
                        best_params = params
                        best_generation_count = avg_gen
                        best_stddev = std_dev
            except Exception as e:
                print(f"Erro ao rodar GA com esses parâmetros: {e}")

    print(f"\nMelhor resultado médio para {objective}: {best_result:.4f}")
    print("Melhores parâmetros:", best_params)
    print("Gerações médias até o melhor resultado:", best_generation_count)
    print("Desvio padrão do fitness:", best_stddev)
    print(f"Resultados salvos em: {csv_filename}")
