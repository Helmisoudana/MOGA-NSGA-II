"""
optimization_core_simplified.py

Corrected and robust NSGA-II wrapper using DEAP when available.
Simplified version with only 3 design variables.

Provides run_nsga2(...) that returns:
    - pareto_solutions: list of {"design": {...}, "objectives": {...}}
    - history: list of {"generation": int, "pareto": [...]} snapshots
"""

from typing import List, Dict, Callable, Optional, Tuple, Any
import random
import copy
import math
import time

# Import simplified ev model
from ev_models_simplified import calculate_objectives

# Try to import DEAP
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except Exception:
    DEAP_AVAILABLE = False

# ----------------------------
# Defaults (variable names and bounds)
# ----------------------------
DEFAULT_VAR_NAMES = [
    "batterie_kWh",
    "puissance_moteur_kW",
    "masse_chassis_kg",
]

DEFAULT_BOUNDS = [
    (20.0, 120.0),    # batterie_kWh
    (30.0, 300.0),    # puissance_moteur_kW
    (400.0, 1200.0),  # masse_chassis_kg
]
algorithm_steps_data = []

def capture_algorithm_step(step_name: str, 
                          generation: int,
                          population: List,
                          selected_indices: List[int] = None,
                          crossover_pairs: List[Tuple[int, int]] = None,
                          mutation_indices: List[int] = None,
                          fronts: List[List[int]] = None):
    """
    Capture les données d'une étape de l'algorithme pour la visualisation.
    """
    step_data = {
        "step_name": step_name,
        "generation": generation,
        "population": copy.deepcopy(population),
        "selected_indices": copy.deepcopy(selected_indices) if selected_indices else [],
        "crossover_pairs": copy.deepcopy(crossover_pairs) if crossover_pairs else [],
        "mutation_indices": copy.deepcopy(mutation_indices) if mutation_indices else [],
        "fronts": copy.deepcopy(fronts) if fronts else []
    }
    
    algorithm_steps_data.append(step_data)
    return step_data

def get_algorithm_steps_data():
    """Retourne les données capturées des étapes de l'algorithme."""
    return algorithm_steps_data

def clear_algorithm_steps_data():
    """Efface les données des étapes de l'algorithme."""
    global algorithm_steps_data
    algorithm_steps_data = []

# Modifiez la fonction run_nsga2 pour capturer les étapes
# Ajoutez ces appels aux endroits appropriés dans votre code NSGA-II

# Exemple dans la boucle principale (pour DEAP):
# Après l'initialisation :
# capture_algorithm_step("Population Initiale", 0, initial_population)

# Après l'évaluation :
# capture_algorithm_step("Évaluation", gen, current_population)

# Après le tri non-dominé :
# capture_algorithm_step("Tri non-dominé", gen, current_population, fronts=fronts)

# Après la sélection :
# capture_algorithm_step("Sélection", gen, current_population, selected_indices=selected_indices)

# Après le croisement :
# capture_algorithm_step("Croisement", gen, current_population, crossover_pairs=crossover_pairs)

# Après la mutation :
# capture_algorithm_step("Mutation", gen, current_population, mutation_indices=mutation_indices)

# Après création de la nouvelle population :
# capture_algorithm_step("Nouvelle Génération", gen+1, new_population)
# ----------------------------
# Utility helpers
# ----------------------------
def _individual_to_design(individual: List[float], bounds: List[Tuple[float, float]], var_names: List[str]) -> Dict[str, float]:
    return {name: float(val) for name, val in zip(var_names, individual)}

def _evaluate_design(design: Dict[str, float]) -> List[float]:
    """
    Evaluate design and return vector of objectives suitable for minimization by NSGA-II:
      [poids_total (min), cout_usd (min), -vitesse_max_kmh (min -> maximize), -autonomie_km (min -> maximize), consommation (min)]
    """
    obj = calculate_objectives(design)
    return [
        obj["poids_total"],
        obj["cout_usd"],
        -obj["vitesse_max_kmh"],     # maximize -> minimize negative
        -obj["autonomie_km"],        # maximize -> minimize negative
        obj["consommation_kwh_100km"],
    ]

def pareto_front_from_population(pop: List[Dict]) -> List[Dict]:
    """Return nondominated solutions (simple Pareto filter). Each entry must contain 'objectives' list."""
    nondominated = []
    for cand in pop:
        dominated = False
        for other in pop:
            if other is cand:
                continue
            less_or_equal = all(o <= c for o, c in zip(other["objectives"], cand["objectives"]))
            strictly_less = any(o < c for o, c in zip(other["objectives"], cand["objectives"]))
            if less_or_equal and strictly_less:
                dominated = True
                break
        if not dominated:
            nondominated.append(cand)
    return nondominated

# ----------------------------
# DEAP helper: safe creator.create (avoid duplicate creation)
# ----------------------------
if DEAP_AVAILABLE:
    # Create fitness/individual types only if not already created (prevents RuntimeError on reload)
    try:
        creator.FitnessMin
    except Exception:
        # weights: 5 objectives (all minimized after sign adjustments)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
    try:
        creator.Individual
    except Exception:
        creator.create("Individual", list, fitness=creator.FitnessMin)

# ----------------------------
# Main run function
# ----------------------------
def run_nsga2(config: Dict,
              initial_population: Optional[Any] = None,
              callback: Optional[Callable[[int, List[Dict]], bool]] = None) -> Dict:
    """
    Run NSGA-II and return results.

    config keys:
      - population_size (int)
      - generations (int)
      - cxpb (float)
      - mutpb (float)
      - var_names (list[str]) optional
      - bounds (list[tuple]) optional
      - history_interval (int) optional

    initial_population can be:
      - None
      - list of dicts: [{var1: val, ...}, ...]
      - list of lists: [[v1, v2, ...], ...]
      - pandas.DataFrame with columns matching var_names

    callback(gen, current_pareto_list) -> bool:
        If provided, called every generation. Return True to stop early.
    """
    var_names = config.get("var_names", DEFAULT_VAR_NAMES)
    bounds = config.get("bounds", DEFAULT_BOUNDS)
    pop_size = int(config.get("population_size", 80))
    ngen = int(config.get("generations", 80))
    cxpb = float(config.get("cxpb", 0.9))
    mutpb = float(config.get("mutpb", 0.2))
    history_interval = int(config.get("history_interval", max(1, ngen // 10)))

    # Helper to convert various initial_population formats into list of individuals (lists of floats)
    initial_individuals: List[List[float]] = []
    if initial_population is not None:
        # DataFrame?
        try:
            import pandas as _pd
            if isinstance(initial_population, _pd.DataFrame):
                df = initial_population
                # Ensure columns order follows var_names
                df = df[list(var_names)]
                for _, row in df.iterrows():
                    initial_individuals.append([float(row[c]) for c in var_names])
        except Exception:
            # not a dataframe or pandas not available; continue to attempt list parsing
            pass

        # If still empty and it's a list-like
        if not initial_individuals:
            if isinstance(initial_population, list):
                for item in initial_population:
                    if isinstance(item, dict):
                        initial_individuals.append([float(item.get(n, random.uniform(low, high))) for n, (low, high) in zip(var_names, bounds)])
                    elif isinstance(item, (list, tuple)):
                        # assume order matches var_names
                        vals = list(item)
                        # if shorter, fill randomly
                        if len(vals) < len(var_names):
                            vals = vals + [random.uniform(lb, ub) for (lb, ub) in bounds[len(vals):]]
                        initial_individuals.append([float(v) for v in vals[:len(var_names)]])
                    else:
                        # skip unknown entries
                        continue

    # ---- If DEAP available, use DEAP NSGA-II ----
    if DEAP_AVAILABLE:
        toolbox = base.Toolbox()

        # Attribute generator uniform within bounds
        for i, (low, high) in enumerate(bounds):
            toolbox.register(f"attr_float_{i}", random.uniform, low, high)

        # Individual / population
        def create_individual():
            return creator.Individual([random.uniform(l, h) for (l, h) in bounds])

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function (DEAP expects a tuple)
        def eval_ind(individual):
            design = _individual_to_design(individual, bounds, var_names)
            return tuple(_evaluate_design(design))
        toolbox.register("evaluate", eval_ind)

        # Genetic operators (bounded SBX and polynomial mutation)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[b[0] for b in bounds], up=[b[1] for b in bounds], eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=[b[0] for b in bounds], up=[b[1] for b in bounds], eta=20.0, indpb=1.0/len(bounds))
        toolbox.register("select", tools.selNSGA2)

        # Create population (respect user-specified pop_size)
        pop = toolbox.population(n=pop_size)

        # Inject initial individuals if provided (overwrite first N)
        for i, ind_vals in enumerate(initial_individuals):
            if i >= len(pop):
                break
            # ensure values are within bounds
            clipped = [max(low, min(high, float(v))) for v, (low, high) in zip(ind_vals, bounds)]
            for j, val in enumerate(clipped):
                pop[i][j] = val

        # If initial population smaller/larger than pop_size, we keep pop_size total individuals.
        # Evaluate population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Ensure population size divisible by 4 (DEAP selTournamentDCD requirement when k==len(pop))
        if len(pop) % 4 != 0:
            # append random individuals until divisible by 4
            while len(pop) % 4 != 0:
                pop.append(toolbox.individual())

        history: List[Dict] = []
        pareto_solutions: List[Dict] = []

        # Main generational loop
        for gen in range(1, ngen + 1):
            # Compute nondominated sorting fronts and assign crowding distances
            fronts = tools.sortNondominated(pop, k=len(pop), first_front_only=False)
            for front in fronts:
                tools.emo.assignCrowdingDist(front)

            # Selection for mating using Tournament DCD
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Variation: crossover and mutation
            # Crossover in pairs
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= cxpb:
                    toolbox.mate(ind1, ind2)
                    del ind1.fitness.values
                    del ind2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() <= mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Create next generation population
            pop = toolbox.select(pop + offspring, pop_size)

            # Build current pareto front
            pop_dicts = []
            for ind in pop:
                design = _individual_to_design(ind, bounds, var_names)
                objs = list(ind.fitness.values)
                pop_dicts.append({"design": design, "objectives": objs})
            current_pareto = pareto_front_from_population(pop_dicts)

            # Save snapshots
            if gen % history_interval == 0 or gen == ngen:
                history.append({"generation": gen, "pareto": copy.deepcopy(current_pareto)})

            # Callback for UI (stop if callback returns True)
            if callback:
                stop = callback(gen, current_pareto)
                if stop:
                    break

        # Final mapping: convert to readable objectives (undo sign inversion)
        final_pop_dicts = []
        for ind in pop:
            design = _individual_to_design(ind, bounds, var_names)
            objs_vec = list(ind.fitness.values)
            objectives_readable = {
                "poids_total": objs_vec[0],
                "cout_usd": objs_vec[1],
                "vitesse_max_kmh": -objs_vec[2],
                "autonomie_km": -objs_vec[3],
                "consommation_kwh_100km": objs_vec[4],
            }
            final_pop_dicts.append({"design": design, "objectives": objectives_readable})

        # Filter nondominated among final_pop_dicts
        temp_for_filter = []
        for p in final_pop_dicts:
            raw_objs = [
                p["objectives"]["poids_total"],
                p["objectives"]["cout_usd"],
                -p["objectives"]["vitesse_max_kmh"],
                -p["objectives"]["autonomie_km"],
                p["objectives"]["consommation_kwh_100km"],
            ]
            temp_for_filter.append({"design": p["design"], "objectives": raw_objs})
        pareto = pareto_front_from_population(temp_for_filter)

        pareto_solutions = []
        for p in pareto:
            des = p["design"]
            objs = _evaluate_design(des)
            pareto_solutions.append({
                "design": des,
                "objectives": {
                    "poids_total": objs[0],
                    "cout_usd": objs[1],
                    "vitesse_max_kmh": -objs[2],
                    "autonomie_km": -objs[3],
                    "consommation_kwh_100km": objs[4],
                }
            })

        return {"pareto_solutions": pareto_solutions, "history": history}

    else:
        # ----------------------------
        # Fallback simple evolutionary strategy
        # ----------------------------
        pop_size_local = pop_size
        population = []
        for i in range(pop_size_local):
            if initial_individuals and i < len(initial_individuals):
                indiv = initial_individuals[i]
                # ensure within bounds
                indiv_clipped = [max(low, min(high, float(v))) for v, (low, high) in zip(indiv, bounds)]
                design = _individual_to_design(indiv_clipped, bounds, var_names)
            else:
                indiv = [random.uniform(b[0], b[1]) for b in bounds]
                design = _individual_to_design(indiv, bounds, var_names)
            objectives = _evaluate_design(design)
            population.append({"design": design, "objectives": objectives})

        history = []
        for gen in range(1, ngen + 1):
            # generate children by perturbation
            children = []
            for _ in range(pop_size_local):
                parent = random.choice(population)
                child_vals = []
                for i, (low, high) in enumerate(bounds):
                    val = parent["design"][var_names[i]]
                    sigma = 0.05 * (high - low)
                    new_val = val + random.gauss(0, sigma)
                    new_val = max(low, min(high, new_val))
                    child_vals.append(new_val)
                design = _individual_to_design(child_vals, bounds, var_names)
                objectives = _evaluate_design(design)
                children.append({"design": design, "objectives": objectives})

            combined = population + children
            pareto = pareto_front_from_population(combined)

            # fill new population: start with pareto, add random others to reach pop_size
            new_pop = pareto.copy()
            remaining = [c for c in combined if c not in new_pop]
            random.shuffle(remaining)
            while len(new_pop) < pop_size_local and remaining:
                new_pop.append(remaining.pop())
            population = new_pop[:pop_size_local]

            if gen % history_interval == 0 or gen == ngen:
                history.append({"generation": gen, "pareto": copy.deepcopy(pareto)})

            if callback:
                stop = callback(gen, pareto)
                if stop:
                    break

        # build final pareto_solutions
        final_pareto = pareto_front_from_population(population)
        pareto_solutions = []
        for p in final_pareto:
            des = p["design"]
            objs = _evaluate_design(des)
            pareto_solutions.append({
                "design": des,
                "objectives": {
                    "poids_total": objs[0],
                    "cout_usd": objs[1],
                    "vitesse_max_kmh": -objs[2],
                    "autonomie_km": -objs[3],
                    "consommation_kwh_100km": objs[4],
                }
            })
        return {"pareto_solutions": pareto_solutions, "history": history}
    