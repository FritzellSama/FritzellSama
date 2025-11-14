"""
Genetic Algorithm for Strategy Optimization

CRITICAL: Production genetic algorithm for parameter optimization
- Population-based search
- Tournament selection
- Crossover and mutation operators
- Elitism for best solutions
"""

from decimal import Decimal
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
import polars as pl
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneticConfig:
    """Genetic algorithm configuration"""
    population_size: int
    n_generations: int
    crossover_rate: Decimal
    mutation_rate: Decimal
    elitism_rate: Decimal
    tournament_size: int
    random_state: Optional[int] = None


@dataclass
class Individual:
    """Single individual in population"""
    genome: Dict[str, Decimal]
    fitness: Optional[Decimal] = None
    generation: int = 0


class GeneticAlgorithm:
    """Genetic Algorithm Optimizer"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = self._load_config(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.rng = np.random.RandomState(self.config.random_state)

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0

        self.logger.info(f"GeneticAlgorithm initialized: pop={self.config.population_size}, gen={self.config.n_generations}")

    def _load_config(self, config: Dict[str, Any]) -> GeneticConfig:
        return GeneticConfig(
            population_size=int(config.get('population_size', os.getenv('GA_POP_SIZE', '100'))),
            n_generations=int(config.get('n_generations', os.getenv('GA_N_GEN', '50'))),
            crossover_rate=Decimal(str(config.get('crossover_rate', os.getenv('GA_CROSSOVER', '0.8')))),
            mutation_rate=Decimal(str(config.get('mutation_rate', os.getenv('GA_MUTATION', '0.1')))),
            elitism_rate=Decimal(str(config.get('elitism_rate', os.getenv('GA_ELITISM', '0.1')))),
            tournament_size=int(config.get('tournament_size', os.getenv('GA_TOURNAMENT', '5'))),
            random_state=config.get('random_state')
        )

    def optimize(
        self,
        fitness_function: Callable[[Dict[str, Decimal]], Decimal],
        parameter_bounds: Dict[str, Tuple[Decimal, Decimal]],
        maximize: bool = True
    ) -> Dict[str, Any]:
        """Run genetic algorithm optimization"""
        try:
            start_time = datetime.utcnow()
            self.logger.info("Starting genetic algorithm optimization")

            # Initialize population
            self._initialize_population(parameter_bounds)

            # Evaluate initial population
            self._evaluate_population(fitness_function, maximize)

            history = []

            # Evolution loop
            for gen in range(self.config.n_generations):
                self.generation = gen

                # Selection
                parents = self._selection()

                # Crossover and mutation
                offspring = self._create_offspring(parents, parameter_bounds)

                # Combine and evaluate
                self.population.extend(offspring)
                self._evaluate_population(fitness_function, maximize)

                # Survival selection
                self._survival_selection()

                # Track best
                gen_best = max(self.population, key=lambda ind: ind.fitness or Decimal('-inf'))
                if self.best_individual is None or gen_best.fitness > (self.best_individual.fitness or Decimal('-inf')):
                    self.best_individual = gen_best

                # Log progress
                avg_fitness = sum(ind.fitness or Decimal('0') for ind in self.population) / Decimal(str(len(self.population)))
                history.append({
                    'generation': gen,
                    'best_fitness': self.best_individual.fitness,
                    'avg_fitness': avg_fitness
                })

                if gen % max(1, self.config.n_generations // 10) == 0:
                    self.logger.info(f"Gen {gen}: Best={self.best_individual.fitness}, Avg={avg_fitness}")

            optimization_time = Decimal(str((datetime.utcnow() - start_time).total_seconds()))

            self.logger.info(f"Optimization complete: best_fitness={self.best_individual.fitness}")

            return {
                'best_params': self.best_individual.genome,
                'best_fitness': self.best_individual.fitness,
                'history': pl.DataFrame(history),
                'optimization_time': optimization_time,
                'n_generations': self.config.n_generations
            }

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            raise

    def _initialize_population(self, bounds: Dict[str, Tuple[Decimal, Decimal]]) -> None:
        """Create initial random population"""
        self.population = []
        for _ in range(self.config.population_size):
            genome = {
                param: Decimal(str(self.rng.uniform(float(bounds[param][0]), float(bounds[param][1]))))
                for param in bounds.keys()
            }
            self.population.append(Individual(genome=genome, generation=0))

    def _evaluate_population(self, fitness_func: Callable, maximize: bool) -> None:
        """Evaluate fitness for all individuals"""
        for ind in self.population:
            if ind.fitness is None:
                try:
                    fitness = fitness_func(ind.genome)
                    ind.fitness = fitness if maximize else -fitness
                except Exception as e:
                    self.logger.warning(f"Fitness evaluation failed: {e}")
                    ind.fitness = Decimal('-inf')

    def _selection(self) -> List[Individual]:
        """Tournament selection"""
        parents = []
        for _ in range(self.config.population_size):
            tournament = self.rng.choice(self.population, size=self.config.tournament_size, replace=False)
            winner = max(tournament, key=lambda ind: ind.fitness or Decimal('-inf'))
            parents.append(winner)
        return parents

    def _create_offspring(self, parents: List[Individual], bounds: Dict[str, Tuple[Decimal, Decimal]]) -> List[Individual]:
        """Create offspring via crossover and mutation"""
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]

            if self.rng.rand() < float(self.config.crossover_rate):
                child1_genome, child2_genome = self._crossover(parent1.genome, parent2.genome)
            else:
                child1_genome, child2_genome = parent1.genome.copy(), parent2.genome.copy()

            # Mutation
            child1_genome = self._mutate(child1_genome, bounds)
            child2_genome = self._mutate(child2_genome, bounds)

            offspring.append(Individual(genome=child1_genome, generation=self.generation + 1))
            offspring.append(Individual(genome=child2_genome, generation=self.generation + 1))

        return offspring

    def _crossover(self, genome1: Dict[str, Decimal], genome2: Dict[str, Decimal]) -> Tuple[Dict[str, Decimal], Dict[str, Decimal]]:
        """Uniform crossover"""
        child1 = {}
        child2 = {}
        for param in genome1.keys():
            if self.rng.rand() < 0.5:
                child1[param] = genome1[param]
                child2[param] = genome2[param]
            else:
                child1[param] = genome2[param]
                child2[param] = genome1[param]
        return child1, child2

    def _mutate(self, genome: Dict[str, Decimal], bounds: Dict[str, Tuple[Decimal, Decimal]]) -> Dict[str, Decimal]:
        """Gaussian mutation"""
        mutated = genome.copy()
        for param, value in mutated.items():
            if self.rng.rand() < float(self.config.mutation_rate):
                lower, upper = bounds[param]
                mutation_range = (upper - lower) * Decimal('0.1')
                mutation = Decimal(str(self.rng.normal(0, float(mutation_range))))
                mutated[param] = min(upper, max(lower, value + mutation))
        return mutated

    def _survival_selection(self) -> None:
        """Elitism + fitness-based selection"""
        # Sort by fitness
        self.population.sort(key=lambda ind: ind.fitness or Decimal('-inf'), reverse=True)

        # Keep top individuals (elitism)
        n_elite = int(float(self.config.elitism_rate) * self.config.population_size)
        self.population = self.population[:max(n_elite, self.config.population_size)]
