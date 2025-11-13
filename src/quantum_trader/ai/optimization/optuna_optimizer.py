"""Optuna Hyperparameter Optimization - PRODUCTION"""
from decimal import Decimal
from typing import Dict, Any, Callable, Tuple, List, Optional
from datetime import datetime
import os, logging
import numpy as np
import polars as pl

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Optuna not installed")

logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """Production Optuna-based hyperparameter optimization"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.n_trials = int(config.get('n_trials', os.getenv('OPTUNA_N_TRIALS', '100')))
        self.timeout = int(config.get('timeout', os.getenv('OPTUNA_TIMEOUT', '3600')))
        self.n_jobs = int(config.get('n_jobs', os.getenv('OPTUNA_N_JOBS', '-1')))
        self.sampler = TPESampler(seed=config.get('random_state', 42))

        self.study: Optional[optuna.Study] = None
        self.logger.info(f"OptunaOptimizer initialized: {self.n_trials} trials")

    def optimize(
        self,
        objective_function: Callable[[Dict[str, Decimal]], Decimal],
        parameter_space: Dict[str, Dict[str, Any]],
        maximize: bool = True
    ) -> Dict[str, Any]:
        try:
            start_time = datetime.utcnow()
            self.logger.info("Starting Optuna optimization")

            # Create study
            direction = 'maximize' if maximize else 'minimize'
            self.study = optuna.create_study(
                direction=direction,
                sampler=self.sampler,
                study_name=f"optimization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )

            # Define objective wrapper
            def objective(trial: optuna.Trial) -> float:
                params = {}
                for param_name, param_config in parameter_space.items():
                    param_type = param_config['type']

                    if param_type == 'float':
                        params[param_name] = Decimal(str(trial.suggest_float(
                            param_name,
                            float(param_config['low']),
                            float(param_config['high']),
                            log=param_config.get('log', False)
                        )))
                    elif param_type == 'int':
                        params[param_name] = Decimal(str(trial.suggest_int(
                            param_name,
                            int(param_config['low']),
                            int(param_config['high'])
                        )))
                    elif param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )

                score = objective_function(params)
                return float(score)

            # Optimize
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=False
            )

            best_params = {k: Decimal(str(v)) if isinstance(v, (int, float)) else v
                          for k, v in self.study.best_params.items()}
            best_value = Decimal(str(self.study.best_value))

            optimization_time = Decimal(str((datetime.utcnow() - start_time).total_seconds()))

            self.logger.info(
                f"Optimization complete: best_value={best_value}, "
                f"trials={len(self.study.trials)}, time={optimization_time}s"
            )

            return {
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(self.study.trials),
                'optimization_time': optimization_time,
                'study': self.study
            }

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            raise RuntimeError(f"Optuna optimization error: {e}")

    def get_optimization_history(self) -> pl.DataFrame:
        """Get optimization history as polars DataFrame"""
        try:
            if self.study is None:
                return pl.DataFrame()

            trials_data = []
            for trial in self.study.trials:
                trial_dict = {
                    'trial_number': trial.number,
                    'value': Decimal(str(trial.value)) if trial.value is not None else None,
                    'state': trial.state.name,
                    **{f'param_{k}': Decimal(str(v)) if isinstance(v, (int, float)) else v
                       for k, v in trial.params.items()}
                }
                trials_data.append(trial_dict)

            return pl.DataFrame(trials_data)

        except Exception as e:
            self.logger.error(f"Failed to get history: {e}")
            return pl.DataFrame()
