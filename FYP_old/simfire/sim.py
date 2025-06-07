from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
import optuna
from optuna.samplers import RandomSampler, TPESampler
import uuid

config = Config("configs/operational_config.yml")
#sim = FireSimulation(config)
#sim.rendering = True

#sim.run("80m") #every minute is 1 tick of the simulation

def optuna_objective(trial):
    sim = FireSimulation(config)
    sim.rendering = True
    return sim.run(trial)

unique_name = f"fire_optimization_{uuid.uuid4()}"

# Run Bayesian optimization with TPE sampler
study_tpe = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),  # optional
    study_name=f"bayesian_search_{unique_name}",
    storage=f"sqlite:///{unique_name}_tpe.db",
    load_if_exists=True
)
study_tpe.optimize(optuna_objective, n_trials=500)
print("TPE - Best area burned:", study_tpe.best_value)
print("TPE - Best params:", study_tpe.best_params)

# Run random search optimization
study_random = optuna.create_study(
    direction="minimize",
    sampler=RandomSampler(seed=42),  # optional: for reproducibility
    study_name=f"random_search_{unique_name}",
    storage=f"sqlite:///{unique_name}_random.db",
    load_if_exists=True
)
study_random.optimize(optuna_objective, n_trials=500)
print("Random - Best area burned:", study_random.best_value)
print("Random - Best params:", study_random.best_params)

# Now save a GIF and fire spread graph from the last 2 hours of simulation

#sim.save_gif("gifs/simulation.gif")
#sim.save_spread_graph()

# Saved to the location specified in the config: simulation.sf_home

# Turn off rendering so the display disappears and the simulation continues to run in the
# background
#sim.rendering = False
