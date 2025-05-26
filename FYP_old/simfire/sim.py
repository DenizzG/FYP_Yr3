from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
import optuna

config = Config("configs/operational_config.yml")
#sim = FireSimulation(config)
#sim.rendering = True

#sim.run("80m") #every minute is 1 tick of the simulation

def optuna_objective(trial):
    sim = FireSimulation(config)
    sim.rendering = True
    return sim.run(trial)

study = optuna.create_study(
    direction="minimize", 
    study_name="fire_optimization", 
    storage="sqlite:///fire_opt.db",  # Optional: for dashboard
    load_if_exists=True
)

study.optimize(optuna_objective, n_trials=1000)
print("Best area burned:", study.best_value)
print("Best params:", study.best_params)
# Now save a GIF and fire spread graph from the last 2 hours of simulation

#sim.save_gif("gifs/simulation.gif")
#sim.save_spread_graph()

# Saved to the location specified in the config: simulation.sf_home

# Turn off rendering so the display disappears and the simulation continues to run in the
# background
#sim.rendering = False
