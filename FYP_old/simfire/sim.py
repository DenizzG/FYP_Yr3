from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
import optuna
from optuna.samplers import RandomSampler, TPESampler, QMCSampler
import uuid

config = Config("configs/operational_config.yml")
#sim = FireSimulation(config)
#sim.rendering = True

#sim.run("80m") #every minute is 1 tick of the simulation

# Step 1: Create one dummy FireSimulation instance to precompute valid points
sim = FireSimulation(config)
sim.assign_valid_points()  # call once here
valid_points_dict = {}

# Extract valid_points from agents, and store externally:
for agent_id, agent in sim.agents.items():
    valid_points_dict[agent_id] = agent.valid_points

# Step 2: Create Optuna objective that uses valid_points_dict
def optuna_objective(trial):
    sim = FireSimulation(config)
    sim.rendering = True
    # Assign precomputed valid_points back into this fresh sim object:
    for agent_id, agent in sim.agents.items():
        agent.valid_points = valid_points_dict[agent_id]
    return sim.run(trial)

unique_name = f"fire_optimization_{uuid.uuid4()}"

sampler = QMCSampler(qmc_type='sobol', seed=42)

study_tpe = optuna.create_study(
    direction="minimize",
    sampler=sampler, 
    study_name=f"bayesian_search_{unique_name}",
    storage=f"sqlite:///{unique_name}_tpe.db",
    load_if_exists=True
)
study_tpe.optimize(optuna_objective, n_trials=10)
print("TPE - Best area burned:", study_tpe.best_value)
print("TPE - Best params:", study_tpe.best_params)

# Now save a GIF and fire spread graph from the last 2 hours of simulation

#sim.save_gif("gifs/simulation.gif")
#sim.save_spread_graph()

# Saved to the location specified in the config: simulation.sf_home

# Turn off rendering so the display disappears and the simulation continues to run in the
# background
#sim.rendering = False
