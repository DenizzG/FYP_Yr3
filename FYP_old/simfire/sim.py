from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
import optuna
from optuna.samplers import RandomSampler, TPESampler, QMCSampler
import uuid
import numpy as np

config = Config("configs/operational_config.yml")
#sim = FireSimulation(config)
#sim.rendering = True

#sim.run("80m") #every minute is 1 tick of the simulation
def extract_best_waypoints(best_params, valid_points_dict, agent_id=0, n_waypoints=1):
    waypoint_list = []
    for i in range(n_waypoints):
        idx = best_params[f"agent_{agent_id}_waypoint_{i}_idx"]
        xy = valid_points_dict[agent_id][idx]
        waypoint_list.extend(xy)  # flatten tuple (x,y) into list
    return tuple(waypoint_list)

    
# Step 1: Create one dummy FireSimulation instance to precompute valid points
sim = FireSimulation(config)
sim.rendering = True  # Enable rendering for the simulation
sim.assign_valid_points()  # call once here
valid_points_dict = {}

for i in range (config.simulation.run_time):
    # Extract valid_points from agents, and store externally:
    for agent_id, agent in sim.agents.items():
        valid_points_dict[agent_id] = agent.valid_points

    # Step 2: Create Optuna objective that uses valid_points_dict
    def optuna_objective(trial):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        sim_trial = sim.copy()
        # Assign precomputed valid_points back into this fresh sim object:
        for agent_id, agent in sim.agents.items():
            agent.valid_points = valid_points_dict[agent_id]
        return sim_trial.run(trial)
    
    #ToDo: get rid of unqiie name for every run
    unique_name = f"fire_optimization_{uuid.uuid4()}"

    sampler = QMCSampler(qmc_type='sobol', seed=1)

    study_tpe = optuna.create_study(
        direction="minimize",
        sampler=sampler, 
        study_name=f"bayesian_search_{unique_name}",
        storage=f"sqlite:///{unique_name}_tpe.db",
        load_if_exists=True
    )
    study_tpe.optimize(optuna_objective, n_trials=10)
    print(f"Study {i+1} - Best area burned:", study_tpe.best_value)
    print(f"Study {i+1} - Best params:", study_tpe.best_params)

    best_params = study_tpe.best_trial.params
    best_waypoints = extract_best_waypoints(best_params, valid_points_dict)

    print(f"Study {i+1} - Best waypoints:", best_waypoints)

    sim.run_for_one_step(best_waypoints)
