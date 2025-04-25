from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation

config = Config("configs/operational_config.yml")

sim = FireSimulation(config)
sim.rendering = True
sim.run("100m")

# Now save a GIF and fire spread graph from the last 2 hours of simulation
sim.save_gif()
sim.save_spread_graph()
# Saved to the location specified in the config: simulation.sf_home

# Update agents for display
# (x, y, agent_id)
"""agent_0 = (50, 50, 0)
agent_1 = (50, 52, 1)

agents = [agent_0, agent_1]"""
# Create the agents on the display

"""sim.update_agent_positions(agents)"""

# Loop through to move agents
"""for i in range(5):
    agent_0 = (5 + i, 5 + i, 0)
    agent_1 = (5 + i, 5 + i, 1)
    # Update the agent positions on the simulation
    sim.update_agent_positions([agent_0, agent_1])
    # Run for 1 update step
    sim.run(1)"""

print("run")
# Turn off rendering so the display disappears and the simulation continues to run in the
# background
sim.rendering = False
