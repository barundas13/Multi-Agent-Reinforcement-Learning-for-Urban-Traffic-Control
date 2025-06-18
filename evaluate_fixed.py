import os
import sys
import traci
from sumolib import checkBinary

# --- Configuration ---
GUI = False  # No GUI needed for evaluation
MAX_STEPS = 3600  # Run for a full hour of simulation time
sumo_config = "grid.sumocfg"

# --- SUMO Setup ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumo_binary = checkBinary('sumo-gui') if GUI else checkBinary('sumo')

if __name__ == "__main__":
    print("Starting evaluation of the default Fixed-Time controller...")

    # Start SUMO with the tripinfo output enabled
    # This automatically saves performance data for every completed vehicle trip
    traci.start([sumo_binary, "-c", sumo_config, "--tripinfo-output", "tripinfo_fixed.xml"])

    step = 0
    # Run simulation until all vehicles that spawned have finished their trip
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        # Print progress every 500 steps
        if step % 500 == 0:
            print(f"  ...at step {step}")

    traci.close()
    print(f"Fixed-Time controller evaluation finished in {step} steps.")
    print("Performance data saved to 'tripinfo_fixed.xml'.")