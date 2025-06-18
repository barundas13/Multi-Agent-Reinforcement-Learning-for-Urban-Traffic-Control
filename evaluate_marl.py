import os
import sys
import traci
from sumolib import checkBinary
import torch
from agent import DQNAgent
import numpy as np

# --- Configuration ---
GUI = False
MIN_GREEN_TIME = 10
sumo_config = "grid.sumocfg"

# --- SUMO Setup ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
sumo_binary = checkBinary('sumo-gui') if GUI else checkBinary('sumo')

# --- Helper Functions ---
def get_green_phase_indices(tl_id):
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
    return [i for i, phase in enumerate(logic.phases) if 'g' in phase.state.lower()]

def get_state(tl_id, green_indices):
    state = []
    incoming_lanes = sorted(list(set(traci.trafficlight.getControlledLanes(tl_id))))
    for lane in incoming_lanes:
        state.append(traci.lane.getLastStepHaltingNumber(lane))
    phase_one_hot = [0] * len(green_indices)
    current_sumo_phase = traci.trafficlight.getPhase(tl_id)
    if current_sumo_phase in green_indices:
        phase_one_hot[green_indices.index(current_sumo_phase)] = 1
    state.extend(phase_one_hot)
    return np.array(state)

if __name__ == "__main__":
    print("Starting evaluation of the trained MARL controller...")

    # Start SUMO with trip info output
    traci.start([sumo_binary, "-c", sumo_config, "--tripinfo-output", "tripinfo_marl.xml"])
    all_tl_ids = traci.trafficlight.getIDList()

    # --- Initialization ---
    green_indices_map = {tl_id: get_green_phase_indices(tl_id) for tl_id in all_tl_ids}
    action_map = {tl_id: {i: idx for i, idx in enumerate(indices)} for tl_id, indices in green_indices_map.items()}
    action_sizes = {tl_id: len(indices) for tl_id, indices in green_indices_map.items()}
    agents = {tl_id: DQNAgent(len(get_state(tl_id, green_indices_map[tl_id])), action_sizes[tl_id]) for tl_id in
              all_tl_ids}

    # --- LOAD TRAINED MODELS and SET TO EVALUATION MODE ---
    print("Loading saved models...")
    for tl_id, agent in agents.items():
        try:
            agent.model.load_state_dict(torch.load(f"models/{tl_id}_model.pth"))
            agent.epsilon = 0.0  # Crucial: force exploitation, no random actions
            agent.model.eval()  # Puts the model in evaluation mode
        except FileNotFoundError:
            print(f"Error: Model for {tl_id} not found. Please run main.py to train first.")
            sys.exit(1)
    print("Models loaded successfully.")

    # --- Evaluation Loop ---
    step = 0
    last_action_time = {tl_id: 0 for tl_id in all_tl_ids}
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_time = traci.simulation.getTime()

        for tl_id in all_tl_ids:
            if current_time >= last_action_time[tl_id] + MIN_GREEN_TIME:
                current_state = get_state(tl_id, green_indices_map[tl_id])
                # Agent chooses the best action based on its learned policy
                action = agents[tl_id].choose_action(current_state)
                traci.trafficlight.setPhase(tl_id, action_map[tl_id][action])
                last_action_time[tl_id] = current_time
        step += 1
        if step % 500 == 0:
            print(f"  ...at step {step}")

    traci.close()
    print(f"MARL controller evaluation finished in {step} steps.")
    print("Performance data saved to 'tripinfo_marl.xml'.")