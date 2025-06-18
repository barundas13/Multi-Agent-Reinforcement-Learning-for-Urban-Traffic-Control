import os
import sys
import traci
from sumolib import checkBinary
import torch
from agent import DQNAgent
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Configuration ---
GUI = False  # Set to False for faster training
TOTAL_EPISODES = 500  # Number of episodes to train for
MAX_STEPS = 3600  # Max steps per episode (1 hour of simulation)
BATCH_SIZE = 32
MIN_GREEN_TIME = 10

# --- SUMO Setup ---
sumo_config = "grid.sumocfg"

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumo_binary = checkBinary('sumo-gui') if GUI else checkBinary('sumo')


# --- Helper Functions ---
def get_green_phase_indices(tl_id):
    # Returns the indices of the green phases for a given traffic light
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
    return [i for i, phase in enumerate(logic.phases) if 'g' in phase.state.lower()]


def get_state(tl_id, green_indices):
    # Retrieves the state for a given traffic light ID
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


def get_total_wait_time(tl_id):
    # Calculates the total waiting time at an intersection
    wait_time = 0
    for lane in set(traci.trafficlight.getControlledLanes(tl_id)):
        wait_time += traci.lane.getWaitingTime(lane)
    return wait_time


# --- Main Execution Block ---
if __name__ == "__main__":
    traci.start([sumo_binary, "-c", sumo_config, "--no-step-log", "true", "--waiting-time-memory", "1000"])
    all_tl_ids = traci.trafficlight.getIDList()

    # --- Initialization ---
    green_indices_map = {tl_id: get_green_phase_indices(tl_id) for tl_id in all_tl_ids}
    action_map = {tl_id: {i: idx for i, idx in enumerate(indices)} for tl_id, indices in green_indices_map.items()}
    action_sizes = {tl_id: len(indices) for tl_id, indices in green_indices_map.items()}
    agents = {tl_id: DQNAgent(len(get_state(tl_id, green_indices_map[tl_id])), action_sizes[tl_id]) for tl_id in
              all_tl_ids}

    episode_rewards_history = []

    # --- Training Loop ---
    for episode in range(TOTAL_EPISODES):
        traci.load(["-c", sumo_config, "--waiting-time-memory", "1000"])
        step = 0
        last_action_time = {tl_id: 0 for tl_id in all_tl_ids}
        transitions = {tl_id: {} for tl_id in all_tl_ids}
        episode_rewards = {tl_id: 0 for tl_id in all_tl_ids}
        start_time = time.time()

        while step < MAX_STEPS:
            traci.simulationStep()
            current_time = traci.simulation.getTime()

            for tl_id in all_tl_ids:
                if current_time >= last_action_time[tl_id] + MIN_GREEN_TIME:
                    if 'state' in transitions[tl_id]:
                        reward = transitions[tl_id]['wait_time'] - get_total_wait_time(tl_id)
                        agents[tl_id].remember(
                            transitions[tl_id]['state'],
                            transitions[tl_id]['action'],
                            reward,
                            get_state(tl_id, green_indices_map[tl_id]),
                            done=(step >= MAX_STEPS - 1)
                        )
                        episode_rewards[tl_id] += reward
                        if len(agents[tl_id].memory) > BATCH_SIZE:
                            agents[tl_id].replay(BATCH_SIZE)

                    current_state = get_state(tl_id, green_indices_map[tl_id])
                    action = agents[tl_id].choose_action(current_state)
                    traci.trafficlight.setPhase(tl_id, action_map[tl_id][action])

                    transitions[tl_id] = {'state': current_state, 'action': action,
                                          'wait_time': get_total_wait_time(tl_id)}
                    last_action_time[tl_id] = current_time
            step += 1
            if traci.simulation.getMinExpectedNumber() == 0: break

        end_time = time.time()
        total_episode_reward = sum(episode_rewards.values())
        episode_rewards_history.append(total_episode_reward)
        print(
            f"Episode {episode + 1}/{TOTAL_EPISODES} | Total Reward: {total_episode_reward:.2f} | Epsilon: {agents[all_tl_ids[0]].epsilon:.4f} | Duration: {end_time - start_time:.2f}s")

    traci.close()

    # --- Save Models and Plot Results ---
    if not os.path.exists('models'): os.makedirs('models')
    for tl_id, agent in agents.items():
        torch.save(agent.model.state_dict(), f"models/{tl_id}_model.pth")
    print("\nAll agent models saved.")

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(8, 6))
    plt.plot(episode_rewards_history)
    plt.title('Training Performanceyou')
    plt.xlabel('Episode')
    plt.ylabel('Total Cumulative Reward')
    plt.tight_layout()
    plt.savefig('reward_history.png')
    print("Reward history plot saved to 'reward_history.png'")
    plt.show()