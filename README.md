# Multi-Agent Reinforcement Learning for Urban Traffic Control

This project explores the use of Multi-Agent Reinforcement Learning (MARL) to develop an intelligent and decentralized traffic control system. The goal is to train independent agents (traffic lights) to cooperatively manage traffic flow in a simulated urban grid, aiming to reduce vehicle wait time and overall trip duration compared to a standard fixed-time controller. The system is built using PyTorch for the agents and SUMO (Simulation of Urban Mobility) for the traffic simulation.

## File Descriptions

This repository contains all the necessary code and configuration files to reproduce the experiment.

- **`grid.nod.xml`, `grid.edg.xml`, `grid.rou.xml`**: These are the raw SUMO definition files for the intersections (nodes), roads (edges), and traffic flows (routes) of the 3x3 grid.
- **`grid.sumocfg`**: The main SUMO configuration file that brings together the network and route definitions to create the simulation scenario.
- **`agent.py`**: Contains the `DQNAgent` class. This is the 'brain' of each traffic light, defining the neural network architecture (a Deep Q-Network) and the logic for learning from experience (the replay method).
- **`main.py`**: The main training script. This file executes the entire learning process by launching SUMO, controlling the agents episode by episode, and saving the final trained models.
- **`evaluate_fixed.py`**: An evaluation script used to run the simulation with SUMO's default fixed-time controller. This generates the baseline performance data.
- **`evaluate_marl.py`**: An evaluation script that loads the saved models from training and runs the MARL controller in 'exploitation' mode (no learning) to measure its performance.
- **`analyze_results.py`**: A script that parses the XML output files generated by the evaluation scripts, calculates key performance metrics, and plots the final comparison bar chart.

## Prerequisites

- **SUMO:** You must have SUMO installed on your system. Please follow the official installation guide at [sumo.dlr.de](https://sumo.dlr.de/docs/Installing/index.html).
