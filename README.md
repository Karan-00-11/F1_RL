# F1 RL Track Simulation Project

## Overview
This project aims to build a reinforcement learning environment for an F1-style car using real track geometry and simplified vehicle physics.

The goal is to study how an agent can learn:
- where to deploy electrical energy,
- where to recharge energy,
- how to switch aero modes,
- and how to optimize lap performance under regulation-like constraints.

## Project Idea
Instead of building a full F1 simulator, this project focuses on a reduced but realistic environment that includes:

- longitudinal speed dynamics
- cornering losses
- tire degradation
- battery state of charge
- energy recovery under braking, lift-off, and coasting
- energy boost deployment
- aero mode switching
- real-life track representation using GPS or track layout data

## Why This Project
Formula 1 is a real-world control problem with strong constraints and trade-offs.  
This makes it a good use case for reinforcement learning, especially for:

- qualifying lap optimization
- race strategy planning
- energy management
- regulation-aware decision making

## Scope
The first version of the project will focus on:

1. Building a track representation from GPS or track data
2. Splitting the lap into segments
3. Simulating simplified vehicle physics
4. Defining an RL environment
5. Training an agent to optimize lap strategy

## Planned Features
- Real track centerline representation
- Segment-based track processing
- Physics-based lap simulation
- Reward function for lap time and energy efficiency
- Qualifying trim benchmark
- Optional race trim extension

## Tech Stack
- Python
- OpenEnv

## Workflow
1. Load track GPS data or track layout
2. Convert it into track segments
3. Simulate the vehicle through the track
4. Define state, action, and reward
5. Train an RL agent
6. Compare against baseline strategies
7. Visualize results

## Expected Output
The final system should produce:
- a simulated track environment
- lap-by-lap performance results
- energy deployment and recovery patterns
- visual plots of the track and agent behavior

## Current Status
This project is in the planning/building stage.

## Future Improvements
- More accurate tire model
- Better aero modeling
- Multi-lap race strategy
- Traffic and overtaking behavior
- Support for different tracks

## Goal
Create a practical RL environment that models an F1-style car strategy problem in a way that is realistic enough to matter, but simple enough to build and train.
