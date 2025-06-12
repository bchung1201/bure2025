# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements simulation code for research on bipartite graph routing with EXP3 and Multiplicative Weights algorithms. The code simulates packet routing in networks with queues and servers, using online learning algorithms to optimize routing decisions.

## Dependencies and Setup

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages: numpy, scipy, matplotlib, seaborn

## Running Simulations

Each figure from the research paper has a corresponding Python file:
- `python figfour_a.py` - Figure 4a simulation
- `python figfour_b.py` - Figure 4b simulation  
- `python figfour_c.py` - Figure 4c simulation
- `python figthree_a.py` - Figure 3a simulation
- `python figthree_b.py` - Figure 3b simulation
- `python figtwo_a.py` - Figure 2a simulation
- `python figtwo_b.py` - Figure 2b simulation

**Note**: Simulations can take several hours to complete depending on hardware.

## Core Architecture

### Algorithm Implementations
- `BipartiteGraphEXP3.py` - EXP3 algorithm implementation for bipartite graph routing
- `BipartiteGraphMW.py` - Multiplicative Weights algorithm implementation
- `MWImplementation.py` - General multiplicative weights implementation with Player/Adversary classes

### Simulation Structure
All figure files follow the same basic pattern:
1. Initialize simulation parameters (time horizon T, number of samples, queues, servers)
2. Set up network topology (input rates, process rates, accessible servers)
3. Run multiple simulation samples with different parameters
4. Collect performance metrics (buildup, regret, clearing rates)
5. Generate and display plots

### Key Parameters
- `T` - Time horizon for simulation
- `sample` - Number of simulation runs for averaging
- `numQueues`/`numServers` - Network topology
- `inputRates`/`processRates` - Arrival and service rates
- `accessibleServers` - Bipartite graph connectivity
- `learning_rate`/`rate`/`gamma` - Learning algorithm parameters

### Performance Metrics
- Queue buildup analysis
- Regret bounds verification  
- Strategy distribution tracking
- Clearing rate measurements