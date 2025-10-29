# Intelligent Cubic Player - 4x4x4 Tic-Tac-Toe

An AI-powered 3D Tic-Tac-Toe game implementing advanced game-playing algorithms.

## Project Overview

This project implements an intelligent player for the Cubic game, a variant of Tic-Tac-Toe played on a 4x4x4 three-dimensional grid. The AI uses sophisticated algorithms including Minimax, Alpha-Beta Pruning, and heuristic evaluation to provide challenging gameplay.

## Features

### ✅ Minimax Algorithm
- Complete game tree exploration
- Optimal move selection
- Recursive evaluation of game states

### ✅ Alpha-Beta Pruning
- Significant performance optimization
- Reduces nodes evaluated by 50-90%
- Maintains optimal play

### ✅ Heuristic Functions
- Evaluates all 76 winning lines
- Scores based on piece alignment
- Strategic position assessment

### ✅ Optimizations
- **Transposition Table**: Caches evaluated positions
- **Move Ordering**: Examines promising moves first
- **Adaptive Depth**: Adjusts based on game state

### ✅ User Interface
- Layer-by-layer 3D visualization
- Color-coded players
- Real-time game statistics
- Intuitive controls

## Installation

1. **Clone or download the project**

2. **Install requirements**:
```bash
pip install -r requirements.txt