from dataclasses import dataclass
import numpy as np, random
from utils.seed import set_seed

# Constants for grid cell states
EMPTY = 0 # No agent
PREY = 1 # Prey agent
PRED = 2 # Predator agent

@dataclass
class ABMParams:
    # Grid
    width: int = 60 # 3600 positions
    height: int = 60
    steps: int = 400
    # Initial populations
    init_prey: int = 900
    init_pred: int = 300
    # Reproducing probabilities
    p_birth_prey: float = 0.05 
    p_birth_pred: float = 0.02 
    # Movement ranges
    prey_move_range: int = 1 
    pred_move_range: int = 1 
    # Energy parameters for predators
    pred_init_energy: int = 5
    pred_energy_gain: int = 3
    pred_energy_cost: int = 1
    
    # Safe zone parameters
    refugia_fraction: float = 0.15 # 15%% of grid cells are refugia
    refugia_pred_penalty: float = 0.5 # 50% reduction in predation success in refugia
    adaptive_window: int = 5 # recent predation count threshold for prey to adaptively move further

# Wrap around edges to avoid artificial boundaries and infinite world
def torus_coords(x, y, W, H):
    return x % W, y % H

# Get neighboring coordinates within a given range r
def neighbors(x, y, W, H, r=1):
    coords = []
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            if dx == 0 and dy == 0: 
                continue
            nx, ny = torus_coords(x+dx, y+dy, W, H)
            coords.append((nx, ny))
    return coords

# Initialize the world with agents and refugia
def init_world(p: ABMParams):
    W, H = p.width, p.height
    grid = np.zeros((H, W), dtype=np.int8)
    energy = np.zeros((H, W), dtype=np.int16)
    refugia = np.zeros((H, W), dtype=np.int8)

    # refugia placement
    n_refugia = int(p.refugia_fraction * W * H)
    idx = np.random.choice(W*H, size=n_refugia, replace=False) 
    for flat in idx:
        y, x = divmod(flat, W)
        refugia[y, x] = 1

    # place agents
    cells = np.arange(W*H)
    np.random.shuffle(cells) # Randomises all cells for unbiased random placement
    # Prey first
    for flat in cells[:p.init_prey]:
        y, x = divmod(flat, W)
        grid[y, x] = PREY
    # Predators
    for flat in cells[p.init_prey:p.init_prey+p.init_pred]:
        y, x = divmod(flat, W)
        grid[y, x] = PRED
        energy[y, x] = 5

    recent_pred = np.zeros((H, W), dtype=np.int16)
    return grid, energy, refugia, recent_pred

# One simulation step
def step(world, p: ABMParams):
    grid, energy, refugia, recent = world
    H, W = grid.shape
    new_grid = grid.copy()
    new_energy = energy.copy()
    new_recent = (recent * 0.9).astype(np.int16)  # decay

    # PREY  
    order = np.random.permutation(H*W)
    for flat in order:
        y, x = divmod(flat, W)
        if grid[y, x] != PREY: 
            continue
        # adaptive move if recent predation high
        move_r = p.prey_move_range + (1 if recent[y, x] >= p.adaptive_window else 0)
        nbrs = neighbors(x, y, W, H, r=move_r)
        np.random.shuffle(nbrs)
        target = None; best_is_ref = -1
        for nx, ny in nbrs:
            if new_grid[ny, nx] == EMPTY:
                is_ref = refugia[ny, nx]
                if is_ref > best_is_ref:
                    best_is_ref = is_ref
                    target = (nx, ny)
        if target:
            nx, ny = target
            new_grid[y, x] = EMPTY
            new_grid[ny, nx] = PREY
            # reproduction
            if np.random.rand() < p.p_birth_prey:
                new_grid[y, x] = PREY

    # PRED
    order = np.random.permutation(H*W)
    for flat in order:
        y, x = divmod(flat, W)
        if grid[y, x] != PRED:
            continue
        # energy upkeep
        new_energy[y, x] -= p.pred_energy_cost
        if new_energy[y, x] <= 0:
            new_grid[y, x] = EMPTY
            new_energy[y, x] = 0
            continue
        # move / hunt
        nbrs = neighbors(x, y, W, H, r=p.pred_move_range)
        np.random.shuffle(nbrs)
        moved = False
        for nx, ny in nbrs:
            if new_grid[ny, nx] == PREY:
                # predation success reduced in refugia
                mult = (1.0 - p.refugia_pred_penalty) if refugia[ny, nx] else 1.0
                if np.random.rand() <= mult:
                    # eat
                    new_grid[ny, nx] = PRED
                    new_grid[y, x] = EMPTY
                    new_energy[ny, nx] = new_energy[y, x] + p.pred_energy_gain
                    new_energy[y, x] = 0
                    new_recent[ny, nx] += 1
                    # reproduction after hunt
                    if np.random.rand() < p.p_birth_pred:
                        new_grid[y, x] = PRED
                        new_energy[y, x] = 5
                    moved = True
                    break
            elif new_grid[ny, nx] == EMPTY:
                new_grid[ny, nx] = PRED
                new_grid[y, x] = EMPTY
                new_energy[ny, nx] = new_energy[y, x]
                new_energy[y, x] = 0
                moved = True
                break
    return (new_grid, new_energy, refugia, new_recent)

def run_abm(p: ABMParams, seed: int = 0):
    set_seed(seed)
    world = init_world(p)
    prey_ts = []
    pred_ts = []
    for t in range(p.steps):
        prey_ts.append(np.count_nonzero(world[0] == PREY))
        pred_ts.append(np.count_nonzero(world[0] == PRED))
        world = step(world, p)
    prey_ts.append(np.count_nonzero(world[0] == PREY))
    pred_ts.append(np.count_nonzero(world[0] == PRED))
    return np.array(prey_ts), np.array(pred_ts)