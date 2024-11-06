import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class TetheredBoatsEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=10, n_boats=2, tether_length=3, 
                 time_penalty=-1, trash_reward=10, complete_reward=50):
        super(TetheredBoatsEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.n_boats = n_boats
        self.tether_length = tether_length
        
        # Rewards
        self.time_penalty = time_penalty
        self.trash_reward = trash_reward
        self.complete_reward = complete_reward
        
        # Grid values
        self.EMPTY = 0
        self.TRASH = 1
        self.BOAT_OR_TETHER = 2
        
        # Action space: 0 - straight, 1 - 45° left, 2 - 45° right
        # Each boat can take one of these actions
        self.action_space = spaces.MultiDiscrete([9] * n_boats)
        
        # Observation space: grid_size x grid_size with values 0, 1, or 2
        self.observation_space = spaces.Box(
            low=0, 
            high=2,
            shape=(grid_size, grid_size), 
            dtype=np.int32
        )
        
        # Initialize state
        self.reset()
    
    def _get_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _get_tether_cells(self, boat1_pos, boat2_pos):
        """Calculate cells occupied by tether between two boats"""
        x1, y1 = boat1_pos
        x2, y2 = boat2_pos
        
        # Get all points on line between boats
        points = []
        n_points = self.tether_length
        for i in range(n_points):
            t = i / (n_points - 1)
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            points.append((x, y))
        
        return points
    
    # def _get_tether_cells(self, boat1_pos, boat2_pos):
    #     """Calculate cells occupied by tether between two boats using Manhattan distance"""
    #     x1, y1 = boat1_pos
    #     x2, y2 = boat2_pos
        
    #     # Get all points on line between boats using Manhattan distance
    #     points = []
        
    #     # Calculate the number of steps in each direction
    #     dx = abs(x2 - x1)
    #     dy = abs(y2 - y1)
    #     num_steps = max(dx, dy) + 1
        
    #     # Iterate over the number of steps and calculate the positions
    #     for i in range(num_steps):
    #         x = x1 + np.sign(x2 - x1) * i
    #         y = y1 + np.sign(y2 - y1) * i
    #         points.append((int(x), int(y)))
        
    #     return points
    
    def _is_valid_move(self, current_pos, new_pos, other_boat_pos):
        """Check if move is valid (within grid, one cell distance, and tether length)"""
        x, y = new_pos
        
        # Check grid boundaries
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
            
        # Check movement distance (can only move one cell in any direction)
        curr_x, curr_y = current_pos
        if abs(x - curr_x) > 1 or abs(y - curr_y) > 1:
            return False
        
        # Check tether length constraint
        new_distance = self._get_distance(new_pos, other_boat_pos)
        if new_distance > self.tether_length:
            return False
        
        # Check if two boats are in same location
        if new_pos == other_boat_pos:
            return False
            
        return True
    
    def _get_new_position(self, pos, action):
        """Get new position based on action"""
        x, y = pos
        
        # Straight movement
        if action == 0:
            return (x + 1, y)
        # 45° left
        elif action == 1:
            return (x + 1, y - 1)
        # 90 left
        elif action == 2:
            return (x, y - 1)
        # 135 degree left
        elif action == 3:
            return (x - 1, y - 1)
        # 180 degree
        elif action == 4:
            return (x - 1, y)
        #135 degree right:
        elif action == 5:
            return (x - 1, y + 1)
        # 90 degree right:
        elif action == 6:
            return (x, y + 1)
        # 45° right
        elif action == 7:
            return (x + 1, y + 1)
        # stay
        else:
            return (x, y)        
    
    def step(self, action):
        """Execute one time step within the environment"""
        assert self.action_space.contains(action)
        
        reward = self.time_penalty
        done = False
        
        # Store previous positions
        old_boat_positions = self.boat_positions.copy()
        
        # Move boats based on actions
        for i, boat_action in enumerate(action):
            new_pos = self._get_new_position(self.boat_positions[i], boat_action)
            other_boat_pos = self.boat_positions[(i + 1) % 2]  # Position of the other boat
            
            # Check if move is valid including tether constraint
            if self._is_valid_move(self.boat_positions[i], new_pos, other_boat_pos):
                self.boat_positions[i] = new_pos
        
        # Calculate tether positions
        tether_cells = self._get_tether_cells(self.boat_positions[0], self.boat_positions[1])
        
        # Update grid
        self.grid.fill(self.EMPTY)
        
        # Place remaining trash
        for trash_pos in self.trash_positions:
            self.grid[trash_pos] = self.TRASH
        
        # Check for trash collection (boat or tether touching trash)
        trash_collected = []
        for trash_pos in self.trash_positions:
            tx, ty = trash_pos
            
            # Check if boats collect trash
            for boat_pos in self.boat_positions:
                bx, by = boat_pos
                if (tx, ty) == (bx, by):
                    trash_collected.append(trash_pos)
                    reward += self.trash_reward
            
            # Check if tether collects trash
            for tether_pos in tether_cells:
                if (tx, ty) == tether_pos:
                    trash_collected.append(trash_pos)
                    reward += self.trash_reward
        
        # Remove collected trash
        self.trash_positions = [pos for pos in self.trash_positions if pos not in trash_collected]
        
        # Place boats and tether
        for boat_pos in self.boat_positions:
            self.grid[boat_pos] = self.BOAT_OR_TETHER
        for tether_pos in tether_cells:
            self.grid[tether_pos] = self.BOAT_OR_TETHER
        
        # Check if all trash is collected
        if len(self.trash_positions) == 0:
            reward += self.complete_reward
            done = True
        
        return self.grid, reward, done, {}
    
    def reset(self):
        """Reset the state of the environment"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Initialize boat positions (start from left side)
        self.boat_positions = [
            (0, self.grid_size // 3),
            (0, 2 * self.grid_size // 3)
        ]
        
        # Initialize random trash positions (on right half of grid)
        n_trash = 50#self.grid_size  # Number of trash pieces
        self.trash_positions = []
        while len(self.trash_positions) < n_trash:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            pos = (x, y)
            if pos not in self.trash_positions:
                self.trash_positions.append(pos)
                self.grid[pos] = self.TRASH

        # self.grid[:, :] = self.TRASH
        
        # Place boats and tether
        for boat_pos in self.boat_positions:
            self.grid[boat_pos] = self.BOAT_OR_TETHER
        for tether_pos in self._get_tether_cells(self.boat_positions[0], self.boat_positions[1]):
            self.grid[tether_pos] = self.BOAT_OR_TETHER
        
        return self.grid
    
    def render(self, mode='human'):
        """Render the environment with clear visualization of boats, trash, and grid"""
        # Clear current figure if it exists
        plt.clf()
        
        # Create figure and axis with specific size
        fig = plt.gcf()
        ax = plt.gca()
        
        # # Set figure size if it hasn't been set
        # if fig.get_size_inches().tolist() != [8, 8]:
        fig.set_size_inches(8, 8)
        
        # Plot the base grid (white background)
        ax.imshow(np.zeros_like(self.grid), cmap='binary', alpha=0.1)
        
        # Add grid lines
        ax.grid(True, which='major', color='black', linewidth=1)
        ax.set_xticks(np.arange(-.5, self.grid_size, 1))
        ax.set_yticks(np.arange(-.5, self.grid_size, 1))
        
        # Remove axis labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Plot trash (red dots)
        for pos in self.trash_positions:
            i, j = pos
            ax.plot(j, i, 'ro', markersize=15, markerfacecolor='red')
        
        # Plot boats (blue triangles) and tether (green line)
        for boat_pos in self.boat_positions:
            i, j = boat_pos
            ax.plot(j, i, '^', color='blue', markersize=15, markerfacecolor='blue')
        
        if len(self.boat_positions) >= 2:
            boat1_y, boat1_x = self.boat_positions[0]
            boat2_y, boat2_x = self.boat_positions[1]

            tether_positions = self._get_tether_cells(self.boat_positions[0], self.boat_positions[1])
            tether_positions = [(boat1_y, boat1_x)] + tether_positions + [(boat2_y, boat2_x)]
            print(tether_positions, self.boat_positions)
            for i in range(0, len(tether_positions) - 1):
                ax.plot([tether_positions[i][1], tether_positions[i+1][1]], [tether_positions[i][0], tether_positions[i+1][0]], 'g-', linewidth=2, alpha=0.6)

            

        # Draw tether between boats


        # if len(self.boat_positions) >= 2:
        #     boat1_y, boat1_x = self.boat_positions[0]
        #     boat2_y, boat2_x = self.boat_positions[1]
        #     ax.plot([boat1_x, boat2_x], [boat1_y, boat2_y], 'g-', linewidth=2, alpha=0.6)
        
        # Set title and display limits
        ax.set_title('Tethered Boats Environment')
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)  # Invert y-axis to match grid coordinates
        
        # Display the plot
        plt.draw()
        plt.pause(0.5)

# Example usage:
if __name__ == "__main__":
    # Create environment
    env = TetheredBoatsEnv()
    
    # Reset environment
    obs = env.reset()
    env.render()
    
    # Run a few random steps
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        env.render()
        
        if done:
            obs = env.reset()
