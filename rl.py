import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import random

st.set_page_config(page_title="RL Grid World", layout="wide")

class GridWorld:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.reset_grid()

    def reset_grid(self):
        self.grid = np.zeros((self.height, self.width))
        self.agent_pos = None
        self.goal_pos = None
        self.barriers = []
        
    def place_agent(self, pos):
        if self.grid[pos] == 0:  # If the cell is empty
            if self.agent_pos is not None:
                self.grid[self.agent_pos] = 0  # Remove agent from previous position
            self.agent_pos = pos
            self.grid[pos] = 1
            return True
        return False
        
    def place_goal(self, pos):
        if self.grid[pos] == 0:  # If the cell is empty
            if self.goal_pos is not None:
                self.grid[self.goal_pos] = 0  # Remove goal from previous position
            self.goal_pos = pos
            self.grid[pos] = 2
            return True
        return False
    
    def place_barrier(self, pos):
        if self.grid[pos] == 0:  # If the cell is empty
            self.barriers.append(pos)
            self.grid[pos] = 3
            return True
        return False
    
    def remove_item(self, pos):
        if pos == self.agent_pos:
            self.agent_pos = None
        elif pos == self.goal_pos:
            self.goal_pos = None
        elif pos in self.barriers:
            self.barriers.remove(pos)
        self.grid[pos] = 0
        
    def is_valid_state(self):
        # Check if we have both agent and goal
        return self.agent_pos is not None and self.goal_pos is not None
    
    def get_state(self, pos):
        # Convert position tuple to a single state number
        return pos[0] * self.width + pos[1]
    
    def get_pos_from_state(self, state):
        # Convert state number back to position tuple
        return (state // self.width, state % self.width)
    
    def get_valid_actions(self, pos):
        # Up, Right, Down, Left
        actions = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for i, (dx, dy) in enumerate(directions):
            new_pos = (pos[0] + dx, pos[1] + dy)
            
            # Check if the new position is within grid bounds and not a barrier
            if (0 <= new_pos[0] < self.height and 
                0 <= new_pos[1] < self.width and
                new_pos not in self.barriers):
                actions.append(i)
                
        return actions
    
    def take_action(self, pos, action):
        # Up, Right, Down, Left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = directions[action]
        new_pos = (pos[0] + dx, pos[1] + dy)
        
        # Check if the new position is valid
        if (0 <= new_pos[0] < self.height and 
            0 <= new_pos[1] < self.width and
            new_pos not in self.barriers):
            
            reward = -1  # Default step penalty
            
            if new_pos == self.goal_pos:
                reward = 100  # Reward for reaching the goal
                done = True
            else:
                done = False
                
            return new_pos, reward, done
            
        # If the new position is invalid, stay in place with a penalty
        return pos, -10, False


class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.height * env.width, 4))  # 4 actions: Up, Right, Down, Left
        
    def choose_action(self, state, training=True):
        valid_actions = self.env.get_valid_actions(self.env.get_pos_from_state(state))
        
        if not valid_actions:
            return None  # No valid actions available
        
        # Epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Choose the action with the highest Q-value among valid actions
            q_values = [self.q_table[state, a] if a in valid_actions else float('-inf') for a in range(4)]
            return np.argmax(q_values)
    
    def train(self, episodes=100):
        paths = []
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.get_state(self.env.agent_pos)
            done = False
            path = [self.env.agent_pos]
            steps = 0
            
            while not done and steps < 1000:  # Prevent infinite loops
                action = self.choose_action(state, training=True)
                if action is None:
                    break  # No valid actions
                
                pos = self.env.get_pos_from_state(state)
                next_pos, reward, done = self.env.take_action(pos, action)
                next_state = self.env.get_state(next_pos)
                
                # Standard Q-learning update:
                # Q(s,a) = Q(s,a) + lr * [R + gamma * max(Q(s',a')) - Q(s,a)]
                best_next_action = self.choose_action(next_state, training=False)
                
                if best_next_action is not None:
                    max_next_q = self.q_table[next_state, best_next_action]
                else:
                    max_next_q = 0
                
                self.q_table[state, action] += self.lr * (reward + self.gamma * max_next_q - self.q_table[state, action])
                
                state = next_state
                path.append(next_pos)
                steps += 1
                
                if done:
                    paths.append(path)
                    steps_history.append(steps)
                    break
            
            # If we didn't reach the goal, don't record this path
            if not done:
                paths.append([])
                steps_history.append(0)
        
        # Return the path of the last successful episode
        successful_episodes = [i for i, s in enumerate(steps_history) if s > 0]
        if successful_episodes:
            best_episode = successful_episodes[np.argmin([steps_history[i] for i in successful_episodes])]
            best_path = paths[best_episode]
            best_steps = steps_history[best_episode]
        else:
            best_path = []
            best_steps = 0
            
        return best_path, best_steps, self.q_table
    
    def get_policy(self):
        policy = np.zeros((self.env.height, self.env.width), dtype=int)
        
        for h in range(self.env.height):
            for w in range(self.env.width):
                state = self.env.get_state((h, w))
                pos = (h, w)
                
                if pos == self.env.goal_pos or pos in self.env.barriers:
                    policy[h, w] = -1  # Goal or barrier
                else:
                    action = self.choose_action(state, training=False)
                    policy[h, w] = action if action is not None else -1
                    
        return policy


class SARSA:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.height * env.width, 4))  # 4 actions: Up, Right, Down, Left
        
    def choose_action(self, state, training=True):
        valid_actions = self.env.get_valid_actions(self.env.get_pos_from_state(state))
        
        if not valid_actions:
            return None  # No valid actions available
        
        # Epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Choose the action with the highest Q-value among valid actions
            q_values = [self.q_table[state, a] if a in valid_actions else float('-inf') for a in range(4)]
            return np.argmax(q_values)
    
    def train(self, episodes=100):
        paths = []
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.get_state(self.env.agent_pos)
            action = self.choose_action(state, training=True)
            done = False
            path = [self.env.agent_pos]
            steps = 0
            
            while not done and steps < 1000 and action is not None:  # Prevent infinite loops
                pos = self.env.get_pos_from_state(state)
                next_pos, reward, done = self.env.take_action(pos, action)
                next_state = self.env.get_state(next_pos)
                
                # Choose next action using current policy
                next_action = self.choose_action(next_state, training=True)
                
                # SARSA update:
                # Q(s,a) = Q(s,a) + lr * [R + gamma * Q(s',a') - Q(s,a)]
                if next_action is not None:
                    self.q_table[state, action] += self.lr * (
                        reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action]
                    )
                else:
                    self.q_table[state, action] += self.lr * (
                        reward - self.q_table[state, action]
                    )
                
                state = next_state
                action = next_action
                path.append(next_pos)
                steps += 1
                
                if done:
                    paths.append(path)
                    steps_history.append(steps)
                    break
            
            # If we didn't reach the goal, don't record this path
            if not done:
                paths.append([])
                steps_history.append(0)
        
        # Return the path of the last successful episode
        successful_episodes = [i for i, s in enumerate(steps_history) if s > 0]
        if successful_episodes:
            best_episode = successful_episodes[np.argmin([steps_history[i] for i in successful_episodes])]
            best_path = paths[best_episode]
            best_steps = steps_history[best_episode]
        else:
            best_path = []
            best_steps = 0
            
        return best_path, best_steps, self.q_table
    
    def get_policy(self):
        policy = np.zeros((self.env.height, self.env.width), dtype=int)
        
        for h in range(self.env.height):
            for w in range(self.env.width):
                state = self.env.get_state((h, w))
                pos = (h, w)
                
                if pos == self.env.goal_pos or pos in self.env.barriers:
                    policy[h, w] = -1  # Goal or barrier
                else:
                    action = self.choose_action(state, training=False)
                    policy[h, w] = action if action is not None else -1
                    
        return policy


def visualize_grid(grid_world, path=None, q_table=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a custom colormap for the grid
    cmap = LinearSegmentedColormap.from_list(
        'gridworld_cmap', 
        ['white', 'skyblue', 'lightgreen', 'lightcoral']
    )
    
    # Plot the grid
    ax.imshow(grid_world.grid, cmap=cmap, interpolation='nearest')
    
    # Add cell text
    for i in range(grid_world.height):
        for j in range(grid_world.width):
            if (i, j) == grid_world.agent_pos:
                ax.text(j, i, '🚚', ha='center', va='center', fontsize=20)
            elif (i, j) == grid_world.goal_pos:
                ax.text(j, i, '📦', ha='center', va='center', fontsize=20)
            elif (i, j) in grid_world.barriers:
                ax.text(j, i, '❌', ha='center', va='center', fontsize=20)
    
    # Add gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='black', linewidth=2)
    ax.set_xticks(np.arange(-0.5, grid_world.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_world.height, 1), minor=True)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks(np.arange(0, grid_world.width, 1))
    ax.set_yticks(np.arange(0, grid_world.height, 1))
    
    # Plot the path if provided
    if path and len(path) > 0:
        path_y = [pos[0] for pos in path]
        path_x = [pos[1] for pos in path]
        ax.plot(path_x, path_y, '-o', color='blue', linewidth=2, markersize=8)
    
    # Add arrows for the policy if Q-table is provided
    if q_table is not None:
        for i in range(grid_world.height):
            for j in range(grid_world.width):
                pos = (i, j)
                if pos not in [grid_world.agent_pos, grid_world.goal_pos] and pos not in grid_world.barriers:
                    state = grid_world.get_state(pos)
                    valid_actions = grid_world.get_valid_actions(pos)
                    
                    if valid_actions:
                        # Get Q-values for valid actions
                        q_values = [q_table[state, a] if a in valid_actions else float('-inf') for a in range(4)]
                        best_action = np.argmax(q_values)
                        
                        # Draw arrow for the best action
                        if best_action == 0:  # Up
                            ax.arrow(j, i, 0, -0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                        elif best_action == 1:  # Right
                            ax.arrow(j, i, 0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
                        elif best_action == 2:  # Down
                            ax.arrow(j, i, 0, 0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                        elif best_action == 3:  # Left
                            ax.arrow(j, i, -0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    plt.tight_layout()
    return fig


# Streamlit UI
st.title("Reinforcement Learning Grid World")

st.markdown("""
This application lets you create a custom grid environment and solve it using either SARSA or Q-Learning algorithms.
1. Set the grid dimensions
2. Place an agent 🚚, barriers ❌, and a goal 📦
3. Choose an algorithm and see it find the optimal path!
""")

# Grid setup
st.sidebar.header("Grid Setup")
height = st.sidebar.slider("Grid Height", 3, 10, 5)
width = st.sidebar.slider("Grid Width", 3, 10, 5)

# Initialize grid world
if 'grid_world' not in st.session_state or st.session_state.grid_height != height or st.session_state.grid_width != width:
    st.session_state.grid_world = GridWorld(height, width)
    st.session_state.grid_height = height
    st.session_state.grid_width = width
    st.session_state.path = None
    st.session_state.steps = None
    st.session_state.q_table = None

# Placement mode selection
st.sidebar.header("Place Items")
placement_mode = st.sidebar.radio(
    "Select item to place:",
    ["Agent 🚚", "Goal 📦", "Barrier ❌", "Eraser ⌫"]
)

# Algorithm selection
st.sidebar.header("Algorithm")
algorithm = st.sidebar.radio(
    "Select algorithm:",
    ["Q-Learning", "SARSA"]
)

# Training parameters
st.sidebar.header("Training Parameters")
episodes = st.sidebar.slider("Number of Episodes", 10, 500, 200)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
discount_factor = st.sidebar.slider("Discount Factor", 0.5, 0.99, 0.9)
exploration_rate = st.sidebar.slider("Exploration Rate", 0.01, 0.5, 0.1)

# Create a button for training
train_button = st.sidebar.button("Train Agent")

# Display the grid
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Grid Environment")
    
    # Create a placeholder for the grid visualization
    grid_placeholder = st.empty()
    
    # Display the current grid
    fig = visualize_grid(st.session_state.grid_world, st.session_state.path, st.session_state.q_table)
    grid_placeholder.pyplot(fig)
    plt.close(fig)  # Close the figure to prevent memory leak
    
    # Create a clickable grid
    clicked_cell = st.button("Update Grid Visualization")
    
    # Grid for placing items
    grid_height = st.session_state.grid_height
    grid_width = st.session_state.grid_width
    
    cols = st.columns(grid_width)
    
    for i in range(grid_height):
        for j in range(grid_width):
            with cols[j]:
                cell_state = st.session_state.grid_world.grid[i, j]
                
                if cell_state == 0:
                    label = f"({i},{j})"
                elif cell_state == 1:
                    label = "🚚"
                elif cell_state == 2:
                    label = "📦"
                elif cell_state == 3:
                    label = "❌"
                
                if st.button(label, key=f"cell_{i}_{j}"):
                    # Handle cell click based on placement mode
                    if placement_mode == "Agent 🚚":
                        st.session_state.grid_world.place_agent((i, j))
                    elif placement_mode == "Goal 📦":
                        st.session_state.grid_world.place_goal((i, j))
                    elif placement_mode == "Barrier ❌":
                        st.session_state.grid_world.place_barrier((i, j))
                    elif placement_mode == "Eraser ⌫":
                        st.session_state.grid_world.remove_item((i, j))
                    
                    # Reset the path when grid changes
                    st.session_state.path = None
                    st.session_state.steps = None
                    st.session_state.q_table = None
                    
                    # Update the grid visualization
                    fig = visualize_grid(st.session_state.grid_world)
                    grid_placeholder.pyplot(fig)
                    plt.close(fig)
                    st.rerun()

with col2:
    st.subheader("Training Results")
    
    if train_button:
        # Check if the grid is valid for training
        if not st.session_state.grid_world.is_valid_state():
            st.error("Please place both an agent and a goal on the grid.")
        else:
            # Run the selected algorithm
            with st.spinner(f"Training using {algorithm}..."):
                if algorithm == "Q-Learning":
                    rl_agent = QLearning(
                        st.session_state.grid_world,
                        learning_rate=learning_rate,
                        discount_factor=discount_factor,
                        exploration_rate=exploration_rate
                    )
                else:  # SARSA
                    rl_agent = SARSA(
                        st.session_state.grid_world,
                        learning_rate=learning_rate,
                        discount_factor=discount_factor,
                        exploration_rate=exploration_rate
                    )
                
                path, steps, q_table = rl_agent.train(episodes)
                st.session_state.path = path
                st.session_state.steps = steps
                st.session_state.q_table = q_table
                
                # Update the grid visualization with the path
                fig = visualize_grid(st.session_state.grid_world, path, q_table)
                grid_placeholder.pyplot(fig)
                plt.close(fig)
    
    # Display training results
    if st.session_state.path is not None:
        if st.session_state.steps > 0:
            st.success(f"Path found in {st.session_state.steps} steps!")
            
            # Display the path
            st.write("Path coordinates:")
            path_text = " → ".join([f"({pos[0]},{pos[1]})" for pos in st.session_state.path])
            st.code(path_text)
        else:
            st.error("No valid path found. Try adjusting the grid or increasing the number of episodes.")
    
    # Reset button
    if st.button("Reset Grid"):
        st.session_state.grid_world.reset_grid()
        st.session_state.path = None
        st.session_state.steps = None
        st.session_state.q_table = None
        fig = visualize_grid(st.session_state.grid_world)
        grid_placeholder.pyplot(fig)
        plt.close(fig)
        st.rerun()