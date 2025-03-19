import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!


class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=50):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        
        self.stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
       
        self.obstacles = set()  # No obstacles in simple version
        self.destination = None

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        self.obstacles = set()
        # sample number of obstacles 0~15
        num_obstacles = random.randint(0, 50)
        for i in range(num_obstacles):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.obstacles.add((x, y)) 
        
        # sample self stations, have to make sure they are not in obstacles and not the same
        self.stations = []
        for i in range(4):
            while True:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                if (x, y) not in self.obstacles and (x, y) not in self.stations:
                    self.stations.append((x, y))
                    break


        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.stations and (x, y) not in self.obstacles
        ]

        self.taxi_pos = random.choice(available_positions)
        
        self.passenger_loc = random.choice([pos for pos in self.stations])
        
        
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)
        
        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                # print("out of range")
                reward -=5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    # print("pick nothing")
                    reward = -10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        # print("drop off wrong place")
                        reward -=10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    # print("drop off nothing")
                    reward -=10
                    
        reward -= 0.1  

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, {}

        

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle
        
        # print("taxi_row: ", taxi_row)
        # print("taxi_col: ", taxi_col)
        # print("obstacle_north: ", obstacle_north)
        # print("obstacle_south: ", obstacle_south)
        # print("obstacle_east: ", obstacle_east)
        # print("obstacle_west: ", obstacle_west)
        # print("passenger_look: ", passenger_look)
        # print("destination_look: ", destination_look)

        
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        '''
        # Place passenger
        py, px = passenger_pos
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'
        '''
        
        grid[self.stations[0][0]][self.stations[0][1]] = 'R'
        grid[self.stations[1][0]][self.stations[1][1]] = 'G'
        grid[self.stations[2][0]][self.stations[2][1]] = 'Y'
        grid[self.stations[3][0]][self.stations[3][1]] = 'B' 

        # find obstacles 'O'
        for x, y in self.obstacles:
            grid[x][y] = 'O'

        # grid[0][0]='R'
        # grid[0][4]='G'
        # grid[4][0]='Y'
        # grid[4][4]='B'
        '''
        # Place destination
        dy, dx = destination_pos
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'
        '''
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        #print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        #print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

class Agent():
    def __init__(self, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.99995, path = None):
        if path == None:
            self.q_table = {}
        else:
            with open(path, 'rb') as f:
                self.q_table = pickle.load(f)
        self.alpha = alpha 
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self._reset()

    def _reset(self):
        MAX_SIZE = 10
        self.visit_count = np.zeros((MAX_SIZE + 2, MAX_SIZE + 2))
        self.wall = np.zeros((MAX_SIZE + 2, MAX_SIZE + 2))
        self.passenger_pos = None
        self.first_near_passenger_pos = None

        self.passenger_on = False

        self.destination_pos = None
        self.first_near_destination_pos = None

        self.is_passenger = [0, 0, 0, 0]
        self.is_destination = [0, 0, 0, 0]

        self.x_min, self.x_max, self.y_min, self.y_max = 0, 0, 0, 0

        self.prev_obs = None
        self.first = True

    def nearby(self, pos1, pos2):
        if abs(pos1[0] - pos2[0]) == 0 and abs(pos1[1] - pos2[1]) == 0:
            return 1
        if abs(pos1[0] - pos2[0]) == 1 and abs(pos1[1] - pos2[1]) == 0: 
            return 1
        if abs(pos1[0] - pos2[0]) == 0 and abs(pos1[1] - pos2[1]) == 1:
            return 1
        return 0

    def get_action(self, obs, debug = False, deterministic = False, eval = False):
        # Make sure env is not changed
        
        # TODO: Train your own agent
        # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
        # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
        #       To prevent crashes, implement a fallback strategy for missing keys. 
        #       Otherwise, even if your agent performs well in training, it may fail during testing.
        self.visit_count[obs[0] + 1][obs[1] + 1] += 1
        self.car_pos = (obs[0] + 1, obs[1] + 1)

        state, goal_pos = self.extract_state(obs)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(6)
        
        if np.random.rand() < self.epsilon and not deterministic and not eval:
            action = np.random.choice([0, 1, 2, 3, 4, 5])
        else:
            prob = self.q_table[state].copy()

            if debug:
                print(f"state: {state[2:6]}")

            # softmax to get the probability
            max_prob = np.max(prob)
            prob -= max_prob
            prob = np.exp(prob)
            prob /= np.sum(prob)

            act = None
            if prob[4] + prob[5] > 0.5:
                act = "pick_drop"
            else:
                act = "move"
            
            down_count = self.visit_count[self.car_pos[0] + 1][self.car_pos[1]]
            up_count = self.visit_count[self.car_pos[0] - 1][self.car_pos[1]]
            right_count = self.visit_count[self.car_pos[0]][self.car_pos[1] + 1]
            left_count = self.visit_count[self.car_pos[0]][self.car_pos[1] - 1]

            # randomly choose the action with the probability
            if deterministic:
                action = np.argmax(prob)
            else:
                if act == "move":
                    new_prob = self.q_table[state][:4].copy()

                    if state[2]:
                        new_prob[0] = -1000000
                    if state[3]:
                        new_prob[1] = -1000000
                    if state[4]:
                        new_prob[2] = -1000000
                    if state[5]:
                        new_prob[3] = -1000000

                    if debug:
                        print(f"new prob: {new_prob}")

                    max_new_prob = np.max(new_prob)
                    new_prob -= max_new_prob
                    new_prob = np.exp(new_prob)

                    new_prob[0] /= 5**down_count
                    new_prob[1] /= 5**up_count
                    new_prob[2] /= 5**right_count
                    new_prob[3] /= 5**left_count

                    new_prob /= np.sum(new_prob)
                    action = np.random.choice([0, 1, 2, 3], p=new_prob)

                    if debug:
                        print(f"new prob: {new_prob}")
                else:
                    # if prob[4] > prob[5]:
                    #     action = 4
                    # else:
                    #     action = 5
                    # if np.random.rand() < (prob[4] / (prob[4] + prob[5])):
                    #     action = 4
                    # else:
                    #     action = 5
                    if prob[4] > prob[5]:
                        action = 4
                    else:
                        action = 5

        if debug:
            # print(f"visit_count: {self.visit_count}")
            print(f"wall: {self.wall}")
            # print(f"passenger_pos: {self.passenger_pos}")
            # print(f"destination_pos: {self.destination_pos}")
            # print(f"passenger_on: {self.passenger_on}")
            # print(f"is_passenger: {self.is_passenger}")
            # print(f"is_destination: {self.is_destination}")
            # print(f"goal_pos: {goal_pos}")
            # print(f"self.car_pos: {self.car_pos}")
            # print(f"state: {state}")
            # print(f"action: {action}")
            # print(f"q_table: {self.q_table[state]}")
            # print(f"prob: {prob}")
            # print(f"down_count: {down_count}")
            # print(f"up_count: {up_count}")
            # print(f"right_count: {right_count}")
            # print(f"left_count: {left_count}")
            # print(f"station_pos: {[(obs[2] + 1, obs[3] + 1), (obs[4] + 1, obs[5] + 1), (obs[6] + 1, obs[7] + 1), (obs[8] + 1, obs[9] + 1)]}")
            # print(f"observation: {obs}")

        # if pickup
        if action == 4 and self.car_pos == self.passenger_pos:
            self.passenger_on = True

        return action
        # You can submit this random agent to evaluate the performance of a purely random strategy.
    def extract_state(self, obs):
        car_pos = (obs[0] + 1, obs[1] + 1)
        station_pos = [(obs[2] + 1, obs[3] + 1), (obs[4] + 1, obs[5] + 1), (obs[6] + 1, obs[7] + 1), (obs[8] + 1, obs[9] + 1)]
        x_min = np.min([station_pos[0][0], station_pos[1][0], station_pos[2][0], station_pos[3][0], car_pos[0]])    
        x_max = np.max([station_pos[0][0], station_pos[1][0], station_pos[2][0], station_pos[3][0], car_pos[0]])
        y_min = np.min([station_pos[0][1], station_pos[1][1], station_pos[2][1], station_pos[3][1], car_pos[1]])
        y_max = np.max([station_pos[0][1], station_pos[1][1], station_pos[2][1], station_pos[3][1], car_pos[1]])
        
        
        # print(x_min, x_max, y_min, y_max)

        if self.passenger_pos is None:
            passenger_nearby = obs[14]
            a = self.nearby(station_pos[0], car_pos)
            b = self.nearby(station_pos[1], car_pos)
            c = self.nearby(station_pos[2], car_pos)
            d = self.nearby(station_pos[3], car_pos)

            if passenger_nearby:
                if a == 0:
                    self.is_passenger[0] = -1
                if b == 0:
                    self.is_passenger[1] = -1
                if c == 0:
                    self.is_passenger[2] = -1
                if d == 0:
                    self.is_passenger[3] = -1
                if a == 1 and self.is_passenger[0] != -1:
                    self.is_passenger[0] = 1
                if b == 1 and self.is_passenger[1] != -1:
                    self.is_passenger[1] = 1
                if c == 1 and self.is_passenger[2] != -1:
                    self.is_passenger[2] = 1
                if d == 1 and self.is_passenger[3] != -1:
                    self.is_passenger[3] = 1
            else:
                if a:
                    self.is_passenger[0] = -1
                if b:
                    self.is_passenger[1] = -1
                if c:
                    self.is_passenger[2] = -1
                if d:
                    self.is_passenger[3] = -1

            ones = np.where(np.array(self.is_passenger) == 1)[0]
            zeros = np.where(np.array(self.is_passenger) == 0)[0]
            minus_ones = np.where(np.array(self.is_passenger) == -1)[0]

            if len(ones) == 1:
                self.passenger_pos = [station_pos[0], station_pos[1], station_pos[2], station_pos[3]][ones[0]]
            elif len(minus_ones) == 3:
                self.passenger_pos = [station_pos[0], station_pos[1], station_pos[2], station_pos[3]][zeros[0]]

        if self.destination_pos is None:
            destination_nearby = obs[15]
            a = self.nearby(station_pos[0], car_pos)
            b = self.nearby(station_pos[1], car_pos)
            c = self.nearby(station_pos[2], car_pos)
            d = self.nearby(station_pos[3], car_pos)

            if destination_nearby:
                if a == 0:
                    self.is_destination[0] = -1
                if b == 0:
                    self.is_destination[1] = -1
                if c == 0:
                    self.is_destination[2] = -1
                if d == 0:
                    self.is_destination[3] = -1
                if a == 1 and self.is_destination[0] != -1:
                    self.is_destination[0] = 1
                if b == 1 and self.is_destination[1] != -1:
                    self.is_destination[1] = 1
                if c == 1 and self.is_destination[2] != -1:
                    self.is_destination[2] = 1
                if d == 1 and self.is_destination[3] != -1:
                    self.is_destination[3] = 1
            else:
                if a:
                    self.is_destination[0] = -1
                if b:
                    self.is_destination[1] = -1
                if c:
                    self.is_destination[2] = -1
                if d:
                    self.is_destination[3] = -1

            ones = np.where(np.array(self.is_destination) == 1)[0]
            zeros = np.where(np.array(self.is_destination) == 0)[0]
            minus_ones = np.where(np.array(self.is_destination) == -1)[0]
            if len(ones) == 1:
                self.destination_pos = [station_pos[0], station_pos[1], station_pos[2], station_pos[3]][ones[0]]
            elif len(minus_ones) == 3:
                self.destination_pos = [station_pos[0], station_pos[1], station_pos[2], station_pos[3]][zeros[0]]

        # if obs[10] or car_pos[0] == x_min:
        #     self.wall[car_pos[0] - 1][car_pos[1]] = 1
        # if obs[11] or car_pos[0] == x_max:
        #     self.wall[car_pos[0] + 1][car_pos[1]] = 1
        # if obs[12] or car_pos[1] == y_max:
        #     self.wall[car_pos[0]][car_pos[1] + 1] = 1
        # if obs[13] or car_pos[1] == y_min:
        #     self.wall[car_pos[0]][car_pos[1] - 1] = 1

        if obs[10]:
            self.wall[car_pos[0] - 1][car_pos[1]] = 1
        if obs[11]:
            self.wall[car_pos[0] + 1][car_pos[1]] = 1
        if obs[12]:
            self.wall[car_pos[0]][car_pos[1] + 1] = 1
        if obs[13]:
            self.wall[car_pos[0]][car_pos[1] - 1] = 1

        goal_pos = None
        if self.passenger_pos is None:
            # Find the nearest station
            min_dist = 100000
            for i in range(4):
                if self.is_passenger[i] == 1:
                    dist = abs(station_pos[i][0] - car_pos[0]) + abs(station_pos[i][1] - car_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        goal_pos = station_pos[i]
            if goal_pos is None:
                for i in range(4):
                    if self.is_passenger[i] == 0:
                        dist = abs(station_pos[i][0] - car_pos[0]) + abs(station_pos[i][1] - car_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            goal_pos = station_pos[i]
        elif not self.passenger_on and self.passenger_pos is not None:
            goal_pos = self.passenger_pos
        elif self.passenger_on and self.destination_pos is None:
            min_dist = 100000
            for i in range(4):
                if self.is_destination[i] == 1:
                    dist = abs(station_pos[i][0] - car_pos[0]) + abs(station_pos[i][1] - car_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        goal_pos = station_pos[i]
            if goal_pos is None:
                for i in range(4):
                    if self.is_destination[i] == 0:
                        dist = abs(station_pos[i][0] - car_pos[0]) + abs(station_pos[i][1] - car_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            goal_pos = station_pos[i]
        else:
            goal_pos = self.destination_pos

        # Find the dir of goal
        if goal_pos[0] > car_pos[0] and goal_pos[1] > car_pos[1]:
            dir = 1
        elif goal_pos[0] == car_pos[0] and goal_pos[1] > car_pos[1]:
            dir = 2
        elif goal_pos[0] < car_pos[0] and goal_pos[1] > car_pos[1]:
            dir = 3
        elif goal_pos[0] < car_pos[0] and goal_pos[1] == car_pos[1]:
            dir = 4
        elif goal_pos[0] < car_pos[0] and goal_pos[1] < car_pos[1]:
            dir = 5
        elif goal_pos[0] == car_pos[0] and goal_pos[1] < car_pos[1]:
            dir = 6
        elif goal_pos[0] > car_pos[0] and goal_pos[1] < car_pos[1]:
            dir = 7
        elif goal_pos[0] > car_pos[0] and goal_pos[1] == car_pos[1]:
            dir = 8
        else:
            dir = 0

        state = (dir, self.passenger_on, self.wall[car_pos[0] + 1][car_pos[1]], self.wall[car_pos[0] - 1][car_pos[1]], self.wall[car_pos[0]][car_pos[1] + 1], self.wall[car_pos[0]][car_pos[1] - 1])
        return state, goal_pos
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.epsilon_end)

    def update_q_table(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(6)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)


# Train your agent here.

def train_agent(episodes=1000, max_steps=100, render_every=100):
    env = SimpleTaxiEnv(grid_size=10, fuel_limit=50)
    agent = Agent(alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.9995)
    
    all_rewards = []
    avg_rewards = []
    success_rate = []
    success_count = 0
    
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        agent._reset()  # Reset agent's tracking variables
        
        total_reward = 0
        episode_success = False
        prev_state = None
        
        for step in range(1, max_steps + 1):
            # Create state representation
            state, goal_pos = agent.extract_state(obs)
            
            # Select action
            action = agent.get_action(obs, debug=(episode == 10000))
            
            # Take action
            next_obs, reward, done, _ = env.step(action)
            prev_dist = abs(goal_pos[0] - obs[0] - 1) + abs(goal_pos[1] - obs[1] - 1)
            next_dist = abs(goal_pos[0] - next_obs[0] - 1) + abs(goal_pos[1] - next_obs[1] - 1)
            if next_dist < prev_dist:
                reward += 1
            else:
                reward -= 1 

            next_state = agent.extract_state(next_obs)
            # If passby the passenger without picking up
            if not agent.passenger_on and agent.car_pos == agent.passenger_pos:
                reward -= 20
            # If passby the destination without dropping off
            if agent.passenger_on and agent.car_pos == agent.destination_pos and action != 5:
                reward -= 50
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state)
            
            # Track success
            if done and step != max_steps:
                episode_success = True
            
            total_reward += reward
            obs = next_obs
            
            # Render if needed
            if episode % render_every == 0 and episode > episodes - render_every:
                print(reward)
                env.render_env(taxi_pos=(obs[0], obs[1]), action=action, step=step, fuel=env.current_fuel)
                time.sleep(0.1) 
                
            if done:
                break
        
        # Update epsilon after each episode
        agent.update_epsilon()
        
        # Record success
        if episode_success:
            success_count += 1
        
        # Track metrics
        all_rewards.append(total_reward)
        avg_reward = np.mean(all_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        if episode % 100 == 0:
            success_rate.append(success_count / 100)
            success_count = 0
            print(f"Episode: {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate[-1]:.2f}, Epsilon: {agent.epsilon:.3f}")
            
    # Save the trained agent
    agent.save('taxi_agent.pkl')
    
    print("Training completed!")
    print(f"Final average reward over last 100 episodes: {avg_rewards[-1]:.2f}")
    print(f"Final success rate over last 100 episodes: {success_rate[-1]:.2f}")
    
    return agent, all_rewards, avg_rewards, success_rate

# Run the training
if __name__ == "__main__":
    agent, rewards, avg_rewards, success_rates = train_agent(episodes=20000, max_steps=50, render_every=3000)