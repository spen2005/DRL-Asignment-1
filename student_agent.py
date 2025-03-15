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
        # print(f"observation: {obs}")
        self.visit_count[obs[0] + 1][obs[1] + 1] += 1
        self.car_pos = (obs[0] + 1, obs[1] + 1)
        # station0_pos = (obs[2] + 1, obs[3] + 1)
        # station1_pos = (obs[4] + 1, obs[5] + 1)
        # station2_pos = (obs[6] + 1, obs[7] + 1)
        # station3_pos = (obs[8] + 1, obs[9] + 1)

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
            if (not self.passenger_on and self.car_pos == self.passenger_pos) or (self.passenger_on and self.car_pos == self.destination_pos):
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
                    # new_prob[0] -= 100 * down_count
                    # new_prob[1] -= 100 * up_count
                    # new_prob[2] -= 100 * right_count
                    # new_prob[3] -= 100 * left_count

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
                    if not self.passenger_on:
                        action = 4
                    else:
                        action = 5

        if debug:
            # print(f"visit_count: {self.visit_count}")
            # print(f"wall: {self.wall}")
            print(f"passenger_pos: {self.passenger_pos}")
            print(f"destination_pos: {self.destination_pos}")
            print(f"passenger_on: {self.passenger_on}")
            # print(f"is_passenger: {self.is_passenger}")
            # print(f"is_destination: {self.is_destination}")
            print(f"goal_pos: {goal_pos}")
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

        if obs[10] or car_pos[0] == x_min:
            self.wall[car_pos[0] - 1][car_pos[1]] = 1
        if obs[11] or car_pos[0] == x_max:
            self.wall[car_pos[0] + 1][car_pos[1]] = 1
        if obs[12] or car_pos[1] == y_max:
            self.wall[car_pos[0]][car_pos[1] + 1] = 1
        if obs[13] or car_pos[1] == y_min:
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

# taxi_agent.pkl
agent = Agent(path = 'taxi_agent.pkl')

def get_action(obs, debug=False):
    action = agent.get_action(obs, debug = False, deterministic=False, eval=True)
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

