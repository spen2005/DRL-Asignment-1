# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

MAX_SIZE = 10
visit_count = np.zeros((MAX_SIZE + 2, MAX_SIZE + 2))
wall = np.zeros((MAX_SIZE + 2, MAX_SIZE + 2))
passenger_pos = None
first_near_passenger_pos = None

passenger_on = False

destination_pos = None
first_near_destination_pos = None

is_passenger = [0, 0, 0, 0]
is_destination = [0, 0, 0, 0]

x_min, x_max, y_min, y_max = 0, 0, 0, 0

prev_obs = None
first = True

def nearby(pos1, pos2):
    if abs(pos1[0] - pos2[0]) == 0 and abs(pos1[1] - pos2[1]) == 0:
        return 1
    if abs(pos1[0] - pos2[0]) == 1 and abs(pos1[1] - pos2[1]) == 0: 
        return 1
    if abs(pos1[0] - pos2[0]) == 0 and abs(pos1[1] - pos2[1]) == 1:
        return 1
    return 0

def get_action(obs):
    global passenger_pos
    global passenger_on
    global destination_pos
    global x_min, x_max, y_min, y_max
    global first

    # Make sure env is not changed
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    # print(f"observation: {obs}")
    visit_count[obs[0] + 1][obs[1] + 1] += 1
    car_pos = (obs[0] + 1, obs[1] + 1)
    station0_pos = (obs[2] + 1, obs[3] + 1)
    station1_pos = (obs[4] + 1, obs[5] + 1)
    station2_pos = (obs[6] + 1, obs[7] + 1)
    station3_pos = (obs[8] + 1, obs[9] + 1)

    x_min = np.min([station0_pos[0], station1_pos[0], station2_pos[0], station3_pos[0], car_pos[0]])    
    x_max = np.max([station0_pos[0], station1_pos[0], station2_pos[0], station3_pos[0], car_pos[0]])
    y_min = np.min([station0_pos[1], station1_pos[1], station2_pos[1], station3_pos[1], car_pos[1]])
    y_max = np.max([station0_pos[1], station1_pos[1], station2_pos[1], station3_pos[1], car_pos[1]])
    
    
    # print(x_min, x_max, y_min, y_max)

    if passenger_pos is None:
        passenger_nearby = obs[14]
        a = nearby(station0_pos, car_pos)
        b = nearby(station1_pos, car_pos)
        c = nearby(station2_pos, car_pos)
        d = nearby(station3_pos, car_pos)

        if passenger_nearby:
            if a == 0:
                is_passenger[0] = -1
            if b == 0:
                is_passenger[1] = -1
            if c == 0:
                is_passenger[2] = -1
            if d == 0:
                is_passenger[3] = -1
            if a == 1 and is_passenger[0] != -1:
                is_passenger[0] = 1
            if b == 1 and is_passenger[1] != -1:
                is_passenger[1] = 1
            if c == 1 and is_passenger[2] != -1:
                is_passenger[2] = 1
            if d == 1 and is_passenger[3] != -1:
                is_passenger[3] = 1

        ones = np.where(np.array(is_passenger) == 1)[0]
        zeros = np.where(np.array(is_passenger) == 0)[0]
        minus_ones = np.where(np.array(is_passenger) == -1)[0]

        if len(ones) == 1:
            passenger_pos = [station0_pos, station1_pos, station2_pos, station3_pos][ones[0]]
        elif len(minus_ones) == 3:
            passenger_pos = [station0_pos, station1_pos, station2_pos, station3_pos][zeros[0]]

    if destination_pos is None:
        destination_nearby = obs[15]
        a = nearby(station0_pos, car_pos)
        b = nearby(station1_pos, car_pos)
        c = nearby(station2_pos, car_pos)
        d = nearby(station3_pos, car_pos)

        if destination_nearby:
            if a == 0:
                is_destination[0] = -1
            if b == 0:
                is_destination[1] = -1
            if c == 0:
                is_destination[2] = -1
            if d == 0:
                is_destination[3] = -1
            if a == 1 and is_destination[0] != -1:
                is_destination[0] = 1
            if b == 1 and is_destination[1] != -1:
                is_destination[1] = 1
            if c == 1 and is_destination[2] != -1:
                is_destination[2] = 1
            if d == 1 and is_destination[3] != -1:
                is_destination[3] = 1

        ones = np.where(np.array(is_destination) == 1)[0]
        zeros = np.where(np.array(is_destination) == 0)[0]
        minus_ones = np.where(np.array(is_destination) == -1)[0]
        if len(ones) == 1:
            destination_pos = [station0_pos, station1_pos, station2_pos, station3_pos][ones[0]]
        elif len(minus_ones) == 3:
            destination_pos = [station0_pos, station1_pos, station2_pos, station3_pos][zeros[0]]

    if obs[10]:
        wall[car_pos[0] - 1][car_pos[1]] = 1
    if obs[11]:
        wall[car_pos[0] + 1][car_pos[1]] = 1
    if obs[12]:
        wall[car_pos[0]][car_pos[1] + 1] = 1
    if obs[13]:
        wall[car_pos[0]][car_pos[1] - 1] = 1



    # action = random.choice([0, 1, 2, 3, 4, 5])
    # Pickup if passenger is not on.
    if not passenger_on and car_pos == passenger_pos:
        action = 4
    # Dropoff
    elif passenger_on and car_pos == destination_pos:
        action = 5
    else:
        # random sample based on visit count and wall state
        if wall[car_pos[0] + 1][car_pos[1]] == 1 or car_pos[0] + 1 > x_max:
            logits0 = -10000
        else:
            logits0 = -visit_count[car_pos[0] + 1][car_pos[1]]
        if wall[car_pos[0] - 1][car_pos[1]] == 1 or car_pos[0] - 1 < x_min:
            logits1 = -10000
        else:
            logits1 = -visit_count[car_pos[0] - 1][car_pos[1]]
        if wall[car_pos[0]][car_pos[1] + 1] == 1 or car_pos[1] + 1 > y_max:
            logits2 = -10000
        else:
            logits2 = -visit_count[car_pos[0]][car_pos[1] + 1]
        if wall[car_pos[0]][car_pos[1] - 1] == 1 or car_pos[1] - 1 < y_min:
            logits3 = -10000
        else:
            logits3 = -visit_count[car_pos[0]][car_pos[1] - 1]

        # sample action based on softmax prob
        logits = np.array([logits0, logits1, logits2, logits3])
        # print(logits)
        prob = np.exp(logits - np.max(logits))
        prob = prob / np.sum(prob)
        # print(prob)
        action = np.random.choice([0, 1, 2, 3], p=prob)
    
    # print(action)

    # if pickup
    if action == 4 and car_pos == passenger_pos:
        passenger_on = True
    if first:
        print(station0_pos, station1_pos, station2_pos, station3_pos, car_pos)
        print(f"visit_count: {visit_count}")
        print(f"wall: {wall}")
        print(f"passenger_pos: {passenger_pos}")
        print(f"destination_pos: {destination_pos}")
        print(f"passenger_on: {passenger_on}")
        print(f"is_passenger: {is_passenger}")
        print(f"is_destination: {is_destination}")
        first = False

    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

