import numpy as np
import random
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import random

def choices(population, weights=None, *, cum_weights=None, k=1):
    """Return a k sized list of population elements chosen with replacement.
    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.
    """
    n = len(population)
    if cum_weights is None:
        if weights is None:
            _int = int
            n += 0.0    # convert to float for a small speed improvement
            return [population[_int(random.random() * n)] for i in _repeat(None, k)]
        cum_weights = list(_accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise ValueError('The number of weights does not match the population')
    bisect = _bisect
    total = cum_weights[-1] + 0.0   # convert to float
    hi = n - 1
    return [population[bisect(cum_weights, random.random() * total, 0, hi)]
            for i in _repeat(None, k)]


def sampling_pairs(num_pair, n=12, fixed_goal=None, fixed_start=None, goal_length=None):
    possibilities = set(range(1, n*(n-1)))
    if fixed_start is not None:
        remove_start = clean_up_cell(n, fixed_start[0], fixed_start[1])
        possibilities -= set(remove_start)
    if fixed_goal is not None:
        remove_goal = clean_up_cell(n, fixed_goal[0], fixed_goal[1])
        possibilities -= set(remove_goal)
    keys = []
    locks = []
    # if fixed_goal is not None:
    #     num_pair = num_pair -1
    for k in range(num_pair):
        key = random.sample(possibilities, 1)[0]
        key_x, key_y = key//(n-1), key%(n-1)
        lock_x, lock_y = key_x, key_y + 1
        to_remove = clean_up_cell(n, key_x, key_y)
        if key_y in [0, n-2] :
            upper_cell_x, upper_cell_y = key_x -1, key_y
            to_remove += clean_up_cell(n, upper_cell_x, upper_cell_y)
            lower_cell_x, lower_cell_y = key_x + 1, key_y
            to_remove += clean_up_cell(n, lower_cell_x, lower_cell_y)
            if key_x == 0:
                lower_cell_x, lower_cell_y = key_x + 2, key_y
                to_remove += clean_up_cell(n, lower_cell_x, lower_cell_y)
                if key_y == n-2:
                    to_remove += clean_up_cell(n, lower_cell_x, lower_cell_y-1)
                    to_remove += clean_up_cell(n, lower_cell_x, lower_cell_y-2)
            if key_x == n-1:
                upper_cell_x, upper_cell_y = key_x - 2, key_y
                to_remove += clean_up_cell(n, upper_cell_x, upper_cell_y)
                if key_y == n-2:
                    to_remove += clean_up_cell(n, upper_cell_x, upper_cell_y-1)
                    to_remove += clean_up_cell(n, upper_cell_x, upper_cell_y-2)
        if key_x == 0:
            lower_cell_x, lower_cell_y = key_x + 1, key_y
            to_remove += clean_up_cell(n, lower_cell_x, lower_cell_y)
        elif key_x == n-1:
            upper_cell_x, upper_cell_y = key_x - 1, key_y
            to_remove += clean_up_cell(n, upper_cell_x, upper_cell_y)
        elif key_x == 1:
            upper_cell_x, upper_cell_y = key_x - 1, key_y
            to_remove += clean_up_cell(n, upper_cell_x, upper_cell_y)
        elif key_x == n-2:
            lower_cell_x, lower_cell_y = key_x + 1, key_y
            to_remove += clean_up_cell(n, lower_cell_x, lower_cell_y)
        possibilities -= set(to_remove)

        keys.append([key_x, key_y])
        locks.append([lock_x, lock_y])
    if fixed_goal is not None and goal_length >1:
        keys[goal_length-1] = [fixed_goal[0], fixed_goal[1]]
        locks[goal_length-1] = [fixed_goal[0], fixed_goal[1]+1]
    if fixed_start is None:
        agent_pos = random.sample(possibilities, 1)
        possibilities -= set(agent_pos)
        agent_pos = np.array([agent_pos[0]//(n-1), agent_pos[0]%(n-1)])
    else:
        agent_pos = np.array(fixed_start)

    first_key = random.sample(possibilities, 1)
    first_key = first_key[0]//(n-1), first_key[0]%(n-1)
    return keys, locks, first_key, agent_pos


def clean_up_cell(n, upper_cell_x, upper_cell_y):
    return [upper_cell_x * (n - 1) + upper_cell_y] + \
           [upper_cell_x * (n - 1) + i + upper_cell_y for i in
            range(1, min(2, n - 2 - upper_cell_y) + 1)] + \
           [upper_cell_x * (n - 1) - i + upper_cell_y for i in range(1, min(2, upper_cell_y) + 1)]




colors = {1: [230, 190, 255], 2: [170, 255, 195], 3: [255, 250, 200],
                       4: [255, 216, 177], 5: [250, 190, 190], 6: [240, 50, 230], 7: [145, 30, 180], 8: [67, 99, 216],
                       9: [66, 212, 244], 10: [60, 180, 75], 11: [191, 239, 69], 12: [255, 255, 25], 13: [245, 130, 49],
                       14: [230, 25, 75], 15: [128, 0, 0], 16: [154, 99, 36], 17: [128, 128, 0], 18: [70, 153, 144],
                       0: [0, 0, 117]}

COLOR_ID = dict([(tuple(v), k) for k, v in colors.items()])  # , "wall"])])

num_colors = len(colors)
agent_color = [128, 128, 128]
goal_color = [255, 255, 255]
grid_color = [220, 220, 220]
wall_color = [0, 0, 0]

def world_gen(n=12, goal_length=3, num_distractor=2, distractor_length=2, seed=None, own_key=False, fixed_start=None, fixed_goal=None):
    """generate BoxWorld
    """
    if seed is not None:
        random.seed(seed)
    n = n-2
    if fixed_goal is not None:
        fixed_goal = np.array(fixed_goal) - 1
    if fixed_start is not None:
        fixed_start = np.array(fixed_start) - 1
    world_dic = {} # dic keys are lock positions, value is 0 if distractor, else 1.
    world = np.ones((n, n, 3)) * 220
    goal_colors = random.sample(range(num_colors), goal_length - 1)
    distractor_possible_colors = [color for color in range(num_colors) if color not in goal_colors]
    distractor_colors = [random.sample(distractor_possible_colors, distractor_length) for k in range(num_distractor)]
    if goal_length > 1:
        distractor_roots = choices(range(goal_length - 1), k=num_distractor)
    else:
        distractor_roots = np.zeros(num_distractor)
    keys, locks, first_key, agent_pos = sampling_pairs(goal_length - 1 + distractor_length * num_distractor, n, fixed_goal, fixed_start, goal_length)
    
    key_loc = [[first_key[0]+1,first_key[1]+1]]
    lock_loc = []

    for i in range(len(keys)):
        key_loc.append([keys[i][0]+1,keys[i][1]+1])
        lock_loc.append([locks[i][0]+1,locks[i][1]+1])
    #print(len(keys))
    #exit(1)
    # Own the first key or not
    #if not own_key:
    #    own_key = random.choice([True, False])

    # if goal_length == 1:
    #     own_key = False
    
    gem_location = np.zeros(2)
    # first, create the goal path
    for i in range(1, goal_length):
        if i == goal_length - 1:
            color = goal_color  # final key is white
            gem_location = np.array( keys[i-1])+1
        else:
            color = colors[goal_colors[i]]
        # print("place a key with color {} on position {}".format(color, keys[i-1]))
        # print("place a lock with color {} on {})".format(colors[goal_colors[i-1]], locks[i-1]))
        world[keys[i-1][0], keys[i-1][1]] = np.array(color)
        world[locks[i-1][0], locks[i-1][1]] = np.array(colors[goal_colors[i-1]])
        world_dic[tuple(locks[i-1] + np.array([1, 1]))] = 1
    first_box = np.array([0,0])
    if len(goal_colors)> 0:
        # keys[0] is an orphand key so skip it
        # if the key is not owned, place it in the grid
        if not own_key:
            world[first_key[0], first_key[1]] = np.array(colors[goal_colors[0]])
            first_box = np.array(first_key) + 1
        # world_dic[first_key[0]+1, first_key[1]+2] = 1
        # print("place the first key with color {} on position {}".format(goal_colors[0], first_key))
    else:
        if fixed_goal is not None:
            gem_location = np.array(fixed_goal)
        else:
            gem_location = np.array(first_key)
        world[gem_location[0], gem_location[1]] = np.array(goal_color)
        gem_location =np.array(gem_location)+1
        first_box = np.array(gem_location)

    # place distractors
    if goal_length > 1:
        for i, (distractor_color, root) in enumerate(zip(distractor_colors, distractor_roots)):
            key_distractor = keys[goal_length-1 + i*distractor_length: goal_length-1 + (i+1)*distractor_length]
            color_lock = colors[goal_colors[root]]
            color_key = colors[distractor_color[0]]
            world[key_distractor[0][0], key_distractor[0][1] + 1] = np.array(color_lock)
            world[key_distractor[0][0], key_distractor[0][1]] = np.array(color_key)
            world_dic[key_distractor[0][0] + 1, key_distractor[0][1] + 2] = 0
            #key_loc.append([key_distractor[0][0]+1,key_distractor[0][0]+1])
            #lock_loc.append([key_distractor[0][0]+1,key_distractor[0][0]+2])
            for k, key in enumerate(key_distractor[1:]):
                color_lock = colors[distractor_color[k]]
                color_key = colors[distractor_color[k-1]]
                world[key[0], key[1]] = np.array(color_key)
                world[key[0], key[1]+1] = np.array(color_lock)
                world_dic[key[0] + 1, key[1] + 2] = 0
                #key_loc.append([key[0]+1,key[1]+1])
                #lock_key.append([key[0]+1,key[1]+2])
    else:
        for i, (distractor_color, root) in enumerate(zip(distractor_colors, distractor_roots)):
            key_distractor = keys[goal_length-1 + i*distractor_length: goal_length-1 + (i+1)*distractor_length]
            color_lock = colors[random.sample(range(num_colors), 1)[0]]
            color_key = colors[distractor_color[0]]
            world[key_distractor[0][0], key_distractor[0][1] + 1] = np.array(color_lock)
            world[key_distractor[0][0], key_distractor[0][1]] = np.array(color_key)
            world_dic[key_distractor[0][0] + 1, key_distractor[0][1] + 2] = 0
            #key_loc.append([key_distractor[0][0]+1,key_distractor[0][0]+1])
            #lock_loc.append([key_distractor[0][0]+1,key_distractor[0][0]+2])
            for k, key in enumerate(key_distractor[1:]):
                color_lock = colors[distractor_color[k]]
                color_key = colors[distractor_color[k-1]]
                world[key[0], key[1]] = np.array(color_key)
                world[key[0], key[1]+1] = np.array(color_lock)
                world_dic[key[0] + 1, key[1] + 2] = 0
                
                #key_loc.append([key[0]+1,key[1]+1])
                #lock_key.append([key[0]+1,key[1]+2])

    # place an agent
    world[agent_pos[0], agent_pos[1]] = np.array(agent_color)
    agent_pos += np.array([1, 1])

    # add black wall
    wall_0 = np.zeros((1, n, 3))
    wall_1 = np.zeros((n+2, 1, 3))
    world = np.concatenate((wall_0, world, wall_0), axis=0)
    world = np.concatenate((wall_1, world, wall_1), axis=1)

    # add key if owned
    if own_key:
        world[0,0]=np.array(colors[goal_colors[0]])
        first_box = np.array(locks[0]) +1

    return world, agent_pos, world_dic, gem_location, first_box, key_loc, lock_loc

def update_color(world, previous_agent_loc, new_agent_loc):
        world[previous_agent_loc[0], previous_agent_loc[1]] = grid_color
        world[new_agent_loc[0], new_agent_loc[1]] = agent_color

def is_empty(room):
    return np.array_equal(room, grid_color) or np.array_equal(room, agent_color)

