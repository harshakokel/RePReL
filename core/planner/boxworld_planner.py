from core.planner.pyhop.pyhop import *
import numpy as np
from box_world_gym.envs.boxworld_gen import colors

obs = None



def collect_gem(state, gem):
    if state.own_key==gem:
        return []
    elif state.own_key is not None:
        return [('open_some_lock', gem),('collect_some_key', gem),('collect_gem', gem )]
    elif state.own_key is None:
        return [('collect_some_key', gem),('collect_gem', gem)]
    return False

def unlock(state, lock):
    state.locks.remove(lock)
    state.own_key=None
    return state

def pick_key(state, key):
    state.keys.remove(key)
    state.own_key = key
    return state


agent_color = [128, 128, 128]
goal_color = [255, 255, 255]
grid_color = [220, 220, 220]
wall_color = [0, 0, 0]

def get_environment_state(obs):
    """
     Converts GYM environment to state.
    :param env: gym env Box World
    :return: State with following attributes
     state.world = (8, 8, 3)
     state.pos = pos
    """
    state = State('state1')
    state.keys = ['gem']
    state.locks = []
    state.inside = {}
    state.needs = {}
    state.color = {'gem':'gem'}
    state.own_key = None
    for predicate in obs:
        if "color(" in predicate:
            object = predicate.split("(")[1].replace(")","").replace(" ","").split(",")
            if "key" in object[0]:
                state.keys.append(object[0])
            if "lock" in object[0]:
                state.locks.append(object[0])
            state.color[object[0]] =object[1]
        if "inside(" in predicate:
            object = predicate.split("(")[1].replace(")","").replace(" ","").split(",")
            state.inside[object[0]] = object[1]
            state.needs[object[1]] = object[0]
        if "own" in predicate:
            object = predicate.split("(")[1].replace(")","").replace(" ","").split(",")
            state.own_key = object[0]
    return state


def add_open_lock_color_pos(color, lock_id):
    def open_lock_dynamic_method(state, goal):
        color_lock = color
        lock = lock_id
        if state.own_key is not None and state.color[state.own_key] == color_lock and lock in state.locks:
            return [ ('unlock', lock)]
        return False
    open_lock_dynamic_method.__name__ = "open_%s_lock_%s" % (color, lock_id)
    return open_lock_dynamic_method

def add_collect_key_color_pos(color, key_id):
    def collect_key_dynamic_method(state, goal):
        color_key = color
        key = key_id
        # if it is a key of same color and is free
        if (key in state.keys) and  state.color[key]==color_key and \
            (key not in state.needs.keys() or state.needs[key] not in state.locks):
            return [('pick_key', key)]
        return False
    collect_key_dynamic_method.__name__ = "collect_%s_%s" % (color, key_id)
    return collect_key_dynamic_method



def define_dynamic_methods(colors):
    dm_open_lock = []
    dm_collect_key = []
    for color in colors:
        dm_collect_key.append(add_collect_key_color_pos(f"{color}", f"key_{color}"))
    dm_collect_key.append(add_collect_key_color_pos("gem", "gem"))
    for color in colors:
        dm_open_lock.append(add_open_lock_color_pos(f"{color}", f"lock_{color}"))
    declare_methods('open_some_lock', *dm_open_lock)
    declare_methods('collect_some_key', *dm_collect_key)

def declare_methods_and_operators():
    define_dynamic_methods(colors.keys())
    declare_methods('collect_gem', collect_gem)
    # print_methods()
    declare_operators(pick_key, unlock)


class RelationalBoxWorldPlanner:

    def __init__(self):
        declare_methods_and_operators()
        self.plan = None
        self.operator_list = self.get_operators()

    def set_goal(self, goal):
        self.goal = goal

    def get_plan(self, state):
        return pyhop(get_environment_state(state), [('collect_gem',"gem")], verbose=0)

    def get_next_operator(self, state):
        if self.plan is None:
            sub_tasks = self.get_plan(state)
            sub_tasks.reverse()
        if not self.plan['sub_task']:
            return None, None, None
        sub_task = self.plan['sub_task'].pop()
        return self.operator_list.index(sub_task[0]), sub_task[1]

    def get_operators(self):
        return list(operators.keys())

    def reset(self):
        self.plan = None


if __name__ == '__main__':
    import box_world_gym
    import matplotlib.pyplot as plt
    import time
    import gym


    env = gym.make('BoxWorld-task2-v1')
    # env.seed(10)

    obs = env.reset()

    a = env.render(mode="rgb_array")
    plt.figure()
    plt.imshow(a)
    plt.show()
    print(obs)
    action = 2
    obs = env.step(action)
    initial_state = get_environment_state(obs)
    define_dynamic_methods(colors.keys())

    declare_methods('collect_gem', collect_gem)
    print_methods()
    declare_operators(pick_key, unlock)
    print_operators()
    try:
        plan = pyhop(initial_state, [('collect_gem', 'gem')], verbose=3)
        assert plan is not False
        print("PLAN: ")
        print(plan)
    except AssertionError as error:
        print("Plan not found")
        pickle_out = open("initial_state" + str(time.time_ns()) + ".pickle", "wb")
        pickle.dump(initial_state, pickle_out)