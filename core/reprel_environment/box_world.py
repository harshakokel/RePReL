import gym
import sys
import logging

sys.path.insert(0, '..')
from core.planner.boxworld_planner import RelationalBoxWorldPlanner
import box_world_gym
from box_world_gym.envs.box_world_env import BoxWorld
import re
import matplotlib.pyplot as plt

num_envs = 32


class RePReLBoxWorld(gym.Env):

    def __init__(self, abstraction, env_id):
        self.env = gym.make(env_id)
        self.sub_goal_count = 0
        self.abstraction = abstraction
        self.operator_order = ["unlock", "pick_key"]
        self.plan_seq = None
        self.pyhop_planner = RelationalBoxWorldPlanner()

    def reset(self):
        obs = self.env.reset()
        self.pyhop_planner.reset()
        self.plan_seq = self.pyhop_planner.get_plan(obs)
        self.sub_goal_count = 0
        return obs

    def get_state_representation(self, state, subgoal):
        if self.abstraction:
            return self.get_abstract_state(state, subgoal)
        return "".join(state)

    # D-FOCI rollout indicates following relevant variables
    # For pick_key(x) : neighbor(Dir,C), agent-at(L1), direction(x, Dir2), own(x)
    # For unlock(x) : neighbor(Dir,C), agent-at(L1), direction(x, Dir2), open(x)
    @staticmethod
    def get_abstract_state(state, sub_goal):
        new_state = []
        relevant_variables=[]
        if sub_goal[0] == "pick_key":
            relevant_variables = ["neighbor","agent-at",f"direction({sub_goal[1]},", f"own({sub_goal[1]})"]
        elif sub_goal[0] == "unlock":
            relevant_variables = ["neighbor","agent-at",f"direction({sub_goal[1]},", f"open({sub_goal[1]})"]
        for predicate in state:
            if any(x in predicate for x in relevant_variables):
                # Only subgoal (X) is relevant, all the other locks and keys are irrelevant (Y)
                # new_state.append( re.sub("((lock|key)_\d{1,2}|gem)", "Y", predicate.replace(sub_goal[1], "X")) )
                new_state.append(  predicate.replace(sub_goal[1], "X") )
        return "".join(new_state)

    @staticmethod
    def is_terminal_state(sub_goal, state):
        terminates = False
        if sub_goal[0] == 'unlock':
            terminates = f"open({sub_goal[1]})" in state
        elif sub_goal[0] == 'pick_key':
            terminates = f"own({sub_goal[1]})" in state
        # if terminates:
        #     logging.debug("terminating")
        return terminates

    def step(self, action):
        return self.env.step(action)
        # s, r, done, info = self.env.step(action)
        # if self.abstraction:
        #     state = self.get_abstract_representation(s, self.plan_seq[self.sub_goal_count])
        # else:
        #     state = self.get_taskable_representation(s)
        # sub_goal_done = self.option_termination(self.plan_seq[self.sub_goal_count], s)
        # option_index = self.option_order.index(self.plan_seq[self.sub_goal_count][0])
        # if sub_goal_done:
        #     r += 1.0
        #     self.sub_goal_count += 1
        #     if self.abstraction:
        #         state = self.get_abstract_representation(s, self.plan_seq[self.sub_goal_count])
        #     else:
        #         state = self.get_taskable_representation(s)
        #     option_index = self.option_order.index(self.plan_seq[self.sub_goal_count][0])
        # return state, r, done, info, option_index

    def render(self, mode=None):
        a = self.env.render(mode="rgb_array")
        plt.figure()
        plt.imshow(a)
        plt.show()

