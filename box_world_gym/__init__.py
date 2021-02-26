from gym.envs.registration import register

#Task 1
kwargs = {'n': 10,
          'goal_length': 1,
          'obs_type': "relational",
          'max_steps': 100,
          'reward_gem': 1,  # 50,1
          'step_cost': 0.1,  # 0.2,0.1
          'num_distractor': 0,
          'no_move_cost': 0.2,
          }

register(
    id='BoxWorld-task1-v1',
    entry_point='box_world_gym.envs:BoxWorld',
    kwargs=kwargs,
)


#Task 2
kwargs = {'n': 10,
          'goal_length': 2,
          'obs_type': "relational",
          'max_steps': 200,
          'reward_gem': 1,  # 50,1
          'step_cost': 0.1,  # 0.2,0.1
          'num_distractor': 0,
          'no_move_cost': 0.2,
          'own_key': True,
          }

register(
    id='BoxWorld-task2-v1',
    entry_point='box_world_gym.envs:BoxWorld',
    kwargs=kwargs,
)


# Task 3
kwargs = {'n': 10,
          'goal_length': 3,
          'obs_type': "relational",
          'max_steps': 300,
          'reward_gem': 1,  # 50,1
          'step_cost': 0.1,  # 0.2,0.1
          'num_distractor': 0,
          'no_move_cost': 0.2,
          }

register(
    id='BoxWorld-task3-v1',
    entry_point='box_world_gym.envs:BoxWorld',
    kwargs=kwargs,
)


# Task 3
kwargs = {'n': 10,
          'goal_length': 4,
          'obs_type': "relational",
          'max_steps': 200,
          'reward_gem': 1,  # 50,1
          'step_cost': 0.1,  # 0.2,0.1
          'num_distractor': 0,
          'no_move_cost': 0.2,
          }

register(
    id='BoxWorld-task4-v1',
    entry_point='box_world_gym.envs:BoxWorld',
    kwargs=kwargs,
)

#Task 2
kwargs = {'n': 10,
          'goal_length': 2,
          'obs_type': "relational",
          'max_steps': 200,
          'reward_gem': 1,  # 50,1
          'step_cost': 0.1,  # 0.2,0.1
          'num_distractor': 0,
          'no_move_cost': 0.2,
          'own_key': False,
          }


register(
    id='BoxWorld-task5-v1',
    entry_point='box_world_gym.envs:BoxWorld',
    kwargs=kwargs,
)