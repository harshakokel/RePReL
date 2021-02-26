from gym.envs.registration import register

register(
    id='BoxWorld_RePReL_task1-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': True, 'env_id': 'BoxWorld-task1-v1'},
)
register(
    id='BoxWorld_RePReL_task2-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': True, 'env_id': 'BoxWorld-task2-v1'},
)
register(
    id='BoxWorld_RePReL_task3-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': True, 'env_id': 'BoxWorld-task3-v1'},
)

register(
    id='BoxWorld_RePReL_task4-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': True, 'env_id': 'BoxWorld-task4-v1'},
)

register(
    id='BoxWorld_RePReL_task5-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': True, 'env_id': 'BoxWorld-task5-v1'},
)

register(
    id='BoxWorld_taskable_task1-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': False, 'env_id': 'BoxWorld-task1-v1'},
)
register(
    id='BoxWorld_taskable_task2-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': False, 'env_id': 'BoxWorld-task2-v1'},
)
register(
    id='BoxWorld_taskable_task3-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': False, 'env_id': 'BoxWorld-task3-v1'},
)

register(
    id='BoxWorld_taskable_task4-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': False, 'env_id': 'BoxWorld-task4-v1'},
)


register(
    id='BoxWorld_taskable_task5-v1',
    entry_point='core.reprel_environment.box_world:RePReLBoxWorld',
    kwargs={'abstraction': False, 'env_id': 'BoxWorld-task5-v1'},
)
