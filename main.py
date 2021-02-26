from csv import writer
import numpy as np
import gym
import matplotlib
import pickle
import logging
import os
from core.RePReL_QLearning import train, test
from box_world_gym.envs.box_world_env import BoxWorld
from core import reprel_environment
import core

logging.basicConfig(level=logging.INFO)

matplotlib.use('TkAgg')

if __name__ == '__main__':
    folder = "data/test/"
    runs = ["run1","run2","run3","run4","run5"]
    # runs = [1]
    for run in runs:
        tasks = ['BoxWorld_RePReL_task2-v1', 'BoxWorld_RePReL_task5-v1','BoxWorld_RePReL_task3-v1']
        model = "RePReL+T"
        Q_lock = {}
        Q_key = {}
        option_Q = [Q_lock, Q_key]
        if not os.path.exists(f"{folder}/{run}"):
            os.makedirs(f"{folder}/{run}")
        logging.info(f"Run {run} {model} started")
        with open(f"{folder}/{run}/boxworld-{model}.csv", "w") as csvfile:
            log_writer = writer(csvfile, delimiter=' ')
            log_writer.writerow(["Domain", "Model", "env_id", "run", "task", "steps", "success_rate", "reward"])
            for i, env_id in enumerate(tasks):
                logging.info(f"{model} {run}: begin experiment task{i} - {env_id}")
                option_Q = train(train_env=gym.make(env_id), test_env=gym.make(env_id), operator_Qtables=option_Q, writer=log_writer, taskid=f"task{i}",
                                 model=model, run=run)
                pickle.dump(Q_key, open(f"{folder}/{run}/reprel-{model}_{run}_task{i}_Q_key.npy", "wb"))
                pickle.dump(Q_lock, open(f"{folder}/{run}/reprel-{model}_{run}_task{i}_Q_lock.npy", "wb"))
                logging.info(f"{model} {run}: end experiment task{i}")
        model = "RePReL"
        logging.info(f"Run {run} {model} started")
        with open(f"{folder}/{run}/boxworld-{model}.csv", "w") as csvfile:
            log_writer = writer(csvfile, delimiter=' ')
            log_writer.writerow(["Domain", "Model", "env_id", "run", "task", "steps", "success_rate", "reward"])
            for i, env_id in enumerate(tasks):
                Q_lock = {}
                Q_key = {}
                option_Q = [Q_lock, Q_key]
                if i == 0:
                    continue
                logging.info(f"{model} {run}: begin experiment task{i} - {env_id}")
                option_Q = train(train_env=gym.make(env_id), test_env=gym.make(env_id), operator_Qtables=option_Q, writer=log_writer, taskid=f"task{i}",
                                 model=model, run=run)
                pickle.dump(Q_key, open(f"{folder}/{run}/reprel-{model}_{run}_task{i}_Q_key.npy", "wb"))
                pickle.dump(Q_lock, open(f"{folder}/{run}/reprel-{model}_{run}_task{i}_Q_lock.npy", "wb"))
                logging.info(f"{model} {run}: end experiment task{i}")

        tasks = ['BoxWorld_taskable_task2-v1','BoxWorld_taskable_task5-v1','BoxWorld_taskable_task3-v1']
        Q_lock = {}
        Q_key = {}
        option_Q = [Q_lock, Q_key]
        model = "trl+T"
        logging.info(f"Run {run} {model} started")
        with open(f"{folder}/{run}/boxworld-{model}.csv", "w") as csvfile:
            log_writer = writer(csvfile, delimiter=' ')
            log_writer.writerow(["Domain", "Model", "env_id","run", "task", "steps", "success_rate", "reward"])
            for i, env_id in enumerate(tasks):
                logging.info(f"{model} {run}: begin experiment task{i} - {env_id}")
                option_Q = train(train_env=gym.make(env_id), test_env=gym.make(env_id), operator_Qtables=option_Q, writer=log_writer, taskid=f"task{i}",
                                 model=model, run=run)
                pickle.dump(Q_key, open(f"{folder}/{run}/trl-{model}_{run}_task{i}_Q_key.npy", "wb"))
                pickle.dump(Q_lock, open(f"{folder}/{run}/trl-{model}_{run}_task{i}_Q_lock.npy", "wb"))
                logging.info(f"{model} {run}: end experiment task{i}")

        model = "trl"
        logging.info(f"Run {run} {model} started")
        with open(f"{folder}/{run}/boxworld-{model}.csv", "w") as csvfile:
            log_writer = writer(csvfile, delimiter=' ')
            log_writer.writerow(["Domain", "Model", "env_id","run", "task", "steps", "success_rate", "reward"])
            for i, env_id in enumerate(tasks):
                if i == 0:
                    continue
                Q_lock = {}
                Q_key = {}
                option_Q = [Q_lock, Q_key]
                logging.info(f"{model} {run}: begin experiment task{i} - {env_id}")
                option_Q = train(train_env=gym.make(env_id), test_env=gym.make(env_id), operator_Qtables=option_Q, writer=log_writer, taskid=f"task{i}",
                                 model=model, run=run)
                pickle.dump(Q_key, open(f"{folder}/{run}/trl-{model}_{run}_task{i}_Q_key.npy", "wb"))
                pickle.dump(Q_lock, open(f"{folder}/{run}/trl-{model}_{run}_task{i}_Q_lock.npy", "wb"))
                logging.info(f"{model} {run}: end experiment task{i}")
