from csv import writer
import numpy as np
import gym
import matplotlib
import pickle
import logging
import os


# Set learning parameters
learning_rate = .01
discount_factor = .99
num_test_episodes = 100
log_step = 10000
total_steps_per_task = 1500000  # 2e6
test_epsilon = 0.01
num_episodes = 20000


def train(train_env, test_env, operator_Qtables = None, writer=None, taskid="task0",
          model="RePReL", run="run1", domain="BoxWorld", action_space=4, terminal_reward=1):
    # create lists to contain total rewards and steps per episode
    episode_rewards = []
    random_ep = np.arange(0.01, 0.75, 0.99 / num_episodes, dtype=float)[::-1]
    steps = 0
    i = 0

    while steps < total_steps_per_task:
        # Reset environment
        s = train_env.reset()
        # Get high-level plan
        plan_seq = train_env.plan_seq
        total_reward = 0
        ep = random_ep[min(i, random_ep.shape[0] - 1)]
        i += 1
        done = False

        # for each grounded operator
        for operator in plan_seq:
            #Get resp. RL policy
            operator_index = train_env.operator_order.index(operator[0])
            Q_table = operator_Qtables[operator_index]

            # Get abstract rep of current state
            s_hat = train_env.get_state_representation(s, operator)
            subgoal_done = train_env.is_terminal_state(operator, s)
            while not subgoal_done:

                # Get action
                a = get_action(Q_table, action_space, ep, s_hat)

                # take step in env and get reward
                s_prime, r, done, _ = train_env.step(a)
                steps += 1

                # Get Abstract State
                s_prime_hat = train_env.get_state_representation(s_prime, operator)

                # Check if the next state is terminal
                subgoal_done = train_env.is_terminal_state(operator, s_prime)

                # add terminal reward
                if subgoal_done:
                    r += terminal_reward
                if done:
                    subgoal_done = done
                # update Q_table
                update_qvalue(Q_table, action_space, s_hat, a, r,  s_prime_hat)

                total_reward += r
                s = s_prime
                s_hat = s_prime_hat

                if steps % log_step == 0:
                    success, reward = test(operator_Qtables, test_env=test_env, action_space=action_space)
                    logging.debug([domain, model, run, taskid, steps, success, reward])
                    writer.writerow([domain, model, train_env.spec.id, run, taskid, steps, success, reward])

        episode_rewards.append(total_reward)
    return operator_Qtables


def update_qvalue(Q_table, action_space, s_hat, a, r, s_prime_hat):
    if s_prime_hat in Q_table.keys():
        q_next = Q_table[s_prime_hat]
    else:
        q_next = np.zeros(action_space)
        Q_table[s_prime_hat] = q_next
    if s_hat not in Q_table:
        Q_table[s_hat] = np.zeros(action_space)
    Q_table[s_hat][a] = Q_table[s_hat][a] + learning_rate * (r + discount_factor * np.max(q_next) - Q_table[s_hat][a])


def get_action(Q_table, action_space, ep, s_hat):
    if np.random.randn(0, 1) < ep:
        a = np.random.randint(action_space)
    else:
        if s_hat in Q_table:
            where_max = np.where(Q_table[s_hat] == np.max(Q_table[s_hat]))[0]
            if len(where_max) == 1:
                a = where_max[0]
            else:
                a = np.random.choice(where_max)
        else:
            a = np.random.randint(action_space)
    return a


def test(operator_Qtables, test_env, display=False, action_space=4):
    is_success = np.zeros(num_test_episodes)
    episode_reward = np.zeros(num_test_episodes)
    for i in range(num_test_episodes):
        # Reset environment and get first new observation
        s = test_env.reset()
        total_reward = 0
        plan_seq = test_env.plan_seq
        for operator in plan_seq:
            operator_index = test_env.operator_order.index(operator[0])
            subgoal_done = test_env.is_terminal_state(operator, s)
            s_hat = test_env.get_state_representation(s, operator)
            Q_table = operator_Qtables[operator_index]
            if display:
                test_env.render()
            while not subgoal_done:

                # get action
                a = get_action(Q_table,action_space,test_epsilon,s_hat)

                # take step in env
                s_prime, r, done, info = test_env.step(a)

                # get abstract representation
                s_prime_hat = test_env.get_state_representation(s_prime, operator)

                # Check if the next state is terminal
                subgoal_done = test_env.is_terminal_state(operator, s_prime)
                if done:
                    subgoal_done = done

                if display:
                    test_env.render()
                s = s_prime
                s_hat = s_prime_hat
                total_reward += r
        is_success[i] = info['is_success']
        episode_reward[i] = total_reward
    return np.average(is_success), np.average(episode_reward)








