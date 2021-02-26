import gym
from gym.spaces.discrete import Discrete
from gym.spaces import Box, Dict
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.pyplot as plt
from box_world_gym.envs.boxworld_gen import *
#from box_world_gen import *
class BoxWorld(gym.Env):
    """Boxworld representation
    Args:
      n: specify the size of the field (n x n)
      goal_length
      num_distractor
      distractor_length
      world: an existing world data. If this is given, use this data.
             If None, generate a new data by calling world_gen() function
    """

    def __init__(self, n=4, goal_length=2, num_distractor=1, distractor_length=1, max_steps=50, world=None, reward_gem=10,
                 obs_type="dict" , navigation=False, step_cost=0, reward_key=1, no_move_cost=0,reward_distractor=-1,
                 no_key_indicator=False, avoid_direct_goal = False, own_key=False,indicate_distractor=False,
                 generate_video = False,
                 fixed_start=None, fixed_goal=None,fixed_goal_length=False):
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.n = n+2
        self.num_pairs = goal_length - 1 + distractor_length * num_distractor
        '''tem = np.arange(7)
        tem = (tem - np.min(tem))/(np.max(tem)-np.min(tem))
        b = np.ones((9,9))
        c = b[1:-1,1:-1] * tem
        self.x_grid = np.transpose(c)[:,:,np.newaxis]
        self.y_grid = c[:,:,np.newaxis]'''
        # Penalties and Rewards
        self.step_cost = step_cost
        self.reward_gem = reward_gem
        self.reward_key = reward_key
        self.no_move_cost = no_move_cost
        self.fixed_goal_length = fixed_goal_length
        self.reward_distractor = reward_distractor
        self.navigation=navigation
        self.own_key = own_key
        self.avoid_direct_goal = avoid_direct_goal
        # Only working for dictionary
        self.indicate_distractor = indicate_distractor
        self.fixed_goal = fixed_goal
        self.fixed_start = fixed_start
        if self.navigation:
            if self.avoid_direct_goal:
                self.goal_length = random.choice(range(3, 4))
            if self.fixed_goal_length:
                self.fixed_goal_length = goal_length
            else:
                self.goal_length = random.choice(range(1,3))
            self.num_distractor = num_distractor
        self.no_key_indicator = no_key_indicator

        self.world = None
        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.obs_type = obs_type

        # Game initialization
        self.owned_key = [220, 220, 220]
        x_coordinate = np.array([x for y in np.arange(self.n) for x in np.arange(self.n)]).reshape(
            (self.n, self.n, 1)) / (self.n)
        y_coordinate = np.array([y for y in np.arange(self.n) for x in np.arange(self.n)]).reshape(
            (self.n, self.n, 1)) / (self.n)
        self._xy_grid = np.append(y_coordinate, x_coordinate, axis=2)
        self.generate_video = generate_video
        if generate_video:
            self.frames = []
        self.np_random_seed = None
        self.achieved_goal = None
        obs = self.reset(world)
        if obs_type=="img":
            self.observation_space = Box(-np.inf, np.inf, shape=[9,9,5], dtype=np.uint8)
        elif obs_type == "dict":
            self.observation_space = Dict(dict(
                desired_goal=Box(0, 1, shape=obs['desired_goal'].shape, dtype='float32'),
                achieved_goal=Box(0, 1, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            ))
        elif obs_type=="relational":
            self.observation_space = len(obs)
        else:
            self.observation_space = Box(0, 1, shape=obs.shape, dtype=np.uint8)
        self.num_env_steps = 0
        self.episode_reward = 0
        self.last_frames = deque(maxlen=3)
        #print(self.observation_space)
        #exit(1)
    def seed(self, seed=None):
        self.np_random_seed = seed
        return [seed]

    def save(self):
        # np.save('box_world.npy', self.world)
        return ( self.world, self.player_position, self.world_dic, self.goal_location, self.first_key_location)



    def step(self, action):
        self.opened_lock = None
        change = CHANGE_COORDINATES[action]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        self.num_env_steps += 1

        reward = -self.step_cost
        done = self.num_env_steps >= self.max_steps
        solved = False

        # Move player if the field in the moving direction is either

        if np.any(new_position < 1) or np.any(new_position >= self.n - 1):
            possible_move = False
            self.achieved_goal = None
        elif is_empty(self.world[new_position[0], new_position[1]]):
            # No key, no lock
            possible_move = True
            self.achieved_goal = None
        elif new_position[1] == 1 or is_empty(self.world[new_position[0], new_position[1] - 1]):
            # It is a key
            if is_empty(self.world[new_position[0], new_position[1] + 1]):
                # Key is not locked
                possible_move = True
                self.owned_key = self.world[new_position[0], new_position[1]].copy()
                self.world[0, 0] = self.owned_key
                if np.array_equal([new_position[0], new_position[1]], self.goal_location):
                    # Goal reached
                    reward += self.reward_gem
                    solved = True
                    done = True
                else:
                    reward += self.reward_key
            else:
                possible_move = False
            self.achieved_goal = self.owned_key
        else:
            # It is a lock
            if np.array_equal(self.world[new_position[0], new_position[1]], self.owned_key):
                # The lock matches the key
                possible_move = True
                self.opened_lock = self.owned_key.copy()

                # loose old key
                self.owned_key = [0, 0, 0]
                # self.world[new_position[0], new_position[1] - 1] = [220, 220, 220]
                self.world[0, 0] = self.owned_key
                if self.world_dic[tuple(new_position)] == 0:
                    reward += self.reward_distractor
                    done = True
                elif np.array_equal([new_position[0], new_position[1]], self.goal_location):
                    # Goal reached
                    reward += self.reward_gem
                    solved = True
                    done = True
                else:
                    reward += self.reward_key
            else:
                possible_move = False
                # print("lock color is {}, but owned key is {}".format(
                #     self.world[new_position[0], new_position[1]], self.owned_key))
            self.achieved_goal = self.owned_key
        if possible_move:
            self.player_position = new_position
            update_color(self.world, previous_agent_loc=current_position, new_agent_loc=new_position)
        else:
            reward -= self.no_move_cost
        self.episode_reward += reward

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": possible_move,
            "bad_transition": self.max_steps == self.num_env_steps,
            "is_success":False
        }
        if done:
            info["episode"] = {"r": self.episode_reward,
                               "length": self.num_env_steps,
                               "solved": solved}
            info["is_success"] = solved
        if self.generate_video:
            self.frames.append(self.world/255.0)
        self.last_frames.append(self.world)
        #print('pl:-',self.player_position,'gl:-',self.goal_location)
        return self._get_obs(), reward, done, info

    def reset(self, world=None):
        #world = self.world
        if world is None:
            if self.navigation:
                if self.avoid_direct_goal:
                    self.goal_length = random.choice(range(3, 4))
                elif not self.fixed_goal_length:
                    self.goal_length = random.choice(range(1, 3))
            #print('None')
            self.world, self.player_position, self.world_dic, self.goal_location, self.first_key_location, self.key_loc, self.lock_loc = world_gen(n=self.n, goal_length=self.goal_length,
                                                         num_distractor=self.num_distractor,
                                                         distractor_length=self.distractor_length,
                                                         seed=self.np_random_seed, own_key=self.own_key,
                                                         fixed_start=self.fixed_start, fixed_goal=self.fixed_goal)
            self.w = [self.world, self.player_position, self.world_dic, self.goal_location, self.first_key_location]
            #self.goal_location = self.first_key_location
            #np.save('world.npy',self.w)
        else:
            self.world, self.player_position, self.world_dic, self.goal_location, self.first_key_location = world
            #print(self.player_position)

        if self.navigation:
            self.goal_location = self.first_key_location
        self.num_env_steps = 0
        self.episode_reward = 0
        self.owned_key =  self.world[0,0]
        if self.generate_video:
            self.frames.append(self.world/255.0)
        return self._get_obs()

    def _get_obs(self):
        if self.obs_type == "img":
            temp = (self.world - grid_color[0])/255 * 2
            temp = np.concatenate([temp,self._xy_grid],axis=2)
            return temp
        elif self.obs_type == "dict":
            if self.no_key_indicator:
                grid = np.sum(self.world, axis=2, keepdims=True) == 0
            else:
                grid =  np.sum((self.world - grid_color[0]), axis=2, keepdims=True) != 0  #[1:self.n + 1, 1:self.n + 1, :]
            grid = grid.astype(int) * 0.2
            if self.indicate_distractor:
                for (location, distractor) in self.world_dic.items():
                    if not distractor:
                        grid[location[0], location[1]] = 0.3
            grid[self.player_position[0], self.player_position[1]][0] = 1
            self.binary_grid = grid.reshape((self.n,self.n))
            return {
                'observation': np.append(grid, self._xy_grid, axis=2).flatten(),
                'achieved_goal': self.player_position/(self.n),
                'desired_goal': self.goal_location/(self.n)
            }
        elif self.obs_type == 'relational':
            return self.get_predicate_rep()

    def get_predicate_rep(self):
        actions = {'n':[-1,0],
                   'ne':[-1,1],
                   'e':[0,1],
                   'se':[1,1],
                   's':[1,0],
                   'sw':[1,-1],
                   'w':[0,-1],
                   'nw':[-1,-1]}
        state =[]
        for (dir, action) in actions.items():
            temp = [self.player_position[0] + action[0], self.player_position[1] + action[1]]
            if self.world[temp[0], temp[1], 0] != 220:
                if temp in self.key_loc:  # Representation for key.
                    if np.array_equal(temp, self.key_loc[-1]):  # goal_location):
                        state += [f'neighbor({dir},gem)']
                    else:
                        key_id = COLOR_ID[tuple(np.array(self.world[temp[0], temp[1]],np.int))]
                        state += [f'neighbor({dir},key_{key_id})']
                elif temp in self.lock_loc:  # Representation for lock.
                    lock_id = COLOR_ID[tuple(np.array(self.world[temp[0], temp[1]],np.int))]
                    state += [f'neighbor({dir},lock_{lock_id})']
                elif np.sum(self.world[temp[0], temp[1], :]) == 0:  # Representation for wall.
                    state += [f'neighbor({dir},wall)']
            else:
                state += [f'neighbor({dir},cell)']

        agent_at ='cell'
        for key in self.key_loc:
            dir= self.get_direction(key)

            if np.array_equal(key, self.key_loc[-1]):  # goal_location):
                if dir == 'on':
                    agent_at = 'gem'
                else:
                    state += [f'direction(gem,{dir})']
            elif  self.world[key[0], key[1], 0] == 220:
                # owned key
                pass
            elif np.array_equal(self.world[key[0], key[1]],grid_color):
                pass
            elif np.array_equal(self.world[key[0], key[1]],agent_color) :
                # Recently owned key
                pass
                # if dir == 'on':
                #     key_id = COLOR_ID[tuple(np.array(self.world[0,0],np.int))]
                #     agent_at = f'key_{key_id}'
            else:
                key_id = COLOR_ID[tuple(np.array(self.world[key[0], key[1]],np.int))]
                state += [f'color(key_{key_id},{key_id})']
                if dir == 'on':
                    agent_at = f'key_{key_id}'
                else:
                    state += [f'direction(key_{key_id},{dir})']
        if tuple(np.array(self.world[0, 0], np.int)) in COLOR_ID.keys():
            key_id = COLOR_ID[tuple(np.array(self.world[0, 0], np.int))]
            state += [f'own(key_{key_id})']
            state += [f'color(key_{key_id},{key_id})']
        for lock in self.lock_loc:
            dir = self.get_direction(lock)
            # Lock opened otherwise
            if tuple(np.array(self.world[lock[0], lock[1]],np.int)) in COLOR_ID:
                lock_id = COLOR_ID[tuple(np.array(self.world[lock[0], lock[1]],np.int))]
                state += [f'direction(lock_{lock_id},{dir})']
                state += [f'color(lock_{lock_id},{lock_id})']
                if np.array_equal(self.world[lock[0], lock[1]-1],goal_color):
                    state += [f'inside(lock_{lock_id},gem)']
                else:
                    key_id = COLOR_ID[tuple(np.array(self.world[lock[0] , lock[1]-1], np.int))]
                    state += [f'inside(lock_{lock_id},key_{key_id})']
        if hasattr(self, 'opened_lock') and self.opened_lock is not None:
            # Lock is opened
            lock_id = COLOR_ID[tuple(np.array(self.opened_lock,np.int))]
            state += [f'open(lock_{lock_id})']
        state += [f'agent-at({agent_at})']
        return state

    def get_task_values(self,level=1):
        # if goal_location is None:
        #     goal_location = self.goal_location
        lidar = []
        lim = 8 + 9*(len(self.key_loc)+len(self.lock_loc))
        lidar = np.zeros(lim)
        count = 0
        
        if level >= 1:
            actions = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]
            # get neighbors
            for action in actions:
                
                temp = [self.player_position[0]+action[0],self.player_position[1]+action[1]]
                
                if self.world[temp[0], temp[1],0] != 220:
                    if temp in self.key_loc:  #Representation for key.
                        lidar[count] = 1
                    elif temp in self.lock_loc: #Representation for lock.
                        lidar[count] = 2
                    elif np.sum(self.world[temp[0],temp[1],:]) == 0:  #Representation for wall.
                        lidar[count] = 4
                    if np.array_equal(temp, self.key_loc[0]):#goal_location):
                        lidar[count] = 3
                
                count += 1

            # for each key and loc

            d_vector = self.get_direction_vector(goal_location)
            lidar[count:count+9] = d_vector

        
        state = ""
        for i in range(lim):
            state += str(int(lidar[i]))

        return np.asarray(state)

    def get_direction_vector(self, goal_location):
        d_vector = np.zeros(9)
        count= 0
        if goal_location[0] < self.player_position[0] and goal_location[1] < self.player_position[1]:
            d_vector[count] = 1
        count += 1
        if goal_location[0] < self.player_position[0] and goal_location[1] > self.player_position[1]:
            d_vector[count] = 1
        count += 1
        if goal_location[0] > self.player_position[0] and goal_location[1] < self.player_position[1]:
            d_vector[count] = 1
        count += 1
        if goal_location[0] > self.player_position[0] and goal_location[1] > self.player_position[1]:
            d_vector[count] = 1
        count += 1
        if goal_location[0] == self.player_position[0] and goal_location[1] < self.player_position[1]:
            d_vector[count] = 1
        count += 1
        if goal_location[0] == self.player_position[0] and goal_location[1] > self.player_position[1]:
            d_vector[count] = 1
        count += 1
        if goal_location[0] < self.player_position[0] and goal_location[1] == self.player_position[1]:
            d_vector[count] = 1
        count += 1
        if goal_location[0] > self.player_position[0] and goal_location[1] == self.player_position[1]:
            d_vector[count] = 1
        count += 1
        if np.array_equal([self.player_position[0], self.player_position[1]], goal_location):
            d_vector[count] = 1
        return d_vector,

    def get_direction(self, goal_location):
        if goal_location[0] < self.player_position[0] and goal_location[1] < self.player_position[1]:
            return 'nw'
        if goal_location[0] < self.player_position[0] and goal_location[1] > self.player_position[1]:
            return 'ne'
        if goal_location[0] > self.player_position[0] and goal_location[1] < self.player_position[1]:
            return 'sw'
        if goal_location[0] > self.player_position[0] and goal_location[1] > self.player_position[1]:
            return 'se'
        if goal_location[0] == self.player_position[0] and goal_location[1] < self.player_position[1]:
            return 'w'
        if goal_location[0] == self.player_position[0] and goal_location[1] > self.player_position[1]:
            return 'e'
        if goal_location[0] < self.player_position[0] and goal_location[1] == self.player_position[1]:
            return 'n'
        if goal_location[0] > self.player_position[0] and goal_location[1] == self.player_position[1]:
            return 's'
        if np.array_equal([self.player_position[0], self.player_position[1]], goal_location):
            return 'on'

    def get_RePReL_presentation(self, level=1, goal_location=None):
        state = self._get_obs()


    def get_hiprl_values(self,level=1,goal_location=None):
        if goal_location is None:
            goal_location = self.goal_location
        lidar = []
        lim = 17
        lidar = np.zeros(lim)
        count = 0
        
        if level >= 1:
            actions = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]
            
            for action in actions:
                
                temp = [self.player_position[0]+action[0],self.player_position[1]+action[1]]
                
                if self.world[temp[0], temp[1],0] != 220:
                    if temp in self.key_loc:  #Representation for key.
                        lidar[count] = 1
                    elif temp in self.lock_loc: #Representation for lock.
                        lidar[count] = 2
                    elif np.sum(self.world[temp[0],temp[1],:]) == 0:  #Representation for wall.
                        lidar[count] = 4
                    if np.array_equal(temp, self.key_loc[0]):#goal_location):
                        lidar[count] = 3
                
                count += 1

            d_vector = self.get_direction_vector(goal_location)
            lidar[count:count + 9] = d_vector


        
        state = ""
        for i in range(lim):
            state += str(int(lidar[i]))
        #print(state)
        #exit(1)
        return np.asarray(state)    

    def get_lidar(self,action):

        def valid_action(temp):
            if temp[0] < 0 or temp[1] < 0 or temp[0] >= self.world.shape[0] or temp[1] >= self.world.shape[0] :
                return False
            return True

        val = 0
        temp = self.player_position.copy()

        for i in range(3):
            temp = temp + action
            if valid_action(temp):
                if self.world[temp[0], temp[1],0] == 220:
                    val += 1
                elif self.world[temp[0],temp[1],0] == 255:
                    val = 4
                else:
                    break
            else:
                break
        return val/4.0

    def get_lidar_values(self,level):

        lidar = []
        lim = 25#57
        lidar = np.zeros(lim)
        count = 0
        
        if level >= 1:
            actions = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]
            #actions = [[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]]
            for action in actions:
                #lidar[count] = self.get_lidar(action)
                temp = [self.player_position[0]+action[0],self.player_position[1]+action[1]]
                if temp[0] < 0 or temp[0] >= 14  or temp[1] <0 or temp[1] >= 14:
                   lidar[count] = 1
                else:
                    if self.world[self.player_position[0]+action[0], self.player_position[1]+action[1],0] != 220:
                       if np.array_equal([self.player_position[0]+action[0], self.player_position[1]+action[1]], self.goal_location):
                          lidar[16+count] = 1
                       else:
                           lidar[count] = 1                
                count += 1
            '''for action in actions:
                #lidar[count] = self.get_lidar(action)
                temp = [self.player_position[0]+action[0],self.player_position[1]+action[1]]
                if temp[0] < 0 or temp[0] >= 14  or temp[1] <0 or temp[1] >= 14:
                   lidar[count] = 1
                elif self.world[temp[0], temp[1],0] != 220:
                    if np.array_equal([temp[0], temp[1]], self.goal_location):
                        lidar[count] = 1
                        #print(lidar)
                        #input('')
                    else:
                    lidar[count] = 1
                count += 1'''
            if self.goal_location[0] < self.player_position[0] and self.goal_location[1] < self.player_position[1]:
                lidar[count] = 1
            count += 1
            if self.goal_location[0] < self.player_position[0] and self.goal_location[1] > self.player_position[1]:
                lidar[count] = 1
            count += 1
            if self.goal_location[0] > self.player_position[0] and self.goal_location[1] < self.player_position[1]:
                lidar[count] = 1
            count += 1
            if self.goal_location[0] > self.player_position[0] and self.goal_location[1] > self.player_position[1]:
                lidar[count] = 1
            count += 1
            if self.goal_location[0] == self.player_position[0] and self.goal_location[1] < self.player_position[1]:
                lidar[count] = 1
            count += 1
            if self.goal_location[0] == self.player_position[0] and self.goal_location[1] > self.player_position[1]:
                lidar[count] = 1
            count += 1
            if self.goal_location[0] < self.player_position[0] and self.goal_location[1] == self.player_position[1]:
                lidar[count] = 1
            count += 1
            if self.goal_location[0] > self.player_position[0] and self.goal_location[1] == self.player_position[1]:
                lidar[count] = 1
            count += 1
        '''lidar[count] = (self.goal_location[0] - self.player_position[0])/14.0
        count += 1
        lidar[count] = (self.goal_location[1] - self.player_position[1])/14.0
        #count += 1'''
        if np.array_equal([self.player_position[0], self.player_position[1]], self.goal_location):
            #lidar = np.zeros(lim)
            #lidar[32:] = np.ones(24)
            lidar[-1] = 1.0
            #print(lidar)
            #exit(1)
            # input('lidar')
        return lidar

    def render(self, mode="human"):
        img = self.world.astype(np.uint8)
        if mode == "rgb_array":
            return img

        else:
            # from gym.envs.classic_control import rendering
            # if self.viewer is None:
            #     self.viewer = rendering.SimpleImageViewer()
            # self.viewer.imshow(img)
            # return self.viewer.isopen
            plt.imshow(img, vmin=0, vmax=255, interpolation='none')
            plt.pause(0.01)
            plt.clf()
    def get_action_lookup(self):
        return ACTION_LOOKUP

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.array_equal(achieved_goal, desired_goal):
            return self.reward_gem
        else:
           return 0

ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
}
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

if __name__ == "__main__":
    # import gym

    # execute only if run as a script
    env = gym.make('BoxWorld-task3-v1')
    # env.seed(10)

    # with open('/home/nathan/PycharmProjects/relational_RL_graphs/images/ex_world.pkl', 'rb') as file:

    obs = env.reset()
    print(obs)
    env.render()
#
#     env.reset()
#     env.render()
#     # with open('/home/nathan/PycharmProjects/relational_RL_graphs/images/ex_world.pkl', 'wb') as file:
#     #     pickle.dump([env.world, env.player_position, env.world_dic], file)
#
#
# # TO DO : impossible lvls ? (keys stacked right made inaccessible)
