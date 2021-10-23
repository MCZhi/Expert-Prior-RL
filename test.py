import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import csv
import glob
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import argparse

from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentPolicy
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType

#### Environment specs ####
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 9))
states = np.zeros(shape=(80, 80, 9))

# observation space
def observation_adapter(env_obs):
    global states

    new_obs = env_obs.top_down_rgb[1] / 255.0
    states[:, :, 0:3] = states[:, :, 3:6]
    states[:, :, 3:6] = states[:, :, 6:9]
    states[:, :, 6:9] = new_obs

    if env_obs.events.collisions or env_obs.events.reached_goal:
        states = np.zeros(shape=(80, 80, 9))

    return np.array(states, dtype=np.float32)

# reward function
def reward_adapter(env_obs, env_reward):
    progress = env_obs.ego_vehicle_state.speed * 0.1
    goal = 1 if env_obs.events.reached_goal else 0
    crash = -1 if env_obs.events.collisions else 0

    return goal + crash

# action space
def action_adapter(model_action): 
    speed = model_action[0] # output (-1, 1)
    speed = (speed - (-1)) * (10 - 0) / (1 - (-1)) # scale to (0, 10)

    speed = np.clip(speed, 0, 10)
    model_action[1] = np.clip(model_action[1], -1, 1)

    # discretization
    if model_action[1] < -1/3:
        lane = -1
    elif model_action[1] > 1/3:
        lane = 1
    else:
        lane = 0

    return (speed, lane)

# information
def info_adapter(observation, reward, info):
    return info

#### Testing ####
parser = argparse.ArgumentParser()
parser.add_argument("algo", help="algorithm to test")
parser.add_argument("scenario", help="scenario to test")
parser.add_argument('model_path', help='path to the trained model')
args = parser.parse_args()

# create env
if args.scenario == 'left_turn':
    scenario_path = ['scenarios/left_turn_test']
    max_episode_steps = 400
elif args.scenario == 'roundabout':
    scenario_path = ['scenarios/roundabout_test']
    max_episode_steps = 600
else:
    raise NotImplementedError

# define agent interface
agent_interface = AgentInterface(
    max_episode_steps=max_episode_steps,
    waypoints=True,
    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
    rgb=RGB(80, 80, 32/80),
    action=ActionSpaceType.LaneWithContinuousSpeed,
)

# define agent specs
agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)

# define the env
AGENT_ID = "Agent-007"
env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=42)
env.observation_space = OBSERVATION_SPACE
env.action_space = ACTION_SPACE

# load trained model
if args.algo == 'value_penalty' or args.algo == 'policy_constraint':
    actor = load_model(args.model_path)
elif args.algo == 'sac' or args.algo == 'ppo' or args.algo == 'gail':
    actor = load_model(args.model_path)
elif args.algo == 'bc':
    actor = [load_model(model) for model in glob.glob(args.model_path+'/*.h5')]
else:
    raise Exception('Undefined Algorithm!')

# set up agent policy
class Policy(AgentPolicy):
    def __init__(self):
        self.actor = actor
        
    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)

        if args.algo == 'bc':
            means = []
            variances = []

            for model in self.actor:
                mean, std = model(obs)
                std += 0.1
                means.append(mean)
                variances.append(tf.square(std))

            mixture_mean = tf.reduce_mean(means, axis=0)
            mixture_var  = tf.reduce_mean(variances + tf.square(means), axis=0) - tf.square(mixture_mean)
            action_dist = tfp.distributions.MultivariateNormalDiag(loc=mixture_mean, scale_diag=tf.sqrt(mixture_var))
            action = action_dist.mean()

            return action.numpy()[0]  

        elif args.algo == 'value_penalty' or args.algo == 'policy_constraint':
            action_mean, action_std = self.actor(obs)
            action_mean = action_mean.numpy()[0]
            action_std = action_std.numpy()[0]

            return action_mean

        else:
            mean, log_std = self.actor(obs)
            dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))
            action_mean = tf.tanh(dist.mean())
            action_mean = action_mean[0].numpy()

            return action_mean

agent_spec.agent_builder = lambda: Policy()
agent = agent_spec.build_agent()

# create test log
if not os.path.exists(f'./test_results/{args.scenario}/{args.algo}'):
    os.makedirs(f'./test_results/{args.scenario}/{args.algo}')

with open(f'./test_results/{args.scenario}/{args.algo}/test_log.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['No.', 'success', 'return', 'length', 'speed', 'heading', 'yaw_rate', 'linear_acceleration_x', 
                    'linear_acceleration_y', 'linear_jerk_x', 'linear_jerk_y', 'angular_velocity_z'])

# begin test
for i in range(50):
    step = 0
    observation = env.reset()
    total_reward = 0.0
    done = False
    print("Progress: {}/{}".format(i+1, 50))
    speed = []
    heading = []
    yaw_rate = []
    linear_acceleration_x = []
    linear_acceleration_y = []
    linear_jerk_x = []
    linear_jerk_y = []
    angular_velocity_z = []

    while not done:
        step += 1
        agent_obs = observation
        agent_obs = agent_obs[AGENT_ID]
        agent_action = agent.act(agent_obs)
        observation, reward, done, info = env.step({AGENT_ID : agent_action})
        reward = reward[AGENT_ID]
        done = done[AGENT_ID]
        info = info[AGENT_ID]
        total_reward += reward
        speed.append(info['env_obs'].ego_vehicle_state.speed)
        heading.append(float(info['env_obs'].ego_vehicle_state.heading))
        yaw_rate.append(info['env_obs'].ego_vehicle_state.yaw_rate)
        linear_acceleration_x.append(info['env_obs'].ego_vehicle_state.linear_acceleration[0])
        linear_acceleration_y.append(info['env_obs'].ego_vehicle_state.linear_acceleration[1])
        linear_jerk_x.append(info['env_obs'].ego_vehicle_state.linear_jerk[0])
        linear_jerk_y.append(info['env_obs'].ego_vehicle_state.linear_jerk[1])
        angular_velocity_z.append(info['env_obs'].ego_vehicle_state.angular_velocity[2])

    print('Success:', info['env_obs'].events.reached_goal)
    with open(f'./test_results/{args.scenario}/{args.algo}/test_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i+1, info['env_obs'].events.reached_goal, total_reward, step, speed, heading, yaw_rate, linear_acceleration_x, 
                         linear_acceleration_y, linear_jerk_x, linear_jerk_y, angular_velocity_z])

env.close()

#### Print results ####
def read_results(args):
    file = f'./test_results/{args.scenario}/{args.algo}/test_log.csv'
    result = pd.read_csv(file)
    success = result['success']
    reward = result['return']		
    length = result['length']

    win = len(success[success == True])
    loss = len(success[success == False])

    success_rate = win / (win + loss) * 100
    
    return success_rate

def read_length(args):
    file = f'./test_results/{args.scenario}/{args.algo}/test_log.csv'
    result = pd.read_csv(file)
    success = result['success']
    reward = result['return']		
    length = result['length']

    time = length[success == True] / 10
    mean = time.mean()
    std = time.std()

    return mean, std

success_rate = read_results(args)
mean, std = read_length(args)

print(f'{args.scenario}/{args.algo}')
print('Success rate:', str(success_rate)+'%')
print('Time: M={:.2f}, SD={:.2f}'.format(mean, std))
