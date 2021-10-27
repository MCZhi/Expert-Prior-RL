from tf2rl.algos.sac import SAC
from tf2rl.algos.expert_prior import ExpertPrior
from tf2rl.algos.ppo import PPO
from tf2rl.algos.gail import GAIL

from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.experiments.irl_trainer import IRLTrainer

import tensorflow as tf
import gym
import numpy as np
import glob
import random
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType

#### Load expert trajectories ####
def load_expert_trajectories(filepath):
    filenames = glob.glob(filepath)

    trajectories = []
    for filename in filenames:
        trajectories.append(np.load(filename))

    obses = []
    next_obses = []
    actions = []
    
    for trajectory in trajectories:
        obs = trajectory['obs']
        action = trajectory['act']

        for i in range(obs.shape[0]-1):
            obses.append(obs[i])
            next_obses.append(obs[i+1])
            act = action[i]
            act[0] += random.normalvariate(0, 0.1) # speed
            act[0] = np.clip(act[0], 0, 10)
            act[0] = 2.0 * ((act[0] - 0) / (10 - 0)) - 1.0 # normalize speed
            act[1] += random.normalvariate(0, 0.1) # lane change
            act[1] = np.clip(act[1], -1, 1)
            actions.append(act)
    
    expert_trajs = {'obses': np.array(obses, dtype=np.float32),
                    'next_obses': np.array(next_obses, dtype=np.float32),
                    'actions': np.array(actions, dtype=np.float32)}

    return expert_trajs

#### Environment specs ####
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 9))
AGENT_ID = 'Agent-007'
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

    if args.algo == 'value_penalty' or args.algo == 'policy_constraint':
        return goal + crash
    else:
        return 0.01 * progress + goal + crash

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

#### RL training ####
parser = Trainer.get_argument()
parser.add_argument("algo", help="algorithm to run")
parser.add_argument("scenario", help="scenario to run")
parser.add_argument("--prior", help="path to the expert prior models", default=None)
args = parser.parse_args()
args.max_steps = 10e4
args.save_summary_interval = 128
args.use_prioritized_rb = False
args.n_experiments = 10
args.logdir = f'./train_results/{args.scenario}/{args.algo}'

# define scenario
if args.scenario == 'left_turn':
    scenario_path = ['scenarios/left_turn']
    max_episode_steps = 400
elif args.scenario == 'roundabout':
    scenario_path = ['scenarios/roundabout']
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

if args.algo == 'gail':
    expert_trajs = load_expert_trajectories(args.prior+'/*.npz')

for i in range(args.n_experiments):
    print(f'Progress: {i+1}/{args.n_experiments}')

    # create env
    env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=i)
    env.observation_space = OBSERVATION_SPACE
    env.action_space = ACTION_SPACE
    env.agent_id = AGENT_ID

    if args.algo == 'sac':
        policy = SAC(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                     auto_alpha=True, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
        trainer = Trainer(policy, env, args)

    elif args.algo == 'value_penalty':
        policy = ExpertPrior(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                             prior=args.prior, auto_alpha=False, alpha=0.002, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
        trainer = Trainer(policy, env, args)

    elif args.algo == 'policy_constraint':
        policy = ExpertPrior(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                             prior=args.prior, auto_alpha=True, epsilon=0.8, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
        trainer = Trainer(policy, env, args)

    elif args.algo == 'ppo':
        policy = PPO(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0],
                     batch_size=32, clip_ratio=0.2, n_epoch=10, entropy_coef=0.01, horizon=512)
        trainer = OnPolicyTrainer(policy, env, args)

    elif args.algo == 'gail':
        policy = PPO(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0],
                     batch_size=32, clip_ratio=0.2, n_epoch=10, entropy_coef=0.01, horizon=512)
        irl = GAIL(state_shape=env.observation_space.shape, action_dim=env.action_space.high.size, batch_size=32, n_training=1)
        trainer = IRLTrainer(policy, env, args, irl, expert_trajs["obses"], expert_trajs["next_obses"], expert_trajs["actions"])

    else:
        raise NotImplementedError

    # begin training
    trainer()

    # close env
    env.close()
