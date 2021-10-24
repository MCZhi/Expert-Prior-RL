import argparse
import os
import random
import gym
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent import AgentSpec, AgentPolicy
from smarts.env.hiway_env import HiWayEnv
from pynput.keyboard import Key, Listener

AGENT_ID = "Agent-007"
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 9))

class HumanKeyboardPolicy(AgentPolicy):
    def __init__(self):
        # initialize the keyboard listener
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

        # initialize desired speed and lane
        self.desired_speed = 2
        self.lane_change = 0

    def on_press(self, key):
        """To control, use the keys:
        Up: to speed up
        Down: to slow down
        Left: to change left
        Right: to change right
        """

        if key == Key.up:
            self.desired_speed += 2
        elif key == Key.down:
            self.desired_speed -= 2
        elif key == Key.right:
            self.lane_change = -1
        elif key == Key.left:
            self.lane_change = 1

    def act(self, obs):
        ego_state = obs.ego_vehicle_state
        wp_paths = obs.waypoint_paths
        ego_lane_index = ego_state.lane_index

        self.lane_index = ego_lane_index + self.lane_change
        self.lane_index = np.clip(self.lane_index, 0, len(wp_paths)-1)

        if self.lane_index == ego_lane_index:
            self.lane_change = 0

        self.desired_speed = np.clip(self.desired_speed, 0, 10)
        sleep(1/30)

        return (self.desired_speed, self.lane_change)


def main(args, scenario, num_episodes, seed):
    agent_spec = AgentSpec(
        interface=AgentInterface(
                    max_episode_steps=None,
                    waypoints=True,
                    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
                    rgb=RGB(80, 80, 32/80),
                    action=ActionSpaceType.LaneWithContinuousSpeed),
        policy_builder=HumanKeyboardPolicy,
    )

    env = HiWayEnv(scenarios=scenario, agent_specs={AGENT_ID: agent_spec}, headless=False, seed=seed)
    episode = 1

    while True:
        # data collection
        obs = []
        act = []
        
        # build agent
        agent = agent_spec.build_agent()

        # start env
        observation = env.reset()
        observation = observation[AGENT_ID]
        done = False
        states = np.zeros(shape=(80, 80, 9))

        while not done:
            agent_obs = observation
            states[:, :, 0:3] = states[:, :, 3:6]
            states[:, :, 3:6] = states[:, :, 6:9]
            states[:, :, 6:9] = agent_obs.top_down_rgb[1] / 255.0
            obs.append(states.astype(np.float32))

            agent_action = agent.act(agent_obs)
            observation, _, done, info = env.step({AGENT_ID: agent_action})
            observation = observation[AGENT_ID]
            done = done[AGENT_ID]
            info = info[AGENT_ID]
            speed = agent_action[0]
            lane = agent_action[1]
            act.append((speed, lane))
        
        if info['env_obs'].events.reached_goal:
            np.savez('expert_data/{}/demo_{}.npz'.format(args.scenario, episode), obs=np.array(obs, dtype=np.float32), act=np.array(act, dtype=np.float32))
            episode += 1

        if episode > num_episodes:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario')
    parser.add_argument('--samples', type=int, default=40)
    args = parser.parse_args()

    if args.scenario == 'left_turn':
        scenario = ["scenarios/left_turn"]
    elif args.scenario == 'roundabout':
        scenario = ["scenarios/roundabout"]
    else:
        raise Exception("Undefined Scenario!")
    
    if not os.path.exists('./expert_data/{}'.format(args.scenario)):
        os.makedirs('./expert_data/{}'.format(args.scenario))

    main(args, scenario=scenario, num_episodes=args.samples, seed=random.randint(0, 100))
