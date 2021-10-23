from pathlib import Path
import os
import numpy as np
from smarts.sstudio import gen_missions, gen_traffic
from smarts.sstudio.types import (
    Scenario,
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    SocialAgentActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
    Mission,
    EndlessMission,
)

scenario = os.path.dirname(os.path.realpath(__file__))

west_east_route = ("west-WE", "east-WE") 
east_west_route = ("east-EW", "west-EW")
turn_left_route = ("south-SN", "west-EW")
turn_right_route = ("south-SN", "east-WE")

# Traffic Flows
for seed in np.random.choice(1000, 20, replace=False):
    actors = {}

    for i in range(4):
        car = TrafficActor(
            name = f'car_type_{i+1}',
            speed=Distribution(mean=np.random.uniform(0.6, 1.0), sigma=0.1),
            min_gap=Distribution(mean=np.random.uniform(2, 4), sigma=0.1),
            imperfection=Distribution(mean=np.random.uniform(0.3, 0.7), sigma=0.1),
            lane_changing_model=LaneChangingModel(speed_gain=np.random.uniform(1.0, 2.0), impatience=np.random.uniform(0, 1.0), cooperative=np.random.uniform(0, 1.0)),
            junction_model=JunctionModel(ignore_foe_prob=np.random.uniform(0, 1.0), impatience=np.random.uniform(0, 1.0)),
        )

        actors[car] = 0.25

    west_east_flow = [Flow(route=Route(begin=("edge-west-WE", i, "random"), end=(f"edge-east-WE", i, "random")),
                           rate=100, actors=actors) for i in range(2)]
    east_west_flow = [Flow(route=Route(begin=("edge-east-EW", i, "random"), end=(f"edge-west-EW", i, "max")),
                           rate=100, actors=actors) for i in range(2)]
    turn_right_flow = [Flow(route=Route(begin=("edge-south-SN", 0, "random"), end=(f"edge-east-WE", 0, "random")),
                            rate=100, actors=actors)]

    traffic = Traffic(flows = west_east_flow + east_west_flow + turn_right_flow)
    
    gen_traffic(scenario, traffic, seed=seed, name=f'traffic_{seed}')

# Agent Missions
gen_missions(scenario=scenario, missions=[Mission(Route(begin=("edge-south-SN", 1, 40), end=("edge-west-EW", 0, 'max')), start_time=15)])

