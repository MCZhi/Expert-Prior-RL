# Expert-Prior-RL

## Getting started
1. Install the [SMARTS](https://github.com/huawei-noah/SMARTS) simulator. Follow the instructions of the official repo.

2. Install tensorflow-probability, cpprb, and seaborn
```shell
pip install tensorflow-probability==0.10.1 cpprb seaborn==0.11.0
```
   
3. Run expert_recoding.py to demonstrate how to drive, and you need to specify the scenario to run.
```shell
python expert_recoding.py --scenario left_turn 
```

4. Run imitation_learning_uncertainty.py to learn the imitative expert policies.
```shell
python imitation_learning_uncertainty.py 
```

5. Run train.py to learn the imitative expert policies.
```shell
python imitation_learning_uncertainty.py 
```

6. Run test.py to learn the imitative expert policies.
```shell
python imitation_learning_uncertainty.py 
```

7. Run plot_results.py to learn the imitative expert policies.
```shell
python imitation_learning_uncertainty.py 
```

## Results
SAC

Policy penalty

## Reference
If you find this repo to be useful in your research, please consider citing our work
```
@article{huang2021driving,
  title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
  author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
```

## License
This repo is released under the MIT License. The RL algorithms are implemented based on [tf2rl](https://github.com/keiohta/tf2rl).
