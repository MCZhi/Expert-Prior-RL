# Expert-Prior-RL

This repo is the implementation of **Imitative Expert Prior-Guided Reinforcement Learning**, from the following paper:

**Efficient Deep Reinforcement Learning with Imitative Expert Priors for Autonomous Driving**
<br> [Zhiyu Huang](https://mczhi.github.io/), [Jingda Wu](https://scholar.google.com/citations?user=icu-ZFAAAAAJ&hl=en), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[Paper]](https://ieeexplore.ieee.org/document/9694460)**&nbsp;**[[arXiv]](https://arxiv.org/abs/2103.10690)**&nbsp;**[[Project Website]](https://mczhi.github.io/Expert-Prior-RL/)**

## Getting started
1. Install the [SMARTS](https://github.com/huawei-noah/SMARTS) simulator (only working with version 0.4.x). Follow the instructions of the official repo.

2. Install tensorflow-probability, cpprb, and seaborn
```shell
pip install tensorflow-probability==0.10.1 cpprb seaborn==0.11.0
```
   
3. Start Envision and run expert_recoding.py to demonstrate how to drive (🠉 speed up, 🠋 slow down, 🠈 turn left, 🠊 turn right), and you need to specify the scenario to run. The available scenarios are left_turn and roundabout. You can optionally specify how many samples you would like to collect.
```shell
scl run --envision expert_recording.py left_turn --samples 40
```

4. Run imitation_learning_uncertainty.py to learn the imitative expert policies. You need to specify the file path to the recorded expert trajectories. You can optionally specify how many samples you would like to use to train the expert policies.
```shell
python imitation_learning_uncertainty.py expert_data/left_turn --samples 40
```

5. Run train.py to train the RL agent. You need to specify the algorithm and scenario to run, and also the file path to the pre-trained imitative models if you are using the expert prior-guided algorithms. The available algorithms are sac, value_penalty, policy_constraint, ppo, gail. If you are using GAIL, the prior should be the path to your demonstration trajectories.
```shell
python train.py value_penalty left_turn --prior expert_model/left_turn 
```

6. Run plot_train.py to visualize the training results. You need to specify the algorithm and scenario that you have trained with, as well as the metric you want to see (success or reward).
```shell
python plot_train.py value_penalty left_turn success
```

7. Run test.py to test the trained policy in the testing situations, along with Envision to visualize the testing process at the same time. You need to specify the algorithm and scenario, and the file path to your trained model. 
```shell
scl run --envision test.py value_penalty left_turn train_results/left_turn/value_penalty/Model/Model_X.h5
```

8. Run plot_test.py to plot the vehicle dynamics states. You need to specify the path to the test log file.
```shell
python plot_test.py test_results/left_turn/value_penalty/test_log.csv
```

## Reference
If you find this repo to be useful in your research, please consider citing our work
```
@article{huang2022efficient,
  title={Efficient Deep Reinforcement Learning with Imitative Expert Priors for Autonomous Driving},
  author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
```

## Acknowledgement
Much of this codebase is based on [tf2rl](https://github.com/keiohta/tf2rl).
