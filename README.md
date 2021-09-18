# Expert-Prior-RL

## Getting started
1. Install the [SMARTS](https://github.com/huawei-noah/SMARTS) simulator. Follow the instructions of the official repo.

2. Install tensorflow-probability, cpprb, and seaborn
```shell
pip install tensorflow-probability==0.10.1 cpprb seaborn==0.11.0
```
   
3. Start Envision and run expert_recoding.py to demonstrate how to drive, and you need to specify the scenario to run.
```shell
scl envision start
# Then to visit the Envision web app in your browser,
http://localhost:8081/
```
```shell
python expert_recoding.py left_turn 
```

4. Run imitation_learning_uncertainty.py to learn the imitative expert policies. You need to specify the file path to the recorded expert trajectories. 
```shell
python imitation_learning_uncertainty.py Expert_data/left_turn
```

5. Run train.py to train the RL agent. You need to specify the algorithm and scenario to run, and also the file path to the pre-trained imitative models if you are using the expert prior-guided algorithms.
```shell
python train.py sac left_turn --prior Expert_model/left_turn 
```

6. Run plot_results.py to visualize the training results. You need to specify the algorithm and scenario that you have trained with, as well as the metric you want to see.
```shell
python plot_results.py sac left_turn success
```

7. Run test.py to test the trained policy in the testing situations. You need to specify the algorithm and scenario, and the file path to your trained model.
```shell
# Start envision and then go to visit the Envision web app in your browser,
scl envision start
http://localhost:8081/
# Begin test
python test.py sac roundabout Train_results/roundabout/sac/Model/Model_328_3.3651.h5
```

## Results
SAC

Policy penalty

## Reference
If you find this repo to be useful in your research, please consider citing our work
```
@article{huang2021efficient,
  title={Efficient Deep Reinforcement Learning with Imitative Expert Priors for Autonomous Driving},
  author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
  journal={arXiv preprint arXiv:2103.10690},
  year={2021}
}
```

## License
This repo is released under the MIT License. The RL algorithms are implemented based on [tf2rl](https://github.com/keiohta/tf2rl).
