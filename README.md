# Expert-Prior-RL

## Getting started
1. Install the [SMARTS](https://github.com/huawei-noah/SMARTS) simulator. Follow the instructions of the official repo.

2. Install tensorflow-probability
```shell
pip install tensorflow-probability==0.10.1
```
Install cpprb
```shell
pip install cpprb
``` 
   
3. Run IRL personalized or IRL general
```shell
python personal_IRL.py 
```
```shell
python general_IRL.py 
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
This repo is released under the MIT License. The NGSIM data processing code is borrowed from [NGSIM interface](https://github.com/Lemma1/NGSIM-interface). The NGSIM env is built on top of [highway env](https://github.com/eleurent/highway-env) which is released under the MIT license.
