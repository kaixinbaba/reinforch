# Reinforch
[![License](https://img.shields.io/github/license/kaixinbaba/reinforch.svg)](https://github.com/kaixinbaba/reinforch/blob/master/LICENSE)


Reinforcement learning + pytorch \
一个使用pytorch实现的强化学习通用框架
## TODO Algorithm list
- [x] DQN [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [x] PG 
- [ ] AC
- [ ] DDPG
- [ ] A3C
- [ ] A2C
- [ ] PPO
- [ ] TRPO
- [ ] D4PG
- [ ] 待补充

## Installation(由于还未实现大部分算法，所以还没有上传至pypi)
方式1：pypi
```
pip install reinforch
```
方式2：从源码安装
```
git clone https://github.com/kaixinbaba/reinforch.git
cd reinforch
python setup.py install
```
## Requirements
```
cd reinforch
pip install -r requirements.txt
```
## Review demo
运行之前先根据上面的提示install并切换到examples目录

### DQN
```
python dqn_CartPole.py
# or
python dqn_MountainCar.py
```
### PG
```
python pg_CartPole.py
# or
python pg_MountainCar_Continuous.py
```
### AC
### DDPG
### A2C
### A3C
### PPO
### TRPO
### D4PG

## Authors
`reinforch` was written by `JunjieXun <452914639@qq.com>`.