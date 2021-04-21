# DQN_Coyote_RoadRunner

This repository is a running example of **Double Deep Q-Network(DDQN)** algorithm on grid world of wile e. coyote & the road runner. [**The DDQN**](https://arxiv.org/pdf/1509.06461v3.pdf) is an improvement to DQN which mitigates the problem of overestimation leading to have better peformance. The reward assignment of the game is represented as follows:
|Object|Reward|
|:---------:|:---------:
|Roadrunner|+100|
|Dynamite|-100|
|Cactus|-10|
|Rock|-1|

![alt text](https://github.com/asalarp/DQN_Wile_RoadRunner/blob/main/Demo.png)


## How to run 

- Create a Python virtual environment:
```
virtualenv NAME_OF_ENV && source NAME_OF_ENV/bin/activate:
```
- install the dependencies by running the following command:
```
pip install -r requirments.txt
```
- Run the program: 
```
python DDQN.py
```


