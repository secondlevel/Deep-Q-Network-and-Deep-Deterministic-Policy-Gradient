# Deep-Q-Network-and-Deep-Deterministic-Policy-Gradient (Deep Learning and Practice homework 6)

The demo video can be seen in this [link](https://www.youtube.com/watch?v=JWXbZfipZzw)

This task is to implement two deep reinforcement algorithms by completing the following two tasks: 

**(1) solve LunarLander-v2 using deep Q-network (DQN)**  

**(2) solve LunarLanderContinuous-v2 using deep deterministic policy gradient (DDPG).**  

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/174265296-e973019d-f29a-4333-adbc-6937edfef1b2.png" title="dqn and ddpg" width="40%" height=40%" hspace="250"/>
</p>

## Hardware
Operating System: Windows 10  

CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  

GPU: NVIDIA GeForce GTX TITAN X  

## Requirement

In this work, you can use the following two option to rebuild the environment.

- #### First option(recommend)

```bash=
$ conda env create -f environment.yml
```
- #### Second option(recommend)

```bash=
$ conda create --name Summer python=3.6 -y
$ conda activate Summer
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install numpy
$ conda install matplotlib -y 
$ conda install pandas -y
$ pip install torchsummary
$ pip install gym
```
