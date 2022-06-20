# Deep-Q-Network-and-Deep-Deterministic-Policy-Gradient (Deep Learning and Practice homework 6)

**The demo video can be seen in this [link](https://www.youtube.com/watch?v=JWXbZfipZzw)**

You can get some detailed introduction and experimental results in this [link](https://github.com/secondlevel/Deep-Q-Network-and-Deep-Deterministic-Policy-Gradient/blob/main/report.pdf).

This task is to implement two deep reinforcement algorithms by completing the following two tasks: 

**(1) solve LunarLander-v2 using deep Q-network (DQN)**  

**(2) solve LunarLanderContinuous-v2 using deep deterministic policy gradient (DDPG).**  

<br>

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/174265296-e973019d-f29a-4333-adbc-6937edfef1b2.png" title="dqn and ddpg" width="50%" height=40%" hspace="250"/>
</p>

## Hardware
Operating System: Windows 10  

CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  

GPU: NVIDIA GeForce GTX TITAN X  

## Requirement

In this work, you can use the following two option to build the environment.

### First option (recommend)

```bash=
$ conda env create -f environment.yml
```
### Second option

```bash=
$ conda create --name Summer python=3.8 -y
$ conda activate Summer
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install numpy
$ conda install matplotlib -y 
$ conda install pandas -y
$ pip install torchsummary
$ pip install gym
```

## System Architecture

You can see the detailed algorithm description in [DQN](https://arxiv.org/pdf/1312.5602.pdf) and [DDPG](https://arxiv.org/pdf/1509.02971.pdf).

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/174431417-e8a38de9-3cfe-405e-a416-e6e210bd0b6b.png" title="dqn and ddpg" width="70%" height=70%" hspace="150"/>
</p>

## Directory Tree

In this project, all you need to do is to git clone this respository. 

You don't need to download another file.

```bash
├─ dqn-example.py
├─ ddpg-example.py
├─ dqn.pth
├─ ddpg.pth
├─ environment.yml
├─ report.pdf
└─ README.md

```

## Training

In the training step, you can train two different model like DQN and DDPG.

### DQN

There have two step to train the DQN model.

The first step is config the DQN training parameters through the following argparse.

```python
## arguments ##
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-m', '--model', default='dqn.pth')
parser.add_argument('--logdir', default='log/dqn')
# train
parser.add_argument('--warmup', default=10000, type=int)
parser.add_argument('--episode', default=2000, type=int)
parser.add_argument('--capacity', default=10000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=.0005, type=float)
parser.add_argument('--eps_decay', default=.995, type=float)
parser.add_argument('--eps_min', default=.01, type=float)
parser.add_argument('--gamma', default=.99, type=float)
parser.add_argument('--freq', default=4, type=int)
parser.add_argument('--target_freq', default=1000, type=int)
# test
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--seed', default=20200519, type=int)
parser.add_argument('--test_epsilon', default=.001, type=float)
args = parser.parse_args()
```

The second step is run the command below.

```python
python dqn-example.py --test_only
```

### DDPG

There have two step to train the DDPG model.

The first step is config the DDPG training parameters through the following argparse.

```python
## arguments ##
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-m', '--model', default='ddpg.pth')
parser.add_argument('--logdir', default='log/ddpg')
# train
parser.add_argument('--warmup', default=50000, type=int)
parser.add_argument('--episode', default=2800, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--capacity', default=500000, type=int)
parser.add_argument('--lra', default=1e-3, type=float)
parser.add_argument('--lrc', default=1e-3, type=float)
parser.add_argument('--gamma', default=.99, type=float)
parser.add_argument('--tau', default=.005, type=float)
# test
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--seed', default=20200519, type=int)
args = parser.parse_args()
```

The second step is run the command below.

```python
python ddpg-example.py --test_only
```

## Testing

You can get some detailed introduction and experimental results in this [link](https://github.com/secondlevel/Deep-Q-Network-and-Deep-Deterministic-Policy-Gradient/blob/main/report.pdf).

In the training step, you also can evaluate two different model like DQN and DDPG.

### DQN

There have two step to evaluate the DQN model.

The first step is config the DQN testing parameters(same to training) through the following argparse. Especially with the evaluate model name dqn.pth.

```python
## arguments ##
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-m', '--model', default='dqn.pth')
parser.add_argument('--logdir', default='log/dqn')
# train
parser.add_argument('--warmup', default=10000, type=int)
parser.add_argument('--episode', default=2000, type=int)
parser.add_argument('--capacity', default=10000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=.0005, type=float)
parser.add_argument('--eps_decay', default=.995, type=float)
parser.add_argument('--eps_min', default=.01, type=float)
parser.add_argument('--gamma', default=.99, type=float)
parser.add_argument('--freq', default=4, type=int)
parser.add_argument('--target_freq', default=1000, type=int)
# test
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--seed', default=20200519, type=int)
parser.add_argument('--test_epsilon', default=.001, type=float)
args = parser.parse_args()
```

The second step is run the command below.

```python
python dqn-example.py
```

### DDPG

There have two step to evaluate the DDPG model.

The first step is config the DDPG testing parameters(same to training) through the following argparse. Especially with the evaluate model name ddpg.pth.

```python
## arguments ##
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-m', '--model', default='ddpg.pth')
parser.add_argument('--logdir', default='log/ddpg')
# train
parser.add_argument('--warmup', default=50000, type=int)
parser.add_argument('--episode', default=2800, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--capacity', default=500000, type=int)
parser.add_argument('--lra', default=1e-3, type=float)
parser.add_argument('--lrc', default=1e-3, type=float)
parser.add_argument('--gamma', default=.99, type=float)
parser.add_argument('--tau', default=.005, type=float)
# test
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--seed', default=20200519, type=int)
args = parser.parse_args()
```

The second step is run the command below.

```python
python ddpg-example.py
```

### Evaluate Result

Then you will get the best result like this, each of the values were the average reward in ten times.

|          | DQN | DDPG |
|:----------:|:------------:|:-----------------:|
| average reward | 269.35   | 285.51        |

## Reference

- https://www.gymlibrary.ml/
- https://arxiv.org/pdf/1312.5602.pdf
- https://arxiv.org/pdf/1509.02971.pdf
