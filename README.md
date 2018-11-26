# Deep Deterministic Policy Gradient (DDPG) in a multi-agent particle environment

This python file implements the [**Deep Deterministic Policy Gradient**](https://arxiv.org/abs/1509.02971) algorithm in a [multiagent tag-game gym environment](https://github.com/openai/multiagent-particle-envs), slightly modified by [Rohan Sawhney](https://github.com/rohan-sawhney/multi-agent-rl). I did it as a final assignment for [Move37](https://www.theschool.ai/courses/move-37-course/), a [School of AI](https://www.theschool.ai) course.


## Getting started

Use the following command with the desired ``NUMBER_OF_EPISODES`` to lauch the file and start training :

```
python episode.py --n_episodes NUMBER_OF_EPISODES
```

Further 


### Prerequisites

You need to install both ``gym`` and ``PyTorch`` to run the file.

```
pip install gym
```
[PyTorch installation](https://pytorch.org/get-started/locally/)


## Result

Here we plot the length of episodes as training goes on :

![alt text](https://github.com/RomDeffayet/DDPG_multi_agent/blob/master/evoltion_plot.png)


## License

This project is under the GNU General Public License - see the [LICENSE](LICENSE) file for details

## Acknowledgements

Inspired from [Rohan Sawhney](https://github.com/rohan-sawhney)'s implementation.
