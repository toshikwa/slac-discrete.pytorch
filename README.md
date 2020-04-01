# SLAC for discrete action settings in PyTorch
A PyTorch implementation of Stochastic Latent Actor-Critic(SLAC)[[1]](#references) extended for discrete action settings. I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions. Also, any pull requests are welcomed.

## Requirements
You can install dependencies using `pip install -r requirements.txt`.

## Examples
You can train Slac-Discrete agent using hyperparameters [here](https://github.com/ku2482/slac-discrete.pytorch/blob/master/config/slac_discrete.yaml).

```
python main.py \
    --cuda \
    --env_id BattleZoneNoFrameskip-v4 \
    --config config/slac_discrete.yaml \
    --seed 0
```

## References
[[1]](https://arxiv.org/abs/1907.00953) Lee, Alex X., et al. "Stochastic latent actor-critic: Deep reinforcement learning with a latent variable model." arXiv preprint arXiv:1907.00953 (2019).
