# RNN-JAX
Implementation of recurrent neural networks and deep state space in Jax and Equinox

# Usage 
**Important:** JAX is a dependency, but it is not explicitly listed as so. Follow [the official Installation guide](https://docs.jax.dev/en/latest/installation.html) to install it for your target architecture.
Then, RNN-JAX can be installed with pip
```
pip install rnn-jax
```


## Example usage
Defining and running a model can be done in few lines

```python
import jax
import equinox as eqx
import jax.random as jr
from rnn_jax.cells import ElmanRNNCell
from rnn_jax.layers import RNN


key = jr.key(0)  # PRNGkey
model_key, data_key = jr.split(key, 2) # split the keys
cell_key, out_key = jr.split(model_key, 2)

rnn = RNN(cell=ElmanRNNCell(idim=1, hdim=16, key=cell_key), odim=1, key=out_key)

x = jr.normal(key=data_key, shape=(100, 1))  # (seq_len, hdim)

outs = rnn(x)
```

For batched inputs, the model should be`vmap`ed over the batch as follows

```python
x = jr.normal(key=data_key, shape=(64, 100, 1)) #(batch, seq_len, hdim)
outs = eqx.filter_vmap(rnn)(x)
```

## Overview of the cell types  (other types will likely be added)

- **Vanilla**: Standard RNNs, following an equation which is roughly equivalent to $h_{t+1} = \sigma(W_{h} h_t + W_{x}x_{t+1} + b)$
    - ElmanRNNCell: standard RNN (Elman, Finding Structure in Time, 1990)
    - LeakyElmanCell: leaky integrator variant of an Elman RNN. This model rescales the update equation by the leakage term $\alpha\in(0,1]$, adding a leakage term $(1-\alpha)h_t$.
    - indRNNCell: independent RNN, where $W_h$ is _diagonal_ (Li et al., [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831), 2018)
     
- **Gated**: Gated RNNs, i.e. architectures with gates designed to adaptively forget past inputs
    - LongShortTermMemoryCell: LSTM cell (Hochreiter and Schmidhuber, Long Short-Term-Memory, 1997)
    - GatedRecurrentUnit: GRU cell (Cho et al. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), 2014)

- **Antisymmetric**: Architectures imposing an antisymmetric structure to the recurrence matrix $W_h$
    - AntiSymmetricRNNCell: Antisymmetric RNN, where the update is described by $h_{t+1} = h_t + \sigma((W_{h} -W_{h}^T) h_t + W_{x}x_{t+1} + b)$ (Chang et al. [AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks](https://arxiv.org/abs/1902.09689), 2019)
    - GatedAntiSymmetricRNNCell: gated version of the antisymmetric RNN (same reference as above)

- **Other Recurrent Models**
    - ClockWorkRNNCell: Clockwork RNN, an architecture that processes inputs at different time scales (Koutník et al. [A Clockwork RNN](https://arxiv.org/abs/1402.3511), 2014)
    - LipschitzRNNCell: Lipschitz RNN, an architecture grounded in continuous time dymamical systems (Erichson et al. [Lipschitz Recurrent Neural Networks](https://arxiv.org/abs/2006.12070), 2020)
    - UnitaryEvolutionRNNCell: a flavor of Unitary RNN, that parametrizes the recurrence matrix to be unitary through Fourier transforms and Householder reflectors (Arjovsky et al. [Unitary Evolution Recurrent Neural Networks](https://arxiv.org/abs/1511.06464), 2016)
    - CoupledOscillatoryRNNCell: an RNN baased on oscillator dynamical systems (Rusch and Mishra, [Coupled Oscillatory Recurrent Neural Network (coRNN)](https://arxiv.org/abs/2010.00951), 2023), and its heterogenous variant (Ceni et al. [Random Oscillators Network for Time Series Processing](https://proceedings.mlr.press/v238/ceni24a/ceni24a.pdf), 2024)

## State Space Models (SSM)
State space models are a class of recurrent network that use linear recurrence to perform forward and backward pass through time. In JAX this can be implemented efficiently using `jax.lax.associative_scan`.
- **S5**: simplified SSM. An SSM that uses a diagonal recurrence matrix. (Smith et al. [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933), 2022).
- **Linear Recurrent Unit**: Linear RNN with a diagonal complex-valued transiton matrix (Orvieto et al. [Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349), 2023).


## Third-Party Attributions

This project includes dataset files sourced from:
reservoirpy (https://github.com/reservoirpy/reservoirpy.git)
Copyright (c)  Xavier Hinaut (2018) <xavier.hinaut@inria.fr>


The dataset retains its original MIT License,
found in `rnn_jax/datasets/_reservoirpy/LICENSE.md`.

## To DOs (roughly in order of importance)
- [ ] code to integrate reservoirpy sets
- [ ] implement some out-of-the-box training methods
- [ ] modular layers (would require models with additional inputs e.g. $\sigma(W_{in} x + W_{h} h + W_{m} m)$) where $m$ is the message from other modules)
- [ ] message-passing nn with recurrent cells. Maybe the modular layer can be viewed as a MPNN.
