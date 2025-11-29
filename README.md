# RNN-Jax
Implementation of various flavours of recurrent neural networks in Jax and Equinox

# Usage 

To start using the library, clone the repository, installing the dependencies in the `pyproject.toml`.

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
    - indRNNCell: independent RNN, where $W_h$ is _diagonal_ (Li et al., [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831), 2018)
     
- **Gated**: Gated RNNs, i.e. architectures with gates designed to adaptively forget past inputs
    - LongShortTermMemoryCell: LSTM cell (Hochreiter and Schmidhuber, Long Short-Term-Memory, 1997)
    - GatedRecurrentUnit: GRU cell (Cho et al. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), 2014)

- **Antisymmetric**: Architectures imposing an antisymmetric structure to the recurrence matrix $W_h$
    - AntiSymmetricRNNCell: Antisymmetric RNN, where the update is described by $h_{t+1} = h_t + \sigma((W_{h} -W_{h}^T) h_t + W_{x}x_{t+1} + b)$ (Chang et al. [AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks](https://arxiv.org/abs/1902.09689), 2019)
    - GatedAntiSymmetricRNNCell: gated version of the antisymmetric RNN (same reference as above)

- **Other Models**
    - ClockWorkRNNCell: Clockwork RNN, an architecture that processes inputs at different time scales (Koutník et al. [A Clockwork RNN](https://arxiv.org/abs/1402.3511), 2014)
    - LipschitzRNNCell: Lipschitz RNN, an architecture grounded in continuous time dymamical systems (Erichson et al. [Lipschitz Recurrent Neural Networks](https://arxiv.org/abs/2006.12070), 2020)
    - UnitaryEvolutionRNNCell: a flavor of Unitary RNN, that parametrizes the recurrence matrix to be unitary through Fourier transforms and Householder reflectors (Arjovsky et al. [Unitary Evolution Recurrent Neural Networks](https://arxiv.org/abs/1511.06464), 2016)
    - CoupledOscillatoryRNNCell: an RNN baased on oscillator dynamical systems ((Rusch and Mishra, [Coupled Oscillatory Recurrent Neural Network (coRNN)](https://arxiv.org/abs/2010.00951), 2023)), and its heterogenous variant (Ceni et al. [Random Oscillators Network for Time Series Processing](https://proceedings.mlr.press/v238/ceni24a/ceni24a.pdf), 2024)



# TODOs:

- Add more cells:
- More examples, benchmarks, use cases
-  (maybe)Advanced Models such as:

  - Linear recurrence models under the same API
    - S4
    - S5
    - LRU

  - (Controlled) Neural ODEs
  - Transformer / RNN hybrids e.g. a prototype of RWKV
