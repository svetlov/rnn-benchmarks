# rnn-benchmarks

All benchmarks are reported for a host with the following specifications :
   * NVIDIA Tesla M40 24GB
   * Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz
   * CUDA 8.0.44, cudnnv6

These benchmarks compare the running time of various recurrent neural networks on different deep-learning libraries.
The networks (LSTM) take as input a 3D Tensor `batch_size x seq_length x hidden_size`
and output the last hidden state, compute a MSE loss, backpropagate the errors through the network and do a simple update of the parameters (`params = params - lr*gradParams`).
The `hidden_size` specifies the size of the output and input layer of the networks.

The code of the scripts we ran are available.
If you are aware of faster implementations, please let me know.

The reported `Train` time is the average time needed to run (forward, backward, and update) a training example (and not a training batch), so the smaller the better.
We also report `Compile` time, which includes symbolic graph optimizations (Theano and TensorFlow compilation), as well as a forward and backward pass (to allocate memory).
While the compilation time isn't really a factor in production, it does increase debugging time, which is why we report it here.


## Theano

Cython==0.26
Theano==0.9.0
pygpu==0.6.8

cmd: `THEANO_FLAGS=mode=FAST_RUN,device=gpu5,floatX=float32 python theano/rnn.py ...`

## tenosflow

tensorflow==1.2.1 (builded from source without XLA)

cmd: `python rnn.py -n 'basic_lstm'`

This LSTM implementation used for these benchmarks does not use peephole connections between cell and gates.

## LSTM (sequence length is constant = 30)

### Batch Size 128

#### Hidden Size 512

| Library | Compile (s) | Train (ms) | Forward only (ms) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 6.09 | 0.4638 | 0.0902 |
| TensorFlow (basic lstm) | 1.420 | 0.2542 | 0.1247 |
| TensorFlow (fused lstm) | 1.313 | 0.2052 | 0.1044 |
| TensorFlow (cudnn lstm) | 1.562 | 0.1404 | 0.0790 |

#### Hidden Size 1024

| Library | Compile (s) | Train (ms) | Forward only (ms) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 6.37 | 0.7446 | 0.1995 |
| TensorFlow (basic lstm) | 1.601 | 0.6852 | 0.2831 |
| TensorFlow (fused lstm) | 1.846 | 0.5723 | 0.2482 |
| TensorFlow (cudnn lstm) | 1.671 | 0.2847 | 0.1554 |

## LSTM (sequence length is constant = 100)

### Batch Size 128

#### Hidden Size 512

| Library | Compile (s) | Train (ms) | Forward only (ms) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 6.38 | 1.3045 | 0.2436 |
| TensorFlow (basic lstm) | 1.564 | 0.8564 | 0.4034 |
| TensorFlow (fused lstm) | 1.391 | 0.6995 | 0.3688 |
| TensorFlow (cudnn lstm) | 1.680 | 0.4921 | 0.2521 |

#### Hidden Size 1024

| Library | Compile (s) | Train (ms) | Forward only (ms) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 6.52 | 2.2268 | 0.5915 |
| TensorFlow (basic lstm) | 1.827 | 2.3519 | 0.9294 |
| TensorFlow (fused lstm) | 1.816 | 1.9621 | 0.8860 |
| TensorFlow (cudnn lstm) | 2.343 | 0.9583 | 0.5656 |
