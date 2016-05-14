# rnn-benchmarks

All benchmarks are reported for a host with the following specifications :
   * NVIDIA GeForce GTX TITAN X GPU 
   * Intel(R) Xeon(R) CPU E5-2630L v3 @ 1.80GHz
   * CUDA 7.5, cudnnv5

These benchmarks compare the running time of various recurrent neural networks on different deep-learning libraries.
The networks (RNN or LSTM) take as input a 3D Tensor `batch_size x seq_length x hidden_size`
and output the last hidden state, compute a MSE loss, backpropagate the errors through the network and do a simple update of the parameters (`params = params - lr*gradParams`). 
The sequence length is always set to `30`. 
The `hidden_size` specifies the size of the output and input layer of the networks.

The code of the scripts we ran are available. 
The implementations of each model on the different libraries each use 
the fastest implementations we were able to find. 
If you are aware of faster implementations, please let me know. 
I've reported results on Theano, Torch and TensorFlow so far, but we will try to include many more libraries in the future (including cudnn very soon).

The reported time is the average time needed to run a training example (and not a training batch), so the smaller the better.
We also report compilation time, which includes symbolic graph optimizations (Theano and TensorFlow), as well as a forward and backward pass (to allocate memory).
While the compilation time isn't really a factor in production, it does increase debugging time.

| Library | Compile (s) | Forward+Backward+Update (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano |  |  |  |
| Torch  |  |  |  |
| TensorFlow |  |  | |

## Fast LSTM

This LSTM implementation does not use peephole connections between cell and gates.

### Batch Size 32

#### Hidden Size 128

| Library | Compile (s) | Forward+Backward+Update (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.46 | 289.6 | 99.1 |
| Torch  | 0.03 | 434.4 | 99.9 |
| TensorFlow | 3.91 | 820.0 | 266.7 |


#### Hidden Size 512

| Library | Compile (s) | Forward+Backward+Update (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.59 | 619.4 | 200.9 |
| Torch  | 0.19 | 610.7 | 201.7 |
| TensorFlow | 3.97 | 886.9 | 324.9 |


#### Hidden Size 1024

| Library | Compile (s) | Forward+Backward+Update (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 9.62 | 1013.5 | 324.1 |
| Torch  | 0.69 | 1139.8 | 346.3 |
| TensorFlow | 3.81 | 1329.2 | 562.7 |


### Batch Size 128

#### Hidden Size 128

| Library | Compile (s) | Forward+Backward+Update (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.38 | 102.9 | 25.6 |
| Torch  | 0.03 | 109.8 | 25.2 |
| TensorFlow | 3.68 | 188.6 | 65.0 |


#### Hidden Size 512

| Library | Compile (s) | Forward+Backward+Update (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.50 | 256.0 | 62.8 |
| Torch  | 0.20 | 214.3 | 51.4 |
| TensorFlow | 3.73 | 255.2 | 114.2 |

#### Hidden Size 1024

| Library | Compile (s) | Forward+Backward+Update (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.45 | 583.4 | 160.2 |
| Torch  | 0.75 | 558.1 | 112.4 |
| TensorFlow | 3.84 | 592.2 | 238.1 |


## RNN

This section benchmarks a simple RNN implementation.

### Batch Size 32

#### Hidden Size 128


#### Hidden Size 512


#### Hidden Size 1024


### Batch Size 128

#### Hidden Size 128


#### Hidden Size 512



#### Hidden Size 1024



### LSTM

This LSTM implementation uses peephole connections between cell and gates.


