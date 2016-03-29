# rnn-benchmarks

All benchmarks are reported using a Nvidia GeForce GTX TITAN X GPU.

These benchmarks compare the running time of various recurrent neural networks on different deep-learning libraries.
The networks (RNN or LSTM) take as input a 3D Tensor (batch_size x seq_length x input_size) and output the last hidden state, compute a MSE loss and backpropagate the errors through the network. Input layer size is always set to 100, and sequence length to 30.

The code of the scripts I ran is available. The implementations of each model on the different libraries each use the fastest implementations I was able to find. If you are aware of faster implementations, please let me know. I've reported results on Theano and Torch so far, but I will try to include many more libraries in the future.

The reported time is the average time needed to run a training example (and not a training batch), so the smaller the better.

### RNN

#### Hidden layer size 100 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano  | 343.8 | 122.9 |
| Torch | 403.3 | 154.4 |


#### Hidden layer size 500 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Torch | 543.2 | 195.8 |
| Theano | 557.5 | 207.6 |



### LSTM

#### Hidden layer size 100 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano (FastLSTM) | 713.1 | 241.7 |
| Theano (LSTM) | 795.4 | 274.4 |
| Torch (Element-Research FastLSTM) | 1991.5 | 430.4 |
| Torch (Element-Research LSTM) | 3549.5 | 1630.8 |


#### Hidden layer size 500 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano (FastLSTM) | 1151.9 | 386.6 |
| Torch (Element-Research FastLSTM) | 2283.2 | 499.4 |
| Theano (LSTM) | 2396.0 | 770.9 |
| Torch (Element-Research LSTM) | 4636.1 | 2923.9 |


FastLSTM implementations (for both Torch and Theano) do not use peephole connections between cell and gates, and compute the input, forget and output gates, as well as the hidden state, in the same operation.
