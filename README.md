# rnn-benchmarks

All benchmarks are reported using a Nvidia GeForce GTX TITAN X GPU.

These benchmarks compare the running time of various recurrent neural networks on different deep-learning libraries.
The networks (RNN or LSTM) take as input a 3D Tensor (batch_size x seq_length x input_size) and output the last hidden state, compute a MSE loss and backpropagate the errors through the network. Input layer size is always set to 100, and sequence length to 30.

The code of the scripts I ran is available. The implementations of each model on the different libraries each use the fastest implementations I was able to find. If you are aware of faster implementations, please let me know. I've reported results on Theano and Torch so far, but I will try to include many more libraries in the future.

### RNN

#### Hidden layer size 100 - Bach size 20

| Library | Time (ms) | Forward only (ms) |
| ------------- | ------------- | ------------- |
| Theano  | 343.8 | 122.9 |
| Torch | 403.3 | 154.4 |


#### Hidden layer size 500 - Bach size 20

| Library | Time (ms) | Forward only (ms) |
| ------------- | ------------- | ------------- |
| Torch | 543.2 | 195.8 |
| Theano | 557.5 | 207.6 |



### LSTM

#### Hidden layer size 100 - Bach size 20

| Library | Time (ms) | Forward only (ms) |
| ------------- | ------------- | ------------- |
| Theano  | 795.4 | 274.4 |
| Torch | 3549.5 | 1630.8 |


#### Hidden layer size 500 - Bach size 20

| Library | Time (ms) | Forward only (ms) |
| ------------- | ------------- | ------------- |
| Theano  | 2396.0 | 770.9 |
| Torch | 4636.1 | 2923.9 |
