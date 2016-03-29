#TensorFlow benchmarks

provided by Maarten Bosma.

I used the build-in rnn libary. ``basic_lstm`` is the Tensorflow equivalent of FastLSTM. 

You could potentially speed up this code further by using feed queue or using ``rnn.dynamic_rnn``.

To install TensorFlow follow [these instructions](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation).

## Results

These results are also produced on a Titan X:

```
python rnn.py -n rnn -b 20 -i 100 -l 100 -s 20
--- 100000 samples in 12.8997249603 seconds (7752.102120 samples/s) —
--- 100000 samples in 38.2646420002 seconds (2613.378479 samples/s) —

python rnn.py -n rnn -b 20 -i 100 -l 500 -s 20
--- 100000 samples in 19.7984211445 seconds (5050.907324 samples/s) —
--- 100000 samples in 48.5306680202 seconds (2060.552639 samples/s) —

python rnn.py -n basic_lstm -b 20 -i 100 -l 500 -s 20
--- 100000 samples in 42.5869998932 seconds (2348.134295 samples/s) ---
--- 100000 samples in 124.468352795 seconds (803.417062 samples/s) —

python rnn.py -n basic_lstm -b 20 -i 100 -l 100 -s 20
--- 100000 samples in 41.8643369675 seconds (2388.667828 samples/s) —
--- 100000 samples in 121.31625104 seconds (824.291854 samples/s) —

python rnn.py -n lstm -b 20 -i 100 -l 100 -s 20
--- 100000 samples in 42.8046689034 seconds (2336.193647 samples/s) —
--- 100000 samples in 120.829017878 seconds (827.615758 samples/s) —

python rnn.py -n lstm -b 20 -i 100 -l 500 -s 20
--- 100000 samples in 48.9260210991 seconds (2043.902076 samples/s) —
--- 100000 samples in 133.602385998 seconds (748.489621 samples/s) ---
```
