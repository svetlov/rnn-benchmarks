#TensorFlow benchmarks

Provided by Maarten Bosma.

I used the build-in rnn libary. ``basic_lstm`` is the Tensorflow equivalent of FastLSTM. 

These results are produced using TensorFlow 0.8, cuda 7.5, cudnnv5, turned off ondemand cpu governor [1], Intel(R) Xeon(R) CPU E5-2630L v3 @ 1.80GHz, Titan X:

To install TensorFlow from source:
   * https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#installing-from-sources
   * http://stackoverflow.com/questions/34239537/how-to-update-tensorflow-from-source
   
## Fast LSTM



### 30 x 32 x 128

```
$ python rnn.py -n basic_lstm -b 32 -l 128 -s 30
Setup : compile + forward/backward x 1
--- 3.91482686996 seconds
Forward:
--- 32000 samples in 8.53500294685 seconds (3749.266427 samples/s, 0.0002667 s/sample) ---
Forward + Backward:
--- 32000 samples in 26.2391839027 seconds (1219.550125 samples/s, 0.0008200 s/sample) ---
``` 

### 30 x 32 x 512

```
python rnn.py -n basic_lstm -b 32 -l 512 -s 30
Setup : compile + forward/backward x 1
--- 3.97159981728 seconds
Forward:
--- 32000 samples in 10.3965659142 seconds (3077.939414 samples/s, 0.0003249 s/sample) ---
Forward + Backward:
--- 32000 samples in 28.3808200359 seconds (1127.522036 samples/s, 0.0008869 s/sample) ---
``` 

### 30 x 32 x 1024


```
python rnn.py -n basic_lstm -b 32 -l 1024 -s 30
Setup : compile + forward/backward x 1
--- 3.81890392303 seconds
Forward:
--- 32000 samples in 18.0062820911 seconds (1777.157541 samples/s, 0.0005627 s/sample) ---
Forward + Backward:
--- 32000 samples in 42.533454895 seconds (752.348947 samples/s, 0.0013292 s/sample) ---
``` 


### 30 x 128 x 128

```
$ python rnn.py -n basic_lstm -b 128 -l 128 -s 30
Setup : compile + forward/backward x 1
--- 3.68258690834 seconds
Forward:
--- 128000 samples in 8.3175599575 seconds (15389.128621 samples/s, 0.0000650 s/sample) ---
Forward + Backward:
--- 128000 samples in 24.1425020695 seconds (5301.853123 samples/s, 0.0001886 s/sample) ---

``` 

### 30 x 128 x 512

```
python rnn.py -n basic_lstm -b 128 -l 512 -s 30
Setup : compile + forward/backward x 1
--- 3.72586607933 seconds
Forward:
--- 128000 samples in 14.6179850101 seconds (8756.336794 samples/s, 0.0001142 s/sample) ---
Forward + Backward:
--- 128000 samples in 32.6627261639 seconds (3918.840067 samples/s, 0.0002552 s/sample) ---

``` 

### 30 x 128 x 1024

```
python rnn.py -n basic_lstm -b 128 -l 1024 -s 30
Setup : compile + forward/backward x 1
--- 3.84206986427 seconds
Forward:
--- 128000 samples in 30.4814198017 seconds (4199.279457 samples/s, 0.0002381 s/sample) ---
Forward + Backward:
--- 128000 samples in 75.8014390469 seconds (1688.622295 samples/s, 0.0005922 s/sample) ---

``` 

## RNN

### 30 x 32 x 128

```
python rnn.py -n rnn -b 32 -l 128 -s 30
Setup : compile + forward/backward x 1
--- 1.6487121582 seconds
Forward:
--- 32000 samples in 3.56794595718 seconds (8968.745711 samples/s, 0.0001115 s/sample) ---
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 2292 get requests, put_count=2236 evicted_count=1000 eviction_rate=0.447227 and unsatisfied allocation rate=0.504363
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 100 to 110
Forward + Backward:
--- 32000 samples in 8.91037988663 seconds (3591.317139 samples/s, 0.0002784 s/sample) ---
``` 

### 30 x 32 x 512

```
python rnn.py -n rnn -b 32 -l 512 -s 30
Setup : compile + forward/backward x 1
--- 1.62368106842 seconds
Forward:
--- 32000 samples in 6.98823904991 seconds (4579.122118 samples/s, 0.0002184 s/sample) ---
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 2292 get requests, put_count=2236 evicted_count=1000 eviction_rate=0.447227 and unsatisfied allocation rate=0.504363
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 100 to 110
Forward + Backward:
--- 32000 samples in 11.1912858486 seconds (2859.367586 samples/s, 0.0003497 s/sample) ---
``` 

### 30 x 32 x 1024

```
python rnn.py -n rnn -b 32 -l 1024 -s 30
Setup : compile + forward/backward x 1
--- 1.72744393349 seconds
Forward:
--- 32000 samples in 7.73560094833 seconds (4136.718041 samples/s, 0.0002417 s/sample) ---
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 2292 get requests, put_count=2236 evicted_count=1000 eviction_rate=0.447227 and unsatisfied allocation rate=0.504363
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 100 to 110
Forward + Backward:
--- 32000 samples in 16.9597899914 seconds (1886.815816 samples/s, 0.0005300 s/sample) ---


``` 

### 30 x 128 x 128

```
python rnn.py -n rnn -b 128 -l 128 -s 30

``` 

### 30 x 128 x 512

```
python rnn.py -n rnn -b 128 -l 512 -s 30

``` 

### 30 x 128 x 1024

```
python rnn.py -n rnn -b 128 -l 1024 -s 30

``` 


### LSTM

### 30 x 32 x 128

```

``` 

### 30 x 32 x 512

```

``` 

### 30 x 32 x 1024

```

``` 

### 30 x 128 x 128

```

``` 

### 30 x 128 x 512

```

``` 

### 30 x 128 x 1024

```

``` 

## Results

These results are produced using TensorFlow 0.7.1, cuda 7.5, cudnnv4, turned off ondemand cpu governor [1], Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz, Titan X:

```
+ python rnn.py -n rnn -b 20 -i 100 -l 100 -s 30
Forward:
--- 100000 samples in 11.039894104 seconds (9058.057117 samples/s) ---
Forward + Backward:
--- 100000 samples in 25.300686121 seconds (3952.461833 samples/s) ---
+ python rnn.py -n rnn -b 20 -i 100 -l 500 -s 30
Forward:
--- 100000 samples in 19.6222681999 seconds (5096.250552 samples/s) ---
Forward + Backward:
--- 100000 samples in 43.0670762062 seconds (2321.959292 samples/s) ---
+ python rnn.py -n basic_lstm -b 20 -i 100 -l 100 -s 30
Forward:
--- 100000 samples in 25.3170599937 seconds (3949.905568 samples/s) ---
Forward + Backward:
--- 100000 samples in 77.6742260456 seconds (1287.428310 samples/s) ---
+ python rnn.py -n basic_lstm -b 20 -i 100 -l 500 -s 30
Forward:
--- 100000 samples in 36.4037480354 seconds (2746.969825 samples/s) ---
Forward + Backward:
--- 100000 samples in 104.032881021 seconds (961.234534 samples/s) ---
+ python rnn.py -n lstm -b 20 -i 100 -l 100 -s 30
Forward:
--- 100000 samples in 26.2394618988 seconds (3811.053590 samples/s) ---
Forward + Backward:
--- 100000 samples in 81.6460819244 seconds (1224.798498 samples/s) ---
+ python rnn.py -n lstm -b 20 -i 100 -l 500 -s 30
Forward:
--- 100000 samples in 36.3097510338 seconds (2754.080981 samples/s) ---
Forward + Backward:
--- 100000 samples in 104.501612902 seconds (956.923021 samples/s) ---
```

 [1] Turning on performance governor: `sudo bash -c 'for i in ls /sys/devices/system/cpu/*/cpufreq/scaling_governor; do echo 'performance' > $i; done;'`
