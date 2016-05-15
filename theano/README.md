# Theano Benchmark Log

Cuda 7.5, cudnnv5, Intel(R) Xeon(R) CPU E5-2630L v3 @ 1.80GHz, Titan X.

## Fast LSTM



### 30 x 32 x 128
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'fastlstm' -l 128 -s 30 -b 32 
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 7.45822191238 seconds
Forward:
--- 32000 samples in 3.17055702209 seconds (10092.863739 samples/s, 0.0000991 s/sample) ---
Forward + Backward:
--- 32000 samples in 9.26702213287 seconds (3453.104950 samples/s, 0.0002896 s/sample) ---
``` 
### 30 x 32 x 512


```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'fastlstm' -l 512 -s 30 -b 32 
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 7.58512711525 seconds
Forward:
--- 32000 samples in 6.42896199226 seconds (4977.475374 samples/s, 0.0002009 s/sample) ---
Forward + Backward:
--- 32000 samples in 19.8206739426 seconds (1614.475880 samples/s, 0.0006194 s/sample) ---
```  

### 30 x 32 x 1024

```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'fastlstm' -l 1024 -s 30 -b 32 
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 9.6281080246 seconds
Forward:
--- 32000 samples in 10.3716170788 seconds (3085.343371 samples/s, 0.0003241 s/sample) ---
Forward + Backward:
--- 32000 samples in 32.4317178726 seconds (986.688406 samples/s, 0.0010135 s/sample) ---
``` 

### 30 x 128 x 128


```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'fastlstm' -l 128 -s 30 -b 128 
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 7.37970685959 seconds
Forward:
--- 128000 samples in 3.27810716629 seconds (39046.923577 samples/s, 0.0000256 s/sample) ---
Forward + Backward:
--- 128000 samples in 13.1759991646 seconds (9714.633281 samples/s, 0.0001029 s/sample) --
``` 

### 30 x 128 x 512

```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'fastlstm' -l 512 -s 30 -b 128 
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 7.49780893326 seconds
Forward:
--- 128000 samples in 8.03891611099 seconds (15922.544561 samples/s, 0.0000628 s/sample) ---
Forward + Backward:
--- 128000 samples in 32.7736029625 seconds (3905.582189 samples/s, 0.0002560 s/sample) ---

``` 

### 30 x 128 x 1024


```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'fastlstm' -l 1024 -s 30 -b 128 
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 7.44703698158 seconds
Forward:
--- 128000 samples in 20.5059478283 seconds (6242.091371 samples/s, 0.0001602 s/sample) ---
Forward + Backward:
--- 128000 samples in 74.6807880402 seconds (1713.961560 samples/s, 0.0005834 s/sample) ---
``` 

## RNN


### 30 x 32 x 128

```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'rnn' -l 128 -s 30 -b 32
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 4.309237957 seconds
Forward:
--- 32000 samples in 0.989920139313 seconds (32325.839963 samples/s, 0.0000309 s/sample) ---
Forward + Backward:
--- 32000 samples in 3.34791088104 seconds (9558.199467 samples/s, 0.0001046 s/sample) ---
``` 

### 30 x 32 x 512


```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'rnn' -l 512 -s 30 -b 32
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 4.36186599731 seconds
Forward:
--- 32000 samples in 3.27020597458 seconds (9785.316353 samples/s, 0.0001022 s/sample) ---
Forward + Backward:
--- 32000 samples in 8.80706095695 seconds (3633.448225 samples/s, 0.0002752 s/sample) ---
``` 

### 30 x 32 x 1024

```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'rnn' -l 1024 -s 30 -b 32
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 4.44132804871 seconds
Forward:
--- 32000 samples in 5.74468803406 seconds (5570.363405 samples/s, 0.0001795 s/sample) ---
Forward + Backward:
--- 32000 samples in 14.2010200024 seconds (2253.359265 samples/s, 0.0004438 s/sample) ---

``` 

### 30 x 128 x 128

```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'rnn' -l 128 -s 30 -b 128
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 4.48347306252 seconds
Forward:
--- 128000 samples in 1.74959516525 seconds (73159.781498 samples/s, 0.0000137 s/sample) ---
Forward + Backward:
--- 128000 samples in 5.81079101562 seconds (22027.982018 samples/s, 0.0000454 s/sample) ---

``` 

### 30 x 128 x 512

```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'rnn' -l 512 -s 30 -b 128
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 4.40771007538 seconds
Forward:
--- 128000 samples in 3.04104089737 seconds (42090.851231 samples/s, 0.0000238 s/sample) ---
Forward + Backward:
--- 128000 samples in 10.1157169342 seconds (12653.576690 samples/s, 0.0000790 s/sample) ---
``` 

### 30 x 128 x 1024

```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn.py -n 'rnn' -l 1024 -s 30 -b 128
Using gpu device 0: GeForce GTX TITAN X (CNMeM is disabled, cuDNN not available)
Compiling...
Setup : compile + forward/backward x 1
--- 4.38037991524 seconds
Forward:
--- 128000 samples in 6.43677687645 seconds (19885.728907 samples/s, 0.0000503 s/sample) ---
Forward + Backward:
--- 128000 samples in 18.919303894 seconds (6765.576615 samples/s, 0.0001478 s/sample) ---
``` 
