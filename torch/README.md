# Torch Benchmark

Provided by [Nicholas Leonard](https://github.com/nicholas-leonard).

Benchmark script uses [Element-Research/rnn](https://github.com/Element-Research/rnn).

Lua 5.2, Cuda 7.5, cudnnv5, Intel(R) Xeon(R) CPU E5-2630L v3 @ 1.80GHz, Titan X:


## Fast LSTM


### 30 x 32 x 128

```
$ th rnn.lua -network 'fastlstm' -batchsize 32 -hiddensize 128 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.024899005889893 seconds ---
Forward:
--- 32000 samples in 3.1959130764008 seconds (10012.885074946 samples/s, 99.871315062046 microsec/samples) ---  
Forward + Backward:
--- 32000 samples in 13.899139881134 seconds (2302.3021987976 samples/s, 434.34784561396 microsec/samples) ---
``` 

### 30 x 32 x 512

```
$ th rnn.lua -network 'fastlstm' -batchsize 32 -hiddensize 512 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.18875980377197 seconds ---
Forward:
--- 32000 samples in 6.4531669616699 seconds (4958.8108406272 samples/s, 201.66125148535 microsec/samples) ---  
Forward + Backward:
--- 32000 samples in 19.541891098022 seconds (1637.5083655011 samples/s, 610.68390309811 microsec/samples) ---
```  

### 30 x 32 x 1024

```
$ th rnn.lua -network 'fastlstm' -batchsize 32 -hiddensize 1024 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.69268393516541 seconds ---
Forward:
--- 32000 samples in 11.082577943802 seconds (2887.4174470646 samples/s, 346.33024781942 microsec/samples) ---  
Forward + Backward:
--- 32000 samples in 36.474525928497 seconds (877.32484315331 samples/s, 1139.8286595941 microsec/samples) ---
``` 

### 30 x 128 x 128


```
$ th rnn.lua -network 'fastlstm' -batchsize 128 -hiddensize 128 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.028716802597046 seconds ---
Forward:
--- 128000 samples in 3.2250719070435 seconds (39689.110895787 samples/s, 25.195827707648 microsec/samples) --- 
Forward + Backward:
--- 128000 samples in 14.058291912079 seconds (9104.9498912316 samples/s, 109.83036831021 microsec/samples) ---
``` 

### 30 x 128 x 512


```
$ th rnn.lua -network 'fastlstm' -batchsize 128 -hiddensize 512 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.19667100906372 seconds ---
Forward:
--- 128000 samples in 6.5813970565796 seconds (19448.779340937 samples/s, 51.417108625174 microsec/samples) --- 
Forward + Backward:
--- 128000 samples in 27.426359891891 seconds (4667.0445070921 samples/s, 214.26836587489 microsec/samples) --- 
``` 

### 30 x 128 x 1024

```
$ th rnn.lua -network 'fastlstm' -batchsize 128 -hiddensize 1024 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.74531388282776 seconds ---
Forward:
--- 128000 samples in 14.383507966995 seconds (8899.0845165442 samples/s, 112.37110942602 microsec/samples) --- 
Forward + Backward:
--- 128000 samples in 71.433391094208 seconds (1791.8792478834 samples/s, 558.07331949472 microsec/samples) --- 
```




## RNN


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


## LSTM


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
