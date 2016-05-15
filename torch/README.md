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
$ th rnn.lua -network 'rnn' -batchsize 32 -hiddensize 128 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.045458793640137 seconds ---
Forward:
--- 32000 samples in 3.2980129718781 seconds (9702.8295844296 samples/s, 103.06271910667 microsec/samples) ---  
Forward + Backward:
--- 32000 samples in 8.305154800415 seconds (3853.0314888602 samples/s, 259.53590124846 microsec/samples) ---

``` 

### 30 x 32 x 512

```
$ th rnn.lua -network 'rnn' -batchsize 32 -hiddensize 512 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.053925037384033 seconds ---
Forward:
--- 32000 samples in 3.6663720607758 seconds (8727.9910213711 samples/s, 114.57390338182 microsec/samples) ---  
Forward + Backward:
--- 32000 samples in 9.2218749523163 seconds (3470.0127856443 samples/s, 288.18337619305 microsec/samples) --- 
``` 

### 30 x 32 x 1024

```
$ th rnn.lua -network 'rnn' -batchsize 32 -hiddensize 1024 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.08701491355896 seconds ---
Forward:
--- 32000 samples in 3.8027799129486 seconds (8414.9119629321 samples/s, 118.83665621281 microsec/samples) ---  
Forward + Backward:
--- 32000 samples in 12.205145835876 seconds (2621.8464374057 samples/s, 381.4105913043 microsec/samples) ---   
``` 

### 30 x 128 x 128

```
$ th rnn.lua -network 'rnn' -batchsize 128 -hiddensize 128 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.078629016876221 seconds ---
Forward:
--- 128000 samples in 4.1859209537506 seconds (30578.752442332 samples/s, 32.702445983887 microsec/samples) --- 
Forward + Backward:
--- 128000 samples in 8.6592428684235 seconds (14781.904624814 samples/s, 67.650280892849 microsec/samples) --- 
``` 

### 30 x 128 x 512

```
$ th rnn.lua -network 'rnn' -batchsize 128 -hiddensize 512 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.088251113891602 seconds ---
Forward:
--- 128000 samples in 4.383120059967 seconds (29203.014867419 samples/s, 34.243039786816 microsec/samples) ---  
Forward + Backward:
--- 128000 samples in 9.4049069881439 seconds (13609.928358313 samples/s, 73.475772514939 microsec/samples) --- 
``` 

### 30 x 128 x 1024

```
$ th rnn.lua -network 'rnn' -batchsize 128 -hiddensize 1024 -seqlen 30
Setup : compile + forward/backward x 1  
--- 0.12880301475525 seconds ---
Forward:
--- 128000 samples in 8.2753868103027 seconds (15467.566064044 samples/s, 64.651412889361 microsec/samples) --- 
Forward + Backward:
--- 128000 samples in 19.230028152466 seconds (6656.2610449056 samples/s, 150.23449249566 microsec/samples) --- 
``` 

