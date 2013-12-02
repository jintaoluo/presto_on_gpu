> It is a branch of PRESTO. We are working to accelerate the program with GPU. The work is based on Scott Ransom's C version,https://github.com/scottransom/presto
About presto, please refer to http://www.cv.nrao.edu/~sransom/presto/

#2013-12-02
####what's new####
1. Texture memory is used to reduce the time required for memory read on GPU.
2. cuFFT is used in a smarter way. cuFFT plans are stored independently outside functions that actually use them.

####test results####
1. CPU platform: Intel Xeon E5-1650, 3.20GHz, 12 Cores, 62GByte memory
2. GPU platform: Nvidia GeForce GTX780, with CUDA 5.5

#####data point = 83886080, dt per bin(s) = 0.00064#####

| -numharm        | -zmax           | CPU runtime(sec) | GPU runtime(sec) | acceleration |
| :-------------: |:-------------:| :-----:| :-----:| :-----:|
| 16 | 256  | 2075.00 | 93.3 | 22.24 |
| 8  | 256  | 1089.82 | 53.96 | 20.19 |
| 1  | 256  | 247.44  | 9.79  | 25.27 |

#2013-10-23#

####The 1st release####


1. You need a CUDA library to compile the code. 
2. Only accelsearch is GPUed. To run accelsearch on GPU, use the -cuda option. Or the program will run on CPU.
For example: accelsearch -numharm 16 -zmax 256 ur_data.dat -cuda 0, 0 means the 1st GPU in your machine.
