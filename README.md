presto_on_gpu
=============

It is a GPU accelerated branch of presto, open source pulsar search and analysis toolkit. http://www.cv.nrao.edu/~sransom/presto/

Based on Scott Ransom's C version,https://github.com/scottransom/presto. Some programs are moved to GPU.

2013-10-23

1, You need a CUDA library to compile the code. 

2, Only accelsearch is GPUed. To run accelsearch on GPU, use the -cuda option. Or the program will run on CPU.
For example: accelsearch -numharm 16 -zmax 256 ur_data.dat -cuda 0
0 means the 1st GPU in your machine.
