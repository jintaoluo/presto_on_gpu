> It is a branch of PRESTO. We are working to accelerate the program with GPU. The work is based on Scott Ransom's C version,https://github.com/scottransom/presto
About presto, please refer to http://www.cv.nrao.edu/~sransom/presto/

##2013-10-23##

####The 1st release####


1. You need a CUDA library to compile the code. 
2. Only accelsearch is GPUed. To run accelsearch on GPU, use the -cuda option. Or the program will run on CPU.
For example: accelsearch -numharm 16 -zmax 256 ur_data.dat -cuda 0, 0 means the 1st GPU in your machine.
