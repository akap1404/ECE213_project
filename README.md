# Parallelized Nussinov RNA Folding Algorithm (CUDA)

This project implements both serial and GPU-parallelized versions of the Nussinov algorithm for predicting RNA secondary structure. The goal of the project is to accelerate the dynamic programming (DP) matrix computation by parallelizing it on a CUDA-enabled GPU.

The project parallelizes the DP matrix filling stage of the Nussinov algorithm to improve runtime performance compared to the serial implementation.

## Repository Structure

```

ECE213_project/
├── run_commands_parallel.sh  
├── run_commands_serial.sh  
├── sequences.txt  
└── src/  
    ├── nussinov_parallel.cu  
    └── nussinov_serial.cu

```  

## File Descriptions

### sequences.txt
This file contains the list of RNA sequences for which we want to compute secondary structures.

### run_commands_parallel.sh
This script runs the parallelized version of the Nussinov algorithm on the GPU.  
Before running the script, make sure to update the path to `sequences.txt` if necessary.

### run_commands_serial.sh
This script runs the serial implementation of the Nussinov algorithm.  
Before running the script, make sure to update the path to `sequences.txt` if necessary.

### src/nussinov_parallel.cu
This file contains the CUDA-based parallel implementation of the Nussinov algorithm. The implementation fills the dynamic programming matrix diagonally and assigns one CUDA block per element of each diagonal, allowing multiple computations to run in parallel on the GPU.

### src/nussinov_serial.cu
This file contains the serial implementation of the Nussinov algorithm. The DP matrix is computed using a single thread, and RNA sequences are processed sequentially without GPU acceleration.
