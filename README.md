This project proposes to parallelize the DP matrix filling for the Nussinov algorithm. The file structure of the repository is as follows:

ECE213_project/
   run_commands.sh 
   sequences.txt
   src/
      nussinov_parallel.cu
      nussinov_serial.cu

Details regarding the files are given below:

sequences.txt: This is the file containing the list of sequences we wish to find the secondary structures for

run_commands_parallel.sh: This is the file that is invoked to run the parallelized version of the Nussinov algorithm on the GPU. The path to the sequences.txt needs to be updated in this file

run_commands_serial.sh: This is the file that is invoked to run the parallelized version of the Nussinov algorithm on the GPU. The path to the sequences.txt needs to be updated in this file

nussinov_parallel.cu: This file contains the parallelized implementation of the Nussinov algorithm which allots a block per element of the diagonals.

nussinov_serial.cu: This is the serial implementation of the Nussinov algorithm which utilizes a single thread for DP matrix computation and processes sequences serially



