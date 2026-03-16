

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <sstream>





#define MIN_LOOP 4

__device__ bool pair_check(char a, char b) {
    return (a=='A' && b=='U') || (a=='U' && b=='A') ||
           (a=='C' && b=='G') || (a=='G' && b=='C');
}

__global__ void nussinov_kernel_serial(int *DP, const char *seq, int N, int k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < N - k; i++) {
            int j = i + k;
            int unpaired = DP[i*N + (j-1)];
            int paired = 0;

            for (int t = i; t < j - MIN_LOOP; t++) {
                if (!pair_check(seq[t], seq[j])) continue;

                int left = (t-1 >= i) ? DP[i*N + (t-1)] : 0;
                int right = DP[(t+1)*N + (j-1)];

                paired = max(paired, 1 + left + right);
            }

            DP[i*N + j] = max(unpaired, paired);
        }
    }
}


void traceback(int i, int j, int N, const std::string &seq,
               const std::vector<int> &DP, std::vector<std::pair<int,int>> &pairs) {
    if (i >= j) return;

    if (DP[i*N + j] == DP[i*N + (j-1)]) {
        traceback(i, j-1, N, seq, DP, pairs);
        return;
    }

    for (int t = i; t < j - MIN_LOOP; t++) {
        if (!((seq[t]=='A' && seq[j]=='U') ||
              (seq[t]=='U' && seq[j]=='A') ||
              (seq[t]=='C' && seq[j]=='G') ||
              (seq[t]=='G' && seq[j]=='C')))
            continue;

        int left = (t-1 >= i) ? DP[i*N + (t-1)] : 0;
        int right = DP[(t+1)*N + (j-1)];

        if (DP[i*N + j] == 1 + left + right) {
            pairs.push_back({t,j});
            traceback(i, t-1, N, seq, DP, pairs);
            traceback(t+1, j-1, N, seq, DP, pairs);
            return;
        }
    }
}


int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " SEQ_FILE.txt\n";
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return 1;
    }

    std::vector<std::string> sequences;
    std::string line;
    while (std::getline(infile, line)) {
        
        
        line.erase(
    std::remove_if(line.begin(), line.end(),
                   [](unsigned char c){ return std::isspace(c); }),
    line.end());
        if (!line.empty())
            sequences.push_back(line);
    }
    infile.close();

    std::cout << "Read " << sequences.size() << " sequences from file\n";
    

    
    auto total_start = std::chrono::high_resolution_clock::now();

    
    for (size_t s = 0; s < sequences.size(); s++) {
        std::string &seq = sequences[s];
        int N = seq.length();

        std::vector<int> DP(N*N, 0);

        
        int *d_DP;
        char *d_seq;
        cudaMalloc(&d_DP, N*N*sizeof(int));
        cudaMalloc(&d_seq, N*sizeof(char));

        cudaMemcpy(d_DP, DP.data(), N*N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seq, seq.c_str(), N*sizeof(char), cudaMemcpyHostToDevice);


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int k = MIN_LOOP; k < N; k++) {
            nussinov_kernel_serial<<<1,1>>>(d_DP, d_seq, N, k);
            cudaDeviceSynchronize();
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float kernel_time;
        cudaEventElapsedTime(&kernel_time, start, stop);

        cudaMemcpy(DP.data(), d_DP, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    
        auto tb_start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<int,int>> pairs;
        traceback(0, N-1, N, seq, DP, pairs);
        auto tb_end = std::chrono::high_resolution_clock::now();
        double traceback_time = std::chrono::duration<double, std::milli>(tb_end - tb_start).count();

     
        std::cout << "Sequence " << s << " optimal pairs: " << DP[N-1] << "\n";
        for (auto &p : pairs)
            std::cout << "(" << p.first << "," << p.second << ") "
                      << seq[p.first] << "-" << seq[p.second] << "\n";
        std::cout << "------------------------\n";

     
        cudaFree(d_DP);
        cudaFree(d_seq);

    
        std::cout << "Kernel (Serial GPU): " << kernel_time << " ms, "
                  << "Traceback (CPU): " << traceback_time << " ms\n\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    std::cout << "Total runtime for all sequences: " << total_time << " ms\n";

    return 0;
}