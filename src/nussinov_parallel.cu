#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <sstream>

#define MIN_LOOP 4
#define THREADS 256


__device__ bool pair_check(char a, char b) {
    return (a=='A' && b=='U') || (a=='U' && b=='A') ||
           (a=='C' && b=='G') || (a=='G' && b=='C');
}

__global__ void nussinov_kernel_element(
    int *DP,
    const char *seq,
    const int *seq_len,
    const int *seq_offset,
    const int *dp_offset,
    int k)
{
    int s = blockIdx.x;     // sequence index
    int elem = blockIdx.y;  // element index along diagonal

    int N = seq_len[s];
    if (k >= N) return;
    if (elem >= N - k) return;

    int i = elem;
    int j = i + k;

    int dp_base = dp_offset[s];
    int seq_base = seq_offset[s];

    int tid = threadIdx.x;

    int local_max = 0;

    
    for (int t = i + tid; t < j - MIN_LOOP; t += blockDim.x) {

        if (!pair_check(seq[seq_base + t], seq[seq_base + j]))
            continue;

        int left = (t-1 >= i) ? DP[dp_base + i*N + (t-1)] : 0;
        int right = DP[dp_base + (t+1)*N + (j-1)];

        local_max = max(local_max, 1 + left + right);
    }


    __shared__ int smax[THREADS];
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smax[tid] = max(smax[tid], smax[tid + stride]);
        __syncthreads();
    }


    if (tid == 0) {
        int unpaired = DP[dp_base + i*N + (j-1)];
        DP[dp_base + i*N + j] = max(unpaired, smax[0]);
    }
}


void traceback(int i, int j, int N, const std::string &seq,
               const std::vector<int> &DP, std::vector<std::pair<int,int>> &pairs)
{
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


int main(int argc, char* argv[])
{

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
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (!line.empty())
            sequences.push_back(line);
    }

    infile.close();

    std::cout << "Read " << sequences.size() << " sequences from file\n";

    int num_seq = sequences.size();


    std::vector<int> seq_len(num_seq);
    std::vector<int> seq_offset(num_seq);
    std::vector<int> dp_offset(num_seq);

    int total_seq_len = 0;
    int total_DP_size = 0;

    for (int s = 0; s < num_seq; s++) {

        seq_len[s] = sequences[s].length();

        seq_offset[s] = total_seq_len;
        dp_offset[s] = total_DP_size;

        total_seq_len += seq_len[s];
        total_DP_size += seq_len[s] * seq_len[s];
    }

    std::vector<char> h_seq(total_seq_len);

    for (int s = 0; s < num_seq; s++)
        std::copy(sequences[s].begin(), sequences[s].end(),
                  h_seq.begin() + seq_offset[s]);

    std::vector<int> h_DP(total_DP_size, 0);

  
    char *d_seq;
    int *d_DP, *d_seq_len, *d_seq_offset, *d_dp_offset;

    cudaMalloc(&d_seq, total_seq_len*sizeof(char));
    cudaMalloc(&d_DP, total_DP_size*sizeof(int));

    cudaMalloc(&d_seq_len, num_seq*sizeof(int));
    cudaMalloc(&d_seq_offset, num_seq*sizeof(int));
    cudaMalloc(&d_dp_offset, num_seq*sizeof(int));


    cudaMemcpy(d_seq, h_seq.data(), total_seq_len*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_DP, h_DP.data(), total_DP_size*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_seq_len, seq_len.data(), num_seq*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq_offset, seq_offset.data(), num_seq*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dp_offset, dp_offset.data(), num_seq*sizeof(int), cudaMemcpyHostToDevice);


    auto total_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    cudaEventRecord(kernel_start);

    int max_N = *std::max_element(seq_len.begin(), seq_len.end());


    for (int k = MIN_LOOP; k < max_N; k++) {

        dim3 grid(num_seq, max_N);
        dim3 block(THREADS);

        nussinov_kernel_element<<<grid, block>>>(
            d_DP,
            d_seq,
            d_seq_len,
            d_seq_offset,
            d_dp_offset,
            k
        );

        cudaDeviceSynchronize();
    }

    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);

    cudaMemcpy(h_DP.data(), d_DP, total_DP_size*sizeof(int), cudaMemcpyDeviceToHost);

    auto tb_start = std::chrono::high_resolution_clock::now();


    for (int s = 0; s < num_seq; s++) {

        int N = seq_len[s];

        std::vector<std::pair<int,int>> pairs;

        int offset = dp_offset[s];

        std::vector<int> DP_matrix(
            h_DP.begin()+offset,
            h_DP.begin()+offset+N*N
        );

        traceback(0, N-1, N, sequences[s], DP_matrix, pairs);

        std::cout << "Sequence " << s
                  << " optimal pairs: "
                  << DP_matrix[N-1]
                  << "\n";

        for (auto &p : pairs)
            std::cout << "(" << p.first << "," << p.second << ") "
                      << sequences[s][p.first] << "-"
                      << sequences[s][p.second] << "\n";

        std::cout << "------------------------\n";
    }

    auto tb_end = std::chrono::high_resolution_clock::now();

   
    auto total_end = std::chrono::high_resolution_clock::now();

    double total_time =
        std::chrono::duration<double>(total_end-total_start).count();

    double traceback_time =
        std::chrono::duration<double>(tb_end-tb_start).count();

    std::cout << "\nTiming (seconds):\n";
    std::cout << "Kernel: " << kernel_ms/1000.0 << "\n";
    std::cout << "Traceback: " << traceback_time << "\n";
    std::cout << "Total: " << total_time << "\n";

 
    cudaFree(d_seq);
    cudaFree(d_DP);
    cudaFree(d_seq_len);
    cudaFree(d_seq_offset);
    cudaFree(d_dp_offset);

    return 0;
}