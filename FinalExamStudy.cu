/*code for doing vector edition on multiple GPUs using 3 different methods

    1.) Standard multi GPU approach
    2.) Using unified memory
    3.) Using page-locked memory


    A few key things to memorize:

    12 things:
    BlockDim.x, BlockDim.y, BlockDim.z
    GridDim.x, GridDim.y, GridDim.z
    blockIdx.x, blockIdx.y, blockIdx.z
    threadIdx.x, threadIdx.y, threadIdx.z

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    # of blocks = (N - 1) / blockDim.x + 1; //this is the same as ceil(N/blockDim.x) but faster
*/

//nvcc FinalExamStudy.cu -o temp

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 10'000 //Vector size
#define BLOCK_SIZE 256 //Number of threads per block

//Global variables
float *a, *b, *c; //Host vectors
float *a_GPU, *b_GPU, *c_GPU; //Device vectors
float *a_GPUm, *b_GPUm, *c_GPUm; //Unified memory vectors
float *a_GPUp, *b_GPUp, *c_GPUp; //Page-locked memory vectors
int methodTogle = 0; //Method to use (0: standard, 1: unified memory, 2: page-locked memory)
int N1, N2; //Number of elements for each GPU in multi-GPU setup
int deviceCount; //Number of devices available

dim3 BlockSize, GridSize0, GridSize1; //Grid and block sizes for kernel launch

//function prototypes
void setup();
void deviceSetup();
void cleanup();
void vectorAddCPU(float *a, float *b, float *c, int n); //for comparison to ensure correctness
__global__ void vectorAddGPU(float *a, float *b, float *c, int n, int offset); //GPU kernel for standard method


void deviceSetup()
{
    BlockSize.x = BLOCK_SIZE;
    BlockSize.y = 1;
    BlockSize.z = 1;

    if(deviceCount == 2)
    {
        GridSize0.x = (N1 - 1) / BlockSize.x + 1; // Number of blocks needed for first half
        GridSize0.y = 1;
        GridSize0.z = 1;
    
        GridSize1.x = (N2 - 1) / BlockSize.x + 1; // Number of blocks needed for second half
        GridSize1.y = 1;
        GridSize1.z = 1;
    }
    else
    {
        //we'll just use gridsize0 for the single GPU case
        GridSize0.x = (N - 1) / BlockSize.x + 1; // Number of blocks needed for single GPU
        GridSize0.y = 1;
        GridSize0.z = 1;
    }

}

void setup()
{
    //allocate host memory
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));

    // get device count for multi-GPU setup
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        cudaMalloc((void**)&a_GPU, N * sizeof(float));
        cudaMalloc((void**)&b_GPU, N * sizeof(float));
        cudaMalloc((void**)&c_GPU, N * sizeof(float));
    }

    //initialize host vectors
    for (int i = 0; i < N; i++)
    {
        a[i] = (float)i;
        b[i] = (float)i;
    }
    

    N1 = N / deviceCount;
    N2 = N - N1;

    //now that we set our halves, we can set the grid and block sizes to ensure we have enough for the largest half
    deviceSetup();

    //copy memory -standard method
    if(deviceCount == 2)
    {
        //Copy first half of vectors to first device
        cudaSetDevice(0);
        cudaMemcpy(a_GPU, a, N1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_GPU, b, N1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(c_GPU, 0, N1 * sizeof(float));

        //copy second alf to other device - offset a and b by N1
        cudaSetDevice(1);
        cudaMemcpy(a_GPU, a + N1, N2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_GPU, b + N1, N2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(c_GPU, 0, N2 * sizeof(float));
    }
    else
    {
        cudaMemcpy(a_GPU, a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_GPU, b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(c_GPU, 0, N * sizeof(float));
    }

    //Unified memory setup

    // Allocate unified memory for vectors
    cudaMallocManaged(&a_GPUm, N * sizeof(float));
    cudaMallocManaged(&b_GPUm, N * sizeof(float));
    cudaMallocManaged(&c_GPUm, N * sizeof(float));
    
    // Initialize data directly in unified memory
    for (int i = 0; i < N; i++) {
        a_GPUm[i] = (float)i;
        b_GPUm[i] = (float)i;
        c_GPUm[i] = 0.0f;
    }
    
    // Prefetch to devices if using multiple GPUs
    if (deviceCount == 2) {
        cudaSetDevice(0);
        cudaMemPrefetchAsync(a_GPUm, N1 * sizeof(float), 0);
        cudaMemPrefetchAsync(b_GPUm, N1 * sizeof(float), 0);
        cudaMemPrefetchAsync(c_GPUm, N1 * sizeof(float), 0);
        
        cudaSetDevice(1);
        cudaMemPrefetchAsync(a_GPUm + N1, N2 * sizeof(float), 1);
        cudaMemPrefetchAsync(b_GPUm + N1, N2 * sizeof(float), 1);
        cudaMemPrefetchAsync(c_GPUm + N1, N2 * sizeof(float), 1);
    }
    else
    {
        cudaMemcpy(a_GPUm, a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_GPUm, b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(c_GPUm, 0, N * sizeof(float));
    }



    //Page-locked memory setup

    // Allocate page-locked memory for vectors
    cudaHostAlloc((void**)&a_GPUp, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&b_GPUp, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&c_GPUp, N * sizeof(float), cudaHostAllocDefault);

    memcpy(a_GPUp, a, N * sizeof(float));
    memcpy(b_GPUp, b, N * sizeof(float));
    memset(c_GPUp, 0, N * sizeof(float));

    if(deviceCount == 2)
    {
        cudaSetDevice(0);
        cudaMemcpy(a_GPUp, a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_GPUp, b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(c_GPUp, 0, N * sizeof(float));
    
        cudaSetDevice(1);
        cudaMemcpy(a_GPUp, a + N1, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_GPUp, b + N1, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(c_GPUp, 0, N * sizeof(float));

    }
    else
    {
        cudaMemcpy(a_GPUp, a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_GPUp, b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(c_GPUp, 0, N * sizeof(float));
    }




}

void vectorAddCPU(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorAddGPU(float *a, float *b, float *c, int n, int offset)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x + offset;

    if (id < n)
    {
        c[id] = a[id] + b[id];
    }
}






int main()
{

    setup(); //calls deviceSetup() as well

    //run the CPU version for comparison
    vectorAddCPU(a, b, c, N);

    int sum = 0;
    //sum the results
    for(int i = 0; i < N; i++)
    {
        sum += c[i];
    }
    printf("CPU result: %d\n", sum);

    if(deviceCount == 2)
    {
        //standard multi-GPU method
        cudaSetDevice(0);
        vectorAddGPU<<<GridSize0, BlockSize>>>(a_GPU, b_GPU, c_GPU, N1, 0); //first half
        cudaDeviceSynchronize(); //wait for kernel to finish

        cudaSetDevice(1);
        vectorAddGPU<<<GridSize1, BlockSize>>>(a_GPU, b_GPU, c_GPU, N2, N1); //second half
        cudaDeviceSynchronize(); //wait for kernel to finish

        //copy result back to host
        cudaSetDevice(0);
        cudaMemcpy(c, c_GPU, N1 * sizeof(float), cudaMemcpyDeviceToHost); //copy first half
        cudaMemcpy(c + N1, c_GPU, N2 * sizeof(float), cudaMemcpyDeviceToHost); //copy second half
        cudaDeviceSynchronize(); //wait for copy to finish

        for(int i = 0; i < N; i++)
        {
            sum += c[i];
        }
        printf("Standard GPU result: %d\n", sum);
        sum = 0; //reset sum for next method

        //unified memory method
        cudaSetDevice(0);
        vectorAddGPU<<<GridSize0, BlockSize>>>(a_GPUm, b_GPUm, c_GPUm, N1, 0); //first half
        cudaDeviceSynchronize(); //wait for kernel to finish

        cudaSetDevice(1);
        vectorAddGPU<<<GridSize1, BlockSize>>>(a_GPUm, b_GPUm, c_GPUm, N2, N1); //second half
        cudaDeviceSynchronize(); //wait for kernel to finish

        //copy result back to host
        cudaSetDevice(0);
        cudaMemcpy(c, c_GPUm, N1 * sizeof(float), cudaMemcpyDeviceToHost); //copy first half
        cudaMemcpy(c + N1, c_GPUm, N2 * sizeof(float), cudaMemcpyDeviceToHost); //copy second half
        cudaDeviceSynchronize(); //wait for copy to finish

        for(int i = 0; i < N; i++)
        {
            sum += c[i];
        }
        printf("Unified GPU result: %d\n", sum);
        sum = 0; //reset sum for next method

        //page-locked memory method
        cudaSetDevice(0);
        vectorAddGPU<<<GridSize0, BlockSize>>>(a_GPUp, b_GPUp, c_GPUp, N1, 0); //first half
        cudaDeviceSynchronize(); //wait for kernel to finish

        cudaSetDevice(1);
        vectorAddGPU<<<GridSize1, BlockSize>>>(a_GPUp, b_GPUp, c_GPUp, N2, N1); //second half
        cudaDeviceSynchronize(); //wait for kernel to finish

        //copy result back to host
        cudaSetDevice(0);
        cudaMemcpy(c, c_GPUp, N1 * sizeof(float), cudaMemcpyDeviceToHost); //copy first half
        cudaMemcpy(c + N1, c_GPUp, N2 * sizeof(float), cudaMemcpyDeviceToHost); //copy second half
        cudaDeviceSynchronize(); //wait for copy to finish

        for(int i = 0; i < N; i++)
        {
            sum += c[i];
        }
        printf("Page-locked GPU result: %d\n", sum);
        sum = 0; //reset sum for next method


    }
    else
    {
        vectorAddGPU<<<GridSize0, BlockSize>>>(a_GPU, b_GPU, c_GPU, N, 0); //single GPU case
        cudaDeviceSynchronize(); //wait for kernel to finish

        cudaMemcpy(c, c_GPU, N * sizeof(float), cudaMemcpyDeviceToHost); //copy result back to host
        cudaDeviceSynchronize(); //wait for copy to finish
        
        int sum = 0;
        //sum the results
        for(int i = 0; i < N; i++)
        {
            sum += c[i];
        }
        printf("Standard GPU result: %d\n", sum);
        sum = 0; //reset sum for next method

        //unified memory method
        vectorAddGPU<<<GridSize0, BlockSize>>>(a_GPUm, b_GPUm, c_GPUm, N, 0); //single GPU case
        cudaDeviceSynchronize(); //wait for kernel to finish

        cudaMemcpy(c, c_GPUm, N * sizeof(float), cudaMemcpyDeviceToHost); //copy result back to host
        cudaDeviceSynchronize(); //wait for copy to finish

        for(int i = 0; i < N; i++)
        {
            sum += c[i];
        }
        printf("Unified GPU result: %d\n", sum);
        sum = 0; //reset sum for next method


        //page-locked memory method
        vectorAddGPU<<<GridSize0, BlockSize>>>(a_GPUp, b_GPUp, c_GPUp, N, 0); //single GPU case
        cudaDeviceSynchronize(); //wait for kernel to finish

        cudaMemcpy(c, c_GPUp, N * sizeof(float), cudaMemcpyDeviceToHost); //copy result back to host
        cudaDeviceSynchronize(); //wait for copy to finish

        for(int i = 0; i < N; i++)
        {
            sum += c[i];
        }
        printf("Page-locked GPU result: %d\n", sum);
        sum = 0; //reset sum for next method

    }



}