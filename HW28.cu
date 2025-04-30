// Name: Mason
// CPU random walk. 
// nvcc HW28.cu -o temp

/*
 What to do:
 Create a function that returns a random number that is either -1 or 1.
 Start at 0 and call this function to move you left (-1) or right (1) one step each call.
 Do this 10000 times and print out your final position.
*/

// Include files
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h> //for random number generation


// Defines
#define NUM_WALKS 10'000 // Number of walks to perform
#define N 10'000 // Number of steps per walk
#define BLOCK_SIZE 256 // Number of threads per block


// Globals
dim3 BlockSize;
dim3 GridSize;
int *distance, *distanceGPU; // Array to store distances for each walk


// Function prototypes
int walk(int);
__global__ void walkGPU(); // Device function to generate random step
bool setup();


bool setup()
{
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;

	GridSize.x = (NUM_WALKS + BlockSize.x - 1) / BlockSize.x; // Number of blocks needed
	GridSize.y = 1;
	GridSize.z = 1;

	distance = (int*)malloc(NUM_WALKS * sizeof(int)); // Allocate memory for distance on host

	cudaMalloc(&distanceGPU, NUM_WALKS * sizeof(int)); // Allocate memory for distance on device

	//no need for copies until we are done with the kernel

	return true;
}


int walk(int steps)
{
	int position = 0; // Reset position for each walk

	for (int i = 0; i < steps; i++)
	{
		// Random number -1 or 1
		int step = (rand() % 2) * 2 - 1; // number%2 [0, 1] then multiply by 2 [0, 2] and subtract 1 [-1, 1]

		//make sure the random number works
		//printf("Step %d: %d\n", i + 1, step);

		// Update position
		position += step;
	}

	return position;
}

__global__ void walkGPU(int *distanceGPU, unsigned int seed)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int position;

	if (id < NUM_WALKS)
	{
		curandState state; // random state for each thread, the state is a data structure that CUDA uses for its RNG algorithms
		curand_init(seed, id, 0, &state); // Initialize the random state, args (seed, thread id, sequence number[allows multiple RNG streams
		position = 0; // Reset position for each walk

		for (int i = 0; i < N; i++)
		{
			// Random number -1 or 1
			int step = curand(&state) % 2 * 2 - 1; // number%2 [0, 1] then multiply by 2 [0, 2] and subtract 1 [-1, 1]

			// Update position
			position += step;
		}

		distanceGPU[id] = position; // Store the final position in global memory
	}
}


int main(int argc, char** argv)
{


	if(!setup()) return -1;

	//Do the GPU walks
	walkGPU<<<GridSize, BlockSize>>>(distanceGPU, time(NULL));

	// Wait for the kernel to finish
	cudaDeviceSynchronize(); 

	// Copy the results back to host
	cudaMemcpy(distance, distanceGPU, NUM_WALKS * sizeof(int), cudaMemcpyDeviceToHost);

	//print results
	printf("Final positions after %d walks:\n", NUM_WALKS);
	for (int i = 0; i < NUM_WALKS; i++)
	{
		printf("Walk %d Final Position: %d\n", i + 1, distance[i]);
	}


	//clean up
	free(distance); 
	cudaFree(distanceGPU);
	
	return 0;
}

