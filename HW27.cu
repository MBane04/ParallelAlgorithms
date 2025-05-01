// Name: Mason
// CPU random walk. 
// nvcc HW27.cu -o temp

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


// Defines
#define N 10000 // Number of steps


// Globals


// Function prototypes
int walk(int);


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



int main(int argc, char** argv)
{
	// Seed the random number generator
	srand(time(NULL));

	// Print the final position
	printf("Final position after %d steps: %d\n", N, walk(N));

	//do it in a loop for fun
	// for (int i = 0; i < 100000; i++)
	// {
	// 	printf("Final position after %d steps: %d\n", N, walk(N));
	// }
	
	return 0;
}

