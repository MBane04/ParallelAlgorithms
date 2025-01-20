// Name:
// nvcc HW1.cu -o temp
/*
 What to do:
 1. Understand every line of code and be able to explain it in class.
 2. Compile, run, and play around with the code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 1000 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; 
float Tolerance = 0.00000001;

// Function prototypes
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

//Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
} //allacotase N number of 32 bit floats (allocate N floats)

//Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}//goes through the array and sets vals for each element

//Adding vectors a and b then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
} // adds the two arrays and stores them in C

// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	int id;
	double sum = 0.0;
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(id = 0; id < n; id++)
	{ 
		sum += c[id]; //sum up all elements in C
	}
	
	if(abs(sum - 3.0*(m*(m+1))/2.0) < Tolerance) //No clue what this calculation is exactly but it checks to make sure it within a tolarnce bc floats are weird
	{
		return(1); //good
	}
	else 
	{
		return(0); //bad
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}//when did I start - when did I end = time elapsed

//Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
}// EZ

int main()
{
	timeval start, end;
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();

	// Starting the timer.	
	gettimeofday(&start, NULL);

	// Add the two vectors.
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);

	// Stopping the timer.
	gettimeofday(&end, NULL);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
	{
		printf("\n\n Something went wrong in the vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the CPU");
		printf("\n The time it took was %ld microseconds", elaspedTime(start, end));
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n");
	
	return(0);
}//Calls functions and prints information based on what happened

