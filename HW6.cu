//Name: Mason Bane
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.

/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;
int N = WindowWidth*WindowHeight; //Number of pixels
float4 *Pixels_GPU, *Pixels_CPU; //This will be the array of pixels we will be using.

dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void escapeOrNotColorGPU(float4 *pixels, int N, int xMax, int xMin, int yMax, int yMin, int windowWidth, int windowHeight);
void setUpDevices();
void allocateMemory();
void initialize();
void display(void);
void cleanUp();

// This will be the layout of the parallel space we will be using.

void setUpDevices()
{
	//Current thoughts: 1024 rows by 1024 columns, each block is one row, each thread will be a pixel in that row.
	//pretty convenient huh?
	BlockSize.x = WindowWidth;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = WindowHeight; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
}

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.
	Pixels_CPU = (float4 *)malloc(N*sizeof(float4));
			

	
	// Device "GPU" Memory.
	cudaMalloc((void**)&Pixels_GPU, N*sizeof(float4));

}

// Loading values into the vectors that we will add.
void initialize()
{
	for(int i = 0; i < N; i++)
	{
		//initialize the pixels to black.
		Pixels_CPU[i].x = 0.0;
		Pixels_CPU[i].y = 0.0;
		Pixels_CPU[i].z = 0.0;
		Pixels_CPU[i].w = 1.0;
	}
}

__global__ void escapeOrNotColorGPU(float4 *pixels, int N, int xMax, int xMin, int yMax, int yMin, int windowWidth, int windowHeight)
{
	//declare variables
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float x, y, mag, tempX;
	int count = 0;
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;

	float stepSizeX = (xMax - xMin)/((float)windowWidth); //step through threads / columns
	float stepSizeY = (yMax - yMin)/((float)windowHeight); //step through blocks / rows

	//rows are blocks, columns are threads
	int row = blockIdx.x;
	int column = threadIdx.x;


	//calculate the x and y values for this pixel
	x = xMin + (stepSizeX * column); //min column in coords + how far over you are in the row (stepsize in coords * thread #)
	y = yMin + (stepSizeY * row); //min row in coords + how far down you are in the row

	//make sure that we don't go out of bounds
	if(id < N)
	{
		mag = sqrt(x*x + y*y);
		while (mag < maxMag && count < maxCount) 
		{	
			tempX = x; //We will be changing the x but we need its old value to find y.
			x = x*x - y*y + A;
			y = (2.0 * tempX * y) + B;
			mag = sqrt(x*x + y*y);
			count++;
		}
		if (count < maxCount) //if we escaped, we're black
		{
			pixels[id].x = 0.0;
			pixels[id].y = 0.0;
			pixels[id].z = 0.0;
			pixels[id].w = 1.0;
		}
		else //we didn't escape, we're red
		{
			pixels[id].x = 1.0;
			pixels[id].y = 0.0;
			pixels[id].z = 0.0;
			pixels[id].w = 1.0;
		}

	}



}

void display(void) 
{ 
	//call the kernel
	escapeOrNotColorGPU<<<GridSize, BlockSize>>>(Pixels_GPU, N, XMax, XMin, YMax, YMin, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);

	//copy the pixels from the GPU to the CPU
	cudaMemcpy(Pixels_CPU, Pixels_GPU, N*sizeof(float4), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGBA, GL_FLOAT, Pixels_CPU);
	glFlush();
}

void cleanUp()
{
	free(Pixels_CPU);
	cudaFree(Pixels_GPU);
	printf("Memory freed. Exiting...\n");

	exit(0);
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);

	//when we exit the glutMainLoop, we want to free the memory using the cleaUp function.
	atexit(cleanUp);

	//set up the devices
	setUpDevices();

	//allocate the memory
	allocateMemory();

	//initialize values on the CPU
	initialize();

	//Transfer the memory to the GPU
	cudaMemcpy(Pixels_GPU, Pixels_CPU, N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//start the glut loop
   	glutMainLoop();
	
}

