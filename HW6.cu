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

dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
float escapeOrNotColor(float, float);
void setUpDevices();
void allocateMemory();
void initialize();

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

	
	// Device "GPU" Memory.


}

// Loading values into the vectors that we will add.
void initialize()
{
	for(int i = 0; i < N; i++)
	{		

	}
}

float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}

void display(void) 
{ 
	float *pixels; 
	float x, y, stepSizeX, stepSizeY;
	int k;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	k=0;
	y = YMin;
	while(y < YMax) 
	{
		x = XMin;
		while(x < XMax) 
		{
			pixels[k] = escapeOrNotColor(x,y);	//Red on or off returned from color
			pixels[k+1] = 0.0; 	//Green off
			pixels[k+2] = 0.0;	//Blue off
			k=k+3;			//Skip to next pixel (3 float jump)
			x += stepSizeX;
		}
		y += stepSizeY;
	}

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

//process

//Each thread will be a pixel and needs 3 float values for RGB, so an array of float4s of size N is needed
//initalize the array with all black pixels (0,0,0,1)
//transfer the memory to the GPU
//manipulate the pixels in parallel
//transfer the memory back to the CPU
//display the pixels
//FREE MEM

