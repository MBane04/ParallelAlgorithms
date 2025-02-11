// Name: Mason Bane
// Not simple Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <curand_kernel.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024; //attempted 2048, 2500
unsigned int WindowHeight = 1024; //attempted 2048

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float, float, float, float, float);

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

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, int windowWidth, int windowHeight) 
{
    float x, y, mag, tempX;
    int count, id;
    
    int maxCount = MAXITERATIONS;
    float maxMag = MAXMAG;


	#pragma unroll //idk if this will make it faster, but might as well try it.
	for (int i = threadIdx.x; i < windowWidth; i += blockDim.x)//<= so we dont do something twice
	{		
		// Calculate the offset into the pixel buffer, needs to be inside the loop
		id = 3 * (blockIdx.x * windowWidth + i); //3 * (what row we are on * size of row + offset in row)
		//just to test above line,  b = 0 threadIdx = 2
		//b=0: tidx = 0 3*(0*2048) + 0, 3*(1024), 3*(2048)[break before doing],3*(1), 3*(1025), 3*(2049)[break before doing]...
		//b=1: 3*(2048), 3*(3072), 3*(4096)
		//b=2: 3*(2*2048 + 0) = 3*(4096)
		
		//calculate offset, sime formula as last time, now with a 1024 pixel jump if needed
		x = xMin + dx * i;
		y = yMin + dy * blockIdx.x;
		
		count = 0;
		mag = sqrt(x * x + y * y);
		while (mag < maxMag && count < maxCount) 
		{
			// We will be changing the x but we need its old value to find y.    
			tempX = x; 
			x = x * x - y * y + A;
			y = (2.0 * tempX * y) + B;
			mag = sqrt(x * x + y * y);
			count++;
		}
		
		float brightness = 12.5;
		//random number between 0 and 1
		curandState state;
		curand_init(0, id, 0, &state);
		float random = curand_uniform(&state);
		if(count < maxCount) //It escaped
		{
			if(count <=2)
			{	if (threadIdx.x % 2 == 0)
				{
					pixels[id]     = 1.0;
					pixels[id + 1] = 0.0;
					pixels[id + 2] = 0.0;
				}
				else if (threadIdx.x % 2 == 1)
				{
					pixels[id]     = 0.0;
					pixels[id + 1] = 1.0;
					pixels[id + 2] = 0.0;
				}
				if(threadIdx.x % 2 == 0 && blockIdx.x % 2 == 0)
				{
					pixels[id]     = min(max(random + 0.5, (float)(count + random*255)/(float)maxCount), 1.0);;
					pixels[id + 1] = 0.0;
					pixels[id + 2] = min(max(random + 0.5, (float)(count + random*255)/(float)maxCount), 1.0);;
				}
				else if(threadIdx.x % 2 == 1 && blockIdx.x % 2 == 1)
				{
					pixels[id]     = min(max(random + 0.5, (float)(count + random*255)/(float)maxCount), 1.0);;
					pixels[id + 1] =  min(max(random + 0.5, (float)(count + random*255)/(float)maxCount), 1.0);;
					pixels[id + 2] = 0.0;
				}
				else
				{
					pixels[id]     = 0.0;
					pixels[id + 1] = 0.0;
					pixels[id + 2] = 1.0;
				}
			}
			else if (count == 3)
			{
				if(blockIdx.x % 3 == 0)
				{
					pixels[id]     = 1.0;
					pixels[id + 2] = 0.0;
					pixels[id + 2] = 0.0;
				}
				else if(blockIdx.x % 3 == 2)
				{
					pixels[id]     = 0.0;
					pixels[id + 1] = 1.0;
					pixels[id + 2] = 0.0;
				}
				else
				{
					pixels[id]     = 0.0;
					pixels[id + 1] = 0.0;
					pixels[id + 2] = 1.0;
				}
			}
			else if (count == 4)
			{
				if(threadIdx.x % 3 == 0)
				{
					pixels[id]     = 1.0;
					pixels[id + 2] = 1.0;
					pixels[id + 2] = 0.0;
				}
				else if(threadIdx.x % 3 == 2)
				{
					pixels[id]     = 1.0;
					pixels[id + 1] = 0.0;
					pixels[id + 2] = 1.0;
				}
				else
				{
					pixels[id]     = random;
					pixels[id + 1] = 0.7;
					pixels[id + 2] = random;
				}
			}
			else if (count == 5)
			{
				pixels[id]     = 94.0/255.0;
				pixels[id + 1] = 85.0/255.0;
				pixels[id + 2] = 225.0/255.0;
			}
			else if (count >= 6 && count < 8)
			{
				pixels[id]     = 250.0/255.0;
				pixels[id + 1] =215.0/255.0;
				pixels[id + 2] = 118.0/255.0;
			}
			else if (count >= 8 && count < 12)
			{
				pixels[id]     = 0.0;
				pixels[id + 1] = 1.0;
				pixels[id + 2] = 1.0;
			}
			else if (count >= 12 && count < 15)
			{
				pixels[id]     = 245.0/255.0;
				pixels[id + 1] = 173.0/255.0;
				pixels[id + 2] = 7.0/255.0;
			}
			else if (count >=15 && count < 18)
			{
				pixels[id]     = 76.0/255.0;
				pixels[id + 1] = 161.0/255.0;
				pixels[id + 2] = 12.0/255.0;
			}
			else if (count >=18 && count < 21)
			{
				pixels[id]     = 250.0/255.0;
				pixels[id + 1] = 152.0/255.0;
				pixels[id + 2] = 253.0/255.0;
			}
			else if (count >=21 && count < 24)
			{
				pixels[id]     = 165.0/255.0;
				pixels[id + 1] = 16.0/255.0;
				pixels[id + 2] = 210.0/255.0;
			}
			else
			{
				pixels[id]     = 141.0/255.0;
				pixels[id + 1] = 1.0;
				pixels[id + 2] = 92.0/255.0;
			}
			

		}
		else //It Stuck around
		{
			pixels[id]     = 1.0;
			pixels[id + 1] = 0.0;
			pixels[id + 2] = min(mag/maxMag * brightness, 1.0);
		}
	}
	
	
}

void display(void) 
{ 
	dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	//Threads in a block
	// if(WindowWidth > 1024)
	// {
	//  	printf("The window width is too large to run with this program\n");
	//  	printf("The window width width must be less than 1024.\n");
	//  	printf("Good Bye and have a nice day!\n");
	//  	exit(0);
	// }
	blockSize.x = 1024; //WindowWidth;
	blockSize.y = 1;
	blockSize.z = 1;
	
	//Blocks in a grid
	gridSize.x = WindowHeight;
	gridSize.y = 1;
	gridSize.z = 1;
	
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
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



