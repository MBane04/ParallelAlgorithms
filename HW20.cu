// Name: Mason Bane
// Optimizing nBody GPU code. 
// nvcc HW20.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean n-body code that runs on the GPU for any number of bodies (within reason). Take this code and make it 
 run as fast as possible using any tricks you know or can find. Try to keep the same general format so we can time it and 
 compare it with others' code. This will be a competition. To focus more on new ideas rather than just using a bunch of if 
 statements to avoid going out of bounds, N will be a power of 2 and 256 < N < 262,144. CHeck in code to make sure this is true.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate.
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).

//1024
//current best: 1'740'077us 
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 1024
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float3 *PGPU, *VGPU, *FGPU;
__constant__ float d_constants[4]; // G, H, Damp, dt
cudaStream_t computeStream, visualStream; //2 streams: one for computation and one for visual updates

float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void nBodyGPU(float3 *, float3 *, float3 *, float, int);
void nBody();
int main(int, char**);

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

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		printf("\n The simulation is running.\n");
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
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
}

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaMemcpyAsync(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost, visualStream);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    		nBody();
    		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
    	gettimeofday(&end, NULL);
    	drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
	float randomAngle1, randomAngle2, randomRadius;
	float d, dx, dy, dz;
	int test;

	float h_constants[4] = {G, H, Damp, DT}; //use constants in the kernel

    	
    BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	int paddedN = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

	GridSize.x = paddedN/BLOCK_SIZE; //Makes enough blocks to deal with the whole vector.
	GridSize.y = 1;
	GridSize.z = 1;
	
	Damp = 0.5;
	
	// Allocate host memory
    cudaMallocHost(&P, N*sizeof(float3)); // Allocate host memory, makes a faster mem copy
    cudaMallocHost(&V, N*sizeof(float3)); // cuda needs to make sure the memory is page locked, so it can be accessed by the GPU
    cudaMallocHost(&F, N*sizeof(float3)); // cudaMallocHost allocates page-locked memory on the host for fast access by the device
    M = (float*)malloc(N*sizeof(float));
    	

	// Allocate device memory with padding
	cudaMalloc(&PGPU, paddedN*sizeof(float3));
	cudaMalloc(&VGPU, paddedN*sizeof(float3));
	cudaMalloc(&FGPU, paddedN*sizeof(float3));

	// Initialize everything to zero
	cudaMemset(PGPU, 0, paddedN*sizeof(float3));
	cudaMemset(VGPU, 0, paddedN*sizeof(float3));
	cudaMemset(FGPU, 0, paddedN*sizeof(float3));

	//create stream for asynchronous memory copy
	cudaStreamCreate(&computeStream);
	cudaStreamCreate(&visualStream);

    	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}

 	// Copy data to GPU after it's fully initialized
    cudaMemcpyToSymbol(d_constants, h_constants, 4*sizeof(float));
    cudaMemcpyAsync(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice, computeStream);
    cudaMemcpyAsync(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice, computeStream);
    cudaMemcpyAsync(FGPU, F, N*sizeof(float3), cudaMemcpyHostToDevice, computeStream);
	
	printf("\n To start timing type s.\n");
}

__global__ void nBodyGPU(float3 *p, float3 *v, float3 *f, float t, int n)
{
    // Shared memory for positions
    __shared__ float3 sh_Pos[BLOCK_SIZE];
    
    // Constants loaded to registers for faster access
    float g = d_constants[0];
    float h = d_constants[1];
    float damp = d_constants[2];
    float dt = d_constants[3];
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int tidx = threadIdx.x;
    
	//It is much faster to calculate things locally in registers than in shared memory.
	//Now each of these threads has it's own copy of the variables to work with stored inside the thread's registers

    // Local force into registers
    float force_x = 0.0f;
    float force_y = 0.0f;
    float force_z = 0.0f;
    
    // Load my position into registers
    float my_x = 0.0f;
	float my_y = 0.0f;
	float my_z = 0.0f;
    
    // Load my position from global memory and zero out forces
    if (i < n) 
	{
        float3 pos_i = p[i];
        my_x = pos_i.x;
        my_y = pos_i.y;
        my_z = pos_i.z;
        
        // Zero out forces
        f[i] = make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Process bodies by block
    for (int currentBlock = 0; currentBlock < gridDim.x; currentBlock++)//for each block
	{
        //Clear shared memory
		sh_Pos[tidx] = make_float3(0.0f, 0.0f, 0.0f);
        __syncthreads();

		// Load current block of positions into shared memory
        int j_idx = tidx +currentBlock * blockDim.x;  //global index of the body being processed within the block
        if (j_idx < n) 
		{
            sh_Pos[tidx] = p[j_idx]; //load valid position into shared memory
        } 
		else //else pad
		{
            sh_Pos[tidx] = make_float3(INFINITY, INFINITY, INFINITY); //Copilot reccomended using Infinity to prevent division by zero
        }
        __syncthreads();
        
        if (i < n) 
		{
            // Process all bodies in this block
            for (int j = 0; j < blockDim.x; j++) 
			{
                int j_global = currentBlock * blockDim.x + j; //
                
                // Skip invalid indices and self-interaction
                if (j_global < n && i != j_global) 
				{
                    // Get other body's position (from shared memory to registers)
                    float3 pos_j = sh_Pos[j];
                    
                    // Calculate displacement
                    float dx = pos_j.x - my_x;
                    float dy = pos_j.y - my_y;
                    float dz = pos_j.z - my_z;
                    float d2 = dx*dx + dy*dy + dz*dz;
                    
                    // Prevent division by zero
                    d2 = fmaxf(d2, 1e-10f);
                    float d = sqrtf(d2);
                    
                    float force_mag = g/d2 - h/(d2*d2);
                    
                    force_x += force_mag * dx/d;
                    force_y += force_mag * dy/d;
                    force_z += force_mag * dz/d;
                }
            }
        }
        __syncthreads(); //make sure this block is done
    }
    
    // Store accumulated force and update position/velocity - only for real bodies
    if (i < n) 
	{
        // Load force from registers to global memory
        f[i] = make_float3(force_x, force_y, force_z);
        
        // Load current velocity
        float3 vel = v[i];
        
        // Update velocity
        float vx, vy, vz;

        if (t == 0.0f) 
		{
            vx = vel.x + (force_x - damp*vel.x)*dt/2.0f;
            vy = vel.y + (force_y - damp*vel.y)*dt/2.0f;
            vz = vel.z + (force_z - damp*vel.z)*dt/2.0f;
        } 
		else 
		{
            vx = vel.x + (force_x - damp*vel.x)*dt;
            vy = vel.y + (force_y - damp*vel.y)*dt;
            vz = vel.z + (force_z - damp*vel.z)*dt;
        }
        
        // Update position and velocity
        v[i] = make_float3(vx, vy, vz);
        p[i] = make_float3(my_x + vx*dt, my_y + vy*dt, my_z + vz*dt);
    }
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;

	while(t < RUN_TIME)
	{

		nBodyGPU<<<GridSize, BlockSize, 0, computeStream>>>(PGPU, VGPU, FGPU, t, N);
		cudaErrorCheck(__FILE__, __LINE__);

		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag) 
			{	
				drawPicture();
			}
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
	}
}

int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}





