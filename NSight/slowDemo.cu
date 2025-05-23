// Name: Mason Bane
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc  slowDemo.cu -G -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the CPU. Rewrite it, keeping the same general format, 
 but offload the compute-intensive parts of the code to the GPU for acceleration.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate, (We will keep the number of bodies under 1024 for this HW so it can be run on one block.)
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F, *P_GPU, *V_GPU, *F_GPU, *d_P, *d_V, *d_F;
float *M, *M_GPU;
float GlobeRadius, Diameter, Radius;
float Damp;
bool slowCode = true; // This is to slow down the code so we can see the bodies move.

dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
cudaStream_t memStream; //This variable will hold the memory stream for the GPU

// Function prototypes
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody();
int main(int, char**);
void cleanUp();

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


//This will be the layout of the parallel space we will be using.
void setUpDevices()
{

	//single block, 1024 threads
	BlockSize.x =1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		timer();
	}
	
	if(key == 'q')
	{
		cleanUp();
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
	
	glColor3d(1.0,1.0,0.5);

	slowCode = true ? cudaMemcpy(P, P_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost) : cudaMemcpyAsync(P_GPU, P, N * sizeof(float3), cudaMemcpyHostToDevice, memStream);
	cudaErrorCheck(__FILE__, __LINE__);

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
	
	Damp = 0.5;

	cudaStreamCreate(&memStream);
	cudaErrorCheck(__FILE__, __LINE__);
	
	M = (float*)malloc(N*sizeof(float));
	P = (float3*)malloc(N*sizeof(float3));
	V = (float3*)malloc(N*sizeof(float3));
	F = (float3*)malloc(N*sizeof(float3));

	// Allocate memory on the GPU, doing it here so all mallocs are in one place.
	cudaMalloc(&P_GPU, N * sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMalloc(&V_GPU, N * sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMalloc(&F_GPU, N * sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMalloc(&M_GPU, N * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
    	
	
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

	//copy the data to the GPU
	cudaMemcpy(P_GPU, P, N * sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(V_GPU, V, N * sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(F_GPU, F, N * sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(M_GPU, M, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
}

__global__ void getForces(float3 *P, float3 *V, float3 *F, float *M, int n)
{
	int idx = threadIdx.x; //rest isn't needed since we're only using one block.

	if(idx < n)
	{
		//Set our forces to 0
		F[idx].x = 0.0;
		F[idx].y = 0.0;
		F[idx].z = 0.0;

		//Calculate my force based on everyone else
		for(int i=0; i<n; i++)
		{
			if(i != idx) //If it's not me, it must be someone else
			{
				float dx = P[i].x-P[idx].x;
				float dy = P[i].y-P[idx].y;
				float dz = P[i].z-P[idx].z;
				float d2 = dx*dx + dy*dy + dz*dz;
				float d  = sqrt(d2);
				
				float force_mag  = (G*M[i]*M[idx])/(d2) - (H*M[i]*M[idx])/(d2*d2);

				//update only my force
				F[idx].x += force_mag*dx/d;
				F[idx].y += force_mag*dy/d;
				F[idx].z += force_mag*dz/d;

				// messing with other people causes race conditions, because we're all trying to mess with the same people who are messing with us and who might have been messed with by someone else
				//Dr. Wyatt is teaching us valuable life lessons, don't mess with other people's stuff cause it screws stuff up
				// F[i].x -= force_mag*dx/d;
				// F[i].y -= force_mag*dy/d;
				// F[i].z -= force_mag*dz/d;
			}
		}
	}
}

__global__ void updatePositions(float3 *P, float3 *V, float3 *F, float *M, int n, float dt, float damp, float time)
{
	int idx = threadIdx.x; //again, rest isn't needed since we're only using one block.

	if(idx < n)
	{
		if(time == 0.0)
		{
			V[idx].x += (F[idx].x/M[idx])*0.5*dt;
			V[idx].y += (F[idx].y/M[idx])*0.5*dt;
			V[idx].z += (F[idx].z/M[idx])*0.5*dt;
		}
		else
		{
			V[idx].x += ((F[idx].x-damp*V[idx].x)/M[idx])*dt;
			V[idx].y += ((F[idx].y-damp*V[idx].y)/M[idx])*dt;
			V[idx].z += ((F[idx].z-damp*V[idx].z)/M[idx])*dt;
		}

		P[idx].x += V[idx].x*dt;
		P[idx].y += V[idx].y*dt;
		P[idx].z += V[idx].z*dt;
	}
}

void nBody()
{

	int    drawCount = 0; 
	float  time = 0.0;
	float dt = 0.0001;

	if(slowCode)
	{
		//Do unnecessary copies to slow down the code 
		cudaMemcpy(P_GPU, P, N * sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaMemcpy(V_GPU, V, N * sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaMemcpy(F_GPU, F, N * sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaMemcpy(M_GPU, M, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}


	while(time < RUN_TIME)
	{


		getForces<<<GridSize, BlockSize>>>(P_GPU, V_GPU, F_GPU, M_GPU, N);
		cudaErrorCheck(__FILE__, __LINE__);

		updatePositions<<<GridSize, BlockSize>>>(P_GPU, V_GPU, F_GPU, M_GPU, N, dt, Damp, time);
		cudaErrorCheck(__FILE__, __LINE__);

		//draw if we need to
		if(drawCount == DRAW_RATE) 
		{
			if(slowCode) cudaMemcpyAsync(P, P_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost); //only copy pos to CPU if drawing
			cudaErrorCheck(__FILE__, __LINE__);
			if(DrawFlag) drawPicture();
			drawCount = 0;
		}
		
		time += dt;
		drawCount++;
	}
	if(slowCode)
	{
		//now that we're done, copy the data back to the CPU
		cudaMemcpy(P, P_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaMemcpy(V, V_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);

		cudaMemcpy(F, F_GPU, N * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}


}

void cleanUp()
{
	free(P);
	free(V);
	free(F);
	free(M);

	cudaFree(P_GPU);
	cudaFree(V_GPU);
	cudaFree(F_GPU);
	cudaFree(M_GPU);

	cudaStreamDestroy(memStream);

	printf("\n Memory has been cleaned up.\n");
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
	
	setUpDevices();
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
	timer();
	
	glutMainLoop();
	atexit(cleanUp);
	return 0;
}




