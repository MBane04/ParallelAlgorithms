// Name: Mason Bane
// nBody code on multiple GPUs. 
// nvcc HW24v2.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 Modify it so it runs on two GPUs.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 128
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
#define RUN_TIME 10.0

// Globals
int N, N0, N1;
float3 *P, *V, *F;
float *M; 
float3 *PGPU0, *VGPU0, *FGPU0;
float3 *PGPU1, *VGPU1, *FGPU1;
float *MGPU0, *MGPU1;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize0, GridSize1;

// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	
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

void setup()
{


	//no. of devices
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount < 2)
	{
		printf("\n\n Look at this idiot (again), you'd think they'd know that this code only runs on 2 GPUs\n");
		printf(" You have %d GPU(s) and I'm not writing this code so that it runs on %d GPU(s) so it's up to your broke self to fix it.\n\n", deviceCount, deviceCount);
		printf(" Maybe stop being a broke bum, get a job (or if you have one already, get a better one), buy another GPU, install it in the machine, then come back and talk to me\n");
		printf(" Until then, don't waste my time.... im outta here loser\n\n\n");

		exit(0);
	}



	float randomAngle1, randomAngle2, randomRadius;
	float d, dx, dy, dz;
	int test;
	
	N = 1000;

	N0 = N/2;
	N1 = N - N0;
	
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
    GridSize0.x = (N0 - 1) / BlockSize.x + 1;
    GridSize0.y = 1;
    GridSize0.z = 1;
    
    GridSize1.x = (N1 - 1) / BlockSize.x + 1;
    GridSize1.y = 1;
    GridSize1.z = 1;

	Damp = 0.5;
	
	M = (float*)malloc(N*sizeof(float));
	P = (float3*)malloc(N*sizeof(float3));
	V = (float3*)malloc(N*sizeof(float3));
	F = (float3*)malloc(N*sizeof(float3));
	
	cudaSetDevice(0);
	cudaMalloc(&PGPU0, N*sizeof(float3));  // All positions (force calc needs all positions)
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU0, N0*sizeof(float3)); // Only first half velocities
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU0, N0*sizeof(float3)); // Only first half forces
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&MGPU0, N*sizeof(float));   // All masses (force calc needs them too)
	cudaErrorCheck(__FILE__, __LINE__);

	cudaSetDevice(1);
	cudaMalloc(&PGPU1, N*sizeof(float3));  // All positions (force calc needs all positions)
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU1, N1*sizeof(float3)); // Only back half velocities
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU1, N1*sizeof(float3)); // Only back half forces
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&MGPU1, N*sizeof(float));   // All masses (force calc needs them too)
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
	

	cudaSetDevice(0);
	cudaMemcpyAsync(PGPU0, P, N*sizeof(float3), cudaMemcpyHostToDevice); // all pos
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU0, V, N0*sizeof(float3), cudaMemcpyHostToDevice); // half vel
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FGPU0, F, N0*sizeof(float3), cudaMemcpyHostToDevice); // half force
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU0, M, N*sizeof(float), cudaMemcpyHostToDevice); // all mass
	cudaErrorCheck(__FILE__, __LINE__);

	cudaSetDevice(1);
	cudaMemcpyAsync(PGPU1, P, N*sizeof(float3), cudaMemcpyHostToDevice); // all pos
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU1, V+N0, N1*sizeof(float3), cudaMemcpyHostToDevice); // half vel
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FGPU1, F+N0, N1*sizeof(float3), cudaMemcpyHostToDevice); // half force
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU1, M, N*sizeof(float), cudaMemcpyHostToDevice); // all mass
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int myN, int totalN, int offset) //added offset to the function so it can be used for both GPUs
{
	float dx, dy, dz,d,d2;
	float force_mag;
	
	int i = threadIdx.x + blockDim.x*blockIdx.x + offset; 
	
	if(i < myN)
	{
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < totalN; j++) //now we loopin through everyone, not just my half
		{
			if(i != j)
			{
				dx = p[j].x-p[i].x;
				dy = p[j].y-p[i].y;
				dz = p[j].z-p[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
	}
}


__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n)
{	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < n)
	{
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
		}

		p[i].x += v[i].x*dt;
		p[i].y += v[i].y*dt;
		p[i].z += v[i].z*dt;
	}
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;

	while(t < RUN_TIME)
	{
		//forces & positions on GPU0
		cudaSetDevice(0);
		getForces<<<GridSize0, BlockSize>>>(PGPU0, VGPU0, FGPU0, MGPU0, G, H, N0, N, 0); //updated so it checks all bodies and only updates the first half of the forces, offset is 0
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSize0,BlockSize>>>(PGPU0, VGPU0, FGPU0, MGPU0, Damp, dt, t, N0);
		cudaErrorCheck(__FILE__, __LINE__);

		//forces & positions on GPU1
		cudaSetDevice(1);
		getForces<<<GridSize1,BlockSize>>>(PGPU1, VGPU1, FGPU1, MGPU1, G, H, N1, N, N0); //same as the first, but not for the first, actually for just the 2nd half, offset is N0
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSize1,BlockSize>>>(PGPU1, VGPU1, FGPU1, MGPU1, Damp, dt, t, N1);
		cudaErrorCheck(__FILE__, __LINE__);

		//make sure both devices are done before we copy the data
		cudaDeviceSynchronize(); //1 is done, go ahead and do it since we here

		cudaSetDevice(0);
		cudaDeviceSynchronize(); //yurrrr

		// Send updated first half 0 to 1
		cudaSetDevice(0);
		cudaMemcpyPeerAsync(PGPU1, 1, PGPU0, 0, N0*sizeof(float3)); //designed for multiple GPUs, args are (dest, destDevice, src, srcDevice, size)
		cudaErrorCheck(__FILE__, __LINE__);

		// Send updated second half (from GPU1) to GPU0
		cudaSetDevice(1);
        cudaMemcpyPeerAsync(PGPU0 + N0, 0, PGPU1 + N0, 1, N1*sizeof(float3));
        cudaErrorCheck(__FILE__, __LINE__);

		// Ensure all copies are complete before next iteration (not sure if this is necesssary, but i guess we'll see)
		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaSetDevice(1);
		cudaDeviceSynchronize();


		// Now we are done with the copy, so we can go ahead and do the next step of the simulation.


		if(drawCount == DRAW_RATE) 
		{	
			//copy positions from GPUs to CPU for drawing
			cudaSetDevice(0);
			cudaMemcpyAsync(P, PGPU0, N0*sizeof(float3), cudaMemcpyDeviceToHost);


			cudaSetDevice(1);
			cudaMemcpyAsync(P+N0, PGPU1, N1*sizeof(float3), cudaMemcpyDeviceToHost);
			cudaErrorCheck(__FILE__, __LINE__);

			//make sure both devices are done before we draw the picture
			cudaDeviceSynchronize(); //already on 1 again

			cudaSetDevice(0);
			cudaDeviceSynchronize();



			drawPicture();
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
	}
}

void cleanup()
{
    // Free host memory
    free(P);
    free(V);
    free(F);
    free(M);
    
    // Free GPU 0 memory
    cudaSetDevice(0);
    cudaFree(PGPU0);
    cudaFree(VGPU0);
    cudaFree(FGPU0);
    cudaFree(MGPU0);
    
    // Free GPU 1 memory
    cudaSetDevice(1);
    cudaFree(PGPU1);
    cudaFree(VGPU1);
    cudaFree(FGPU1);
    cudaFree(MGPU1);
}

int main(int argc, char** argv)
{
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Nbody Two GPUs");
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
	glutDisplayFunc(drawPicture);
	glutIdleFunc(nBody);
	
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

	cleanup();

	return 0;
}

