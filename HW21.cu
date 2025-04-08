// Name: Mason Bane
// Optimizing nBody GPU code. 
//nvcc -O3 -use_fast_math -arch=sm_86 --ptxas-options=-v -Xptxas -dlcm=ca -Xptxas -dlcm=cs  -maxrregcount=48 -o temp HW21best.cu -lglfw -lGLEW -lGL -lGLU

//  !!!!!!!!!!  YOU MIGHT NEED TO RUN dependencies.sh TO INSTALL THE DEPENDENCIES FOR THIS CODE TO WORK.  !!!!!!!!!!!

/*
for BS in 128 256 512 1024; do
    nvcc -O3 -use_fast_math -DBLOCK_SIZE=$BS -arch=sm_86 --ptxas-options=-v -Xptxas -dlcm=ca HW21.cu -o temp_$BS -lglfw -lGLEW -lGL -lGLU
    ./temp_$BS 1024 0
done

*/

/*
 What to do:
 It has been optimized using standard techniques like shared memory, using float4s, setting the block sizes to powers of 2, 
 and ensuring that the number of bodies is an exact multiple of the block size to reduce if statement.
 Take this code and make it run as fast as possible using any tricks you know or can find.
 Try to keep the general format the same so we can time it and compare it with others' code.
 This will be a competition. 
 You can remove all the restrictions from the blocks. We will be competing with 10,752 bodies. 
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate.
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).

Starting time:
12'219'667 us

Best time:
7'877'181 us

*/

// Include files
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <GL/glew.h> // Needed for VBOs + CUDA, using GLAD in main code
#include <GLFW/glfw3.h>
#include <vector>


// Defines
#define BLOCK_SIZE 256
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
float4 *P, *V, *F;
float *M; 
float4 *PGPU, *VGPU, *FGPU;
__constant__ float d_constants[4]; // G, H, Damp, dt
cudaStream_t computeStream, visualStream; //2 streams: one for computation and one for visual updates
size_t sharedMemSize;

GLFWwindow* Window;

float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;


//To use VBOs for sphere rendering
GLuint sphereVBO, sphereIBO; // Vertex Buffer Object and Index Buffer Object for sphere rendering, Vertex is the sphere's vertices and Index is the order in which to draw them
GLuint numSphereVertices, numSphereIndices; // Number of vertices and indices in the sphere geometry

// Function prototypes
void cudaErrorCheck(const char *, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void nBodyGPUFirstIteration(float4 *, float4 *, float4 *, int);
__global__ void nBodyGPU(float4 *, float4 *, float4 *, int);
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

void keyPressed(GLFWwindow* Window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS) // Only handle key presses
    {
        if (key == GLFW_KEY_S) // 'S' key to run the simulation
        {
            printf("\n The simulation is running.\n");
            timer(); // Run the simulation
        }
        else if (key == GLFW_KEY_Q) // 'Q' key to quit
        {
            glfwSetWindowShouldClose(Window, GLFW_TRUE); // Close the Window
        }
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

// Function to render a sphere using VBOs
void createSphereVBO(float radius, int slices, int stacks) //this one builds a single sphere VBO that we can use for rendering each body in the simulation.
{
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    
    // Generate sphere vertices with positions and normals
    for (int i = 0; i <= stacks; ++i) 
    {
        float phi = PI * i / stacks;
        float sinPhi = sin(phi);
        float cosPhi = cos(phi);
        
        for (int j = 0; j <= slices; ++j) 
        {
            float theta = 2.0f * PI * j / slices;
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);
            
            // Vertex position (x, y, z)
            float x = radius * sinPhi * cosTheta;
            float y = radius * cosPhi;
            float z = radius * sinPhi * sinTheta;
            
            // Normal vector (normalized position for sphere)
            float nx = sinPhi * cosTheta;
            float ny = cosPhi;
            float nz = sinPhi * sinTheta;
            
            // Add vertex (position + normal)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }
    
    // Generate indices for triangle strips
    for (int i = 0; i < stacks; ++i) 
    {
        for (int j = 0; j < slices; ++j) 
        {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;
            
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);
            
            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
    
    numSphereVertices = vertices.size() / 6; // 6 floats per vertex (pos + normal)
    numSphereIndices = indices.size();
    
    // Create and bind the vertex buffer
    glGenBuffers(1, &sphereVBO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    // Create and bind the index buffer
    glGenBuffers(1, &sphereIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    // Unbind buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void renderSphereVBO() 
{
    // Bind the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIBO);
    
    // Enable vertex and normal arrays
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    
    // Set up pointers to vertex and normal data
    glVertexPointer(3, GL_FLOAT, 6 * sizeof(float), 0);
    glNormalPointer(GL_FLOAT, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    
    // Draw the sphere
    glDrawElements(GL_TRIANGLES, numSphereIndices, GL_UNSIGNED_INT, 0);
    
    // Disable arrays
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    
    // Unbind buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	
	//copy mem
	cudaMemcpyAsync(P, PGPU, N*sizeof(float4), cudaMemcpyDeviceToHost, visualStream); // Copy positions from GPU to CPU for rendering

	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		renderSphereVBO(); 
		glPopMatrix();
	}

	glfwSwapBuffers(Window); // Swap buffers to display the rendered frame

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

    	
    if (N <= 1024) 
    {
        BlockSize.x = 128;
    } 
    else
    {
        BlockSize.x = 256;
    } 
	BlockSize.y = 1;
	BlockSize.z = 1;

	GridSize.x = ((N + BlockSize.x - 1) / BlockSize.x); //Makes enough blocks to deal with the whole vector.
	GridSize.y = 1;
	GridSize.z = 1;

    // Calculate shared memory size for the kernel, this is used in the kernel for shared memory allocation
    sharedMemSize = BlockSize.x * sizeof(float4);
	
	Damp = 0.5;
	
	// Allocate host memory
    cudaMallocHost(&P, N*sizeof(float4)); // Allocate host memory, makes a faster mem copy
    cudaMallocHost(&V, N*sizeof(float4)); // cuda needs to make sure the memory is page locked, so it can be accessed by the GPU
    cudaMallocHost(&F, N*sizeof(float4)); // cudaMallocHost allocates page-locked memory on the host for fast access by the device
    M = (float*)malloc(N*sizeof(float));
    	

	// Allocate device memory with padding
	cudaMalloc(&PGPU, N*sizeof(float4));
	cudaMalloc(&VGPU, N*sizeof(float4));
	cudaMalloc(&FGPU, N*sizeof(float4));

	// Initialize everything to zero
	cudaMemset(PGPU, 0, N*sizeof(float4));
	cudaMemset(VGPU, 0, N*sizeof(float4));
	cudaMemset(FGPU, 0, N*sizeof(float4));

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
				d = sqrtf(dx*dx + dy*dy + dz*dz);
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
    cudaMemcpyAsync(PGPU, P, N*sizeof(float4), cudaMemcpyHostToDevice, computeStream);
    cudaMemcpyAsync(VGPU, V, N*sizeof(float4), cudaMemcpyHostToDevice, computeStream);
    cudaMemcpyAsync(FGPU, F, N*sizeof(float4), cudaMemcpyHostToDevice, computeStream);
	
	printf("\n To start timing type s.\n");
}

__global__ void nBodyGPUFirstIteration(float4 *__restrict__ p, float4 *__restrict__ v, float4 *__restrict__ f, int n)
{
    extern __shared__ float4 shPos[];
    const float g = d_constants[0];
    const float h = d_constants[1];
    const float damp = d_constants[2];
    const float dt = d_constants[3];
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    
    // Register storage
    float3 force = {0.0f, 0.0f, 0.0f};
    float4 my_pos = p[i];
    float4 my_vel = v[i];
    
    // Main loop over memory tiles
    for (int tile = 0; tile < gridDim.x; ++tile) {
        const int load_idx = tile * blockDim.x + tid;
        shPos[tid] = (load_idx < n) ? p[load_idx] : make_float4(INFINITY, INFINITY, INFINITY, 0);
        __syncthreads();

        // Process current tile
        #pragma unroll 8
        for (int j = 0; j < blockDim.x; ++j) {
            // Skip self-interaction
            if (tile == blockIdx.x && j == tid) continue;
            
            const float4 other_pos = shPos[j];
            const float dx = other_pos.x - my_pos.x;
            const float dy = other_pos.y - my_pos.y;
            const float dz = other_pos.z - my_pos.z;
            
            // Fast force calculation with softening
            const float r2 = dx*dx + dy*dy + dz*dz + 1e-10f;
            const float inv_r = rsqrtf(r2);
            const float inv_r2 = inv_r * inv_r;
            const float f_mag = (g * inv_r2 - h * inv_r2 * inv_r2) * inv_r;
            
            force.x += f_mag * dx;
            force.y += f_mag * dy;
            force.z += f_mag * dz;
        }
        __syncthreads();
    }

    // Update velocity (mass = 1.0)
    my_vel.x += (force.x - damp * my_vel.x) * dt/2.0f;
    my_vel.y += (force.y - damp * my_vel.y) * dt/2.0f;
    my_vel.z += (force.z - damp * my_vel.z) * dt/2.0f;

    // Update position
    my_pos.x += my_vel.x * dt;
    my_pos.y += my_vel.y * dt;
    my_pos.z += my_vel.z * dt;

    // Store results
    p[i] = my_pos;
    v[i] = my_vel;
    f[i] = make_float4(force.x, force.y, force.z, 0.0f); // Optional force storage
}

__global__ void nBodyGPU(float4 *__restrict__ p, float4 *__restrict__ v, float4 *__restrict__ f, int n)
{
    extern __shared__ float4 shPos[];
    const float g = d_constants[0];
    const float h = d_constants[1];
    const float damp = d_constants[2];
    const float dt = d_constants[3];
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    
    // Register storage
    float3 force = {0.0f, 0.0f, 0.0f};
    float4 my_pos = p[i];
    float4 my_vel = v[i];
    
    // Main loop over memory tiles
    for (int tile = 0; tile < gridDim.x; ++tile) {
        const int load_idx = tile * blockDim.x + tid;
        shPos[tid] = (load_idx < n) ? p[load_idx] : make_float4(INFINITY, INFINITY, INFINITY, 0);
        __syncthreads();

        // Process current tile
        #pragma unroll 8
        for (int j = 0; j < blockDim.x; ++j) {
            // Skip self-interaction
            if (tile == blockIdx.x && j == tid) continue;
            
            const float4 other_pos = shPos[j];
            const float dx = other_pos.x - my_pos.x;
            const float dy = other_pos.y - my_pos.y;
            const float dz = other_pos.z - my_pos.z;
            
            // Fast force calculation with softening
            const float r2 = dx*dx + dy*dy + dz*dz + 1e-10f;
            const float inv_r = rsqrtf(r2);
            const float inv_r2 = inv_r * inv_r;
            const float f_mag = (g * inv_r2 - h * inv_r2 * inv_r2) * inv_r;
            
            force.x += f_mag * dx;
            force.y += f_mag * dy;
            force.z += f_mag * dz;
        }
        __syncthreads();
    }

    // Update velocity (mass = 1.0)
    my_vel.x += (force.x - damp * my_vel.x) * dt;
    my_vel.y += (force.y - damp * my_vel.y) * dt;
    my_vel.z += (force.z - damp * my_vel.z) * dt;

    // Update position
    my_pos.x += my_vel.x * dt;
    my_pos.y += my_vel.y * dt;
    my_pos.z += my_vel.z * dt;

    // Store results
    p[i] = my_pos;
    v[i] = my_vel;
    f[i] = make_float4(force.x, force.y, force.z, 0.0f); // Optional force storage
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;


	// First iteration, won't worry about it for now, optimize then copy
    nBodyGPUFirstIteration<<<GridSize, BlockSize, sharedMemSize, computeStream>>>(PGPU, VGPU, FGPU, N);
	cudaErrorCheck(__FILE__, __LINE__);

	t += dt;

	while(t < RUN_TIME)
	{

		nBodyGPU<<<GridSize, BlockSize, sharedMemSize, computeStream>>>(PGPU, VGPU, FGPU, N);
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
    if (argc < 3)
    {
        printf("\n You need to enter the number of bodies (an int)");
        printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
        printf("\n on the command line.\n");
        exit(0);
    }

    // Parse command-line arguments
    N = atoi(argv[1]);
    DrawFlag = atoi(argv[2]);

    // Setup simulation
    setup();

    int XWindowSize = 1000;
    int YWindowSize = 1000;

    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Set compatibility mode for legacy OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);

    // Create a windowed mode Window and its OpenGL context
    Window = glfwCreateWindow(XWindowSize, YWindowSize, "N-Body Simulation", NULL, NULL);
    if (!Window)
    {
        glfwTerminate();
        fprintf(stderr, "Failed to create Window\n");
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(Window);
    glfwSwapInterval(1); // Enable vsync

    glfwSetKeyCallback(Window, keyPressed);

    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        fprintf(stderr, "GLEW initialization error: %s\n", glewGetErrorString(err));
        glfwTerminate();
        return -1;
    }

    // Initialize the sphere VBO for rendering
    createSphereVBO(Radius, 20, 20);

    // Set the viewport size and aspect ratio
    glViewport(0, 0, XWindowSize, YWindowSize);

    // Set up projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 5.0 * GlobeRadius);

    // Set up modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0f, 0.0f, 2.0f * GlobeRadius, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // Set background color
    glClearColor(0.0, 0.0, 0.0, 0.0);

    // Lighting and material properties
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
    GLfloat light_ambient[] = {0.0, 0.0, 0.0, 1.0};
    GLfloat light_diffuse[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};

    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

    GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

    GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat mat_shininess[] = {10.0};
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

    glEnable(GL_DEPTH_TEST);


	drawPicture(); // Initial draw to show the starting positions of the bodies
	// Wait for user to close the window
	while (!glfwWindowShouldClose(Window))
	{
		glfwPollEvents();
	}


    // Cleanup and exit
    glfwDestroyWindow(Window);
    glfwTerminate();

    return 0;
}