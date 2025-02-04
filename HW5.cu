// Name: Mason Bane
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

/*********************ALL DEFS FOUND IN THE WEBSITE BELOW************************/
// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
/********************************************************************************/
int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPU(s) in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name); //Name of the GPU
		printf("Compute capability: %d.%d\n", prop.major, prop.minor); //Compute capability of the GPU, major and minor
		printf("Clock rate: %d\n", prop.clockRate); //Clock frequency in kilohertz, how many instructions per second
		printf("Device copy overlap: "); //Can memory copy operations be overlapped
		if (prop.deviceOverlap) printf("Enabled\n"); //can the device concurrently copy memory and execute a kernel (Deprecated)
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem); //Global memory available on the GPU in bytes
		printf("Total constant Mem: %ld\n", prop.totalConstMem); //Constant memory available on the GPU in bytes, constant memory is a read-only cache that is shared by all threads in a block
		printf("Max mem pitch: %ld\n", prop.memPitch); //Maximum pitch in bytes allowed by memory copies
		printf("Texture Alignment: %ld\n", prop.textureAlignment); //Alignment requirement for textures
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //# of MPs on the GPU
		printf("Shared mem per block: %ld\n", prop.sharedMemPerBlock); //Shared memory available per block in bytes
		printf("Registers per block: %d\n", prop.regsPerBlock); //# of 32-bit registers per block
		printf("Threads in warp: %d\n", prop.warpSize); //Warp size in threads, a warp is a group of threads that are concurrently executing, 32 on all current GPUs
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); //Maximum # of threads per block
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); //Max # of threads per block in each dimension
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); //Maximum size of each dimension of a grid
		printf("\n");


		//missing info to print out
		printf(" ---Additional Information for device %d ---\n", i);
		printf("Texture pitch alignment: %ld\n", prop.texturePitchAlignment); //Pitch alignment requirement for textures bound to pitched memory
		printf("Integrated: %d\n", prop.integrated); //is the GPU integrated with the CPU oppsite is discrete
		printf("Can map host memory: %d\n", prop.canMapHostMemory); //Can map host memory into the device address space
		printf("Compute mode: %d\n", prop.computeMode); //what compute mode the GPU is in
		printf("maxTexture1D: %d\n", prop.maxTexture1D); //Maximum size of 1D texture
		printf("maxTexture1DMipmap: %d\n", prop.maxTexture1DMipmap); // maximum size of 1D mipmapped texture
		printf("maxTexture1DLinear: %d\n", prop.maxTexture1DLinear); //deprecated, document said dont use it
		printf("maxTexture2D: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]); //Maximum size of 2D texture
		printf("maxTexture2DMipmap: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]); //Maximum size of 2D mipmapped texture
		printf("maxTexture2DLinear: (%d, %d, %d)\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]); //Max dims (w, h, pitch) for 2D textures bound to piched memory
		printf("maxTexture2DGather: (%d, %d)\n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]); //Maximum dims of 2D texture if texture gather operations are performed
		printf("maxTexture3D: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]); //Maximum dims of 3D texture
		printf("maxTexture3DAlt: (%d, %d, %d)\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]); //Maximum alternate 3D texture dims
		printf("maxTextureCubemap: %d\n", prop.maxTextureCubemap); //Maximum dims of cubemap texture
		printf("maxTexture1DLayered: (%d, %d)\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]); //Maximum size of 1D layered texture
		printf("maxTexture2DLayered: (%d, %d, %d)\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]); //Maximum dims of 2D layered texture
		printf("maxTextureCubemapLayered: (%d, %d)\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);//Maximum dims of cubemap layered texture
		printf("maxSurface1D: %d\n", prop.maxSurface1D); //Maximum size of 1D surface
		printf("maxSurface2D: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]); //Maximum 2D surface dims
		printf("maxSurface3D: (%d, %d, %d)\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]); //Maximum 3D surface dims
		printf("maxSurface1DLayered: (%d, %d)\n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]); //Maximum 1D layered surface dims
		printf("maxSurface2DLayered: (%d, %d, %d)\n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]); //Maximum 2D layered surface dims
		printf("maxSurfaceCubemap: %d\n", prop.maxSurfaceCubemap); //Maximum cubemap surface dims
		printf("maxSurfaceCubemapLayered: (%d, %d)\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]); //Maximum cubemap layered surface dims
		printf("surfaceAlignment: %ld\n", prop.surfaceAlignment); //Alignment requirement for surfaces
		printf("concurrentKernels: %d\n", prop.concurrentKernels); //Can the GPU run multiple kernels concurrently
		printf("ECCEnabled: %d\n", prop.ECCEnabled); //is Error Correcting Code enabled
		printf("pciBusID: %d\n", prop.pciBusID); //PCI bus ID of the GPU
		printf("pciDeviceID: %d\n", prop.pciDeviceID); //PCI device ID of the GPU
		printf("pciDomainID: %d\n", prop.pciDomainID); //PCI domain ID of the GPU
		printf("tccDriver: %d\n", prop.tccDriver); //is the GPU running a Tesla device using TCC driver
		printf("asyncEngineCount: %d\n", prop.asyncEngineCount); //Number of asynchronous engines
		printf("unifiedAddressing: %d\n", prop.unifiedAddressing); //does the GPU support unified addressing
		printf("memoryClockRate: %d\n", prop.memoryClockRate); //Peak memory clock frequency in kilohertz (deprecated)
		printf("memoryBusWidth: %d\n", prop.memoryBusWidth); //Global memory bus width in bits
		printf("l2CacheSize: %d\n", prop.l2CacheSize); //size of the L2 cache
		printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor); //Maximum number of resident threads per multiprocessor
		printf("streamPrioritiesSupported: %d\n", prop.streamPrioritiesSupported); //does the GPU support stream priorities
		printf("globalL1CacheSupported: %d\n", prop.globalL1CacheSupported); //can you cache global memory in L1
		printf("localL1CacheSupported: %d\n", prop.localL1CacheSupported); //can you cache local memory in L1
		printf("sharedMemPerMultiprocessor: %ld\n", prop.sharedMemPerMultiprocessor); //shared memory per multiprocessor in bytes
		printf("regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor); //number of 32-bit registers per multiprocessor
		printf("managedMemory: %d\n", prop.managedMemory); //does the GPU support managed memory
		printf("isMultiGpuBoard: %d\n", prop.isMultiGpuBoard); //is the GPU  on a multi GPU board
		printf("multiGpuBoardGroupID: %d\n", prop.multiGpuBoardGroupID); //unique ID for a group of GPUs on the same multi GPU board
		printf("hostNativeAtomicSupported: %d\n", prop.hostNativeAtomicSupported); //can the host perform native atomic operations on device memory
		printf("singleToDoublePrecisionPerfRatio: %d\n", prop.singleToDoublePrecisionPerfRatio); //ratio of single precision performance (in FLOPS) to double precision performance
		printf("pageableMemoryAccess: %d\n", prop.pageableMemoryAccess); //can the GPU access pageable memory
		printf("concurrentManagedAccess: %d\n", prop.concurrentManagedAccess); //Can the GPU access managed memory concurrently with the CPU
		printf("computePreemptionSupported: %d\n", prop.computePreemptionSupported); //does the GPU support compute preemption
		printf("canUseHostPointerForRegisteredMem: %d\n", prop.canUseHostPointerForRegisteredMem); //can the GPU use the host pointer for registered memory
		printf("cooperativeLaunch: %d\n", prop.cooperativeLaunch); //does the GPU support launch cooperative kernels
		printf("cooperativeMultiDeviceLaunch: %d\n", prop.cooperativeMultiDeviceLaunch); //deprecated
		printf("sharedMemPerBlockOptin: %ld\n", prop.sharedMemPerBlockOptin); //max shared memory per block by special opt-in flag per device
		printf("pageableMemoryAccessUsesHostPageTables: %d\n", prop.pageableMemoryAccessUsesHostPageTables); //does the GPU use the host page tables for pageable memory access
		printf("directManagedMemAccessFromHost: %d\n", prop.directManagedMemAccessFromHost);
		printf("Access policy maxWindowSize: %d\n", prop.accessPolicyMaxWindowSize); //Maximum window size for access policy
		printf("clusterLaunched: %d\n", prop.clusterLaunch); //does the GPU support cluster launch
		//printf("defferedMappingCudaArray: %d\n", prop.deferredMappingCudaArray); //does the GPU support deferred mapping of cuda arrays
		//printf("gpuDirectRDMAFlushWrites: %d\n", prop.gpuDirectRDMAFlushWrites); //bitmask of flags that indicate the GPU supports direct RDMA writes
		printf("gpuDirectRDMASupported: %d\n", prop.gpuDirectRDMASupported); //does the GPU support direct RDMA Access Protocol Interface(APIs)
		printf("gpuDirectRDMAWritesOrdering: %d\n", prop.gpuDirectRDMAWritesOrdering); //documentation is unclear, says look at enum
		printf("hostRegisterReadOnlySupported: %d\n", prop.hostRegisterReadOnlySupported); //can the CPU register memory as read only to the GPU
		printf("hostRegisterSupported: %d\n", prop.hostRegisterSupported); //can the CPU register memory to the GPU
		printf("ipcEventSupported: %d\n", prop.ipcEventSupported); //does the GPU support IPC events
		printf("kernelExecTimeoutEnabled: %d\n", prop.kernelExecTimeoutEnabled); //is the kernel execution timeout enabled (deprecated)
		printf("luid: %d\n", prop.luid); //8-byte Local Unique Identifier, undefined on non-Windows platforms
		printf("luidDeviceNodeMask: %d\n", prop.luidDeviceNodeMask); //device node mask of the LUID, undefined on non-Windows platforms
		printf("memoryPoolSupportedHandleTypes: %d\n", prop.memoryPoolSupportedHandleTypes); //bitmask of handle types supported with mempool-based IPC
		printf("memoryPoolsSupported: %d\n", prop.memoryPoolsSupported); //does the GPU support memory pools
		printf("persistingL2CacheMaxSize: %d\n", prop.persistingL2CacheMaxSize); // GPU's maximum size of persisting L2 cache in bytes
		printf("reserved: %d\n", prop.reserved); //reserved for future use and must be zero
		printf("reservedSharedMemPerBlock: %ld\n", prop.reservedSharedMemPerBlock); //shared mem per block reserved by CUDA in bytes
		printf("sparseCudaArraySupported: %d\n", prop.sparseCudaArraySupported); //does the GPU support sparse CUDA arrays
		printf("timelineSemaphoreInteropSupported: %d\n", prop.timelineSemaphoreInteropSupported); //does the GPU support timeline semaphore interop
		printf("unifiedFunctionPointers: %d\n", prop.unifiedFunctionPointers); //does the GPU support unified function pointers
		printf("uuid: %d\n", prop.uuid); //16-byte globally unique identifier






	}	
	return(0);
}

