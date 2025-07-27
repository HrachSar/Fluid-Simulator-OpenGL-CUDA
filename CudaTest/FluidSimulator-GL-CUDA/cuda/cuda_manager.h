#ifndef CUDA_MANAGER
#define CUDA_MANAGER


#include <cuda_runtime.h>
#include "../core/renderer.h"

class cuda_manager
{
public:
	cudaGraphicsResource* m_cudaVBOResource;


	cuda_manager();

	~cuda_manager();
	void free();
	fluid_particle* mapResources();
	void unmapResources();
};

#endif

