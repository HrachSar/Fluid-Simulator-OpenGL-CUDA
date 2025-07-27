#include "cuda_manager.h"

cuda_manager::cuda_manager() : m_cudaVBOResource(nullptr) {}

cuda_manager::~cuda_manager() {
	free();
}

fluid_particle* cuda_manager::mapResources() {
	if (m_cudaVBOResource != nullptr) {
		fluid_particle* d_ptr;
		size_t size;
		cudaGraphicsMapResources(1, &m_cudaVBOResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &size, m_cudaVBOResource);

		return d_ptr;
	}
	return nullptr;
}
void cuda_manager::unmapResources() {
	if (m_cudaVBOResource != nullptr) {
		cudaGraphicsUnmapResources(1, &m_cudaVBOResource, 0);
	}
}
void cuda_manager::free() {
	if (m_cudaVBOResource != nullptr) {
		cudaGraphicsUnregisterResource(m_cudaVBOResource);
		m_cudaVBOResource = nullptr;
	}
}
