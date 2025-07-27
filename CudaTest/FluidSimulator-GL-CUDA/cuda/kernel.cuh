#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../core/renderer.h"


__device__ int IX(int x, int y); //current index
__global__ void updateVertices(fluid_particle* particles, float time); // Update particle positions
__global__ void fluidCopyToPrev(fluid_particle* particles); //copy current particles to the prev
__global__ void fluidAddDensity(fluid_particle* particles, int sourceX, int sourceY, float amount); // Add density to a particle
__global__ void fluidAddVelocity(fluid_particle* particles, int sourceX, int sourceY, float amountX, float amountY); //Add velicity to a particle 
__global__ void fluidSetBounds(int b, fluid_particle* particles); //Set the edge particles to act like walls
__global__ void fluidLinSolve(int b, fluid_particle* particles, float a, float c);
__global__ void fluidProjectStep1(fluid_particle* particles, fluid_particle* temp); //Calculate divergence 
__global__ void fluidProjectStep2(fluid_particle* particles, fluid_particle* temp); //Calculate pressure gradient
__global__ void fluidAdvect(int b, fluid_particle* particles, float dt); //This function is responsible for actually moving things around

/*
	The parameter b represents to which fields must be changed -> velocity.u, velocity.v or density. 
	The parameter a represents the diffusion strenght -> a = VISC * dt * (GRID_SIZE - 2)^2
	The parameter c = 1 + 6 * a = normalization factor (1 + 4a for 2D, but 6a accounts for boundaries)
*/

extern "C" {
	void fluidUpdateWithCuda(fluid_particle* particles, fluid_particle* temp, float dt); // Update fluid using CUDA
	void fluidAddDensityWithCuda(fluid_particle* particles, int sourceX, int sourceY, float amount);
	void fluidAddVelocityWithCuda(fluid_particle* particles, int sourceX, int sourceY, float amountX, float amountY);
}

#endif