#include "kernel.cuh"
#include <math.h>

__global__ void updateVertices(fluid_particle* particles, float angle) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < GRID_SIZE && y < GRID_SIZE) {

		int idx = (y * GRID_SIZE + x);

        fluid_particle& p = particles[idx];

        float cosA = cosf(angle);
        float sinA = sinf(angle);

        float newX = p.x * cosA - p.y * sinA;
        float newY = p.x * sinA + p.y * cosA;

        p.x = newX;
        p.y = newY;

    }

}

__device__ int IX(int x, int y) {
    return GRID_SIZE * y + x;
}
__global__ void fluidCopyToPrev(fluid_particle* particles) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < GRID_SIZE && idy < GRID_SIZE) {
        int index = IX(idx, idy);
        particles[index].u_prev = particles[index].u;
        particles[index].v_prev = particles[index].v;
        particles[index].density_prev = particles[index].density;
    }
}

__global__ void fluidAddDensity(fluid_particle* particles, int sourceX, int sourceY, float amount) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int n = GRID_SIZE;

    if (idx > 0 && idx < n && idy > 0 && idy < n) {
        int index = IX(idx, idy);
        int dx = abs(idx - sourceX);
        int dy = abs(idy - sourceY);

        if (dx <= 2 && dy <= 2) {
            float falloff = 1.0f / (1.0f + dx + dy);
            particles[index].density += amount * falloff;
        }
    }
}
__global__ void fluidAddVelocity(fluid_particle* particles, int sourceX, int sourceY, float amountX, float amountY) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int n = GRID_SIZE;

    if (idx >= 0 && idx < n && idy >= 0 && idy < n) {
        int index = IX(idx, idy);
        int dx = abs(idx - sourceX);
        int dy = abs(idy - sourceY);

        if (dx <= 2 && dy <= 2) {
            float falloff = 1.0f / (1.0f + dx + dy);
            particles[index].u += amountX * falloff;
            particles[index].v += amountY * falloff ;
        }
    }

}
__global__ void fluidSetBounds(int b, fluid_particle* particles) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;


    int n = GRID_SIZE;

    if ( idx < n &&  idy < n) {

        if (idx == 0 && idy > 0 && idy < n - 1) {
            particles[IX(0, idy)].u = b == 1 ? -particles[IX(1, idy)].u : particles[IX(1, idy)].u;
            particles[IX(0, idy)].v = b == 2 ? -particles[IX(1, idy)].v : particles[IX(1, idy)].v;
            if (b == 0) particles[IX(0, idy)].density = particles[IX(1, idy)].density;
        }
        else if (idx == n - 1 && idy > 0 && idy < n - 1) {
            particles[IX(n - 1, idy)].u = b == 1 ? -particles[IX(n - 2, idy)].u : particles[IX(n - 2, idy)].u;
            particles[IX(n - 1, idy)].v = b == 2 ? -particles[IX(n - 2, idy)].v : particles[IX(n - 2, idy)].v;
            if (b == 0) particles[IX(n - 1, idy)].density = particles[IX(n - 2, idy)].density;
        }

        if (idy == 0 && idx > 0 && idx < n - 1) {
            particles[IX(idx, 0)].u = b == 1 ? -particles[IX(idx, 1)].u : particles[IX(idx, 1)].u;
            particles[IX(idx, 0)].v = b == 2 ? -particles[IX(idx, 1)].v : particles[IX(idx, 1)].v;
            if (b == 0) particles[IX(idx, 0)].density = particles[IX(idx, 1)].density;
        }
        else if (idy == n - 1 && idx > 0 && idx < n - 1) {
            particles[IX(idx, n - 1)].u = b == 1 ? -particles[IX(idx, n - 2)].u : particles[IX(idx, n - 2)].u;
            particles[IX(idx, n - 1)].v = b == 2 ? -particles[IX(idx, n - 2)].v : particles[IX(idx, n - 2)].v;
            if (b == 0) particles[IX(idx, n - 1)].density = particles[IX(idx, n - 2)].density;
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {

        particles[IX(0, 0)].u = 0.33f * (particles[IX(1, 0)].u + particles[IX(0, 1)].u);
        particles[IX(0, 0)].v = 0.33f * (particles[IX(1, 0)].v + particles[IX(0, 1)].v);
        particles[IX(0, 0)].density = 0.33f * (particles[IX(1, 0)].density + particles[IX(0, 1)].density);

        particles[IX(0, n - 1)].u = 0.33f * (particles[IX(1, n - 1)].u + particles[IX(0, n - 2)].u);
        particles[IX(0, n - 1)].v = 0.33f * (particles[IX(1, n - 1)].v + particles[IX(0, n - 2)].v);
        particles[IX(0, n - 1)].density = 0.33f * (particles[IX(1, n - 1)].density + particles[IX(0, n - 2)].density);

        particles[IX(n - 1, 0)].u = 0.33f * (particles[IX(n - 2, 0)].u + particles[IX(n - 1, 1)].u);
        particles[IX(n - 1, 0)].v = 0.33f * (particles[IX(n - 2, 0)].v + particles[IX(n - 1, 1)].v);
        particles[IX(n - 1, 0)].density = 0.33f * (particles[IX(n - 2, 0)].density + particles[IX(n - 1, 1)].density);

        particles[IX(n - 1, n - 1)].u = 0.33f * (particles[IX(n - 2, n - 1)].u + particles[IX(n - 1, n - 2)].u);
        particles[IX(n - 1, n - 1)].v = 0.33f * (particles[IX(n - 2, n - 1)].v + particles[IX(n - 1, n - 2)].v);
        particles[IX(n - 1, n - 1)].density = 0.33f * (particles[IX(n - 2, n - 1)].density + particles[IX(n - 1, n - 2)].density);

    }
    
}
__global__ void fluidLinSolve(int b, fluid_particle* particles, float a, float c) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int n = GRID_SIZE;
    float cRecip = 1.0f / c;

    if (idx > 0 && idx < n - 1 && idy > 0 && idy < n - 1) {
        int index = IX(idx, idy);
        if(b == 1) particles[index].u = (particles[index].u_prev + a * (particles[IX(idx + 1, idy)].u + particles[IX(idx - 1, idy)].u +
            particles[IX(idx, idy + 1)].u + particles[IX(idx, idy - 1)].u)) * cRecip;
        if(b == 2) particles[index].v = (particles[index].v_prev + a * (particles[IX(idx + 1, idy)].v + particles[IX(idx - 1, idy)].v +
            particles[IX(idx, idy + 1)].v + particles[IX(idx, idy - 1)].v)) * cRecip;
        if(b == 0) particles[index].density = (particles[index].density_prev + a * (particles[IX(idx + 1, idy)].density + particles[IX(idx - 1, idy)].density +
            particles[IX(idx, idy + 1)].density + particles[IX(idx, idy - 1)].density)) * cRecip;
    }
}
__global__ void fluidAdvect(int b, fluid_particle* particles, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int n = GRID_SIZE;

    if (idx > 0 && idx < n - 1 && idy > 0 && idy < n - 1) {
        int index = IX(idx, idy);

        float dtx = dt * (n - 2);
        float dty = dt * (n - 2);

        // Use previous velocity for advection
        float x = idx - dtx * particles[index].u_prev;
        float y = idy - dty * particles[index].v_prev;

        if (x < 0.5f) x = 0.5f;
        if (x > n - 1.5f) x = n - 1.5f;
        int i0 = (int)x;
        int i1 = i0 + 1;

        if (y < 0.5f) y = 0.5f;
        if (y > n - 1.5f) y = n - 1.5f;
        int j0 = (int)y;
        int j1 = j0 + 1;

        float s1 = x - i0;
        float s0 = 1.0f - s1;
        float t1 = y - j0;
        float t0 = 1.0f - t1;
    
        if (b == 0) { // Density
            particles[index].density = s0 * (t0 * particles[IX(i0, j0)].density_prev + t1 * particles[IX(i0, j1)].density_prev) +
                s1 * (t0 * particles[IX(i1, j0)].density_prev + t1 * particles[IX(i1, j1)].density_prev);
        }
        else if (b == 1) { // Velocity U
            particles[index].u = s0 * (t0 * particles[IX(i0, j0)].u_prev + t1 * particles[IX(i0, j1)].u_prev) +
                s1 * (t0 * particles[IX(i1, j0)].u_prev + t1 * particles[IX(i1, j1)].u_prev);
        }
        else if (b == 2) { // Velocity V
            particles[index].v = s0 * (t0 * particles[IX(i0, j0)].v_prev + t1 * particles[IX(i0, j1)].v_prev) +
                s1 * (t0 * particles[IX(i1, j0)].v_prev + t1 * particles[IX(i1, j1)].v_prev);
        }
    }
}
__global__ void fluidProjectStep1(fluid_particle* particles, fluid_particle* temp) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int n = GRID_SIZE;

    if (idx > 0 && idx < n - 1 && idy > 0 && idy < n - 1) {
        int index = IX(idx, idy);

        // Store divergence in temp buffer's density field
        temp[index].density = -0.5f * (particles[IX(idx + 1, idy)].u - particles[IX(idx - 1, idy)].u +
            particles[IX(idx, idy + 1)].v - particles[IX(idx, idy - 1)].v) / n;

        // Initialize pressure to zero in temp buffer's u field
        temp[index].u = 0.0f; // Pressure
    }
}
__global__ void fluidProjectStep2(fluid_particle* particles, fluid_particle* temp) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int n = GRID_SIZE;

    if (idx > 0 && idx < n - 1 && idy > 0 && idy < n - 1) {
        int index = IX(idx, idy);

        // Subtract pressure gradient using temp buffer
        particles[index].u -= 0.5f * (temp[IX(idx + 1, idy)].u - temp[IX(idx - 1, idy)].u) * n;
        particles[index].v -= 0.5f * (temp[IX(idx, idy + 1)].u - temp[IX(idx, idy - 1)].u) * n;
    }
}


extern "C" {
    void fluidUpdateWithCuda(fluid_particle* particles, fluid_particle* temp, float dt) {

        dim3 blockDim(16, 16);
        dim3 gridDim((GRID_SIZE + blockDim.x - 1) / blockDim.x,
	        (GRID_SIZE + blockDim.y - 1) / blockDim.y);

        int centerX = GRID_SIZE / 2;
        int centerY = GRID_SIZE / 2;
        fluidAddDensity <<<gridDim, blockDim >>> (particles, centerX, centerY, 100.0f * dt);
        fluidAddVelocity <<<gridDim, blockDim >>> (particles, centerX, centerY, 0.0f, 50.0f * dt);

        fluidCopyToPrev <<<gridDim, blockDim >>> (particles);
        cudaDeviceSynchronize();

        // Velocity step - diffusion
        for (int i = 0; i < ITER; i++) {
            fluidLinSolve <<<gridDim, blockDim >>> (1, particles, VISC * dt * (GRID_SIZE - 2) * (GRID_SIZE - 2), 1 + 6 * VISC * dt * (GRID_SIZE - 2) * (GRID_SIZE - 2));
            cudaDeviceSynchronize();
            fluidSetBounds <<<gridDim, blockDim >>> (1, particles);
            cudaDeviceSynchronize();

            fluidLinSolve <<<gridDim, blockDim >>> (2, particles, VISC * dt * (GRID_SIZE - 2) * (GRID_SIZE - 2), 1 + 6 * VISC * dt * (GRID_SIZE - 2) * (GRID_SIZE - 2));
            cudaDeviceSynchronize();
            fluidSetBounds <<<gridDim, blockDim >>> (2, particles);
            cudaDeviceSynchronize();
        }

        // PROJECTION STEP 1: Make velocity field incompressible
        fluidProjectStep1 << <gridDim, blockDim >> > (particles, temp);
        cudaDeviceSynchronize();

        // Solve for pressure
        for (int i = 0; i < ITER; i++) {
            fluidLinSolve << <gridDim, blockDim >> > (0, temp, 1.0f, 6.0f); // Solve pressure in temp buffer
            cudaDeviceSynchronize();
            fluidSetBounds << <gridDim, blockDim >> > (0, temp);
            cudaDeviceSynchronize();
        }

        fluidProjectStep2 << <gridDim, blockDim >> > (particles, temp);
        cudaDeviceSynchronize();
        fluidSetBounds << <gridDim, blockDim >> > (1, particles);
        fluidSetBounds << <gridDim, blockDim >> > (2, particles);
        cudaDeviceSynchronize();

        fluidCopyToPrev <<<gridDim, blockDim >>> (particles);
        cudaDeviceSynchronize();

        // Advect velocity
        fluidAdvect <<<gridDim, blockDim >>> (1, particles, dt);
        cudaDeviceSynchronize();
        fluidAdvect <<<gridDim, blockDim >>> (2, particles, dt);
        cudaDeviceSynchronize();
        fluidSetBounds <<<gridDim, blockDim >>> (1, particles);
        fluidSetBounds <<<gridDim, blockDim >>> (2, particles);
        cudaDeviceSynchronize();

        // Density step - diffusion
        fluidCopyToPrev <<<gridDim, blockDim >>> (particles);
        fluidProjectStep1 << <gridDim, blockDim >> > (particles, temp);
        cudaDeviceSynchronize();


        for (int i = 0; i < ITER; i++) {
            fluidLinSolve <<<gridDim, blockDim >>> (0, particles, DIFF * dt * (GRID_SIZE - 2) * (GRID_SIZE - 2), 1 + 6 * DIFF * dt * (GRID_SIZE - 2) * (GRID_SIZE - 2));
            cudaDeviceSynchronize();
            fluidSetBounds <<<gridDim, blockDim >>> (0, particles);
            cudaDeviceSynchronize();
        }

        // Advect density
        fluidCopyToPrev <<<gridDim, blockDim >>> (particles);
        cudaDeviceSynchronize();
        fluidAdvect <<<gridDim, blockDim >>> (0, particles, dt);
        cudaDeviceSynchronize();
        fluidSetBounds <<<gridDim, blockDim >>> (0, particles);
        cudaDeviceSynchronize();

        //updateVertices <<<gridDim, blockDim >>> (particles, dt);
    }
    void fluidAddDensityWithCuda(fluid_particle* particles, int sourceX, int sourceY, float amount) {
        dim3 blockDim(16, 16);
        dim3 gridDim((GRID_SIZE + blockDim.x - 1) / blockDim.x,
            (GRID_SIZE + blockDim.y - 1) / blockDim.y);
        fluidAddDensity << <gridDim, blockDim >> > (particles, sourceX, sourceY, amount);      
    }
    void fluidAddVelocityWithCuda(fluid_particle* particles, int sourceX, int sourceY, float amountX, float amountY) {
        dim3 blockDim(16, 16);
        dim3 gridDim((GRID_SIZE + blockDim.x - 1) / blockDim.x,
            (GRID_SIZE + blockDim.y - 1) / blockDim.y);
        fluidAddVelocity << <gridDim, blockDim >> > (particles, sourceX, sourceY, amountX, amountY);
    }
}


