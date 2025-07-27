# FluidSimulator‑GL‑CUDA

**OpenGL + CUDA Fluid Simulation Demo**

This project demonstrates real-time fluid rendering and computation by combining OpenGL visualization with CUDA-accelerated computation in a C++ application. It simulates fluid behavior using the principles described in **Mike Ash’s “Fluid Simulation for Dummies”**, based on simplified **Navier-Stokes equations**.

This simulation implements a grid-based fluid solver inspired by [Mike Ash’s article: “Fluid Simulation for Dummies”](https://mikeash.com/pyblog/fluid-simulation-for-dummies.html).  
It solves the **incompressible Navier-Stokes equations**. 

##  Features

-  **CUDA-accelerated** fluid simulation on uniform 2D/3D grids (the z component is set to 0, but you can easily modify it under the 3D following Mike Ash's article). 
-  **OpenGL real-time rendering** of fluid density or particles
-  **CUDA–OpenGL interop**: Zero-copy memory sharing between simulation and visualization
-  **Modular C++ structure** for easy experimentation with simulation components

 ##  Prerequisites

- **CUDA Toolkit** (ensure `nvcc` and libs are set)
- **GLFW**, **GLAD**, and **GLM** installed or included

  ##  Build & Run

1. Clone the repository:
   ```bash
   git clone https://github.com/HrachSar/Fluid-Simulator-OpenGL-CUDA.git
   cd Fluid-Simulator-OpenGL-CUDA/CudaTest/FluidSimulator-GL-CUDA
