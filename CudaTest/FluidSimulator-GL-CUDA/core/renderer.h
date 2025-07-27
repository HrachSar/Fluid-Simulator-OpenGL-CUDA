#ifndef RENDERER_H
#define RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "../resources/resource_manager.h"
#include <vector>
#include <cuda_runtime.h>

#define VISC 0.0001f
#define DIFF 0.0001f
#define ITER 4

const float FOV = 45.0f;
const int GRID_SIZE = 256;
const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 800;

struct fluid_particle {
    float x, y, z; // Position
    float u, v;   //  velocity
    float density; // Density
    float u_prev, v_prev; // Previous velocity
    float density_prev; // Previous density
};

const int STRIDE = sizeof(fluid_particle);

class renderer {
public:
    unsigned int m_VBO, m_VAO, m_EBO; //Buffers
    glm::mat4 m_model, m_view, m_projection; // matrices
    std::string m_name;
    shader m_shader;
    char* m_vertexShader; // path to vertex shader
    char* m_fragmentShader; //path to fragment shader
    char* m_texture;
    std::vector<fluid_particle> m_particles; 
    fluid_particle* d_temp_particles; //Previous state particles -> pass to GPU as temp

    renderer(std::string name, char* vShader, char* fShader, char* texture = nullptr, glm::mat4 model = glm::mat4(1.0f), glm::mat4 view = glm::mat4(1.0f), glm::mat4 projection = glm::mat4(1.0f));
    ~renderer();

    void init();
    void render();
    void free();
    void setViewMatrix(glm::mat4 matrix);
    void initFluidBuffersTemp();
};



#endif //RENDERER_H
