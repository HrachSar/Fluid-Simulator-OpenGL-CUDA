#include "renderer.h"

renderer::renderer(std::string name, char* vShader, char* fShader, char* texture, glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
    m_name = name;
    m_vertexShader = vShader;
    m_fragmentShader = fShader;
    m_texture = texture;
    m_model = model;
    m_view = view;
    m_projection = projection;
}
renderer::~renderer() {
    free();
}

void renderer::init() {

	m_particles.resize(GRID_SIZE * GRID_SIZE);
    for (int y = 0; y < GRID_SIZE; ++y) {
        for (int x = 0; x < GRID_SIZE; ++x) {
			int index = y * GRID_SIZE + x;

            float u = x / float(GRID_SIZE - 1); //conver x and y from 0 to 1
            float v = y / float(GRID_SIZE - 1);
            
            float xpos = u * 2.0f - 1.0f; //convert to ndc -> [-1, 1]
            float ypos = v * 2.0f - 1.0f;

            m_particles[index] = {
				xpos, ypos, 0.0f, // Position
				0.0f, 0.0f, // Velocity
				1.0f, // Density
				0.0f, 0.0f, // Previous velocity
				1.0f // Previous density
            };

        }
    }

    resource_manager::loadShader(m_vertexShader, m_fragmentShader, nullptr, m_name);
    if (m_texture != nullptr)
        resource_manager::loadTexture(m_texture, true, m_name);

    m_shader = resource_manager::getShader(m_name);
    m_shader.use();
    m_projection = glm::perspective(glm::radians(FOV), (float) WIDTH / (float) HEIGHT, 0.1f, 100.0f);
    m_shader.setMat4x4("projection", m_projection);

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, m_particles.size() * STRIDE, m_particles.data(), GL_DYNAMIC_DRAW);
    //Position 
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, STRIDE, (void*)(offsetof(fluid_particle, x)));
    glEnableVertexAttribArray(0);
	//Velocity
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, STRIDE, (void*)(offsetof(fluid_particle, u)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    initFluidBuffersTemp();
}

void renderer::render() {

    m_shader.use();
    //resource_manager::getTexture(m_name).bind(); //use if texture exists.
    m_shader.setMat4x4("model", m_model); //set vertex shader uniforms.
    m_shader.setMat4x4("view", m_view);
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_POINTS, 0, GRID_SIZE*GRID_SIZE);
}

void renderer::initFluidBuffersTemp() {
    size_t size = GRID_SIZE * GRID_SIZE * sizeof(fluid_particle);
    cudaMalloc(&d_temp_particles, size);

    cudaMemset(d_temp_particles, 0, size);
}

void renderer::setViewMatrix(glm::mat4 matrix) {
    m_view = matrix;
}

void renderer::free() {

    if (d_temp_particles) {
        cudaFree(d_temp_particles);
        d_temp_particles = nullptr;
    }

	m_particles.clear();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glDeleteVertexArrays(1, &m_VAO);
    glDeleteBuffers(1, &m_VBO);
    resource_manager::clear();
}
