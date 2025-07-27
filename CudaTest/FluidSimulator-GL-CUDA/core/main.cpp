#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_gl_interop.h>

#include "renderer.h"
#include "../resources/resource_manager.h"
#include "../camera/camera.h"
#include "../cuda/cuda_manager.h"
#include  "../cuda/kernel.cuh"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void process_input(GLFWwindow* window);

camera _camera(glm::vec3(0.0f, 0.0f, 10.0f));

renderer particleRenderer("cube", "shader/shader.vertex", "shader/shader.frag",
            "gfx/awesomeface.png", glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f));

float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;


float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "GL_example", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetWindowSizeCallback(window, framebuffer_size_callback);

   //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);

    particleRenderer.init();
	cuda_manager cudaManager;
	cudaGraphicsGLRegisterBuffer(&cudaManager.m_cudaVBOResource, particleRenderer.m_VBO, cudaGraphicsRegisterFlagsNone);

    while (!glfwWindowShouldClose(window)) {

        process_input(window);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		fluid_particle* d_ptr = cudaManager.mapResources();
       
        float currentTime = glfwGetTime();
        deltaTime = currentTime - lastFrame;
        lastFrame = currentTime;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            float x_ndc = (float(xpos) / WIDTH) * 2.0f - 1.0f;
            float y_ndc = 1.0f - (float(ypos) / HEIGHT) * 2.0f;

            int grid_x = (int)(((x_ndc + 1.0f) / 2.0f) * (GRID_SIZE - 1));
            int grid_y = (int)(((y_ndc + 1.0f) / 2.0f) * (GRID_SIZE - 1));

            grid_x = glm::clamp(grid_x, 0, GRID_SIZE - 1);
            grid_y = glm::clamp(grid_y, 0, GRID_SIZE - 1);

            if (d_ptr) {

                fluidAddDensityWithCuda(d_ptr, grid_x, grid_y, 10);
                fluidAddVelocityWithCuda(d_ptr, grid_x, grid_y, 10, 10);
                cudaDeviceSynchronize();
            }
        }

        if (d_ptr)
            fluidUpdateWithCuda(d_ptr, particleRenderer.d_temp_particles, deltaTime);
        else
			std::cout << "Failed to map CUDA resources." << std::endl;

        particleRenderer.setViewMatrix(_camera.getViewMatrix());
        particleRenderer.render();

		cudaManager.unmapResources();

        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    cudaDeviceSynchronize();
    cudaManager.free();
    particleRenderer.free();
    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
}

void process_input(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        _camera.processKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        _camera.processKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        _camera.processKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        _camera.processKeyboard(RIGHT, deltaTime);
}
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xOffset = xpos - lastX;
    float yOffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

   //_camera.processMouseMovement(xOffset, yOffset);
}
