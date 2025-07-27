cmake_minimum_required(VERSION 3.28)
project(GL_example)

set(OpenGlLinkers -lglfw -lglut -lGLU -lGL -lGLEW -lm -lX11 -lpthread -lXi -lXrandr -ldl)

add_executable(GL_example
        core/main.cpp
        core/glad.c
        shader/shader.cpp
        shader/shader.h
        resources/resource_manager.cpp
        resources/resource_manager.h
        textures/texture.cpp
        textures/texture.h
        core/renderer.cpp
        core/renderer.h
        camera/camera.cpp
        camera/camera.h
)

target_link_libraries(GL_example ${OpenGlLinkers})