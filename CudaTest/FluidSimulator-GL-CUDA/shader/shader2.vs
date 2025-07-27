#version 330 core
layout (location = 1) in vec3 aPos;

void main(){
    gl_Position = vec4(aPos.x + 0.5f, aPos.yz, 1.0f);
}