#version 330 core
out vec4 FragColor;

uniform sampler2D m_texture;
in vec2 velocity;

void main(){
    vec2 vel = velocity;
    float mag = length(vel) * 10.0f;

    vec3 color = vec3(0.5f);
    color.r = clamp(abs(vel.x) * 100.0f, 0.0f, 1.0f);
    color.g = clamp(abs(vel.y) * 100.0f, 0.0f, 1.0f);
    color.b = clamp(mag, 0.0f, 1.0f);

    FragColor = vec4(color, 1.0); // Default color
}