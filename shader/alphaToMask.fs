#version 330
uniform sampler2D colorTex;
uniform vec4 screenSize;
in vec2 texcoord;
out vec4 FragColor;
void main(void)
{
    FragColor = vec4(texture(colorTex, texcoord).aaa,1.0);
}
