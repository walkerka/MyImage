#version 330
uniform sampler2D colorTex;
uniform sampler2D alphaTex;
in vec2 texcoord;
out vec4 FragColor;

void main(void)
{
    vec4 c = texture(colorTex, texcoord);
    float a = texture(alphaTex, texcoord).r;
    FragColor = vec4(c.rgb,a);
}
