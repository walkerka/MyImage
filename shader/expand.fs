#version 330
uniform sampler2D colorTex;
in vec2 texcoord;
out vec4 FragColor;

void main(void)
{
    vec4 c = texture(colorTex, texcoord);
    if (c.a == 0.0)
    {
        // find nearest solid color

        vec4 n = textureOffset(colorTex, texcoord, ivec2(-1,0));
        if (n.a > 0.0)
        {
            FragColor = n;
            return;
        }
        n = textureOffset(colorTex, texcoord, ivec2(1,0));
        if (n.a > 0.0)
        {
            FragColor = n;
            return;
        }
        n = textureOffset(colorTex, texcoord, ivec2(0,-1));
        if (n.a > 0.0)
        {
            FragColor = n;
            return;
        }
        n = textureOffset(colorTex, texcoord, ivec2(0,1));
        if (n.a > 0.0)
        {
            FragColor = n;
            return;
        }
    }
    FragColor = c;
}
