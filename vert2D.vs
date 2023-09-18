#version 330 core
layout (location = 0) in vec4 vertex;

out vec2 TexCoords;

uniform mat4 projection2D;
uniform mat4 model2D;

//mat4 m = mat4(1.0);
//mat4 p = mat4(1.0);

void main()
{
	TexCoords = vertex.zw;
	gl_Position = projection2D * model2D * vec4(vertex.xy, 0.0f, 1.0f);
	//gl_Position = p * m * vec4(vertex.xy, 0.0f, 1.0f);
}
