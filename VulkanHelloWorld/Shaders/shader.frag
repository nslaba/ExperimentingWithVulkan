#version 450

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	float time;
	float aspectRatio;
} ubo;
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

const int MAX_PARTICLES = 1000;


struct Particle {
	vec2 position;
	vec3 color;
	float size;
	float speed;
};

Particle particles[MAX_PARTICLES];

// rgb2hsv & hsv2rgb are taken from: https://www.shadertoy.com/view/lt3GWj
vec3 rgb2hsv(vec3 c)
{
	vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
	vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void initParticles() {
	for (int i = 0; i < MAX_PARTICLES; i ++){
		
		
	}

}

void main() {
	vec3 texColor = texture(texSampler, fragTexCoord).rgb;
	//if (texColor.z > 0.1) {
	//	texColor = mix(texColor, texColor.zxy, ubo.time*0.1);
	//}
	// Trying
	float angle = ubo.time * 0.03;
	vec2 normalizedCoord = texColor.xy / ubo.aspectRatio * 2.0 - 1.0;
	for (float i = 0.0; i < 128; i++){
		normalizedCoord = abs(normalizedCoord);
		normalizedCoord -= 0.5;
		normalizedCoord *= 1.03;
		normalizedCoord *= mat2(
			cos(angle), -sin(angle),
			sin(angle), cos(angle)
		);
	
	}
	float len = length(normalizedCoord);

	outColor = vec4((len), ((length(normalizedCoord + vec2(0.2, -0.3)))), ((length(normalizedCoord + vec2(-0.4, -0.1)))), 1.0);
	
	//outColor = min(mix(vec4(texColor, 1.0), color, ubo.time), vec4(0.8, 0.8, 0.8, 1.0));
	
	
}