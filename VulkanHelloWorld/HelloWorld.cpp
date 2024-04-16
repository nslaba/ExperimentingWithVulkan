#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#include <vulkan/vulkan.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "vec4.hpp"
#include "mat4x4.hpp"

#include <iostream>
#include <stdexcept>
#include <cstdlib> // provides exit success and exit failure macros

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class HelloTriangleApplication {

public:
	GLFWwindow* window;
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}
	
private:
	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", nullptr, nullptr);
	}


	void initVulkan() {
		createInstance();
	}
	vk::Instance instance;
	vk::InstanceCreateInfo createInfo{};
	void createInstance() {
		vk::ApplicationInfo appInfo{};
		appInfo.sType = vk::StructureType::eApplicationInfo;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		createInfo.sType = vk::StructureType::eInstanceCreateInfo;
		createInfo.pApplicationInfo = &appInfo;

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;

		createInfo.enabledLayerCount = 0;

		vk::Result result = vk::createInstance(&createInfo, nullptr, &instance);
	}

	
	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		glfwDestroyWindow(window);

		glfwTerminate();
	}
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	
	return EXIT_SUCCESS;


	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

	std::cout << extensionCount << " extensions supported\n";

	glm::mat4 matrix;
	glm::vec4 vec;

	auto test = matrix * vec;

	return 0;
}