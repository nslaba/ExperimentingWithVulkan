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
#include <optional>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class HelloTriangleApplication {

public:
	
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}
	
private:
	/* Member vars */
	GLFWwindow* window;
	vk::Instance instance;
	vk::InstanceCreateInfo createInfo{};
	vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;

	/* Member structs */
	struct QueueFamilyIndeces {
		std::optional<uint32_t> graphicsFamily; // std optional is a wrapper that contains no value until ou assign something to it.
		
		// generic check inline
		bool isComplete() {
			return graphicsFamily.has_value();
		}
	};


	/* Member functions */
	void initWindow() {
		
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", nullptr, nullptr);
	}


	void initVulkan() {

		// 1. INIT
		createInstance();

		// 2. Find physical device
		pickPhysicalDevice();

		// 3. Find logical device
		
	}
	
	/* 1. INIT VULKAN */
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

		//vk::Result result = vk::createInstance(&createInfo, nullptr, &instance);

		// check if successful
		if (vk::createInstance(&createInfo, nullptr, &instance) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	/* 2. Pick physical device */
	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		auto devices = instance.enumeratePhysicalDevices();
		if (!devices.empty()) {
			physicalDevice = devices[0];
		}
		else {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		// If successfully found, then allocate an array to hold all of the vk physical device handles
		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}

	}

	
	// Physical device helper func
	bool isDeviceSuitable(vk::PhysicalDevice device) {
		vk::PhysicalDeviceProperties deviceProperties;
		deviceProperties = device.getProperties();

		vk::PhysicalDeviceFeatures deviceFeatures;
		deviceFeatures = device.getFeatures();

		// deal with queues
		QueueFamilyIndeces indices = findQueueFamilies(device);

		return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && deviceFeatures.geometryShader && indices.graphicsFamily.has_value();
	}

	/* 3. Find logical device */
	QueueFamilyIndeces findQueueFamilies(vk::PhysicalDevice device) {
		QueueFamilyIndeces indices;

		uint32_t queueFamilyCount = 0;
		std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
			{
				indices.graphicsFamily = i;
			}
			// Early exit
			if (indices.isComplete()) break;

			i++;
		}
		return indices;
	}


	
	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		instance.destroy();
		
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