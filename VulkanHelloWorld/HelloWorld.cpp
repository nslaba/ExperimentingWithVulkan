#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw3native.h>
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
	vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
	
	// validation layers
#ifdef NDEBUG
	const bool enableValidationLayers = true;
#else
	const bool enableValidationLayers = false;
#endif

	// logical device
	vk::Device logicalDevice;
	vk::Queue graphicsQueue;

	// surface creation
	vk::SurfaceKHR surface;

	
	/* Member structs */
	
	// vulkan creation
	vk::InstanceCreateInfo createInfo{};

	// physical dvice
	struct QueueFamilyIndeces {
		std::optional<uint32_t> graphicsFamily; // std optional is a wrapper that contains no value until ou assign something to it.
		
		// generic check inline
		bool isComplete() {
			return graphicsFamily.has_value();
		}
	};
	

	
	vk::PhysicalDeviceFeatures deviceFeatures{};

	// logical device
	vk::DeviceCreateInfo deviceCreateInfo{};


	/* Member functions */
	void initWindow() {
		
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", nullptr, nullptr);
	}


	void initVulkan() {

		// 1. INIT
		createInstance();

		// 2. Create Surface
		createSurface();

		// 3. Find physical device
		pickPhysicalDevice();

		// 4. Find logical device (to interface with physical)
		createLogicalDevice();
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

		// create instance && throw run time error if unsuccessful
		if (vk::createInstance(&createInfo, nullptr, &instance) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create instance!");
		}
	}
	
	/* 2. Create Surface (vulkan object) */
	void createSurface()
	{
		vk::Win32SurfaceCreateInfoKHR createSurfaceInfo{};
		createSurfaceInfo.sType = vk::StructureType::eWin32SurfaceCreateInfoKHR;
		createSurfaceInfo.hwnd = glfwGetWin32Window(window);
		createSurfaceInfo.hinstance = GetModuleHandle(nullptr);

		vk::SurfaceKHR surface;

		try {
			surface = instance.createWin32SurfaceKHR(createSurfaceInfo);
		}
		catch (vk::SystemError& err)
		{
			throw std::runtime_error("Failed to create a surface");
		}

	}

	/* 3. Pick physical device */
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

		//vk::PhysicalDeviceFeatures deviceFeatures;
		deviceFeatures = device.getFeatures();

		// deal with queues
		QueueFamilyIndeces indices = findQueueFamilies(device);

		return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && deviceFeatures.geometryShader && indices.graphicsFamily.has_value();
	}

	/* 3. Queue families for physical device */
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

	/* 4. Find logical device */
	void createLogicalDevice()
	{
		QueueFamilyIndeces indices = findQueueFamilies(physicalDevice);
		vk::PhysicalDeviceFeatures deviceFeatures = physicalDevice.getFeatures();

		// Have to create a local var
		vk::DeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = vk::StructureType::eDeviceQueueCreateInfo;
		queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
		queueCreateInfo.queueCount = 1;

		// Priorities
		float queuePriority = 1.0f;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		// device create info
		deviceCreateInfo.sType = vk::StructureType::eDeviceCreateInfo;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

		// extensions and layers
		deviceCreateInfo.enabledExtensionCount = 0;
		if (enableValidationLayers)
		{
			//ignore for now
			//deviceCreateInfo.enabledLayerCount = static_cast<uint_32_t>(validationLayers.size());
		}


		// create logical device		
		try
		{
			logicalDevice = physicalDevice.createDevice(deviceCreateInfo);
		}
		catch (vk::SystemError& err)
		{
			throw std::runtime_error("Failed to create a logical device");
		}

		// If successful, retrieve queue
		graphicsQueue = logicalDevice.getQueue(indices.graphicsFamily.value(), 0);
	}
	
	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {

		logicalDevice.destroy();

		instance.destroySurfaceKHR(surface, nullptr);

		instance.destroy(nullptr);
		
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