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
#include <set>




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
	vk::Queue presentQueue;

	// surface creation
	vk::SurfaceKHR surface;

	
	/* Member structs */
	
	// vulkan creation
	vk::InstanceCreateInfo createInfo{};

	// physical dvice
	struct QueueFamilyIndeces {
		std::optional<uint32_t> graphicsFamily; // std optional is a wrapper that contains no value until ou assign something to it.
		std::optional<uint32_t> presentFamily;
		// generic check inline
		bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};
	

	
	vk::PhysicalDeviceFeatures deviceFeatures{};

	// logical device
	vk::DeviceCreateInfo deviceCreateInfo{};

	// swap chains
	const std::vector<const char*> deviceExtension = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
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

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && deviceFeatures.geometryShader && indices.isComplete() && extensionsSupported;
	}

	bool checkDeviceExtensionSupport(vk::PhysicalDevice device)
	{

		uint32_t extensionCount;
		 
		if (device.enumerateDeviceExtensionProperties(nullptr, &extensionCount, nullptr) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to enumerate device extension properties");
		}

		std::vector<vk::ExtensionProperties> availableExtensions(extensionCount);
		if (device.enumerateDeviceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to enumerate device extension properties with available extension data");
		}

		std::set<std::string> requiredExtensions(deviceExtension.begin(), deviceExtension.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	/* 3. Queue families for physical device */
	QueueFamilyIndeces findQueueFamilies(vk::PhysicalDevice device) {
		QueueFamilyIndeces indices;

		uint32_t queueFamilyCount = 0;
		std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

		vk::Bool32 presentSupport = false; 
		
		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			presentSupport = device.getSurfaceSupportKHR(i, surface);
			
			// Regarding graphics families
			if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
			{					
				indices.graphicsFamily = i;
			}
			
			 //Regarding surface families
			if (presentSupport)
			{
				indices.presentFamily = i;
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

		// Create struct for both presentation and family queue indices:

		std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() }; // ensure that each queue family index is stored only once.

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			vk::DeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = vk::StructureType::eDeviceQueueCreateInfo; 
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}
		

		vk::PhysicalDeviceFeatures deviceFeatures = physicalDevice.getFeatures();

		// device create info
		deviceCreateInfo.sType = vk::StructureType::eDeviceCreateInfo;
		deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
		deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

		std::vector<vk::ExtensionProperties> deviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();

		// extensions and layers
		deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());

		// extract the names to const char*
		std::vector<const char*> extensionNames;
		for (const auto& extension : deviceExtensions)
		{
			extensionNames.push_back(extension.extensionName);
		}
		deviceCreateInfo.ppEnabledExtensionNames = extensionNames.data();

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
		presentQueue = logicalDevice.getQueue(indices.presentFamily.value(), 0);
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