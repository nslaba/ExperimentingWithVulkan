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

// swap extend
#include <cstdint> // for uint32_t
#include <limits> // for std::numeric_limits
#include <algorithm> // for std::clamp

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <glm.hpp> // library used for lin alg related types
#include <array>


#define GLM_FORCE_RADIANS // forces using radians
#include <gtc/matrix_transform.hpp>

#include <chrono> // timekeeping

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	//  mem function to populate input binding description structure
	static vk::VertexInputBindingDescription getBindingDescription() {
		vk::VertexInputBindingDescription bindingDescription{}; // describes at which rate to load data from memory throughout the vertices
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = vk::VertexInputRate::eVertex;

		return bindingDescription;
	}



	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);
							
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
		attributeDescriptions[1].offset = offsetof(Vertex, color);
							
		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}

};

// indices to represent the contents of the index buffer
const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0 // order of rectangle vertices
};

const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
}; // This is known as 'interleaving vertex attributes'


struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	float time;
	float aspectRatio;
};

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2; //to allow multiple frames to be in flight at once

// Validation layers
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


// Proxy function to help with debug msg callback
vk::Result CreateDebugUtilsMessengerEXT(vk::Instance instance, const vk::DebugUtilsMessengerCreateInfoEXT* pCreateInfo, const vk::AllocationCallbacks* pAllocator, vk::DebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)instance.getProcAddr("vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		VkResult result = func(instance, &pCreateInfo->operator const VkDebugUtilsMessengerCreateInfoEXT & (), &pAllocator->operator const VkAllocationCallbacks & (), reinterpret_cast<VkDebugUtilsMessengerEXT*>(pDebugMessenger));
		return static_cast<vk::Result>(result);
	}
	else {
		return vk::Result::eErrorExtensionNotPresent;
	}
}

void DestroyDebugUtilsMessengerEXT(vk::Instance instance, vk::DebugUtilsMessengerEXT debugMessenger, const vk::AllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT");

	if (func != nullptr) {
		func(instance, debugMessenger, &pAllocator->operator const VkAllocationCallbacks &());
	}

}



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
	vk::DebugUtilsMessengerEXT debugMessenger;
	vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
	vk::DispatchLoaderDynamic dldi;

	// logical device
	vk::Device logicalDevice;
	vk::Queue graphicsAndComputeQueue;
	vk::Queue presentQueue;

	// surface creation
	vk::SurfaceKHR surface;

	// swap chain
	vk::SwapchainKHR swapChain;
	vk::Extent2D swapChainExtent;
	// retrieving swap chain images
	std::vector<vk::Image> swapChainImages;
	vk::Format swapChainImageFormat;
	std::vector<vk::Framebuffer> swapChainFramebuffers;

	// graphics Pipeline
	vk::RenderPass renderPass;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline graphicsPipeline;
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> commandBuffers;

	// compute Pipeline
	vk::DescriptorSetLayout computeDescriptorSetLayout;
	vk::PipelineLayout computePipelineLayout;
	vk::Pipeline computePipeline;

	// Synchronization
	std::vector<vk::Semaphore> imageAvailableSemaphores;
	std::vector<vk::Semaphore> renderFinishedSemaphores;
	std::vector<vk::Fence> inFlightFences;
	bool framebufferResized = false;
	
	// Frames
	uint32_t currentFrame = 0;

	// vertex buffer
	vk::DeviceMemory vertexBufferMemory;
	vk::Buffer vertexBuffer;

	//index buffer
	vk::DeviceMemory indexBufferMemory;
	vk::Buffer indexBuffer;

	// UBOs
	std::vector<vk::Buffer> uniformBuffers;
	std::vector<vk::DeviceMemory> uniformBuffersMemory;
	std::vector<void *> uniformBuffersMapped;

	// descriptors
	vk::DescriptorPool descriptorPool;
	std::vector<vk::DescriptorSet> descriptorSets;

	// Textures
	vk::Image textureImage;
	vk::DeviceMemory textureImageMemory;

	vk::ImageView textureImageView;
	vk::Sampler textureSampler;

	/* Member structs */
	
	// physical dvice
	struct QueueFamilyIndeces {
		// std optional is a wrapper that contains no value until you assign something to it.
		std::optional<uint32_t> graphicsAndComputeFamily;
		std::optional<uint32_t> presentFamily;
		// generic check inline
		bool isComplete() {
			return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
		}
	};
	

	
	

	

	// swap chains
	const std::vector<const char*> deviceExtension = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	struct SwapChainSupportDetails {
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

	// Image views
	std::vector<vk::ImageView> swapChainImageViews;
	

	/* Member functions */
	void initWindow() {
		
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan() {

		// 1. INIT
		createInstance();

		// DEBUGGING
		setupDebugMessenger();

		// 2. Create Surface
		createSurface();

		// 3. Find physical device
		pickPhysicalDevice();

		// 4. Find logical device (to interface with physical)
		createLogicalDevice();

		// 5. Create Swap chains
		createSwapChain();

		// 6. create Image views
		createImageViews();

		// 7. Create Render Pass
		createRenderPass();

		// 8. Create Descriptor layout
		createDescriptorSetLayout();

		// 8. create Graphics Pipeline
		createGraphicsPipeline();

		createComputeDescriptorSetLayout();
		
		// 9. Create Compute Pipeline
		createComputePipeline();
		
		// 9. Create Framebuffer
		createFramebuffers();

		// 10. Create Command pool
		createCommandPool();

		// 11. create Texture image
		createTextureImage();

		// 12. create texture image view
		createTextureImageView();

		// 13. create texture sampler
		createTextureSampler();

		// 11. Create Vertex Buffer
		createVertexBuffer();

		// 12. Create Index Buffer
		createIndexBuffer();

		// 13. Create Uniform Buffers
		createUniformBuffers();

		// 14. Create Descriptor pools
		createDescriptorPool();

		// 15. Create descriptor sets
		createDescriptorSets();

		// 13. Create Command Buffer
		createCommandBuffers();

		// 14. Create Synhronization Objects
		createSyncObjects();
	}

	// validation layers
	// checks if all the requested layers are available
	bool checkValidationLayerSupport() {
		std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
		

		// check if all the layers exist
		for (const char* layerName : validationLayers) {
			bool layerFound = false;
		
			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}
		
			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	// validation layer callback
	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	static VKAPI_ATTR vk::Bool32 debugCallback(
		vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		vk::DebugUtilsMessageTypeFlagsEXT messageType,
		const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) 
	{
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
		return vk::False;
	}

	
	/* 1. INIT VULKAN */
	void createInstance() {

		// validation layers
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}

		vk::ApplicationInfo appInfo{};
		appInfo.sType = vk::StructureType::eApplicationInfo;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// vulkan creation
		vk::InstanceCreateInfo createInfo{};
		createInfo.sType = vk::StructureType::eInstanceCreateInfo;
		createInfo.pApplicationInfo = &appInfo;

		vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

		// validation layers
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (vk::DebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();
		

		// create instance && throw run time error if unsuccessful
		try {
			instance = vk::createInstance(createInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create a vulkan instance!" + std::string(err.what()));
		}

		// checking extensions
		
		std::vector<vk::ExtensionProperties> generalExtensions = vk::enumerateInstanceExtensionProperties();
		for (vk::ExtensionProperties& ext : generalExtensions) {
			std::cout << "Extension Name: " << ext.extensionName << "\n";
			std::cout << "Extension Version: " << ext.specVersion << "\n";
		}

	}

	void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo) {
		//createInfo = {};
		createInfo.sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT;
		createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError; // what you'd like the call back to be called for
		createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
		createInfo.pfnUserCallback = reinterpret_cast<PFN_vkDebugUtilsMessengerCallbackEXT>(debugCallback);
		createInfo.pUserData = nullptr;

	}


	// DEBUGGING -- debug messenger setup
	void setupDebugMessenger() {
		if (!enableValidationLayers) return;
		
		vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);
		
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to set up debug messenger");
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
		if (devices.empty()) {
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
		vk::PhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures = device.getFeatures();

		// deal with queues
		QueueFamilyIndeces indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);
		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && deviceFeatures.geometryShader && indices.isComplete() && extensionsSupported && swapChainAdequate && deviceFeatures.samplerAnisotropy;
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
			if ((queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) && (queueFamily.queueFlags & vk::QueueFlagBits::eCompute))
			{					
				indices.graphicsAndComputeFamily = i;
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
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() }; // ensure that each queue family index is stored only once.


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
		deviceFeatures.samplerAnisotropy = vk::True; // for texture sampling
		// device create info
		vk::DeviceCreateInfo deviceCreateInfo{};
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
			deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			deviceCreateInfo.enabledLayerCount = 0;
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
		graphicsAndComputeQueue = logicalDevice.getQueue(indices.graphicsAndComputeFamily.value(), 0);
		presentQueue = logicalDevice.getQueue(indices.presentFamily.value(), 0);
	}

	// SWAP CHAINS
	SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device)
	{
		SwapChainSupportDetails details;
		// Capabilities
		details.capabilities = device.getSurfaceCapabilitiesKHR(surface);

		// Formats
		details.formats = device.getSurfaceFormatsKHR(surface);
		assert(!details.formats.empty());
		details.formats[0].format = (details.formats[0].format == vk::Format::eUndefined) ? vk::Format::eB8G8R8A8Srgb : details.formats[0].format; // ensure there's a val

		// Present Modes
		uint32_t presentModeCount;
		assert (device.getSurfacePresentModesKHR(surface, &presentModeCount, nullptr) == vk::Result::eSuccess);
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			assert(device.getSurfacePresentModesKHR(surface, &presentModeCount, details.presentModes.data()) == vk::Result::eSuccess);
		}
		
		return details;
	}

	// Choosing the right swap chain settings helper functions -- optional since I already set it to srgb colorspace if not found

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
	{
		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];

	}

	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
	{
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == vk::PresentModeKHR::eMailbox)
			{
				return availablePresentMode;
			}
		}

		return vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			vk::Extent2D actualExtend = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtend.width = std::clamp(actualExtend.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtend.height = std::clamp(actualExtend.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtend;
		}
	}

	// 5. Create Swap Chain

	void recreateSwapChain() {
		// handle minimization
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		logicalDevice.waitIdle(); // shouldn't touch resources that may still be in use.
		
		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createFramebuffers();
	}

	void cleanupSwapChain() {
		for (auto framebuffer : swapChainFramebuffers) {
		for (auto framebuffer : swapChainFramebuffers) 
			logicalDevice.destroyFramebuffer(framebuffer);
		}
		for (auto imageView : swapChainImageViews)
		{
			logicalDevice.destroyImageView(imageView);
		}

		logicalDevice.destroySwapchainKHR(swapChain);
	}

	void createSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		swapChainExtent = chooseSwapExtent(swapChainSupport.capabilities);
		swapChainImageFormat = surfaceFormat.format;

		// num of images
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		QueueFamilyIndeces indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

		vk::SwapchainCreateInfoKHR createSwapChainInfo{};

		if (indices.presentFamily != indices.graphicsAndComputeFamily)
		{
			createSwapChainInfo.imageSharingMode = vk::SharingMode::eConcurrent;
			createSwapChainInfo.queueFamilyIndexCount = 2;
			createSwapChainInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createSwapChainInfo.imageSharingMode = vk::SharingMode::eExclusive;
			createSwapChainInfo.queueFamilyIndexCount = 0;
			createSwapChainInfo.pQueueFamilyIndices = nullptr;
		}

		createSwapChainInfo.sType = vk::StructureType::eSwapchainCreateInfoKHR;
		createSwapChainInfo.surface = surface;

		createSwapChainInfo.minImageCount = imageCount;
		createSwapChainInfo.imageFormat = swapChainImageFormat;
		createSwapChainInfo.imageColorSpace = surfaceFormat.colorSpace;
		createSwapChainInfo.imageExtent = swapChainExtent;
		createSwapChainInfo.imageArrayLayers = 1;
		createSwapChainInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
		createSwapChainInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createSwapChainInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
		createSwapChainInfo.presentMode = presentMode;
		createSwapChainInfo.clipped = true; // don't care about color of pixels that are obscured.
		createSwapChainInfo.oldSwapchain = nullptr;

		// create swap chain
		try {
			swapChain = logicalDevice.createSwapchainKHR(createSwapChainInfo);
		}
		catch (const vk::SystemError& err)
		{
			throw std::runtime_error("Failed to create swap chain!");
		}

		// retrieve images
		assert(logicalDevice.getSwapchainImagesKHR(swapChain, &imageCount, nullptr) == vk::Result::eSuccess);
		swapChainImages.resize(imageCount);
		assert(logicalDevice.getSwapchainImagesKHR(swapChain, &imageCount, swapChainImages.data()) == vk::Result::eSuccess);
	}


	// 6. Create Image views
	void createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
		}
	}

	// 7. Create Render Pass
	void createRenderPass() {
		// single color buffer attachment
		vk::AttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = vk::SampleCountFlagBits::e1;
		colorAttachment.loadOp = vk::AttachmentLoadOp::eClear; // what to do with data before rendering -- clears framebuffer to black before drawing new frame
		colorAttachment.storeOp = vk::AttachmentStoreOp::eStore; // what to do with data after rendering -- renderred contents will be stored in mem and can be read later
		colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare; // Not using stencil buffer for now
		colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		colorAttachment.initialLayout = vk::ImageLayout::eUndefined; // means we don't care what previous image was in
		colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR; //images to be presented in the swap chain

		// Subpasses and Attachment refs
		vk::AttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0; // since the array consists of a single attachment description and so this is its index
		colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
		
		vk::SubpassDescription subpass{};
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		// subpass dependency
		vk::SubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL; // refers to the implicit subpass before or after the render pass depending on whether it is specified in srcSubpass or dstSubpass.
		dependency.dstSubpass = 0; //index of subpass. dstSubpass must always be higher than srcSubpass to prevent cycles in the dependency graph.
		dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependency.srcAccessMask = vk::AccessFlagBits(0);
		dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

		// create the render pass
		vk::RenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = vk::StructureType::eRenderPassCreateInfo;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		try {
			renderPass = logicalDevice.createRenderPass(renderPassInfo);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to create Render Pass!");
		}

	}

	//8. Create Descriptor set layout
	void createDescriptorSetLayout() {
		vk::DescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
		uboLayoutBinding.pImmutableSamplers = nullptr;

		vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

		std::array<vk::DescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
		

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = vk::StructureType::eDescriptorSetLayoutCreateInfo;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		try {
			descriptorSetLayout = logicalDevice.createDescriptorSetLayout(layoutInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create descriptor set layout!" + std::string(err.what()));
		}

	}

	void createComputeDescriptorSetLayout() {
		std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{};
		layoutBindings[0].binding = 0;
		layoutBindings[0].descriptorCount = 1;
		layoutBindings[0].descriptorType = vk::DescriptorType::eUniformBuffer;
		layoutBindings[0].pImmutableSamplers = nullptr;
		layoutBindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

		layoutBindings[1].binding = 1;
		layoutBindings[1].descriptorCount = 1;
		layoutBindings[1].descriptorType = vk::DescriptorType::eStorageBuffer;
		layoutBindings[1].pImmutableSamplers = nullptr;
		layoutBindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

		layoutBindings[2].binding = 2;
		layoutBindings[2].descriptorCount = 1;
		layoutBindings[2].descriptorType = vk::DescriptorType::eStorageBuffer;
		layoutBindings[2].pImmutableSamplers = nullptr;
		layoutBindings[2].stageFlags = vk::ShaderStageFlagBits::eCompute;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = vk::StructureType::eDescriptorSetLayoutCreateInfo;
		layoutInfo.bindingCount = 3;
		layoutInfo.pBindings = layoutBindings.data();


		try {
			computeDescriptorSetLayout = logicalDevice.createDescriptorSetLayout(layoutInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create compute descriptor set layout! " + std::string(err.what()));
		}
	}

	void createComputePipeline() {
		auto computeShaderCode = readFile("Shaders/compute.spv");

		vk::ShaderModule computeShaderModule = createShaderModule(computeShaderCode);

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{};
		computeShaderStageInfo.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
		computeShaderStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
		computeShaderStageInfo.module = computeShaderModule;
		computeShaderStageInfo.pName = "main";

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = vk::StructureType::ePipelineLayoutCreateInfo;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;

		// create layout
		try {
			computePipelineLayout = logicalDevice.createPipelineLayout(pipelineLayoutInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create compute pipeline layout" + std::string(err.what()));
		}

		vk::ComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = vk::StructureType::eComputePipelineCreateInfo;
		pipelineInfo.layout = computePipelineLayout;
		pipelineInfo.stage = computeShaderStageInfo;

		vk::ResultValue<vk::Pipeline> result = logicalDevice.createComputePipeline(VK_NULL_HANDLE, pipelineInfo);
		if (result.result != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to create the Graphics Pipeline!");
		}
		computePipeline = result.value;


		logicalDevice.destroyShaderModule(computeShaderModule);
	}

	// 8. create graphics pipeline
	void createGraphicsPipeline()
	{
		auto vertShaderCode = readFile("Shaders/vert.spv");
		auto fragShaderCode = readFile("Shaders/frag.spv");
		

		vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);
		

		// Need to assign shaders to pipeline stages
		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
		vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main"; // entrypoint in the glsl code -- could combine multiple shaders with different entry points


		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
		fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		

		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// Vertex input
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vertexInputInfo.sType = vk::StructureType::ePipelineVertexInputStateCreateInfo;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
		


		// Input Assembly
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo;
		inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
		inputAssembly.primitiveRestartEnable = vk::False;

		// Viewport
		vk::Viewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width; // swap chain images will be used as framebuffers later on
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		// Scissor (want to draw to the entire framebuffer
		vk::Rect2D scissor{};
		scissor.offset = vk::Offset2D{ 0, 0 };
		scissor.extent = swapChainExtent;

		// Option to keep dynamic states out of pipeline -- this will cause the config of these vals to be ignored. Required to specify the data at drawing time.
		std::vector<vk::DynamicState> dynamicStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};

		vk::PipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = vk::StructureType::ePipelineDynamicStateCreateInfo;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();


		// THIS HAS TO BE AT PIPELINE CREATION TIME -- setting pViewports and pScissors makes it immutable the way I have it OR I can set that at drawing time.
		vk::PipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = vk::StructureType::ePipelineViewportStateCreateInfo;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		// Rasterizer
		vk::PipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = vk::StructureType::ePipelineRasterizationStateCreateInfo;
		rasterizer.depthClampEnable = vk::False; // if this is true, then fragments beyond near and far planes are clamped -- useful for shadowmaps.
		rasterizer.rasterizerDiscardEnable = vk::False; // of this is true, then geom doesn't pass through rasterizer stage.
		rasterizer.polygonMode = vk::PolygonMode::eFill; // how fragments are generated for geom
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = vk::CullModeFlagBits::eBack;
		rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
		rasterizer.depthBiasEnable = vk::False;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		// Multisampling
		vk::PipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = vk::StructureType::ePipelineMultisampleStateCreateInfo;
		multisampling.sampleShadingEnable = vk::False;
		multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = vk::False;
		multisampling.alphaToOneEnable = vk::False;

		// Depth and stencil testing will go here.

		// Color Blending -- per-Framebuffer struct
		vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		colorBlendAttachment.blendEnable = vk::False;
		colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eOne;
		colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eZero;
		colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
		colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
		colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;

		vk::PipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = vk::StructureType::ePipelineColorBlendStateCreateInfo;
		colorBlending.logicOpEnable = vk::False;
		colorBlending.logicOp = vk::LogicOp::eCopy;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		// Pipeline Layout
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = vk::StructureType::ePipelineLayoutCreateInfo;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0; // another way of passing dynamic data to shaders
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		// create the layout
		try {
			pipelineLayout = logicalDevice.createPipelineLayout(pipelineLayoutInfo);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to create pipeline Layout!");
		}

		// Create graphics pipeline!!
		vk::GraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = vk::StructureType::eGraphicsPipelineCreateInfo;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0; //index
		pipelineInfo.basePipelineHandle = nullptr;
		pipelineInfo.basePipelineIndex = -1;
		
		vk::ResultValue<vk::Pipeline> result = logicalDevice.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo);
		if (result.result != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to create the Graphics Pipeline!");
		}
		graphicsPipeline = result.value;

		// Clean up
		logicalDevice.destroyShaderModule(fragShaderModule);
		logicalDevice.destroyShaderModule(vertShaderModule);
	}

	// helper func to create shader mod obj
	vk::ShaderModule createShaderModule(const std::vector<char>& code)
	{
		vk::ShaderModuleCreateInfo createInfo{};
		createInfo.sType = vk::StructureType::eShaderModuleCreateInfo;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		vk::ShaderModule shaderModule;

		try {
			shaderModule =logicalDevice.createShaderModule(createInfo);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to create shader module!");
		}

		return shaderModule;
	}

	static std::vector<char> readFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary); // ate starts reading eof . . . binary reads as binary

		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file!");
		}

		// get size knowing eof
		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		// back to beginning
		file.seekg(0);
		file.read(buffer.data(), fileSize);
		
		// close file
		file.close();
		return buffer;
	}

	// 9. Create Framebuffer
	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());
		// Iterate through image views and create frame buffer for them
		for (size_t i = 0; i < swapChainImageViews.size(); i++)
		{
			vk::ImageView attachments[] = {
				swapChainImageViews[i]
			};

			vk::FramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = vk::StructureType::eFramebufferCreateInfo;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			try {
				swapChainFramebuffers[i] = logicalDevice.createFramebuffer(framebufferInfo);
			}
			catch (const vk::SystemError& err) {
				std::ostringstream oss;
				oss << "Failed to create a framebuffer for: " << i;
				throw std::runtime_error(oss.str());
			}
		}
	}

	// 10. Create Command pool
	void createCommandPool() {
		QueueFamilyIndeces queueFamilyIndices = findQueueFamilies(physicalDevice);

		vk::CommandPoolCreateInfo poolInfo{};
		poolInfo.sType = vk::StructureType::eCommandPoolCreateInfo;
		poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer; // allow command buffers to be rerecorded individually
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

		try {
			commandPool = logicalDevice.createCommandPool(poolInfo);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to create command pool!");
		}
	}

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
		vk::BufferCreateInfo bufferInfo{};
		bufferInfo.sType = vk::StructureType::eBufferCreateInfo;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = vk::SharingMode::eExclusive;

		try {
			buffer = logicalDevice.createBuffer(bufferInfo, nullptr);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create vertex buffer" + std::string(err.what()));
		}

		// assign mem to vertex buffer
		vk::MemoryRequirements memRequirements;
		memRequirements = logicalDevice.getBufferMemoryRequirements(buffer);

		// mem allocation
		vk::MemoryAllocateInfo allocInfo{};
		allocInfo.sType = vk::StructureType::eMemoryAllocateInfo;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemotyType(memRequirements.memoryTypeBits, properties);

		try {
			bufferMemory = logicalDevice.allocateMemory(allocInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create vertex buffer memory!");
		}

		// bind mem with buffer
		logicalDevice.bindBufferMemory(buffer, bufferMemory, 0);
	}

	// 11. Create texture image
	void createTextureImage() {
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load("Textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		vk::DeviceSize imageSize = texWidth * texHeight * 4; // pixels are laid out row by row with 4 bytes per pixel in the case of STBI_rgb_alpha

		if (!pixels) {
			throw std::runtime_error("Failed to load texture image!");
		}

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;

		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);


		void* data;
		try {
			data = logicalDevice.mapMemory(stagingBufferMemory, 0, imageSize, vk::MemoryMapFlags()); // mapping creates a link between cpu and gpu memory, but keeping it mapped forever can create contention

		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to map memory!" + std::string(err.what()));
		}
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		logicalDevice.unmapMemory(stagingBufferMemory);

		stbi_image_free(pixels);

		// eSampled: sampled image in fragment shader and eStorage: storage image in compute shader
		createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);
		
		transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		
		//cleanup
		logicalDevice.destroyBuffer(stagingBuffer);
		logicalDevice.freeMemory(stagingBufferMemory);	
	}

	void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory) {
		vk::ImageCreateInfo imageInfo{};
		imageInfo.sType = vk::StructureType::eImageCreateInfo;
		imageInfo.imageType = vk::ImageType::e2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = vk::ImageLayout::eUndefined;
		imageInfo.usage = usage;
		imageInfo.sharingMode = vk::SharingMode::eExclusive;
		imageInfo.samples = vk::SampleCountFlagBits::e1;
		imageInfo.flags = vk::ImageCreateFlags();

		try {
			textureImage = logicalDevice.createImage(imageInfo, nullptr);

		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create an image!" + std::string(err.what()));
		}

		vk::MemoryRequirements memRequirements;
		try {
			memRequirements = logicalDevice.getImageMemoryRequirements(textureImage);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create memory requirements for image!" + std::string(err.what()));
		}

		vk::MemoryAllocateInfo allocInfo{};
		allocInfo.sType = vk::StructureType::eMemoryAllocateInfo;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemotyType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

		try {
			textureImageMemory = logicalDevice.allocateMemory(allocInfo, nullptr);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to allocate memory for texture image!");

		}

		logicalDevice.bindImageMemory(textureImage, textureImageMemory, 0);
	}


	// Helper func for recording and executing a command buffer
	vk::CommandBuffer beginSingleTimeCommands() {
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = vk::StructureType::eCommandBufferAllocateInfo;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		vk::CommandBuffer commandBuffer;
		try {
			commandBuffer = logicalDevice.allocateCommandBuffers(allocInfo)[0];
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to allocate command buffer: " + std::string(err.what()));
		}

		vk::CommandBufferBeginInfo beginInfo{};
		beginInfo.sType = vk::StructureType::eCommandBufferBeginInfo;
		beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

		try {
			commandBuffer.begin(beginInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to execuete command buffer: " + std::string(err.what()));
		}

		return commandBuffer;
	}

	void endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
		try {
			commandBuffer.end();
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to end command buffer! " + std::string(err.what()));
		}

		vk::SubmitInfo submitInfo{};
		submitInfo.sType = vk::StructureType::eSubmitInfo;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		try {
			graphicsAndComputeQueue.submit(submitInfo, nullptr);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to submit to graphics queue!" + std::string(err.what()));
		}

		try {
			graphicsAndComputeQueue.waitIdle();
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to wait on graphics queue!" + std::string(err.what()));
		}

		try {
			logicalDevice.freeCommandBuffers(commandPool, 1, &commandBuffer);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to free command buffer!" + std::string(err.what()));
		}
	}

	// helper func for layout
	void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
		vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

		// Layout transition use image memory barrier
		vk::ImageMemoryBarrier barrier{};
		barrier.sType = vk::StructureType::eImageMemoryBarrier;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
		barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1; // since not mip mapped
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		/*barrier.srcAccessMask = vk::AccessFlagBits::z;*/
		/*barrier.dstAccessMask = 0;*/ // Will get back to once transitions are figured out

		vk::PipelineStageFlags sourceStage;
		vk::PipelineStageFlags destinationStage;

		if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
			barrier.srcAccessMask = vk::AccessFlags(); // no need to wait on anything since image is undefined
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite; // destination written by a transfer operation

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eTransfer;

		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) { // image was used for a transfer op and is now being used to read by a shader
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eTransfer;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}


		// submit barrier 
		commandBuffer.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags(), nullptr, nullptr, barrier);

		endSingleTimeCommands(commandBuffer);
	}

	// Helper func for coping buffer to image
	void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
		vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

		// specify what is being copied
		vk::BufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = vk::Offset3D{ 0, 0, 0 };
		region.imageExtent = vk::Extent3D{
			width,
			height,
			1
		};

		try {
			commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to copy buffer to an image!" + std::string(err.what()));
		}

		endSingleTimeCommands(commandBuffer);
	}

	// 12. Create texture image view
	void createTextureImageView() {
		textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
	}

	vk::ImageView createImageView(vk::Image image, vk::Format format) {
		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.sType = vk::StructureType::eImageViewCreateInfo;
		viewInfo.image = image;
		viewInfo.viewType = vk::ImageViewType::e2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		vk::ImageView imageView;
		try {
			imageView = logicalDevice.createImageView(viewInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create an image view!" + std::string(err.what()));
		}

		return imageView;
	}

	// 13. create texture sampler
	void createTextureSampler() {
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.sType = vk::StructureType::eSamplerCreateInfo;
		samplerInfo.magFilter = vk::Filter::eLinear; // oversampling: more fragments than texels
		samplerInfo.minFilter = vk::Filter::eLinear; // undersampling: more texels than fragments

		// For texture space coordinates
		samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;

		samplerInfo.anisotropyEnable = vk::True;
	

		vk::PhysicalDeviceProperties properties{};
		properties = physicalDevice.getProperties();

		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy; // limits that max number of texel samples that can be used to calculate the final color
		samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
		samplerInfo.unnormalizedCoordinates = vk::False;
		samplerInfo.compareEnable=vk::False;
		samplerInfo.compareOp = vk::CompareOp::eAlways;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		try {
			textureSampler = logicalDevice.createSampler(samplerInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create texture sampler!" + std::string(err.what()));
		}
	}

	// 11. Create Vertex Buffer
	void createVertexBuffer() {
		// use staging buffer as an 'intermediate' between cpu and gpu.

		vk::DeviceSize bufferSize = static_cast<vk::DeviceSize>(sizeof(vertices[0]) * vertices.size());

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);
		// note, host visible means visible by cpu

		// map memory
		void* data = nullptr;

		try {
			data = logicalDevice.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags()); // mapping creates a link between cpu and gpu memory, but keeping it mapped forever can create contention

		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to map memory!" + std::string(err.what()));
		}
		memcpy(data, vertices.data(), (size_t)bufferSize);
		logicalDevice.unmapMemory(stagingBufferMemory);

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);
		
		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		try {
			logicalDevice.destroyBuffer(stagingBuffer);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to destroy staging buffer!" + std::string(err.what()));
		}

		try {
			logicalDevice.freeMemory(stagingBufferMemory);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to free staging buffer memory!" + std::string(err.what()));
		}
	}

	void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
	{
		vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::BufferCopy copyRegion{};
		copyRegion.size = size;

		try {
			commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to transfer data between buffers!" + std::string(err.what()));
		}

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemotyType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
		vk::PhysicalDeviceMemoryProperties memProperties;
		memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	// 12. Create index buffer
	void createIndexBuffer() {
		
		vk::DeviceSize bufferSize = static_cast<vk::DeviceSize>(sizeof(indices[0]) * indices.size());

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);
		
		// map memory
		void* data = nullptr;

		try {
			data = logicalDevice.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags()); // mapping creates a link between cpu and gpu memory, but keeping it mapped forever can create contention

		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to map memory!" + std::string(err.what()));
		}
		memcpy(data, indices.data(), (size_t)bufferSize);
		logicalDevice.unmapMemory(stagingBufferMemory);

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		try {
			logicalDevice.destroyBuffer(stagingBuffer);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to destroy staging buffer!" + std::string(err.what()));
		}

		try {
			logicalDevice.freeMemory(stagingBufferMemory);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to free staging buffer memory!" + std::string(err.what()));
		}
	}

	// 13. Create Uniform Buffers
	void createUniformBuffers() {
		vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], uniformBuffersMemory[i]);
			
			try {
				uniformBuffersMapped[i] = logicalDevice.mapMemory(uniformBuffersMemory[i], 0, bufferSize, vk::MemoryMapFlags());
			}
			catch (vk::SystemError& err) {
				throw std::runtime_error("Failed to map memory to a uniform buffer!" + std::string(err.what()));
			}
		}
	}

	// 14. Create descriptor pools
	void createDescriptorPool() {
		std::array<vk::DescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = vk::StructureType::eDescriptorPoolCreateInfo;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		try {
			descriptorPool = logicalDevice.createDescriptorPool(poolInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to create a descriptor pool!" + std::string(err.what()));
		}
	}
	
	// 15. Create descriptor sets
	void createDescriptorSets() {
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = vk::StructureType::eDescriptorSetAllocateInfo;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT); // one descriptor set for each frame in flight
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

		try {
			descriptorSets = logicalDevice.allocateDescriptorSets(allocInfo);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to allocate descriptor sets!" + std::string(err.what()));
		}

		// configure descriptors
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vk::DescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			vk::DescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			imageInfo.imageView = textureImageView;
			imageInfo.sampler = textureSampler;

			std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};

			
			descriptorWrites[0].sType = vk::StructureType::eWriteDescriptorSet;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;
			
			descriptorWrites[1].sType = vk::StructureType::eWriteDescriptorSet;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;
			
			//descriptorWrites[1].pImageInfo = nullptr;
			//descriptorWrites[1].pTexelBufferView = nullptr;

			try {
				logicalDevice.updateDescriptorSets(descriptorWrites, {});
			} // MIGHT HAVE TO CHANGE
			catch (vk::SystemError& err) {
				throw std::runtime_error("Failed to update descriptor set!" + std::string(err.what()));
			}
		}
	}

	// 13. Create Command Buffer
	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = vk::StructureType::eCommandBufferAllocateInfo;
		allocInfo.commandPool = commandPool;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size()); // if primary or secondary

		try {
			commandBuffers = logicalDevice.allocateCommandBuffers(allocInfo);			
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to create a command buffers!" + std::string(err.what()));
		}
	}

	void recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex) {

		vk::CommandBufferBeginInfo beginInfo{};
		beginInfo.sType = vk::StructureType::eCommandBufferBeginInfo;
		beginInfo.flags = vk::CommandBufferUsageFlags();
		beginInfo.pInheritanceInfo = nullptr;

		try {
			commandBuffer.begin(beginInfo);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to begin recording command buffer" + std::string(err.what()));
		}

		// render pass
		vk::RenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = vk::StructureType::eRenderPassBeginInfo;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = vk::Offset2D{ 0,0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		vk::ClearColorValue clearColor(std::array<float, 4> {{0.0f, 0.0f, 0.0f, 1.0f}});
		vk::ClearValue clearValue;
		clearValue.color = clearColor;
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearValue;

		commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
		
		// Drawing commands
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

		vk::Buffer vertexBuffers[] = { vertexBuffer };
		vk::DeviceSize offsets[] = { 0 };
		try {
			commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to bind vertex buffer" + std::string(err.what()));
		}

		try {
			commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to bind index buffer!" + std::string(err.what()));
		}
		
		// handle dynamic viewport and scissor state
		vk::Viewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		commandBuffer.setViewport(0, 1, &viewport);

		vk::Rect2D scissor{};
		scissor.offset = vk::Offset2D{ 0, 0 };
		scissor.extent = swapChainExtent;
		commandBuffer.setScissor(0, 1, &scissor);

		try {
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
		}
		catch (vk::SystemError& err) {
			throw std::runtime_error("Failed to bind descriptor sets!" + std::string(err.what()));
		}
		
		try {
			commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
			//commandBuffer.draw(static_cast<uint32_t>(vertices.size()), 1, 0, 0);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to draw!" + std::string(err.what()));
		}

		commandBuffer.endRenderPass();

		try {
			commandBuffer.end();
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to record command buffer!" + std::string(err.what()));
		}

	}

	// 14. Create Synhronization Objects
	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		vk::SemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = vk::StructureType::eSemaphoreCreateInfo;

		vk::FenceCreateInfo fenceInfo{};
		fenceInfo.sType = vk::StructureType::eFenceCreateInfo;
		fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled; // So that the first frame doesn't halt the fence indefinitely

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			try {
				imageAvailableSemaphores[i] = logicalDevice.createSemaphore(semaphoreInfo);
				renderFinishedSemaphores[i] = logicalDevice.createSemaphore(semaphoreInfo);
				inFlightFences[i] = logicalDevice.createFence(fenceInfo);
			}
			catch (const vk::SystemError& err) {
				throw std::runtime_error("Failed to create synchronization objects for a frame!" + std::string(err.what()));
			}
		}

		
	}

	void drawFrame() {
		// wait until previous frame has finished
		assert(logicalDevice.waitForFences(1, &inFlightFences[currentFrame], vk::True, UINT64_MAX) == vk::Result::eSuccess); // 64 bit unsigned it is max time out
		
		// acquire image from swap chain
		uint32_t imageIndex;
		vk::Result result;
		
		result = logicalDevice.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex); // 3rd param is timeout in nanoseconds -- using the 64 bit unsigned int means timeout is disabled
		

		if (result == vk::Result::eErrorOutOfDateKHR) { // swap chain becomes incompatible with the surface and can no longer be used for rendering. Usually after window resize
			recreateSwapChain();
			return;
		}
		else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) { // swap chain can be used for surface, but the surface properties don't match
			throw std::runtime_error("Failed to acquire swap chain image!");
		}

		// manually reset fence after waiting
		assert(logicalDevice.resetFences(1, &inFlightFences[currentFrame]) == vk::Result::eSuccess);

		commandBuffers[currentFrame].reset();

		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
		
		// UBO
		updateUniformBuffer(currentFrame);

		// Queue submission and synchronization
		vk::SubmitInfo submitInfo{};
		submitInfo.sType = vk::StructureType::eSubmitInfo;
		
		vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame]};
		vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput }; // the stage of the pipeline to wait in
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
		vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame]};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;
		
		try {
			assert(graphicsAndComputeQueue.submit(1, &submitInfo, inFlightFences[currentFrame]) == vk::Result::eSuccess);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to submit draw command buffer!" + std::string(err.what()));
		}

		// submit result back to the swap chain
		vk::PresentInfoKHR presentInfo{};
		presentInfo.sType = vk::StructureType::ePresentInfoKHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		vk::SwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		try {
			vk::Result resultPresent;
			resultPresent = presentQueue.presentKHR(presentInfo);

			if (resultPresent == vk::Result::eSuboptimalKHR || framebufferResized) {
				framebufferResized = false;
				recreateSwapChain();
			}
		}
		catch (const vk::OutOfDateKHRError& err )
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		catch (const vk::SystemError& err) {
			std::cerr << "Vulkan system error: " << err.what() << std::endl;
		}
		
		catch (const std::exception& e) {
			std::cerr << "Failed to present swap chain image: " << e.what() << std::endl;
			throw;
		}
		catch (...) {
			std::cerr << "Caught an unecognized exception" << std::endl;
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	// Will generate a new transformation every frame to make the geometry spin around
	void updateUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
		float aspectRatio = swapChainExtent.width / (float)swapChainExtent.height;

		// UBO nescesities for rotation
		UniformBufferObject ubo{};
		ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)); // eye, center, up axis
		ubo.proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 10.0f); // 45 deg vert FOV, aspect ratio, near, far planes
		ubo.proj[1][1] *= -1; //if I don't do this image rendered upside down since GLM originally designed for openGL where Y coordinate in clip coord is inverted
		ubo.time = time;
		ubo.aspectRatio = aspectRatio;

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));


	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}
		logicalDevice.waitIdle();
	}

	void cleanup() {
		// clean up swap chains
		cleanupSwapChain();

		logicalDevice.destroySampler(textureSampler);
		logicalDevice.destroyImageView(textureImageView);

		logicalDevice.destroyImage(textureImage);
		logicalDevice.freeMemory(textureImageMemory);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			logicalDevice.destroyBuffer(uniformBuffers[i]);
			logicalDevice.freeMemory(uniformBuffersMemory[i]);
		}

		logicalDevice.destroyDescriptorPool(descriptorPool);
		logicalDevice.destroyDescriptorSetLayout(descriptorSetLayout);
		logicalDevice.destroyBuffer(indexBuffer);
		logicalDevice.freeMemory(indexBufferMemory);
		logicalDevice.destroyBuffer(vertexBuffer);
		logicalDevice.freeMemory(vertexBufferMemory);
		logicalDevice.destroyPipeline(graphicsPipeline);
		logicalDevice.destroyPipelineLayout(pipelineLayout);
		logicalDevice.destroyRenderPass(renderPass);
		
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			logicalDevice.destroySemaphore(imageAvailableSemaphores[i]);
			logicalDevice.destroySemaphore(renderFinishedSemaphores[i]);
			logicalDevice.destroyFence(inFlightFences[i]);
		}

		logicalDevice.destroyCommandPool(commandPool);

		logicalDevice.destroy();

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

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