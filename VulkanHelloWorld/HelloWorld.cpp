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




const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Validation layers
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


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
	

	// logical device
	vk::Device logicalDevice;
	vk::Queue graphicsQueue;
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

	// Pipeline
	vk::RenderPass renderPass;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline graphicsPipeline;
	vk::CommandPool commandPool;
	vk::CommandBuffer commandBuffer;

	// Synchronization
	vk::Semaphore imageAvailableSemaphore;
	vk::Semaphore renderFinishedSemaphore;
	vk::Fence inFlightFence;
	
	
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

		// 8. create Graphics Pipeline
		createGraphicsPipeline();
		
		// 9. Create Framebuffer
		createFramebuffers();

		// 10. Create Command pool
		createCommandPool();

		// 11. Create Command Buffer
		createCommandBuffer();

		// 12. Create Synhronization Objects
		createSyncObjects();
	}

	// validation layers
	// checks if all the requested layers are available
	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		assert(vk::enumerateInstanceLayerProperties(&layerCount, nullptr) == vk::Result::eSuccess); // lists all available layers

		std::vector<vk::LayerProperties> availableLayers(layerCount);
		assert(vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data()) == vk::Result::eSuccess);


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

		createInfo.sType = vk::StructureType::eInstanceCreateInfo;
		createInfo.pApplicationInfo = &appInfo;

		// validation layers
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();
		
		//vk::Result result = vk::createInstance(&createInfo, nullptr, &instance);

		// create instance && throw run time error if unsuccessful
		if (vk::createInstance(&createInfo, nullptr, &instance) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	// DEBUGGING -- debug messenger setup
	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
		createInfo.sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT;
		createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
		createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
		createInfo.pfnUserCallback = reinterpret_cast<PFN_vkDebugUtilsMessengerCallbackEXT>(debugCallback);
		createInfo.pUserData = nullptr;
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
		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && deviceFeatures.geometryShader && indices.isComplete() && extensionsSupported && swapChainAdequate;
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
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		vk::SwapchainCreateInfoKHR createSwapChainInfo{};

		if (indices.presentFamily != indices.graphicsFamily)
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
			vk::ImageViewCreateInfo createInfo{};
			createInfo.sType = vk::StructureType::eImageViewCreateInfo;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = vk::ImageViewType::e2D;
			createInfo.format = swapChainImageFormat;
			// Could get creative with this later.
			createInfo.components.r = vk::ComponentSwizzle::eIdentity;
			createInfo.components.g = vk::ComponentSwizzle::eIdentity;
			createInfo.components.b = vk::ComponentSwizzle::eIdentity;
			createInfo.components.a = vk::ComponentSwizzle::eIdentity;
			createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			// create image view
			try {
				swapChainImageViews[i] = logicalDevice.createImageView(createInfo);
			}
			catch (const vk::SystemError& err){
				throw std::runtime_error("Failed to create Image View!");
			}

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

		// create the render pass
		vk::RenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = vk::StructureType::eRenderPassCreateInfo;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;

		try {
			renderPass = logicalDevice.createRenderPass(renderPassInfo);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to create Render Pass!");
		}

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
		vertexInputInfo.sType = vk::StructureType::ePipelineVertexInputStateCreateInfo;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr;
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr;


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
		rasterizer.frontFace = vk::FrontFace::eClockwise;
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
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;
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
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		try {
			commandPool = logicalDevice.createCommandPool(poolInfo);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to create command pool!");
		}
	}
	
	// 11. Create Command Buffer
	void createCommandBuffer() {
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = vk::StructureType::eCommandBufferAllocateInfo;
		allocInfo.commandPool = commandPool;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = 1; // if primary or secondary

		try {
			std::vector<vk::CommandBuffer, std::allocator<vk::CommandBuffer>> result = logicalDevice.allocateCommandBuffers(allocInfo);
			commandBuffer = result[0];
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

		commandBuffer.draw(3, 1, 0, 0); // vertex count (3 vertices to draw), instance count, first vertex (offset into the vertex buffer), first instance (offset for instanced rendering)

		commandBuffer.endRenderPass();

		try {
			commandBuffer.end();
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to record command buffer!" + std::string(err.what()));
		}

	}

	// 12. Create Synhronization Objects
	void createSyncObjects() {
		vk::SemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = vk::StructureType::eSemaphoreCreateInfo;

		vk::FenceCreateInfo fenceInfo{};
		fenceInfo.sType = vk::StructureType::eFenceCreateInfo;
		fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled; // So that the first frame doesn't halt the fence indefinitely

		try {
			imageAvailableSemaphore = logicalDevice.createSemaphore(semaphoreInfo);
			renderFinishedSemaphore = logicalDevice.createSemaphore(semaphoreInfo);
			inFlightFence = logicalDevice.createFence(fenceInfo);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to create synchronization objects" + std::string(err.what()));
		}
	}

	void drawFrame() {
		// wait until previous frame has finished
		assert(logicalDevice.waitForFences(1, &inFlightFence, vk::True, UINT64_MAX) == vk::Result::eSuccess); // 64 bit unsigned it is max time out
		// manually reset fence after waiting
		assert(logicalDevice.resetFences(1, &inFlightFence) == vk::Result::eSuccess);
		
		// acquire image from swap chain
		uint32_t imageIndex;
		assert(logicalDevice.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex)== vk::Result::eSuccess); // 3rd param is timeout in nanoseconds -- using the 64 bit unsigned int means timeout is disabled
		
		commandBuffer.reset();

		recordCommandBuffer(commandBuffer, imageIndex);
		
		// Queue submission and synchronization
		vk::SubmitInfo submitInfo{};
		submitInfo.sType = vk::StructureType::eSubmitInfo;
		
		vk::Semaphore waitSemaphores[] = { imageAvailableSemaphore };
		vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput }; // the stage of the pipeline to wait in
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		vk::Semaphore signalSemaphores[] = { renderFinishedSemaphore };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;
		
		try {
			assert(graphicsQueue.submit(1, &submitInfo, inFlightFence)== vk::Result::eSuccess);
		}
		catch (const vk::SystemError& err) {
			throw std::runtime_error("Failed to submit draw command buffer!" + std::string(err.what()));
		}

	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			//drawFrame();
		}
	}

	void cleanup() {
		logicalDevice.destroySemaphore(imageAvailableSemaphore);
		logicalDevice.destroySemaphore(renderFinishedSemaphore);
		logicalDevice.destroyFence(inFlightFence);
		logicalDevice.destroyCommandPool(commandPool);

		for (auto framebuffer : swapChainFramebuffers) {
			logicalDevice.destroyFramebuffer(framebuffer);
		}
		logicalDevice.destroyPipeline(graphicsPipeline);
		logicalDevice.destroyPipelineLayout(pipelineLayout);
		logicalDevice.destroyRenderPass(renderPass);
		// clean up image views
		for (auto imageView : swapChainImageViews)
		{
			logicalDevice.destroyImageView(imageView);
		}

		logicalDevice.destroySwapchainKHR(swapChain);

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