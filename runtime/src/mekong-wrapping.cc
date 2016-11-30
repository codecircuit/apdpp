/*! \file mekong-wrapping.cc
    \brief File containing functions, which substitutes the original ones.

    This file is the high level Ansatz to understand Mekong's runtime
    functionality. Here you can see the definitions of the functions, which
    will be used to substitute the original cuda driver functions. We use four
    global variables to store the state of the running program. The analysis of
    the kernels (bsp_database.h) and the user configuration file (user_config.h)
    is statically linked while the runtime library is compiled. Thus you have to
    recompile the runtime if you change the analysis of the kernels or the
    runtime configuration file.

*/

#include "mekong-wrapping.h"
#include "alias_handle.h"
#include "kernel_info.h"
#include "virtual_buffer.h"
#include "log_statistics.h"
#include "kernel_launch.h"
#include "user_config.h" // generated of $PROJECT_DIR/CONFIG.txt
#include "dependency_resolution.h"
#include "mekong-cuda.h"
#include "bsp_database.h" // generated of $PROJECT_DIR/bsp_analysis/dbs/kernel_info.dbb
#include "communicator.h" // dominiks memcpy lib

#include <string>
#include <unordered_set>
#include <functional> // std::hash
#include <queue>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <utility>
#include <memory>
#include <stdexcept>
#include <vector>


using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

/*******************************
 * GLOBAL VARIABLE DEFINITIONS *
 *******************************/
// Here we init a set, which saves only different kernel launches.
std::unordered_set<std::shared_ptr<Mekong::KernelLaunch>,
                   Mekong::KernelLaunch::hash,
                   Mekong::KernelLaunch::equal_to>
Mekong::KernelLaunch::all(0, Mekong::KernelLaunch::hash(),
                          Mekong::KernelLaunch::equal_to());

#ifdef SOFIRE
// Dominiks memcopy library
communicator Mekong::MemCpyHtoD::comm;
#endif

// FUTURE WORK: Here we save the order of current kernel launches
// see WORKLIST.md
// std::queue<std::shared_ptr<Mekong::KernelLaunch>> MEKONG_launches;

// Statistics
static Mekong::Statistics MEKONG_statistics;

// Log
#if USER_OPTION_LOG_ON == true
#define LOG(text) MEKONG_log << text;
static Mekong::Log<char> MEKONG_log;
#else
#define LOG(text)
#endif

// Save the dependency resolution objects
static std::unordered_set<std::shared_ptr<Mekong::DepResolution>>
MEKONG_depResolutions;

// give one dev pointer to the alias handler and it will give you back the
// pointers belonging to that pointer on different gpus
static std::shared_ptr<Mekong::AliasHandle> MEKONG_aliasH(new Mekong::AliasHandle);

// here we store which kernel wrote last to a certain memory location
static std::shared_ptr<Mekong::Buffer> MEKONG_buffer(new Mekong::Buffer);

// contains the information of the static kernel analysis
static std::vector<std::shared_ptr<const Mekong::bsp_KernelInfo>>
MEKONG_kinfos(Mekong::bsp_KernelInfo::createKInfos(Mekong::bspAnalysisStr));

/************************
 * FUNCTION DEFINITIONS *
 ************************/
/*! \brief Substitution for cuInit()
*/
Mekong::MErawresult wrapInit(unsigned flags) {
	LOG("[MEKONG] [+] FUNC wrapInit():\n")

	Mekong::MEresult res;
	res &= Mekong::meInit(flags);

	LOG("[MEKONG] [-] FUNC wrapInit()\n")
	return res.getRaw();
}

/*! \brief Check for any devices and persuades a one gpu system to the user.

    The function will throw an error if there is no GPU in the system.
    \param num Will be equal to one if the are one or more GPUs in the system
*/
Mekong::MErawresult wrapDeviceGetCount(int* num) {
	LOG("[MEKONG] [+] FUNC wrapDeviceGetCount():\n")

	Mekong::MEresult res;
	int devCount = - 1;
	res &= Mekong::meDeviceGetCount(&devCount);
	if (devCount < 1) {
		throw std::runtime_error("Mekong can not detect any device in the system");
	}
	if (res.isSuccess()) {
		LOG("[MEKONG] recognized " + std::to_string(devCount)
		    + " devices in the system\n")
	}
	*num = 1; // We persuade a one gpu system to the user

	if (USER_OPTION_COLLECT_STATISTICS) {
		MEKONG_statistics.setNumDev(devCount);
	}

	LOG("[MEKONG] [-] FUNC wrapDeviceGetCount()\n")
	return res.getRaw();
}

/*! \brief Register all devices in a global variable
    \param dev will represent the first device found in the system.
*/
Mekong::MErawresult wrapDeviceGet(Mekong::MEdevice* dev, int dummy) {
	LOG("[MEKONG] [+] FUNC wrapDeviceGet():\n")
	Mekong::MEresult res;
	int devCount = -1;
	res &= Mekong::meDeviceGetCount(&devCount);
	if (devCount < 1) {
		throw std::runtime_error("Mekong can not detect any device in the system");
	}
	std::vector<Mekong::MEdevice> devs(devCount);
	unsigned short gpu = 0;
	for (auto& device : devs) {
		res &= Mekong::meDeviceGet(&device, gpu);
		++gpu;
	}

	if (res.isSuccess()) {
		LOG("[MEKONG] registered " + std::to_string(devCount)
		    + " device pointers\n")
	}
	else {
		LOG("[MEKONG] failed to register device pointers\n")
	}

	// here we link the first device to the other devices
	*dev = devs[0];
	if (USER_OPTION_COLLECT_STATISTICS) {
		MEKONG_statistics.setNumDev(devs.size());
	}
	(*MEKONG_aliasH)[*dev] = std::move(devs); // access on global variable
	LOG("[MEKONG] [-] FUNC wrapDeviceGet()\n")
	return res.getRaw();
}

/*! \brief Returns the smallest compute capability of all devices
*/
Mekong::MErawresult wrapDeviceComputeCapability(int* major, int* minor,
                                                Mekong::MEdevice dev) {
	LOG("[MEKONG] [+] wrapDeviceComputeCapability():\n")
	Mekong::MEresult res;
	int min_minor;
	int min_major;
	int curr_minor;
	int curr_major;
	res &= Mekong::meDeviceComputeCapability(&min_major, &min_minor, dev);
	for (auto& otherDev : (*MEKONG_aliasH)[dev]) {
		res &= Mekong::meDeviceComputeCapability(&curr_major, &curr_minor, otherDev);
		if (curr_major < min_major) {
			min_major = curr_major;
		}
		if (curr_major == min_major && curr_minor < min_minor) {
			min_minor = curr_minor;
		}

	}
	*major = min_major;
	*minor = min_minor;
	if (res.isSuccess()) {
		LOG("[MEKONG] minimum compute capability sm_"
		    + std::to_string(*major) + std::to_string(*minor) + '\n')
	}
	else {
		LOG("[MEKONG] cuda error while detecting the compute capability\n")
	}
	LOG("[MEKONG] [-] wrapDeviceComputeCapability()\n")
	return res.getRaw();
}

/*! \brief Creates one context per registered GPU.

    Creates one context for each device registered on key `dev`.
    The contexts are registered in a global variable.
    \param ctx will be the context living on the first registered device.
    \sa wrapDeviceGet registers devices. 
*/
Mekong::MErawresult wrapCtxCreate(Mekong::MEcontext* ctx, unsigned flags,
                                  Mekong::MEdevice dev) {
	LOG("[MEKONG] [+] FUNC wrapCtxCreate():\n")
	Mekong::MEresult res;
	std::vector<Mekong::MEcontext> ctxs(MEKONG_aliasH->getNumDev());
	LOG("  Registered " + std::to_string(ctxs.size()) + " GPUs\n")
	unsigned short gpu = 0;
	for (auto& ctx : ctxs) {
		LOG("  Going to create context on GPU " + std::to_string(gpu) + '\n') 
		res &= Mekong::meCtxCreate(&ctx, flags, (*MEKONG_aliasH)[dev][gpu]);
		++gpu;
	}
	LOG("[MEKONG] created " + std::to_string(ctxs.size()) + " contexts\n")
	*ctx = ctxs[0];
	(*MEKONG_aliasH)[*ctx] = std::move(ctxs);
	LOG("[MEKONG] [-] FUNC wrapCtxCreate()\n")
	return res.getRaw();
}

/*! \brief Loads the module `name` for each registered context.

    All loaded modules are saved in a global variable.
    \param module will be the module living in the first registered context.
    \sa wrapCtxCreate to register contexts
*/ 
Mekong::MErawresult wrapModuleLoad(Mekong::MEmodule* module, const char* name) {
	LOG("[MEKONG] [+] FUNC wrapModuleLoad():\n")
	Mekong::MEresult res;
	std::vector<Mekong::MEmodule> mods(MEKONG_aliasH->getNumDev());
	unsigned short gpu = 0;
	for (auto& mod : mods) {
		res &= Mekong::meCtxPushCurrent(MEKONG_aliasH->getCtx()[gpu]);
		res &= Mekong::meModuleLoad(&mod, name);
		res &= Mekong::meCtxPopCurrent(0);
		++gpu;
	}
	if (res.isSuccess()) {
		LOG("[MEKONG] loaded " + std::to_string(mods.size())
		    + " module from file ") LOG(name) LOG('\n')
	}
	else {
		LOG("[MEKONG] could not load modules from file ") LOG(name) LOG('\n')
	}
	*module = mods[0];
	(*MEKONG_aliasH)[*module] = std::move(mods);
	LOG("[MEKONG] [-] FUNC wrapModuleLoad()\n")
	return res.getRaw();
}

/*! \brief Gets a function for each registered module.

    The functions will be saved in a global variable and linked to
    the value of `mod`.
    \param func will be the function loaded in the first registered module.
    \sa wrapModuleLoad loads all modules
*/
Mekong::MErawresult wrapModuleGetFunction(Mekong::MEfunction* func,
                                          Mekong::MEmodule mod,
                                          const char* fname) {
	LOG("[MEKONG] [+] FUNC wrapModuleGetFunction():\n")
	Mekong::MEresult res;
	std::string superName = std::string(fname) + "_super";
	std::vector<Mekong::MEfunction> funcs(MEKONG_aliasH->getNumDev());
	unsigned short gpu = 0;
	for (auto& func : funcs) {
		res &= Mekong::meCtxPushCurrent(MEKONG_aliasH->getCtx()[gpu]);
		res &= Mekong::meModuleGetFunction(&func, (*MEKONG_aliasH)[mod][gpu],
		                                   superName.c_str());
		res &= Mekong::meCtxPopCurrent(0);
		++gpu;
	}
	if (res.isSuccess()) {
		LOG("[MEKONG] loaded " + std::to_string(funcs.size())
		    + " functions with name ") LOG(fname) LOG('\n')
	}
	else {
		LOG("[MEKONG] could not load functions with name ") LOG(fname) LOG('\n')
	}

	*func = funcs[0];
	(*MEKONG_aliasH)[*func] = std::move(funcs);

	// link function pointer to the function name
	MEKONG_aliasH->atName(*func) = std::string(fname);
	LOG("[MEKONG] [-] FUNC wrapModuleGetFunction()\n")
	return res.getRaw();
}

/*! \brief Allocates `size` Bytes on every registered device.

    Thus in total we allocate `numGPUs * size` Bytes. The buffer `ptr`
    will be linked to the other `numGPUs - 1` buffers pointing to memory
    locations on the other devices. Thus, if the user refers to the
    value of `ptr`, we will associate `numGPU` buffers on different devices
    with it.
    \param ptr will point to the memory allocated on the first gpu.
*/
Mekong::MErawresult wrapMemAlloc(Mekong::MEdeviceptr* ptr, size_t size) {
	LOG("[MEKONG] [+] FUNC wrapMemAlloc():\n")
	Mekong::MEresult res;
	std::vector<Mekong::MEdeviceptr> devptrs(MEKONG_aliasH->getNumDev());
	unsigned short gpu = 0;
	for (auto& devptr : devptrs) {
		res &= Mekong::meCtxPushCurrent(MEKONG_aliasH->getCtx()[gpu]);
		res &= Mekong::meMemAlloc(&devptr, size);
		res &= Mekong::meCtxPopCurrent(0);
		++gpu;
	}
	*ptr = devptrs[0];
	if (res.isSuccess()) {
		LOG("[MEKONG] allocated " + std::to_string((double) size/ 1e6)
		    + " MB/device with "  + std::to_string(gpu) + " devices in total\n")
	}
	else {
		LOG("[MEKONG] could not allocate " + std::to_string((double) size/ 1e6)
		    + " MB/device with " + std::to_string(gpu) + " devices in total\n")

	}
	(*MEKONG_aliasH)[*ptr] = std::move(devptrs);
	LOG("[MEKONG] [-] FUNC wrapMemAlloc()\n")
	return res.getRaw();
}

/*! \brief Broadcasts the data to all buffers linked to `dstDevPtr`.

    \sa wrapMemAlloc allocates memory on every device.
    \todo if we recognize that the user calls the MemcpyHtoD
          inside an iterative loop we should reuse the created
          memcpy object. Thus we need another wrapping function
          which searches for a created memcpy object.
*/
Mekong::MErawresult wrapMemcpyHtoD(Mekong::MEdeviceptr dstDevPtr,
                                   void* srcHostPtr,
                                   size_t size) {
	Mekong::MEresult res;
	// UP TO NOW WE COPY TO EVERY GPU
	auto broadcast = Mekong::MemCpyHtoD::createBroadcast(dstDevPtr, srcHostPtr,
	                                                     size, MEKONG_aliasH);
	res &= broadcast->exec();

	if (USER_OPTION_COLLECT_STATISTICS) {
		MEKONG_statistics.addCpyHtoD(broadcast);
	}

	// If we copy on a device ptr we should invalidate the results made
	// by a kernel launch on that device ptr
	MEKONG_buffer->erase(dstDevPtr); 
	// Mark the broadcast in the virtual buffer object. If we had not
	// registered the broadcast and the buffer had not been written by any
	// kernel, but the user called a device to host memory copy on that buffer,
	// the runtime would not know how to get the data from the different GPUs.
	MEKONG_buffer->setBroadcast(dstDevPtr);
	return res.getRaw();
}

/*! \brief Creates partitions, checks for dependencies and launches the kernels.

    In this wrapping function the bulk of the runtime's functionality is
    located. Firstly, we use the appropriate static kernel analysis to create
    the kernel launch object. The partitions get created in the constructor of
    the kernel launch class. After the launch has been created we check for
    existing dependencies, which is in fact a previous kernel launch, which
    wrote to a buffer the current kernel reads. After resolving the dependencies
    by creating and executing an appropriate dependency resoulution object
    (if it did not exist already), we execute the kernel launch object.
    This results in as many kernel launches as there are partitions belonging to
    that particular kernel launch object.
    \param func must be the function loaded with wrapModuleGetFunction. This is
           the transformed kernel function.
    \sa wrapModuleGetFunction to load the transformed kernel function
    \sa Mekong::KernelLaunch Constructor which initializes the partition creation
    \sa Mekong::Partition for the specific code which creates the partitions
    \sa Mekong::DepResolution for dependency resolutions
    \todo Support shared memory and stream usage.
*/
Mekong::MErawresult wrapLaunchKernel(Mekong::MEfunction func,
                                     unsigned blocksX,
                                     unsigned blocksY,
                                     unsigned blocksZ,
                                     unsigned threadsX,
                                     unsigned threadsY,
                                     unsigned threadsZ,
                                     unsigned shmem, Mekong::MEstream stream,
                                     void** args, void** extra) {

	auto throwError = [] (const std::string& msg) {
		throw std::runtime_error("SPACE Mekong, FUNC wrapLaunchKernel():\n" + msg);
	};

	// measure time for kernel launch object creation (not execution)
	auto timestamp = Clock::now(); 

	LOG("[MEKONG] [+] FUNC wrapLaunchKernel():\n")
	Mekong::MEresult res;

	// SEARCH FOR THE APPROPRIATE KERNEL INFO OBJECT
	std::shared_ptr<const Mekong::bsp_KernelInfo> currKernInfo;
	for (auto kinfo : MEKONG_kinfos) {
		if (kinfo->getName() == MEKONG_aliasH->atName(func)) {
			currKernInfo = kinfo;
			break;
		}
	}
	if (currKernInfo == nullptr) {
		throwError("I could not find any valid kernel analysis");
	}
	LOG("  * arg type size = "
	    + std::to_string(currKernInfo->getArgTypes().size()) + '\n')
	LOG("  * Creating bare KernelLaunch object\n")

	auto kl_and_bool = Mekong::KernelLaunch::getOrInsert(func,
	                                         { blocksX, blocksY, blocksZ },
	                                         { threadsX, threadsY, threadsZ },
	                                         shmem, args, currKernInfo,
	                                         MEKONG_aliasH);

	auto kl = std::get<0>(kl_and_bool);
	if (std::get<1>(kl_and_bool)) { // element was inserted successfully
		LOG("  * Inserted new launch into set (set size = "
		    + std::to_string(Mekong::KernelLaunch::all.size()) + ")\n")
	}
	else { // element was not inserted
		LOG("  * Launch already exists; I will take old one; "
		    "set size = " + std::to_string(Mekong::KernelLaunch::all.size())
		    + ";\n")
	}

	// Macro will expand to to a stream operator
	// which calls the overloaded stream operator of class kernelLaunch
	LOG("  * Configuration ") LOG(*kl) LOG('\n')
	LOG("  * Partitions:\n")

	if (USER_OPTION_LOG_ON) {
		for (const auto& partition : kl->getPartitions()) {
			LOG("    * ") LOG(*partition) LOG('\n')
		}
	}
	

	// CHECK DEVICE LIMITS IF MARKED IN USER CONFIGURATION
	// TODO this can also be optimized if we ask for the
	// device properties only once and save them.
	if (USER_OPTION_CHECK_DEVICE_LIMITS) {
		LOG("  * checking device limits...\n")
		for (auto& dev : MEKONG_aliasH->getDevs()) {
			auto gridMax = Mekong::meGetGridLimits(dev);
			auto blockMax = Mekong::meGetBlockLimits(dev);
			size_t maxThreads = Mekong::meGetThreadsPerBlockLimit(dev);
			size_t maxShMem = Mekong::meShMemPerBlockLimit(dev);
			for (const auto& prt : kl->getPartitions()) {
				if (prt->getGrid()[0] > gridMax[0] ||
				    prt->getGrid()[1] > gridMax[1] ||
				    prt->getGrid()[2] > gridMax[2]) {
					throw std::invalid_argument(
						"[MEKONG] FUNC wrapLaunchKernel(): "
						"Your kernel configuration "
						"exceeds the device limits: "
						"grid size is too big."
					);
				}
				if (prt->getBlock()[0] > blockMax[0] ||
				    prt->getBlock()[1] > blockMax[1] ||
				    prt->getBlock()[2] > blockMax[2]) {
					throw std::invalid_argument(
						"[MEKONG] FUNC wrapLaunchKernel(): "
						"Your kernel configuration "
						"exceeds the device limits: "
						"block size is too big."
					);
				}
				if (prt->getBlock()[0] *
				    prt->getBlock()[1] *
				    prt->getBlock()[2] > maxThreads) {
					throw std::invalid_argument(
						"[MEKONG] FUNC wrapLaunchKernel(): "
						"Your kernel configuration "
						"exceeds the device limits: "
						"maximum number of threads per "
						"block is too big."
					);
				}
				if (shmem > maxShMem) {
					throw std::invalid_argument(
						"[MEKONG] FUNC wrapLaunchKernel(): "
						"Your kernel configuration"
						"exceeds the device limits: "
						"not enough shared memory."
					);
				}
			}
		}
	}

	if (USER_OPTION_COLLECT_STATISTICS) {
		Duration klCreationTime = Clock::now() - timestamp;
		MEKONG_statistics.addKernelLaunchCreationTime(klCreationTime.count());
	}

	// restart clock for dep res creation
	timestamp = Clock::now();

	// SOLVE KERNEL DEPENDENCIES
	// 1. Search for Kernels (called 'master') which wrote to pointers 
	//    this kernel reads.
	// 2. Check if there is already a dependency resolve object of every
	//    'master' and this launch
	// 3. If not create a dependency resolve objects
	// 4. Execute the dependency resolve objects

	// 1.
	std::unordered_set<std::shared_ptr<Mekong::KernelLaunch>> masters;
	std::vector<std::shared_ptr<Mekong::DepResolution>> resolves;
	for (auto ptr : kl->getReads()) {
		if (MEKONG_buffer->isWritten(ptr)) {
			masters.insert((*MEKONG_buffer)[ptr]);
		}
	}
	LOG("  * found " + std::to_string(masters.size())
	    + " dependencies for this launch\n")
	bool foundResolve;
	size_t createdRes = 0;
	size_t foundRes = 0;
	for (auto master : masters) {
		foundResolve = false;
		for (auto oldResolve : MEKONG_depResolutions) {
			// 2.
			if (oldResolve->isResolutionOf(master, kl)) {
				resolves.push_back(oldResolve);
				foundResolve = true;
				++foundRes;
				break;
			}
		}
		// 3.
		if (!foundResolve) { 
			std::shared_ptr<Mekong::DepResolution> resolution(
				new Mekong::DepResolution(master, kl, MEKONG_aliasH)
			);
			++createdRes;
			MEKONG_depResolutions.insert(resolution);
			resolves.push_back(resolution);
			if (USER_OPTION_COLLECT_STATISTICS) {
				MEKONG_statistics.addResolution(resolution);
			}
		}
	}
	LOG("  * created " + std::to_string(createdRes) + " and found "
	    + std::to_string(foundRes) + " dependency resolver\n")

	// save the time needed to create the dep res objects
	Duration time_depResManagement = Clock::now() - timestamp;

	if (USER_OPTION_COLLECT_STATISTICS) {
		Duration depResCreationTime = Clock::now() - timestamp;
		MEKONG_statistics.addDepResCreationTime(depResCreationTime.count());
	}

	// 4.
	for (auto resolve : resolves) {
		res &= resolve->exec();
	}

	if (res.isSuccess() && !resolves.empty()) {
		LOG("  * dependencies resolved\n")
		LOG("    executed the following sub mem copies:\n")
		if (USER_OPTION_LOG_ON) {
			for (auto resolve : resolves) {
				for (auto& memcpy : resolve->getMemCpys()) {
					for (const auto& subcpy : *memcpy->getPattern()) {
						LOG("    ") LOG(subcpy) LOG('\n')
					}
					
				}
			}
		}
	}
	else if (!res.isSuccess()) {
		LOG("  * failed to execute the dependency resolver\n")
	}
	else {
		LOG("  * no dependency resolution neccessary\n")
	}

	kl->depsResolved(); // marked solved dependencies in kernel launch

	// EXECUTE KERNEL LAUNCH
	res &= kl->exec();
	if (res.isSuccess()) {
		LOG("  * submitted kernel launch\n")
	}
	else {
		LOG("  * failed to submit kernel launch\n")
	}

	// MARK NEW WRITES IN THE VIRTUAL MEMORY BUFFER
	for (Mekong::MEdeviceptr writePtr : kl->getWrites()) {
		MEKONG_buffer->setWritten(writePtr, kl);
	}

	LOG("[MEKONG] [-] FUNC wrapLaunchKernel()\n")

	if (USER_OPTION_COLLECT_STATISTICS) {
		MEKONG_statistics.addLaunch(kl);
	}

	return res.getRaw();
}

/*! \brief Synchronizes with all registered contexts.
    \sa wrapCtxCreate for context creation
*/
Mekong::MErawresult wrapCtxSynchronize() {
	LOG("[MEKONG] [+] FUNC wrapCtxSynchronize():\n")
	Mekong::MEresult res;
	for (auto& ctx : MEKONG_aliasH->getCtx()) {
		res &= Mekong::meCtxPushCurrent(ctx);
		res &= Mekong::meCtxSynchronize();
		res &= Mekong::meCtxPopCurrent(0);
	}
	if (res.isSuccess()) {
		LOG("[MEKONG] synchronized with all contexts\n")
	}
	else {
		LOG("[MEKONG] failed to synchronize with all contexts\n")
	}

	LOG("[MEKONG] [-] FUNC wrapCtxSynchronize()\n")
	return res.getRaw();
}

/*! \brief Calculate where the written elements are and copy them to the host.

    The assumption that one kernel writing to a buffer invalidates all previous
    writes enables us to search only for the last kernel, which wrote to the
    given device buffer. Once the kernel launch object is found we use it to
    calculate the written elements and to know which GPU holds them. Lastly, we
    copy the elements to the host buffer.  In fact this means that not written
    elements won't be copied to the host buffer (e.g. the border elements of a
    stencil).  A special case is an complete unwritten buffer, which we copy
    back to the host completely.

    \todo Support partially written buffers (e.g. a mesh refinement code on a
          stencil). To achieve this we have to keep track which GPU holds the
          lastly written elements of a buffer. We can use `numGPUs` `isl_set`
          objects to save that information for a certain buffer.
*/
Mekong::MErawresult wrapMemcpyDtoH(void* dstHostPtr,
                                   Mekong::MEdeviceptr srcDevPtr,
                                   size_t size) {
	LOG("[MEKONG] [+] FUNC wrapMemcpyDtoH():\n")
	Mekong::MEresult res;
	std::shared_ptr<Mekong::MemCpyDtoH> cpy; 

	// if pointer was not written by any kernel
	if (!MEKONG_buffer->isWritten(srcDevPtr)) {
		// and the wrapMemcpyHtoD copied it as a broadcast to all devices
		if (MEKONG_buffer->isBroadcast(srcDevPtr)) { 
			cpy = std::shared_ptr<Mekong::MemCpyDtoH>(
				      new Mekong::MemCpyDtoH(dstHostPtr, srcDevPtr, size,
				                             MEKONG_aliasH)
				  );
			res = cpy->exec(); // simply copy from first device to host
			if (res.isSuccess()) {
				LOG("[MEKONG] copied untouched broadcast data back to host "
				    "memory\n")
			}
			else {
				LOG("[MEKONG] failed to copy untouched broadcast data back to "
				    "host memory\n")
			}
		}
		else {
			throw std::invalid_argument(
				"SPACE Mekong, FUNC wrapMemcpyDtoH(): "
				"You want to copy data from a device "
				"pointer you never " "copied to or launched "
				"a kernel, which wrote to that pointer. "
				"This case is not supported"
			);
		}
	}
	else { // if the pointer has been written
		auto kl = (*MEKONG_buffer)[srcDevPtr];
		cpy = kl->getWrittenData(srcDevPtr, dstHostPtr);
		auto pattern = cpy->getPattern();
		LOG("[MEKONG] Going to exec memcpy: ") LOG(*cpy) LOG('\n')
		if (USER_OPTION_LOG_ON) {
			for (const auto& subcpy : *pattern) {
				LOG("  ") LOG(subcpy) LOG('\n')
			}
			if ((*pattern).empty()) {
				LOG("[MEKONG] WARNING: No memcpys executed\n")
			}
		}
		res = cpy->exec();
		if (USER_OPTION_LOG_ON) {
			if (res.isSuccess()) {
				LOG("[MEKONG] copied kernel data back to host memory\n")
			}
			else {
				LOG("[MEKONG] failed to copy data back to host memory\n")
			}
		}
	}
	LOG("[MEKONG] [-] FUNC wrapMemcpyDtoH()\n")
	if (USER_OPTION_COLLECT_STATISTICS) {
		MEKONG_statistics.addCpyDtoH(cpy);
	}
	return res.getRaw();
}

/*! \brief free all buffers linked to `ptr`.
    \sa wrapMemAlloc to link and allocate buffers.
*/
Mekong::MErawresult wrapMemFree(Mekong::MEdeviceptr ptr) {
	LOG("[MEKONG] [+] FUNC wrapMemFree():\n")
	Mekong::MEresult res;
	unsigned short gpu = 0;
	for (auto& devptr : (*MEKONG_aliasH)[ptr]) {
		res &= Mekong::meCtxPushCurrent(MEKONG_aliasH->getCtx()[gpu]);
		res &= Mekong::meMemFree(devptr);
		res &= Mekong::meCtxPopCurrent(0);
		++gpu;
	}
	if (USER_OPTION_LOG_ON) {
		if (res.isSuccess()) {
			LOG("[MEKONG] freed device memory pointer\n")
		}
		else {
			LOG("[MEKONG] failed to free device memory pointer\n")
		}
	}
	MEKONG_aliasH->erase(ptr);

	LOG("[MEKONG] [-] FUNC wrapMemFree()\n")
	return res.getRaw();
}

/*! \brief Destroy all contexts linked to `ctx`.
    \sa wrapCtxCreate to create and link contexts.
*/
Mekong::MErawresult wrapCtxDestroy(Mekong::MEcontext ctx) {
	LOG("[MEKONG] [+] FUNC wrapCtxDestroy():\n") 
	Mekong::MEresult res;
	for (auto& context : (*MEKONG_aliasH)[ctx]) {
		res &= Mekong::meCtxDestroy(context);
	}
	if (USER_OPTION_LOG_ON) {
		if (res.isSuccess()) {
			LOG("[MEKONG] context destroyed\n")
		}
		else {
			LOG("[MEKONG] failed to destroy context\n")
		}
	}
	MEKONG_aliasH->erase(ctx);
	LOG("[MEKONG] [-] FUNC wrapCtxDestroy()\n")
	return res.getRaw();
}

/*! \brief Can be used for debugging purposes
*/
Mekong::MErawresult getDataFromDevice(void* dst, Mekong::MEdeviceptr src,
                                      size_t size, int dev) {
	Mekong::MEresult res;
	res &= Mekong::meMemcpyDtoH(dst, (*MEKONG_aliasH)[src][dev], size);
	return res.getRaw();
}

/*! \brief Reports Mekong's statistics to the standard output.

    The function immediately returns if the user configured no output
    in the configuration file. The host code transformation inserts a call of
    this function in front of every return statement. If 
    USER_OPTION_COLLECT_STATISTICS is set to true in the user configuration
    file, the output will be more detailed.
*/
void MEKONG_report() {
	if (!USER_OPTION_MAKE_REPORT) {
		return;
	}

	using std::cout;
	using std::endl;
	cout << std::setprecision(6);

	std::map<std::string, std::string> kernel2partitioning;
	for (auto kinfo : MEKONG_kinfos) {
		kernel2partitioning[kinfo->getName()] =
			kinfo->getPartitioning()->getSplitStr();
	}

	int numDev;
	try {
		numDev = MEKONG_aliasH->getNumDev();
	}
	catch (...) {
		cout << "[MEKONG] Program Report: "
		     << "You did not register any device" << endl;
		return;
	}

	cout << endl;
	cout << "[MEKONG] Program Report:" << endl;
	cout << endl;
	cout << "# Alias Handle Information" << endl;
	cout << endl;
	cout << "  - number of devices = " << numDev << endl;
	//cout << "  - number of application device pointers = ";
	//cout << MEKONG_aliasH->getDevPtrMap().size() << endl;

	cout << "  - number of application kernel functions = ";
	cout << MEKONG_aliasH->getFuncMap().size() << endl;

	if (!USER_OPTION_COLLECT_STATISTICS) {
		return;
	}

	cout << endl;
	cout << "# Memory Copy Information:" << endl;
	cout << "This excludes all memory copy operations, which had to be done ";
	cout << "due to inter kernel dependencies." << endl;
	cout << endl;

	cout << "  - total num memcpy executions = ";
	cout << MEKONG_statistics.getNumMemCpy() << endl;

	cout << "  - num HtoD memcpy executions = ";
	cout << MEKONG_statistics.getNumMemCpy(Mekong::HtoD) << endl;

	cout << "  - num DtoH memcpy executions = ";
	cout << MEKONG_statistics.getNumMemCpy(Mekong::DtoH) << endl;

	cout << "  - total memcpy time = ";
	cout << MEKONG_statistics.getMemCpyTime() << " s" << endl;

	cout << "  - HtoD memcpy time = ";
	cout << MEKONG_statistics.getMemCpyTime(Mekong::HtoD) << " s" << endl;

	cout << "  - DtoH memcpy time = ";
	cout << MEKONG_statistics.getMemCpyTime(Mekong::DtoH) << " s" << endl;

	cout << "  - total memcpy size = ";
	cout << (double) MEKONG_statistics.getMemCpySize() / 1e6 << " MB" << endl;

	cout << "  - HtoD memcpy size = ";
	cout << (double) MEKONG_statistics.getMemCpySize(Mekong::HtoD) / 1e6;
	cout << " MB" << endl;

	cout << "  - DtoH memcpy size = ";
	cout << (double) MEKONG_statistics.getMemCpySize(Mekong::DtoH) / 1e6;
	cout << " MB" << endl;

	cout << "  - total Bandwidth = ";
	cout << MEKONG_statistics.getMemBW() << " GB/s" << endl;

	cout << "  - HtoD Bandwidth = ";
	cout << MEKONG_statistics.getMemBW(Mekong::HtoD) << " GB/s" << endl;

	cout << "  - DtoH Bandwidth = ";
	cout << MEKONG_statistics.getMemBW(Mekong::DtoH) << " GB/s" << endl;

	cout << endl;
	cout << "# Dependency Resolution Information:" << endl;
	cout << endl;

	cout << "  - num dep resolution executions = ";
	cout << MEKONG_statistics.getNumDepResExecs() << endl;

	cout << "  - num dep res objects = ";
	cout << MEKONG_statistics.getNumDepResObjects() << endl;

	cout << "  - total dep res creation time = ";
	cout << MEKONG_statistics.getDepResCreationTime() << " s" << endl;

	cout << "  - total dep res time = ";
	cout << MEKONG_statistics.getDepResExecTime() << " s" << endl;

	cout << "  - total dep res memcpy size = ";
	cout << (double) MEKONG_statistics.getDepResCpySize() / 1e6 << " MB" << endl;

	cout << endl;
	cout << "# Kernel Launch Information" << endl;
	cout << endl;

	cout << "  - num launch executions = ";
	cout << MEKONG_statistics.getNumLaunchExecs() << endl;

	cout << "  - num launch objects = ";
	cout << MEKONG_statistics.getNumLaunchObjects() << endl;

	cout << "  - kernel launch object creation time = ";
	cout << MEKONG_statistics.getLaunchCreationTime() << " s" << endl;

	for (auto& kernel_part : kernel2partitioning) {
		cout << "  - kernel name = " << std::get<0>(kernel_part) << endl;
		cout << "    partitioning = " << std::get<1>(kernel_part) << endl;
	}

	cout << "  - arg access time = ";
	cout << MEKONG_statistics.getArgAccessTime() << " s" << endl;

	cout << "  - linearization time = ";
	cout << MEKONG_statistics.getLinearizationTime() << " s" << endl;

	cout << "  - num arg access calls = ";
	cout << MEKONG_statistics.getNumArgAccessCalls() << endl;
	
	cout << "  - num arg access calcs = ";
	cout << MEKONG_statistics.getNumArgAccessCalcs() << endl;

	cout << endl;
	cout << "[MEKONG] Report End" << endl;
}
