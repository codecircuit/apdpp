#include "mekong-cuda.h"

#include <stdexcept>

namespace Mekong {

using namespace std;

MEresult::MEresult() { res_ = CUDA_SUCCESS; }

MEresult::MEresult(MErawresult res) : res_(res) {}

MEresult::MEresult(const MEresult& other) { res_ = other.res_; }

bool MEresult::isSuccess() const {
	return res_ == CUDA_SUCCESS;
}

MErawresult MEresult::getRaw() const {
	return res_;
}

MEresult& MEresult::operator&=(const MEresult& other) {
	if (!other.isSuccess() && this->isSuccess()) {
		this->res_ = other.res_;
	}
	return *this;
}

MEresult operator&&(const MEresult& a, const MEresult& b) {
	return MEresult(a) &= b;
}

MEresult meInit(unsigned flags) {
	return cuInit(flags);
}

MEresult meDeviceGetCount(int* count) {
	return cuDeviceGetCount(count);
}

MEresult meDeviceGet(MEdevice* device, int ordinal) {
	return cuDeviceGet(device, ordinal);
}

MEresult meDeviceComputeCapability(int* major, int* minor, MEdevice dev) {
	return cuDeviceComputeCapability(major, minor, dev);
}

MEresult meCtxCreate(MEcontext* pctx, unsigned int flags, MEdevice dev) {
	return cuCtxCreate(pctx, flags, dev);
}

MEresult meCtxSynchronize() {
	return cuCtxSynchronize();
}

MEresult meCtxPushCurrent(MEcontext ctx) {
	return cuCtxPushCurrent(ctx);
}

MEresult meCtxDestroy(MEcontext ctx) {
	return cuCtxDestroy(ctx);
}

MEresult meCtxPopCurrent(MEcontext* ctx) {
	return cuCtxPopCurrent(ctx);
}

MEresult meModuleLoad(MEmodule* module, const char* fname) {
	return cuModuleLoad(module, fname);
}

MEresult meModuleGetFunction(MEfunction* hfunc, MEmodule hmod, const char* name) {
	return cuModuleGetFunction(hfunc, hmod, name);
}

MEresult meMemAlloc(MEdeviceptr* dptr, size_t size) {
	return cuMemAlloc(dptr, size);
}

MEresult meMemcpyHtoD(MEdeviceptr dst, const void* src, size_t size) {
	return cuMemcpyHtoD(dst, src, size);
}

MEresult meMemcpyDtoH(void* dst, MEdeviceptr src, size_t size) {
	return cuMemcpyDtoH(dst, src, size);
}

MEresult meMemcpyDtoD(MEdeviceptr dst, MEdeviceptr src, size_t size) {
	return cuMemcpyDtoD(dst, src, size);
}

MEresult meMemcpyHtoDAsync(MEdeviceptr dst, const void* src, size_t size, MEstream hStream) {
	return cuMemcpyHtoDAsync(dst, src, size, hStream);
}

MEresult meMemcpyDtoHAsync(void* dst, MEdeviceptr src, size_t size, MEstream hStream) {
	return cuMemcpyDtoHAsync(dst, src, size, hStream);
}

MEresult meMemcpyDtoDAsync(MEdeviceptr dst, MEdeviceptr src, size_t size, MEstream hStream) {
	return cuMemcpyDtoDAsync(dst, src, size, hStream);
}

MEresult meMemFree(MEdeviceptr dptr) {
	return cuMemFree(dptr);
}


MEresult meLaunchKernel(MEfunction f,
						unsigned gridDimX,
						unsigned gridDimY,
						unsigned gridDimZ,
						unsigned blockDimX,
						unsigned blockDimY,
						unsigned blockDimZ,
						unsigned sharedMemBytes,
						MEstream hStream,
						void** kernelArgs,
						void** extra) {
	return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
	                         blockDimX, blockDimY, blockDimZ,
	                         sharedMemBytes, hStream, kernelArgs, extra);
}

array<unsigned, 3> meGetGridLimits(MEdevice dev) {
	CUdevprop prop;
	MEresult res;
	res &= cuDeviceGetProperties(&prop, dev);
	if (!res.isSuccess()) {
		throw runtime_error("Error when checking device limits. You can turn this check off.");
	}
	return {(unsigned) prop.maxGridSize[0],
	        (unsigned) prop.maxGridSize[1],
	        (unsigned) prop.maxGridSize[2]};
}

array<unsigned, 3> meGetBlockLimits(MEdevice dev) {
	CUdevprop prop;
	MEresult res;
	res &= cuDeviceGetProperties(&prop, dev);
	if (!res.isSuccess()) {
		throw runtime_error("Error when checking device limits. You can turn this check off.");
	}
	return { (unsigned) prop.maxThreadsDim[0],
	         (unsigned) prop.maxThreadsDim[1],
	         (unsigned) prop.maxThreadsDim[2] };
}

size_t meGetThreadsPerBlockLimit(MEdevice dev) {
	CUdevprop prop;
	MEresult res;
	res &= cuDeviceGetProperties(&prop, dev);
	if (!res.isSuccess()) {
		throw runtime_error("Error when checking device limits. You can turn this check off.");
	}
	return prop.maxThreadsPerBlock;
}

size_t meShMemPerBlockLimit(MEdevice dev) {
	CUdevprop prop;
	MEresult res;
	res &= cuDeviceGetProperties(&prop, dev);
	if (!res.isSuccess()) {
		throw runtime_error("Error when checking device limits. You can turn this check off.");
	}
	return prop.sharedMemPerBlock;
}

} // namespace end
