#include "mekong-cuda.h"
#include "alias_handle.h"

#include <map>
#include <vector>
#include <stdexcept>
#include <string>

namespace Mekong {

using namespace std;

//! Return linked devices
vector<MEdevice>& AliasHandle::operator[](const MEdevice& dev) {
	return devMap_[dev];
}

//! Return linked contexts
vector<MEcontext>& AliasHandle::operator[](const MEcontext& ctx) {
	return ctxMap_[ctx];
}

//! Return linked modules
vector<MEmodule>& AliasHandle::operator[](const MEmodule& mod) {
	return modMap_[mod];
}

//! Return linked functions.

//! We have to load a kernel function into every module for every GPU,
//! although it is always logically the same function.
vector<MEfunction>& AliasHandle::operator[](const MEfunction& func) {
	return funcMap_[func];
}

//! Return linked device pointers
vector<MEdeviceptr>& AliasHandle::operator[](const MEdeviceptr& ptr) {
	return ptrMap_[ptr];
};

/*! \brief Give the name to a registered kernel function.

     We need this to map the correct kernel analysis to the kernel function,
     when the kernel launch object is created.
     \sa wrapLaunchKernel The kernel launch objects are created here.
*/
string& AliasHandle::atName(const MEfunction& func) {
	return nameMap_[func];
}

//! In case of a cuMemFree, we want to delete the stored information
void AliasHandle::erase(const MEdeviceptr& ptr) {
	ptrMap_.erase(ptr);
}

//! In case of a cuCtxDestroy, we want to delete the stored information
void AliasHandle::erase(const MEcontext& ctx) {
	ctxMap_.erase(ctx);
}

//! Get the registered contexts
const vector<MEcontext>& AliasHandle::getCtx() const {
	// this is going to be interesting in case of a multithreaded
	// host code
	if (ctxMap_.size() != 1) {
		throw runtime_error("SPACE Mekong, CLASS AliasHandle, FUNC getCtx(): "
		                    "context mapping is ambigious. Maybe you created "
		                    "more than one context in the host code?");
	}
	return ctxMap_.begin()->second;
}

//! Get the number of registered devices
unsigned short AliasHandle::getNumDev() const {
	if (devMap_.empty()) {
		throw runtime_error("Mekong AliasHandle object does registered any devices!");
	}
	return devMap_.begin()->second.size();
}

//! Get the registered devices
const vector<MEdevice>& AliasHandle::getDevs() const {
	if (devMap_.size() != 1) {
		throw runtime_error("SPACE Mekong, CLASS AliasHandle, FUNC getDevs(): "
		                    "number of registered device is ambigious. Maybe you"
		                    "called cuGetDevice() more than once in your code?");
	}
	return devMap_.begin()->second;
}

//! Returns a map representing the linkage between device buffers.
const AliasHandle::ptrMap_t& AliasHandle::getDevPtrMap() const {
	return ptrMap_;
}

//! Returns a map representing the linkage between the device functions.
const AliasHandle::funcMap_t& AliasHandle::getFuncMap() const {
	return funcMap_;
}

};
