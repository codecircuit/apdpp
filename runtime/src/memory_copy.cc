/*! \file memory_copy.cc
    \brief Source file to handle memory copy operations in Mekong's context.

    \todo This file and all classes should be rewritten in a more simple style.
          Up to now we have one base class which contains a lot functionality
          and 3 (HtoD, DtoH, DtoD) child classes, which are merely wrapper
          classes. They should be replaced by one simple written class.
          Moreover the complex usage of the MemSubCopy is not a good style.
*/
#include "mekong-cuda.h"
#include "alias_handle.h"
#include "memory_copy.h"

#include <memory>
#include <vector>
#include <string>
#include <ostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <chrono>

namespace Mekong {

using namespace std;

using Clock = chrono::high_resolution_clock;
using Duration = chrono::duration<double>;

MemCpyDtoD::MemCpyDtoD(MEdeviceptr dstsrc, shared_ptr<const MemPattern> pmp,
                       shared_ptr<AliasHandle> aliasH, bool sync) :
	MemCpy<MEdeviceptr, const MEdeviceptr>(dstsrc, dstsrc, DtoD, pmp, aliasH, sync) {}

MemCpyDtoH::MemCpyDtoH(void* dst, const MEdeviceptr src, shared_ptr<const MemPattern> pmp,
		   shared_ptr<AliasHandle> aliasH, bool sync) :
		MemCpy<void*, const MEdeviceptr>(dst, src, DtoH, pmp, aliasH, sync) {}

//! trivial constructor, which wraps a simple single gpu to single host memory copy operation
MemCpyDtoH::MemCpyDtoH(void* dst, const MEdeviceptr src, size_t size,
		   shared_ptr<AliasHandle> aliasH) :
		MemCpy<void*, const MEdeviceptr>(dst, src, DtoH, createTrivialMemPattern(size), aliasH, true) {}


MemCpyHtoD::MemCpyHtoD(MEdeviceptr dst, const void* src,
                       shared_ptr<const MemPattern> pmp,
                       shared_ptr<AliasHandle> aliasH, bool sync, size_t size) :
	MemCpy<MEdeviceptr, const void*>(dst, src, HtoD, pmp, aliasH, sync, size) {}



ostream& operator<<(ostream& out, const MemSubCopy& subcpy) {
	out << "(src: " << subcpy.src << ", dst: " << subcpy.dst
	    << ", from: " << subcpy.from << ", to: " << subcpy.to
	    << ", size: " << subcpy.size << " Byte)";
	return out;
}

bool operator==(const MemSubCopy& a, const MemSubCopy& b) {
	return a.src == b.src && a.dst == b.dst && a.from == b.from && a.to == b.to && a.size == b.size;
}

MEresult MemCpyHtoD::exec() {
	if (aliasH_->getCtx().size() != aliasH_->getNumDev()) {
		throw runtime_error("CLASS MemCpyHtoD, FUNC exec(): not enough device contexts"
		                    "for number of devices in alias handle object");
	}
	if ((*aliasH_)[dst_].size() != aliasH_->getNumDev()) {
		throw runtime_error("CLASS MemCpyHtoD, FUNC exec(): not enough alias pointer for every device in"
		                    "alias handle object. NumDev = " + to_string(aliasH_->getNumDev()) +
		                    ", Num alias pointer = " + to_string((*aliasH_)[dst_].size()));
	}
	MEresult res;
	auto time_exec_begin = Clock::now();
#ifdef SOFIRE
	if (!this->isBroadcast_) {
		for (auto subcpy : *pmp_) {
			if (subcpy.src != -1 || subcpy.dst < 0) {
				throw invalid_argument("Namespace: Mekong, Class MemCpyHtoD, Func exec():\n"
				                       "memcpy marked as Host to Device, but configuration of\n"
				                       "sub copy objects is not consistent with this.");
			}
			res &= meCtxPushCurrent(aliasH_->getCtx().at(subcpy.dst));
			res &= meMemcpyHtoDAsync((*aliasH_)[dst_].at(subcpy.dst) + subcpy.to,
			       (unsigned char*) src_ + subcpy.from, subcpy.size, 0);
			res &= meCtxPopCurrent(nullptr);
			if (!res.isSuccess()) {
				break;
			}
		}
	}
	else { // is broadcast, thus use dominiks library
		comm.destroyCircle();
		for (int gpu = 0; gpu < aliasH_->getNumDev(); ++gpu) {
			auto dstPointer = (*aliasH_)[dst_].at(gpu);
			comm.setDeviceMemory(gpu, (char*) dstPointer);
		}
		comm.setDeviceMemory(comm.getHostDevNum(), (char*) src_);
		int src = comm.getHostDevNum();
		vector<int> dests;
		for (int gpu = 0; gpu < aliasH_->getNumDev(); ++gpu) {
			dests.push_back(gpu);
		}
		comm.createCircle(dests, src);
		// syncs automatically
		comm.broadcast(dests, comm.getHostDevNum(), orgSize_);
	}
	// synchronize with each context
	if (sync_ && !isBroadcast_) {
		for (auto ctx : aliasH_->getCtx()) {
			if (!res.isSuccess()) { 
				break;
			}
			res &= meCtxPushCurrent(ctx);
			res &= meCtxSynchronize();
			res &= meCtxPopCurrent(nullptr);
		}
	}
#else
	for (auto subcpy : *pmp_) {
		if (subcpy.src != -1 || subcpy.dst < 0) {
			throw invalid_argument("Namespace: Mekong, Class MemCpyHtoD, Func exec():\n"
			                       "memcpy marked as Host to Device, but configuration of\n"
			                       "sub copy objects is not consistent with this.");
		}
		res &= meCtxPushCurrent(aliasH_->getCtx().at(subcpy.dst));
		res &= meMemcpyHtoDAsync((*aliasH_)[dst_].at(subcpy.dst) + subcpy.to,
			   (unsigned char*) src_ + subcpy.from, subcpy.size, 0);
		res &= meCtxPopCurrent(nullptr);
		if (!res.isSuccess()) {
			break;
		}
	}
	// synchronize with each context
	if (sync_) {
		for (auto ctx : aliasH_->getCtx()) {
			if (!res.isSuccess()) { 
				break;
			}
			res &= meCtxPushCurrent(ctx);
			res &= meCtxSynchronize();
			res &= meCtxPopCurrent(nullptr);
		}
	}
#endif
	Duration time_exec = Clock::now() - time_exec_begin;
	time_ += time_exec.count();
	++executions_;
	return res;
}

MEresult MemCpyDtoH::exec() {
	if (aliasH_->getCtx().size() != aliasH_->getNumDev()) {
		throw runtime_error("CLASS MemCpyDtoH, FUNC exec(): not enough device contexts"
		                    "for number of devices in alias handle object");
	}
	if ((*aliasH_)[src_].size() != aliasH_->getNumDev()) {
		throw runtime_error("CLASS MemCpyDtoH, FUNC exec(): not enough alias pointer for every device in"
		                    "alias handle object. NumDev = " + to_string(aliasH_->getNumDev()) +
		                    ", Num alias pointer = " + to_string((*aliasH_)[src_].size()));
	}
	MEresult res;
	auto time_exec_begin = Clock::now();
	for (auto subcpy : *pmp_) {
		if (subcpy.src < 0 || subcpy.dst != -1) {
			throw invalid_argument("Namespace: Mekong, Class MemCpyDtoH, Func exec():\n"
								   "memcpy marked as Device to Host, but configuration of\n"
								   "sub copy objects is not consistent with this.");
		}
		res &= meCtxPushCurrent(aliasH_->getCtx().at(subcpy.src));
		auto aim = (*aliasH_)[src_].at(subcpy.src) + subcpy.from;
		res &= meMemcpyDtoHAsync((unsigned char*) dst_ + subcpy.to , aim
								 , subcpy.size, 0);
		res &= meCtxPopCurrent(nullptr);
		if (!res.isSuccess()) {
			break;
		}
	}
	// synchronize with each context
	if (sync_) {
		for (auto ctx : aliasH_->getCtx()) {
			if (!res.isSuccess()) { 
				break;
			}
			res &= meCtxPushCurrent(ctx);
			res &= meCtxSynchronize();
			res &= meCtxPopCurrent(nullptr);
		}
	}
	Duration time_exec = Clock::now() - time_exec_begin;
	time_ += time_exec.count();
	++executions_;
	return res;
}

shared_ptr<const MemCpyDtoH::MemPattern> MemCpyDtoH::createTrivialMemPattern(size_t size) const {
	MemSubCopy msb;
	msb.src = 0;
	msb.dst = -1;
	msb.from = 0;
	msb.to = 0;
	msb.size = size;
	return shared_ptr<const MemPattern>(new vector<MemSubCopy>(1, msb));
}

MEresult MemCpyDtoD::exec() {
	if (aliasH_->getCtx().size() != aliasH_->getNumDev()) {
		throw runtime_error("CLASS MemCpyDtoD, FUNC exec(): not enough device contexts"
		                    "for number of devices in alias handle object");
	}
	if ((*aliasH_)[src_].size() != aliasH_->getNumDev()) {
		throw runtime_error("CLASS MemCpyDtoD, FUNC exec(): not enough alias pointer for every device in"
		                    "alias handle object. NumDev = " + to_string(aliasH_->getNumDev()) +
		                    ", Num alias pointer = " + to_string((*aliasH_)[src_].size()));
	}
	MEresult res;
	auto time_exec_begin = Clock::now();
	for (auto subcpy : *pmp_) {
		if (subcpy.src < 0 || subcpy.dst < 0) {
			throw invalid_argument("Namespace: Mekong, Class MemCpyDtoD, Func exec():\n"
								   "memcpy marked as Device to Device, but configuration of\n"
								   "sub copy objects is not consistent with this.");
		}
		res &= meCtxPushCurrent(aliasH_->getCtx().at(subcpy.dst));
		res &= meMemcpyDtoDAsync((*aliasH_)[dst_].at(subcpy.dst) + subcpy.to,
								 (*aliasH_)[src_].at(subcpy.src) + subcpy.from, subcpy.size, 0);
		res &= meCtxPopCurrent(nullptr);
		if (!res.isSuccess()) {
			break;
		}
	}
	// synchronize with each context
	if (sync_) {
		for (auto ctx : aliasH_->getCtx()) {
			if (!res.isSuccess()) { 
				break;
			}
			res &= meCtxPushCurrent(ctx);
			res &= meCtxSynchronize();
			res &= meCtxPopCurrent(nullptr);
		}
	}

	Duration time_exec = Clock::now() - time_exec_begin;
	time_ += time_exec.count();
	++executions_;
	return res;
}

/*! \brief Makes a broadcast from from host to all devices.
    \param master the device id with the source data
*/
shared_ptr<MemCpyHtoD>
MemCpyHtoD::createBroadcast(MEdeviceptr dst, const void* src, size_t size,
                            shared_ptr<AliasHandle> aliasH) {
	// CREATE MEM PATTERN FIRST
#ifndef SOFIRE
	vector<MemSubCopy> subcpys(aliasH->getNumDev());
	unsigned short gpu = 0;
	for (auto& sc : subcpys) {
		sc.src = -1;
		sc.dst = gpu++;
		sc.from = 0;
		sc.to = 0;
		sc.size = size;
	}
	// CREATE MemCpyHtoD Object
	shared_ptr<const MemPattern> pattern(new vector<MemSubCopy>(move(subcpys)));
	shared_ptr<MemCpyHtoD> res(new MemCpyHtoD(dst, src, pattern, aliasH, true,
	                                          size));
#else
	shared_ptr<MemCpyHtoD> res(new MemCpyHtoD(dst, src, nullptr, aliasH, true,
	                                          size));
#endif
	res->isBroadcast_ = true;
	return res;
}

/*! \brief Makes a broadcast from from one device to all the others.
    \param master the device id with the source data
*/
shared_ptr<MemCpyDtoD>
MemCpyDtoD::createBroadcast(MEdeviceptr dstsrc, size_t size,
                            shared_ptr<AliasHandle> aliasH, unsigned short master) {
	// CREATE MEM PATTERN FIRST
	vector<MemSubCopy> subcpys(aliasH->getNumDev() - 1);
	unsigned short gpu = 0;
	for (auto& sc : subcpys) {
		if (gpu == master) {
			++gpu;
		}
		sc.src = master;
		sc.dst = gpu;
		sc.from = 0;
		sc.to = 0;
		sc.size = size;
		++gpu;
	}
	// CREATE MemCpyDtoD Object
	shared_ptr<const MemPattern> pattern(new vector<MemSubCopy>(move(subcpys)));
	shared_ptr<MemCpyDtoD> res(new MemCpyDtoD(dstsrc, pattern, aliasH));
	return res;
}

}; // namespace end
