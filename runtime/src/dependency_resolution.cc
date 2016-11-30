#include "kernel_launch.h"
#include "memory_copy.h"
#include "mekong-cuda.h"
#include "alias_handle.h"
#include "dependency_resolution.h"

#include <stdexcept>
#include <memory>
#include <vector>
#include <ostream>

namespace Mekong {

DepResolution::DepResolution(shared_ptr<KernelLaunch> master,
                       shared_ptr<KernelLaunch> slave,
                       shared_ptr<AliasHandle> aliasH) 
		: master_(master),
		  slave_(slave),
		  aliasH_(aliasH),
		  memcpys_(initMemcpys()) {}

//! Sync all devices; Execute resolving mem copies; Sync all devices again;
MEresult DepResolution::exec() {
	auto time_exec_begin = Clock::now();
	MEresult res;

	// Ensure that the data is already calculated
	// which we want to copy
	res &= syncWithMaster();

	for (const auto& memcpy : memcpys_) {
		res &= memcpy->exec();
	}

	// SYNCHRONIZE
	// TODO later we should have the possibility to synchronize with the
	// specific kernel launch only. Up to now we make a total sync
	for (auto& ctx : aliasH_->getCtx()) {
		res &= meCtxPushCurrent(ctx);
		res &= meCtxSynchronize();
		res &= meCtxPopCurrent(0);
	}
	++executions_;
	Duration time_exec = Clock::now() - time_exec_begin;
	time_ += time_exec.count();
	return res;
}

//! Ensures that all data needed by slave is finished in the master launch.

//! This means not neccessarily a total synchronization between the host
//! and all devices. It can be possible to synchronize with certain
//! launched partitions, which contains the dependencies.
MEresult DepResolution::syncWithMaster() const {
	MEresult res;
	for (auto& ctx : aliasH_->getCtx()) {
		res &= meCtxPushCurrent(ctx);
		res &= meCtxSynchronize();
		res &= meCtxPopCurrent(0);
	}
	return res;
}

/*! \brief Pointer based comparison.

    This is valid, because we create only unique kernel launch objects.
    Thus if both kernel launches are equal (the same grid size, block size, ...)
    the pointers must be equal.
    \sa Mekong::KernelLaunch::all
    \sa Mekong::KernelLaunch::equal_to
    \sa Mekong::KernelLaunch::operator==
*/
bool DepResolution::isResolutionOf(shared_ptr<KernelLaunch> master,
                                   shared_ptr<KernelLaunch> slave) const {
	return master == master_ && slave == slave_;
}

//! If a dep res object contains no mem cpy objects it is empty
bool DepResolution::isEmpty() const {
	return getMemCpys().empty();
}

shared_ptr<KernelLaunch> DepResolution::getMaster() const {
	return master_;
}

shared_ptr<KernelLaunch> DepResolution::getSlave() const {
	return slave_;
}

double DepResolution::getTime() const {
	return time_;
}

size_t DepResolution::getExecs() const {
	return executions_;
}

const vector<unique_ptr<MemCpyDtoD>>& DepResolution::getMemCpys() const {
	return memcpys_;
}

/*! \brief Creates the mem copies out of the accessed indices

     Example: you have two GPUs and kernel launch `master` writes
     to a buffer launch `slave` reads. Both launches have an
     `ArgAccess` object belonging to that buffer. The `ArgAccess`
     object describes which _indices_ are read and written. This
     function calculates the intersection of the indices and creates
     the appropriate `MemSubCopy` objects.
     \param type is necessary, because we have to transfer the indices
            to accessed Bytes. Thus we need to know the size of one
            array element.
     \sa ArgAccess::intersectIntervals
     \sa Mekong::MemSubCopy
*/
vector<MemSubCopy>
DepResolution::memCpyIntersections(const ArgAccess& master,
                                   const ArgAccess& slave,
                                   shared_ptr<const bsp_ArgType> type) {
	vector<MemSubCopy> res;
	const auto& mMap = master.getMap();
	const auto& sMap = slave.getMap();
	for (auto sit = sMap.begin(); sit != sMap.end(); ++sit) {
		for (auto mit = mMap.begin(); mit != mMap.end(); ++mit) {
			if (mit->first != sit->first) { // if gpus are not equal
				for (const auto& sRange : sit->second) {
					for (const auto& mRange : mit->second) {
						auto isect = ArgAccess::intersectIntervals(sRange, mRange);
						// if intersection is not range is not null
						if (get<0>(isect) != 0 || get<1>(isect) != 0) {
							// TODO optimize the construction of
							// mem sub copy objects
							MemSubCopy subcpy;
							subcpy.src  = mit->first;
							subcpy.dst  = sit->first;
							// as we do no reshaping yet we have the same start
							// position on both arrays
							subcpy.from = type->getElSize() * get<0>(isect);
							subcpy.to   = type->getElSize() * get<0>(isect);
							subcpy.size = (get<1>(isect) - get<0>(isect))
							              * type->getElSize();
							res.push_back(move(subcpy));
						}
					}
				}
			}
		}
	}
	return res;
}

/*! \brief Creates the dependency resolving memory copy objects.

    You can have a `master` kernel launch which writes to multiple
    buffers read by kernel launch `slave`. Thus you need to resolve
    dependencies on multiple buffers. That is the purpose of this
    function.
*/
vector<unique_ptr<MemCpyDtoD>> DepResolution::initMemcpys() const {

	vector<unique_ptr<MemCpyDtoD>> res;

	// We search now for pointer kernel arguments, which
	// are read or written
	int slaveArgId = 0;
	for (const auto& arg : slave_->getArgs()) {
		if (arg->getType()->getPtrlvl() == 1  && arg->getType()->isRead()) {
			// ask the master if it has a argument, which holds the
			// value of the device ptr
			int masterArgId = master_->getArgId(arg->asDevPtr());
			if (masterArgId != -1) {
				auto masterArg = master_->getArgFromId(masterArgId);
				// ask the master if that argument is modified in
				// the kernel, thus we have inter kernel
				// dependencies
				if (masterArg->getType()->isModified()) {
					auto masterAcc = master_->getWriteArgAccess(masterArgId);
					auto slaveAcc = slave_->getReadArgAccess(slaveArgId);
					// get the intersection of this two accesses
					auto subcpys = memCpyIntersections(*masterAcc, *slaveAcc, arg->getType());
					shared_ptr<const vector<MemSubCopy>> memPattern(
						new vector<MemSubCopy>(move(subcpys)));
					unique_ptr<MemCpyDtoD> uptr(
						new MemCpyDtoD(arg->asDevPtr(), memPattern, aliasH_, false)); // false-> no sync in memcpys
					res.push_back(move(uptr));
				}
			}
		}
		++slaveArgId;
	}
	return res;
}

ostream& operator<<(ostream& out, const DepResolution& depRes) {
	out << "DepResObj has the memsubcpys:" << endl;
	for (const auto& cpy : depRes.getMemCpys()) {
		for (const auto& subcpy : *cpy->getPattern()) {
		out << "\t" << subcpy << endl;
		}
	}
	return out;
}

}; // namespace end
