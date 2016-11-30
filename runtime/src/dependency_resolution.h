#ifndef MEKONG_DEPRESOLVE_H
#define MEKONG_DEPRESOLVE_H

#include "kernel_launch.h"
#include "memory_copy.h"
#include "mekong-cuda.h"
#include "alias_handle.h"

#include <stdexcept>
#include <memory>
#include <vector>
#include <ostream>

namespace Mekong {

class DepResolution {
	public:
		//! master was already executed and slave's execution depends on master
		DepResolution(shared_ptr<KernelLaunch> master,
		           shared_ptr<KernelLaunch> slave,
		           shared_ptr<AliasHandle> aliasH);
		DepResolution(vector<unique_ptr<MemCpyDtoD>>&& memcpys);

		MEresult exec();
		MEresult syncWithMaster() const;

		bool isResolutionOf(shared_ptr<KernelLaunch> master,
		                    shared_ptr<KernelLaunch> slave) const;
		bool isEmpty() const;
		shared_ptr<KernelLaunch> getMaster() const;
		shared_ptr<KernelLaunch> getSlave() const;
		double getTime() const;
		size_t getExecs() const;
		const vector<unique_ptr<MemCpyDtoD>>& getMemCpys() const;


	private:

		//! intersects the two arg access objects and creates the
		//! correct memSubCopy objects to resolve the dependencies.
		//! The function returs the sub cpys and the tot size of
		//! copied bytes
		static vector<MemSubCopy>
		memCpyIntersections(const ArgAccess& master,
		                    const ArgAccess& slave,
		                    shared_ptr<const bsp_ArgType> type);

		size_t executions_ = 0;
		double time_ = 0;

		const shared_ptr<AliasHandle> aliasH_ = nullptr;
		const shared_ptr<KernelLaunch> master_ = nullptr;
		const shared_ptr<KernelLaunch> slave_ = nullptr;

		// a dependency resolve can have multiple memcpys
		// as we can have inter kernel dependencies on
		// different device ptrs
		const vector<unique_ptr<MemCpyDtoD>> memcpys_;
		vector<unique_ptr<MemCpyDtoD>> initMemcpys() const;
};

ostream& operator<<(ostream& out, const DepResolution& depRes); 

}; // namespace end

#endif
