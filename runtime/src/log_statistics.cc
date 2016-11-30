#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>

#include "memory_copy.h"
#include "dependency_resolution.h"
#include "partitioning.h"
#include "kernel_launch.h"
#include "log_statistics.h"

namespace Mekong {

using namespace std;

//! Returns the total time for copy operations in seconds.

//! This refers only to the wrapped memory copy operations, thus the
//! memory copy operations performed for dependency resolution are
//! not included.
double Statistics::getMemCpyTime() const {
	double time = 0;
	for (auto pmc : host2dev_) {
		time += pmc->getTime();
	}
	for (auto pmc : dev2host_) {
		time += pmc->getTime();
	}
	for (auto pmc : dev2dev_) {
		time += pmc->getTime();
	}
	return time;
}

//! Returns the time for copy operations of a certain kind (HtoD, DtoH, DtoD).

//! This refers only to the wrapped memory copy operations, thus the
//! memory copy operations performed for dependency resolution are
//! not included.
double Statistics::getMemCpyTime(MemCpyKind kind) const {
	double time = 0;
	if (kind == HtoD) {
		for (auto pmc : host2dev_) {
			time += pmc->getTime();
		}
	}
	else if (kind == DtoH) {
		for (auto pmc : dev2host_) {
			time += pmc->getTime();
		}
	}
	else if (kind == DtoD) {
		for (auto pmc : dev2dev_) {
			time += pmc->getTime();
		}
	}
	return time;
}

//! Returns the total (HtoD, DtoH, DtoD) amount of copied Bytes.

//! This refers only to the wrapped memory copy operations, thus the
//! memory copy operations performed for dependency resolution are
//! not included.
size_t Statistics::getMemCpySize() const {
	size_t size = 0;
	for (auto pmc : host2dev_) {
		size += pmc->getSize();
	}
	for (auto pmc : dev2host_) {
		size += pmc->getSize();
	}
	for (auto pmc : dev2dev_) {
		size += pmc->getSize();
	}
	return size;
}

/*! \brief Returns the amount of copied Bytes of a certain kind
           (e.g. HtoD, DtoH, DtoD)

    This refers only to the wrapped memory copy operations, thus the
    memory copy operations performed for dependency resolution are
    not included.
*/
size_t Statistics::getMemCpySize(MemCpyKind kind) const {
	size_t size = 0;
	if (kind == HtoD) {
		for (auto pmc : host2dev_) {
			size += pmc->getSize();
		}
	}
	else if (kind == DtoH) {
		for (auto pmc : dev2host_) {
			size += pmc->getSize();
		}
	}
	else if (kind == DtoD) {
		for (auto pmc : dev2dev_) {
			size += pmc->getSize();
		}
	}
	return size;
}

//! Returns the copied Bytes due to inter kernel dependency resolutions.
size_t Statistics::getDepResCpySize() const {
	size_t res = 0;
	for (auto r : resolutions_) {
		for (const auto& memcpy : r->getMemCpys()) {
			res += memcpy->getSize();
		}
	}
	return res;
}

//! Returns the number of memcpy executions
size_t Statistics::getNumMemCpy() const {
	size_t num = 0;
	for (auto h2d : host2dev_) {
		num += h2d->getExecutions();
	}
	for (auto d2h : dev2host_) {
		num += d2h->getExecutions();
	}
	for (auto d2d : dev2dev_) {
		num += d2d->getExecutions();
	}
	return num;
}

/*! \brief Returns the number of memcpy executions of a certain kind
    \sa MemCpyKind
*/
size_t Statistics::getNumMemCpy(MemCpyKind kind) const {
	size_t num = 0;
	if (kind == HtoD) {
		for (auto h2d : host2dev_) {
			num += h2d->getExecutions();
		}
	}
	else if (kind == DtoH) {
		for (auto d2h : dev2host_) {
			num += d2h->getExecutions();
		}
	}
	else if (kind == DtoD) {
		for (auto d2d : dev2dev_) {
			num += d2d->getExecutions();
		}
	}
	return num;
}

/*! \brief Returns the total number of calls of the getArgAccess
           function for every registered kernel launch object.

    This function is used to check for a correct program execution.
    Moreover you can recognize if the caching for getArgAccess calls
    works correctly, as it is an expensive function. A getArgAccess call
    does not neccessarily result in an ArgAccess creation.

    \sa getNumArgAccessCalcs not only calling but calculating an ArgAccess.
    \sa KernelLaunch for the getArgAccess function.
*/
unsigned Statistics::getNumArgAccessCalls() const {
	unsigned res = 0;
	for (auto l : launches_) {
		res += l->getNumArgAccessCalls();
	}
	return res;
}

/*! \brief Returns the total number of calculations of the getArgAccess
           function for every registered kernel launch object.

    This function is used to check for a correct program execution.
    Moreover you can recognize if the caching for getArgAccess calls
    works correctly, as it is an expensive function.

    \sa getNumArgAccessCalls
    \sa KernelLaunch for the getArgAccess function.
*/
unsigned Statistics::getNumArgAccessCalcs() const {
	unsigned res = 0;
	for (auto l : launches_) {
		res += l->getNumArgAccessCalcs();
	}
	return res;
}

/*! \brief Returns the number of dependency resolution executions.

    The execution of one dependency resolution object can result in multiple
    memory copy operations.
*/
unsigned Statistics::getNumDepResExecs() const {
	unsigned res = 0;
	for (auto r : resolutions_) {
		res += r->getExecs();
	}
	return res;
}

//! Returns the number of dependency resolution objects
unsigned Statistics::getNumDepResObjects() const {
	return resolutions_.size();
}

//! Returns the number of kernel launch executions
unsigned Statistics::getNumLaunchExecs() const {
	unsigned res = 0;
	for (auto kl : launches_) {
		res += kl->getExecs();
	}
	return res;
}

unsigned Statistics::getNumLaunchObjects() const {
	return launches_.size();
}

/*! \brief Returns the average memory bandwidth in GB/s of all memory copy
           operations.

    This refers only to the wrapped memory copy operations, thus the
    memory copy operations performed for dependency resolution are
    not included.
*/
double Statistics::getMemBW() const {
	if (getMemCpyTime() == 0) {
		return 0;
	}
	return (double) getMemCpySize() / 1e9 / getMemCpyTime();
}

/*! \brief Returns the average memory bandwidth in GB/s of a specified memcpy
           kind.

    This refers only to the wrapped memory copy operations, thus the
    memory copy operations performed for dependency resolution are
    not included.
*/
double Statistics::getMemBW(MemCpyKind kind) const {
	if (getMemCpyTime(kind) == 0) {
		return 0;
	}
	return (double) getMemCpySize(kind) / 1e9 / getMemCpyTime(kind);
}

/*! \brief Time to **calculate** the inter kernel dependencies.

    This time excludes the time needed for exchanging data between the GPUs.
*/
double Statistics::getDepResCreationTime() const {
	return depResCreationTime_;
}

/*! \brief Returns the total arg access time by all
           registered kernel launch objects.

    The getArgAccess function can be indirectly called by the
    wrapLaunchKernel and the wrapMemcpyDtoH function. Thus this is not only the
    time needed to calculate inter kernel dependencies. In fact the ArgAccess
    time is a subset of application_kernel_time + application_dtoh_time.
    \sa KernelLaunch for the getArgAccess function
    \sa wrapLaunchKernel
    \sa wrapMemcpyDtoH
*/
double Statistics::getArgAccessTime() const {
	double res = 0;
	for (auto l : launches_) {
		res += l->getArgAccessTime();
	}
	return res;
}

/*! \brief Returns the total linearization time by all
           registered kernel launch objects.

    The linearization happens in the getArgAccess function. Thus this time
    is a subset of the getArgAccess time. In particular the linearization time
    is the time needed to transform ISL access sets into intervals saved as
    std::tuples. This process is more intensive for 2D ISL sets, but must still
    be done for 1D ISL sets.
    \sa KernelLaunch for the getArgAccess function
    \sa wrapLaunchKernel
    \sa wrapMemcpyDtoH
*/
double Statistics::getLinearizationTime() const {
	double res = 0;
	for (auto l : launches_) {
		res += l->getLinearizationTime();
	}
	return res;
}

/*! \brief Returns the time needed to resolve the inter kernel dependencies.
*/
double Statistics::getDepResExecTime() const {
	double res = 0;
	for (auto r : resolutions_) {
		res += r->getTime();
	}
	return res;
}

//! Returns the time needed to create and cache all kernel launch objects.
double Statistics::getLaunchCreationTime() const {
	return kernelLaunchCreationTime_;
}

void Statistics::setNumDev(unsigned numDev) {
	numDev_ = numDev;
}

void Statistics::addResolution(shared_ptr<DepResolution> dres) {
	resolutions_.insert(dres);
}

void Statistics::addLaunch(shared_ptr<KernelLaunch> kl) {
	launches_.insert(kl);
}

void Statistics::addDepResCreationTime(double t) {
	depResCreationTime_ += t;
}

void Statistics::addKernelLaunchCreationTime(double t) {
	kernelLaunchCreationTime_ += t;
}

void Statistics::addCpyDtoH(shared_ptr<const MemCpyDtoH> mc) {
	dev2host_.insert(mc);
}

void Statistics::addCpyHtoD(shared_ptr<const MemCpyHtoD> mc) {
	host2dev_.insert(mc);
}

void Statistics::addCpyDtoD(shared_ptr<const MemCpyDtoD> mc) {
	dev2dev_.insert(mc);
}

}; // namespace end
