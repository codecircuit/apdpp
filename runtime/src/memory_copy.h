/*! \file memory_copy.h
    \brief Header file to handle memory copy operations in Mekong's context.

    \todo This file and all classes should be rewritten in a more simple style.
          Up to now we have one base class which contains a lot functionality
          and 3 (HtoD, DtoH, DtoD) child classes, which are merely wrapper
          classes. They should be replaced by one simple written class.
          Moreover the complex usage of the MemSubCopy is not a good style.
*/

#ifndef MEKONG_MEMCPY_H
#define MEKONG_MEMCPY_H

#include <memory>
#include <vector>
#include <string>
#include <ostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <chrono>

#include "mekong-cuda.h"
#include "alias_handle.h"
#ifdef SOFIRE
#include "communicator.h"
#endif

namespace Mekong {

using namespace std;

using Clock = chrono::high_resolution_clock;
using Duration = chrono::duration<double>;

enum MemCpyKind { HtoH, DtoH, DtoD, HtoD };

//! Small struct to save information needed to execute one cuda memory copy
struct MemSubCopy {
	int src;      ///< -1 refers to host memory, 0-i targets gpu 0 to i
	int dst;      ///< -1 refers to host memory, 0-i targets gpu 0 to i
	size_t from;  ///< marks the position of the start Byte
	size_t to;    ///< offset on the destination array
	size_t size;  ///< size of the copy in Bytes
};

bool operator==(const MemSubCopy& a, const MemSubCopy& b);

ostream& operator<<(ostream& out, const MemSubCopy& subcpy);

//! Represents a memory copy operation
template<class DstPtrT, class SrcPtrT>
class MemCpy {
	public:
		//TODO: maybe a list structure is more efficient here
		typedef vector<MemSubCopy> MemPattern;

		MemCpy(DstPtrT dst, SrcPtrT src, MemCpyKind kind,
		       shared_ptr<const MemPattern> pmp,
		       shared_ptr<AliasHandle> aliasH, bool sync = true,
		       size_t size = 0);

		virtual MEresult exec() = 0;

		bool isHtoD() const;
		bool isDtoH() const;
		bool isDtoD() const;
		bool isHtoH() const;
		bool isSync() const;
		size_t getSize() const;
		size_t getExecutions() const;
		double getTime() const;
		double getBW() const;
		DstPtrT getDst() const;
		SrcPtrT getSrc() const;
		shared_ptr<const MemPattern> getPattern() const;
		string getKindStr() const;

		void setDst(const DstPtrT& dst);

	protected:
		size_t executions_ = 0;
		double time_ = 0;
		MemCpyKind kind_;
		DstPtrT dst_;
		SrcPtrT src_;
		size_t orgSize_; ///< original size
		shared_ptr<const MemPattern> pmp_; ///< contains detailed information about copy operations \sa MemSubCopy
		shared_ptr<AliasHandle> aliasH_;
		bool sync_;
		bool isBroadcast_ = false;
};

class MemCpyDtoD : public MemCpy<MEdeviceptr, const MEdeviceptr> {
	public:
		MemCpyDtoD(MEdeviceptr dstsrc, shared_ptr<const MemPattern> pmp,
		           shared_ptr<AliasHandle> aliasH, bool sync = true);

		static shared_ptr<MemCpyDtoD>
		createBroadcast(MEdeviceptr dstsrc, size_t size,
		                shared_ptr<AliasHandle> aliasH, unsigned short master);

		MEresult exec() override; 
};

class MemCpyDtoH : public MemCpy<void*, const MEdeviceptr> {
	public:
		MemCpyDtoH(void* dst, const MEdeviceptr src,
		           shared_ptr<const MemPattern> pmp,
		           shared_ptr<AliasHandle> aliasH, bool sync = true);

		MemCpyDtoH(void* dst, const MEdeviceptr src, size_t size,
		           shared_ptr<AliasHandle> aliasH);

		MEresult exec() override; 
	private:
		shared_ptr<const MemPattern> createTrivialMemPattern(size_t size) const;
};

class MemCpyHtoD : public MemCpy<MEdeviceptr, const void*> {
	public:
#ifdef SOFIRE
		static communicator comm;
#endif
		MemCpyHtoD(MEdeviceptr dst, const void* src,
		           shared_ptr<const MemPattern> pmp,
		           shared_ptr<AliasHandle> aliasH, bool sync = true, size_t size = 0);

		static shared_ptr<MemCpyHtoD>
		createBroadcast(MEdeviceptr dst, const void* src, size_t size,
		                shared_ptr<AliasHandle> aliasH);

		MEresult exec() override; 
};

/*! \brief Constructor of base class which is used by its children.

    The class is not intended to be used directly.
*/
template<class DstPtrT, class SrcPtrT>
MemCpy<DstPtrT, SrcPtrT>::MemCpy(DstPtrT dst, SrcPtrT src, 
                                 MemCpyKind kind,
                                 shared_ptr<const MemPattern> pmp,
                                 shared_ptr<AliasHandle> aliasH, bool sync,
                                 size_t size) :
				kind_(kind),
				dst_(dst),
				src_(src),
				pmp_(pmp),
				aliasH_(aliasH),
				sync_(sync),
				orgSize_(size) {}


template<class DstPtrT, class SrcPtrT>
bool MemCpy<DstPtrT, SrcPtrT>::isHtoD() const {
	return kind_ == HtoD;
}

template<class DstPtrT, class SrcPtrT>
bool MemCpy<DstPtrT, SrcPtrT>::isDtoH() const {
	return kind_ == DtoH;
}

template<class DstPtrT, class SrcPtrT>
bool MemCpy<DstPtrT, SrcPtrT>::isDtoD() const {
	return kind_ == DtoD;
}

template<class DstPtrT, class SrcPtrT>
bool MemCpy<DstPtrT, SrcPtrT>::isHtoH() const {
	return kind_ == HtoH;
}

/*! \brief True if all devices are synchronized after execution.

    Thus before function exec() returns it will call cudaDeviceSynchronize
    for each device.
    \sa exec
*/
template<class DstPtrT, class SrcPtrT>
bool MemCpy<DstPtrT, SrcPtrT>::isSync() const {
	return sync_;
}

//! Returns the total amount of copied Bytes.
template<class DstPtrT, class SrcPtrT>
size_t MemCpy<DstPtrT, SrcPtrT>::getSize() const {
	size_t res = 0;
	if (!isBroadcast_) {
		for (const auto& subcpy : *pmp_) {
			res += subcpy.size;
		}
	}
	else {
		res = orgSize_ * aliasH_->getNumDev();
	}
	return res * executions_;
}

//! returns zero if the memcpy was not executed yet
template<class DstPtrT, class SrcPtrT>
double MemCpy<DstPtrT, SrcPtrT>::getTime() const {
	if (!sync_) {
		throw runtime_error("SPACE Mekong, CLASS MemCpy, FUNC getTime(): "
		                    "you can not get time of a non synchronizing "
		                    "memory copy operation!");
	}
	return time_;
}

//! Bandwidth in GB/s, returns zero if the memcpy was not executed yet
template<class DstPtrT, class SrcPtrT>
double MemCpy<DstPtrT, SrcPtrT>::getBW() const {
	return executions_ > 0 ? (double) getSize() / 1e9 / getTime() : 0;
}

//! Returns destination pointer
template<class DstPtrT, class SrcPtrT>
DstPtrT MemCpy<DstPtrT, SrcPtrT>::getDst() const {
	return dst_;
}

//! Returns source pointer
template<class DstPtrT, class SrcPtrT>
SrcPtrT MemCpy<DstPtrT, SrcPtrT>::getSrc() const {
	return src_;
}

/*! \brief Returns number of calls of function exec().
    \sa exec
*/
template<class DstPtrT, class SrcPtrT>
size_t MemCpy<DstPtrT, SrcPtrT>::getExecutions() const {
	return executions_;
}

//! Returns "HtoD" or "DtoH" or "DtoD" or "HtoH"
template<class DstPtrT, class SrcPtrT>
string MemCpy<DstPtrT, SrcPtrT>::getKindStr() const {
	if (isHtoD()) {
		return "HtoD";
	}
	if (isDtoH()) {
		return "DtoH";
	}
	if (isDtoD()) {
		return "DtoD";
	}
	if (isHtoH()) {
		return "HtoH";
	}
	return "0";
}

/*! \brief Returns the vector of sub memory copy objects.
    \sa MemSubCopy 
*/
template<class DstPtrT, class SrcPtrT>
shared_ptr<const vector<MemSubCopy>> MemCpy<DstPtrT, SrcPtrT>::getPattern() const {
	return pmp_;
}

/*! \brief Sets the destination pointer.

    The destination pointer must be the pointer which is used in the
    application. The object will automatically use the alias handle object to
    get the appropriate pointers on the other devices.
*/
template<class DstPtrT, class SrcPtrT>
void MemCpy<DstPtrT, SrcPtrT>::setDst(const DstPtrT& dst) {
	dst_ = dst;
}

template<class DstPtrT, class SrcPtrT>
ostream& operator<<(ostream& out, const MemCpy<DstPtrT, SrcPtrT>& mc) {
	out << boolalpha << "MemCpy(executions: " << mc.getExecutions() << ", " << mc.getKindStr();
	if (mc.isSync()) {
		out << ", time: " << mc.getTime() << "s";
	}
	out << ", " << mc.getSize() << " Bytes" << ", dst: " << mc.getDst();
	out << ", src: " << mc.getSrc() << ")";
	return out;
}

}; // namespace end

#endif
