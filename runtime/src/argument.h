#ifndef MEKONG_ARGUMENT_H
#define MEKONG_ARGUMENT_H

#include <memory>
#include <cstdint>

#include "argument_type.h"
#include "bitop.h"
#include "mekong-cuda.h"

namespace Mekong {

using namespace std;
using namespace bitop;

/*! \brief Represents value and type of a kernel argument of a kernel launch.

    It is handy to have a kernel argument modeled by this class, because we
    need various representations of a kernel argument value. E.g. the
    wrapLaunchKernel function gets the kernel argument values with type void*.
    We want to have a generic interface to cast the arguments into their
    intended types.
*/
class KernelArg {
	public:
		static vector<shared_ptr<const KernelArg>>
		createArgs(const vector<shared_ptr<const bsp_ArgType>>& types, void** rawArgs);

		KernelArg(shared_ptr<const bsp_ArgType> type,
		          unique_ptr<const charPack> cp,
				  const vector<size_t>& dimSizes = {});
		KernelArg(shared_ptr<const bsp_ArgType> type, const void* vptr,
		          const vector<size_t>& dimSizes = {});

		bool isEqualInBits(const KernelArg& other) const;
		shared_ptr<const bsp_ArgType> getType() const;
		const vector<size_t>& getDimSizes() const;
		size_t getDimSize(unsigned axis) const;
		intmax_t asInt() const;
		float asFloat() const;
		double asDouble() const;
		MEdeviceptr asDevPtr() const;

		// as the cuda kernel launch function expect a void* and not a const void* we
		// need this copy function to get the raw void* pointer
		unique_ptr<charPack> cpyCharPack() const;

		                  friend bool operator==(const KernelArg& a, const KernelArg& b);
		template<class T> friend bool operator==(const KernelArg& a, T obj);
		template<class T> friend bool operator==(T obj, const KernelArg& a);

	private:
		//! If the argument is a pointer to a multi dimensional kernel array we
		//! must know the size of the kernel array to linearize the access
		//! patterns on that array. For Linearization it is sufficient to know
		//! the dimension limits with exception of one dimension. Thus
		//! dimSizes_.size() = numDims - 1.
		vector<size_t> dimSizes_;
		unique_ptr<const charPack> cp_;
		shared_ptr<const bsp_ArgType> type_;
};

ostream& operator<<(ostream& out, const KernelArg& arg);

bool operator!=(const KernelArg& a, const KernelArg& b);

//! Compares the bits of the two objects.
template<class T>
bool operator==(const KernelArg& a, T obj) {
	return *a.cp_ == charPack(obj);
}

//! Compares the bits of the two objects.
template<class T>
bool operator==(T obj, const KernelArg& a) {
	return *a.cp_ == charPack(obj);
}

//! Compares the bits of the two objects.
template<class T>
bool operator!=(const KernelArg& a, T obj) {
	return !(a == obj);
}

//! Compares the bits of the two objects.
template<class T>
bool operator!=(T obj, const KernelArg& a) {
	return !(obj == a);
}

}; // namespace end

#endif
