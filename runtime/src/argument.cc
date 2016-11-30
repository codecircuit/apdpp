#include <memory>
#include <cstdint>

#include "argument.h"
#include "argument_type.h"
#include "bitop.h"
#include "mekong-cuda.h"

namespace Mekong {

using namespace std;

//! Creates all kernel argument objects for a certain kernel launch object.
vector<shared_ptr<const KernelArg>>
KernelArg::createArgs(const vector<shared_ptr<const bsp_ArgType>>& types,
                      void** rawArgs) {
	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS KernelArg, FUNC createArgs():\n" + msg);
	};

	vector<shared_ptr<const KernelArg>> res;
	vector<vector<size_t>> allDimSizes(types.size());
	vector<unique_ptr<const charPack>> charPacks;

	// COLLECT RAW KERNEL VALUES FIRST
	unsigned short arg_nr = 0;
	for (auto arg_type : types) {
		charPacks.push_back(unique_ptr<const charPack>(
		                    new charPack(rawArgs[arg_nr], arg_type->getSize() * 8))); // *8 as charPacks deal with bits
		++arg_nr;
	}

	auto parseArrayDimSizes = [&] (shared_ptr<const bsp_ArgType> arg_type) {
		vector<size_t> dimSizes(arg_type->getNumDims() - 1);
		unsigned short dim_nr = 0;
		for (string dim_pattern : arg_type->getDimSizePatterns()) {
			if (dim_pattern.empty()) {
				throwError("Could not deduce array size from empty pattern");
			}
			if (dim_pattern.substr(0,3) == "arg") {
				string argNumberStr = dim_pattern.substr(3, dim_pattern.size() - 3);
				int arg_nr = stoi(argNumberStr);
				if (!types[arg_nr]->isInt()) {
					throwError("I can not deduce an array size from a non-integral "
					           "kernel argument with type " + types[arg_nr]->getName());
				}
				dimSizes[dim_nr] = charPacks[arg_nr]->asInt<int>();
			}
			else {
				throwError("Could not deduce array size from pattern '" + dim_pattern + "'");
			}

			++dim_nr;
		}
		return dimSizes;
	};

	// COLLECT ARRAY DIMENSION SIZES
	arg_nr = 0;
	for (auto arg_type : types) {
		if (arg_type->getPtrlvl() == 1) {
			if (arg_type->getNumDims() > 1) {
				allDimSizes[arg_nr] = parseArrayDimSizes(arg_type);
			}
		}
		++arg_nr;
	}

	// FINALLY CREATE THE ARGUMENT OBJECTS
	arg_nr = 0;
	for (auto arg_type : types) {
		res.push_back(
			shared_ptr<const KernelArg>(
				new KernelArg(arg_type,
				              move(charPacks[arg_nr]),
				              allDimSizes[arg_nr])
			)
		);
		++arg_nr;
	}
	return res;
}

/*! \brief Constructs a kernel argument object.

    The binary data will be taken from the given character pack pointer. The 
    pointer will be left in an undefined state after object construction.
    Moreover the constructor throws if a wrong dimension sizes are given.
    \param type E.g. contains the information of the argument's size in Bytes.
    \param pc undefined after object creation.
    \param dimSizes contains the dimension size in case of a multi dimensional
           array.
    \sa getDimSize for the array dimension size explanation.
*/
KernelArg::KernelArg(shared_ptr<const bsp_ArgType> type,
                     unique_ptr<const charPack> cp,
                     const vector<size_t>& dimSizes) :
			type_(type),
			cp_(move(cp)),
			dimSizes_(dimSizes) {

	if (dimSizes.size() != type->getNumDims() - 1 &&
	    !(dimSizes.empty() && type->getNumDims() == 0)) {

		throw invalid_argument(
			"SPACE Mekong, CLASS KernelArg, FUNC KernelArg():\n"
			"I got a type with num dims " + to_string(type->getNumDims()) +
			", but you put " + to_string(dimSizes.size()) + " dimension sizes" +
			" in the constructor function." +
			"But I expect " + to_string(type->getNumDims() - 1) +
			" dimension sizes\n"
		);
	}

}

/*! \brief Constructs a kernel argument object.

    At the construction of the object the binary data of the given memory
    location will be copied and saved. Thus you can free **vptr** after object
    creation.
    \param type E.g. contains the information of the argument's size in Bytes.
    \param vptr points the the memory location of the argument's value.
    \param dimSizes contains the dimension size in case of a multi dimensional
           array.
    \sa getDimSize for the array dimension size explanation.
*/
KernelArg::KernelArg(shared_ptr<const bsp_ArgType> type, const void* vptr,
                     const vector<size_t>& dimSizes) :
			type_(type),
			cp_(new charPack(vptr, type->getSize() * 8)),
			dimSizes_(dimSizes) {

	if (dimSizes.size() != type->getNumDims() - 1
	    && !(dimSizes.empty() && type->getNumDims() == 0)) {
		throw invalid_argument("SPACE Mekong, CLASS KernelArg, FUNC KernelArg():\n"
		                       "I got a type with num dims "
		                       + to_string(type->getNumDims()) +
		                       ", but you put " + to_string(dimSizes.size())
		                       + " dimension sizes in the constructor function."
		                       "But I expect " + to_string(type->getNumDims() - 1)
		                       + string(" dimension sizes\n"));
	}
}

//! If all bits of the values are equal, the function returns true
bool KernelArg::isEqualInBits(const KernelArg& other) const {
	return *cp_ == *other.cp_; // compare value of character packs
}

//! Returns an object containing the argument type.
shared_ptr<const bsp_ArgType> KernelArg::getType() const {
	return type_;
}

/*! \brief Returns the vector containing the array dimension sizes.
    \sa getDimSize for the array size explanation.
*/
const vector<size_t>& KernelArg::getDimSizes() const {
	return dimSizes_;
}

/*! \brief Returns a dimension size.

    If polly analyzes an 2D memory access **A[x + N * y]** it will identify
    variable **x** as the variable representing the second dimension and **y**
    representing the first dimension, because there is polly's constraint that 
    the first dimension must always be unlimited. In our example variable **x**
    must be limited to **N - 1** in case of a two dimensional access pattern.
    Thus if you call this function with **axis = 0**, it will return the size of
    the second dimension **x**.
    \sa **const SCEV *getDimensionSize(unsigned Dim)** in **ScopInfo.h**
    of the polly library.
*/
size_t KernelArg::getDimSize(unsigned axis) const {
	try {
		return dimSizes_.at(axis);
	}
	catch(...) {
		throw range_error("SPACE Mekong, CLASS KernelArg, FUNC getDimSize():\n"
		                  "I do not have that many dimensions");
	}
}

//! Converts the argument value to a intmax_t.
intmax_t KernelArg::asInt() const {
	return cp_->asInt<intmax_t>();
}

//! Converts the argument value to a double of IEEE754 standard.
float KernelArg::asFloat() const {
	return cp_->asFloat_IEEE754();
}

//! Converts the argument value to a double of IEEE754 standard.
double KernelArg::asDouble() const {
	return cp_->asDouble_IEEE754();
}

/*! \brief Returns the value of the argument as an intmax_t.

    The return type have been choosen because a device buffer pointer in CUDA
    is an integer type.
*/
MEdeviceptr KernelArg::asDevPtr() const {
	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS KernelArg, FUNC asDevPtr():\n"
		                    + msg);
	};
	if (getType()->getSize() != sizeof(MEdeviceptr)
	    || sizeof(intmax_t) < sizeof(MEdeviceptr)) {
		throw runtime_error("Kernel argument of type '" + getType()->getName() +
		                    "' can not be converted to a device pointer, "
		                    "because the size of the argument's type is "
		                    + to_string(getType()->getSize())
		                    + ", whereas the size of a device ptr is equal to"
		                    + " " + to_string(sizeof(MEdeviceptr)) + " Bytes.");
	}
	return cp_->asInt<intmax_t>();
}

unique_ptr<charPack> KernelArg::cpyCharPack() const {
	return unique_ptr<charPack>(new charPack(*cp_));
}

ostream& operator<<(ostream& out, const KernelArg& arg) {
	if (arg.getType()->getNumDims() > 1) {
		out << "(Bits: " << *arg.cpyCharPack() << ", Type: " << *arg.getType();
		out << ", NumDims: " << arg.getType()->getNumDims();
		out << ", DimSizes: ";
		for (auto dimSize : arg.getDimSizes()) {
			out << dimSize << " ";
		}
		out << ")";
	}
	else {
		out << "(Bits: " << *arg.cpyCharPack() << ", Type: " << *arg.getType();
		out << ", NumDims: " << arg.getType()->getNumDims();
		out << ")";
	}
	return out;
}

/*! \brief Compares the value of two kernel arguments.

    The functions returns true if the binary values and the pointer levels
    of the two kernel arguments are equal. This function is used to compare
    two kernel launch objects for equality.

    \sa bsp_ArgType for pointer level.

    \todo If the function is only used to compare two kernel launch objects
    it should be sufficient to compare the binary values only, because the
    position of the argument in the kernel argument list implicitly fixes the
    argument type.
*/
bool operator==(const KernelArg& a, const KernelArg& b) {
	if (a.getType()->getPtrlvl() != b.getType()->getPtrlvl()) {
		return false;
	}
	if (*a.cp_ != *b.cp_) {
		return false;
	}
	return true;
}

bool operator!=(const KernelArg& a, const KernelArg& b) {
	return !(a == b);
}

}; // namespace end
