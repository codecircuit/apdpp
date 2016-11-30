#include <stdexcept>
#include <ostream>

#include "mekong-cuda.h"
#include "argument_type.h"

#include "isl/union_map.h"

namespace Mekong {

using namespace std;

/*! \brief Constructs an argument type object with kernel analysis information.

    \param name LLVM IR type name
    \param ptrlevel e.g. int* -> 1; int** -> 2; float -> 0; double -> 0;
    \param fundT 'f' -> Float; 'd' -> Double; 'i' -> Int; otherwise None;
    \param size of this type in Bytes
    \param elsize size of pointed type in Bytes; No relevance for non pointer
           types.
    \param isModified is the buffer changed during kernel calculation?
           This has no relevance for non pointer types.
    \param isRead is the buffer read during kernel calculation?
           This has no relevance for non pointer types.
    \param numDims number of array dimensions. This has no relevance for non
           pointer types.
    \param dimSizePatterns patterns which describe how to calculate the array
           dimension size with the values of the kernel arguments.
           This has no relevance for non pointer types.

    \sa Mekong::KernelArg::getDimSize for a detailed explanation of the kernel
                                      array sizes
*/
bsp_ArgType::bsp_ArgType(const string& name, unsigned ptrlevel,
                         char fundT, unsigned size,
                         unsigned elsize, bool isModified,
                         bool isRead, unsigned numDims,
                         const vector<string>& dimSizePatterns) :
		name_(name),
		ptrlevel_(ptrlevel),
		fundT_(fundT == 'i' ? Int :
		       fundT == 'f' ? Float :
		       fundT == 'd' ? Double :
		       /* else */ None),
		size_(size),
		elsize_(elsize),
		isModified_(isModified),
		isRead_(isRead),
		numDims_(numDims),
		dimSizePatterns_(dimSizePatterns) {}

/*! \brief Returns true if fundamental type != None.

    In fact the function will return true if the argument type represents a
    double, float, an Integer type (chars are ints in LLVM IR), or pointers
    on them.

    Examples:
    int**       --> true
    float*      --> true
    float       --> true
    struct Foo* --> false // unknown type
    char        --> true
*/
bool bsp_ArgType::isFundType() const {
	return fundT_ != None;
}

/*! \brief Returns true if an argument of this type is modified in the kernel.

    Non pointer kernel arguments are per definition never modified. But the
    values stored in a device buffer can be changed while a kernel is running.
    This information helps improving the runtime dependency resolution.
*/
bool bsp_ArgType::isModified() const {
	return isModified_;
}

/*! \brief Inverse of isModified()
    \sa isModified
*/
bool bsp_ArgType::isConst() const {
	return !isModified_;
}

/*! \brief Returns true if an argument of this type is read in the kernel.

    Non pointer kernel arguments are per definition never read. But the
    values stored in a device buffer can be read while a kernel is running.
    This information helps improving the runtime dependency resolution.
    E.g. a device buffer can only be written but not read.
*/
bool bsp_ArgType::isRead() const {
	return isRead_;
}

bool bsp_ArgType::isFloat() const {
	return fundT_ == Float;
}

bool bsp_ArgType::isDouble() const {
	return fundT_ == Double;
}

bool bsp_ArgType::isInt() const {
	return fundT_ == Int;
}

/*! \brief e.g. `int**` has pointer level two, `int*` has pointer level one.

      `int`     has pointer level zero; 
      `float`   has pointer level zero; 
      `double*` has pointer level one
      
*/
unsigned bsp_ArgType::getPtrlvl() const {
	return ptrlevel_;
}

//! Returns the size of this type in Bytes.
unsigned bsp_ArgType::getSize() const {
	return size_;
}

/*! \brief Returns the size of the type this type points to.

    Throws for non pointer types.
*/
unsigned bsp_ArgType::getElSize() const {
	if (ptrlevel_ < 1) {
		throw runtime_error(
			"Namespace Mekong, Class bsp_ArgType, Func getElSize():\n"
			"bsp_ArgType " + name_ + " is no pointer type!"
		);
	}
	return elsize_;
}

/*! \brief Returns the number of dimensions determined by the kernel analysis.

    This only makes sense for pointer types, because a scalar type can not
    represent an array.
*/
unsigned bsp_ArgType::getNumDims() const {
	return numDims_;
}


/*! \brief describes how to calc the dim size
    \sa Mekong::KernelArg::getDimSize for a detailed explanation
*/ 
const vector<string>& bsp_ArgType::getDimSizePatterns() const {
	return dimSizePatterns_;
}

//! Returns Float, Double, Int, or None
bsp_ArgType::FundType bsp_ArgType::getFundType() const {
	return fundT_;
}

//! Returns the LLVM IR type name.
string bsp_ArgType::getName() const {
	return name_;
}

//! Help function, which determines if an isl map is empty.
bool bsp_ArgType::isEmptyIslMap(const string& mapStr) {
	if (mapStr.empty()) {
		return true;
	}
	if (mapStr == "null" || mapStr == "empty") {
		return true;
	}
	isl_ctx* ctx = isl_ctx_alloc();
	isl_union_map* umap = isl_union_map_read_from_str(ctx, mapStr.c_str());
	bool result = isl_union_map_is_empty(umap);
	isl_union_map_free(umap);
	isl_ctx_free(ctx);
	return result;
}

ostream& operator<<(ostream& out, const bsp_ArgType& argT) {
	out << argT.getName();
	return out;
}

};
