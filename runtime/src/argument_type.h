#ifndef MEKONG_ARGUMENT_TYPE_H
#define MEKONG_ARGUMENT_TYPE_H

#include <ostream>
#include <vector>

#include "mekong-cuda.h"

namespace Mekong {

using namespace std;

/*! \brief Represents argument type information given by kernel analysis.

    In the runtime there should only exist const bsp_ArgType objects, as all
    information contained by an bsp_ArgType object is given on compile time by
    the kernel analysis.
    \sa bsp_KernelInfo here the bsp_ArgType objects are created and the kernel
                       analysis is read.
*/
class bsp_ArgType {
	public:
		//! We support these basic types and pointers to them as kernel arguments.
		enum FundType { Float, Double, Int, None };
		bsp_ArgType(const string& name, unsigned ptrlevel,
		            char fundT, unsigned size,
		            unsigned elsize, bool isModified,
		            bool isRead, unsigned numDims,
		            const vector<string>& dimSizePatterns);
		bool isFundType() const;
		bool isModified() const; //!< devptr's content can be changed;
		bool isConst() const;    //!< devptr's content can be constant;
		bool isRead() const;
		bool isFloat() const;
		bool isDouble() const;
		bool isInt() const;
		unsigned getPtrlvl() const;
		unsigned getSize() const;
		unsigned getElSize() const;
		unsigned getNumDims() const;
		const vector<string>& getDimSizePatterns() const;
		FundType getFundType() const;
		string getName() const;

	private:
		// all these type do not have to be const, because we in the runtime
		// we create only const bsp_ArgType objects anyways

		FundType fundT_;     //!< int** has fund type Int and int has also fund type Int
		unsigned ptrlevel_;  //!< int** has ptr level 2, whereas int has ptr level 0
		unsigned size_;      //!< size in Bytes
		unsigned elsize_;    //!< if arg type is a pointer, this is the element size in Bytes
		bool isModified_;    //!< if type is a pointer type, the content can be modified
		bool isRead_;        //!< if type is a pointer type, the content can be read
		unsigned numDims_;   //!< if type is a pointer type, it can represent a multi dim array

		//! dimSizePatterns.size() = numDims_ - 1; for pointer types
		//! \sa Mekong::KernelArg::getDimSize for a detailed explanation
		vector<string> dimSizePatterns_; //!< describes how to calc the dim size

		string name_;        //!< llvm name of this type

		static bool isEmptyIslMap(const string& mapStr);
};

ostream& operator<<(ostream& out, const bsp_ArgType& argT);

};

#endif
