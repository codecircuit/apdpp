/******************************************************************
 *
 * FILENAME    - database_writer.h
 *
 * DESCRIPTION -
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-06-12
 *
 ******************************************************************/ 

#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <utility> // std::pair, std::make_pair //
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

#include "dashdb.h"

using namespace std;
using namespace llvm;

#define DEBUG_TYPE "DATABASE_WRITER"

namespace Mekong {

class DatabaseWriter {
	public:
		void setKernels(const vector<const Function*>& kernels)
		                   { kernels_ = kernels; }
		void setArgumentTypes(const Function* kernel, const vector<Type*>& types)
		                   { argTypes_[kernel] = types; }
		void setPartitioning(const Function* kernel, const string& part)
		                   { partitioning_[kernel] = part; }
		void setIslReadWriteStrings(const map<const Function*, vector<string>>& islReadMaps,
		                            const map<const Function*, vector<string>>& islWriteMaps);
		void setIslReadWriteParameterStrings(
                           const map<const Function*, vector<vector<string>>>& islReadParameters,
                           const map<const Function*, vector<vector<string>>>& islWriteParameters);
		void setIslNumDimArrays(const map<const Function*, vector<int>>& islNumDimArrays)
		                   { islNumDimArrays_ = islNumDimArrays; }
		void setIslArrayDimSizes(const map<const Function*, vector<vector<string>>>& islArrayDimSizes)
		                   { islArrayDimSizes_ = islArrayDimSizes; }
		const vector<const Function*>& getKernels() const { return kernels_; }

		void write(const char* filename); // can not be const, as root_ is changed

	private:
		vector<const Function*> kernels_;
		map<const Function*, vector<Type*>> argTypes_;
		map<const Function*, string> partitioning_;
		//! mapping of function to memory access on arguments
		//! e.g. islReads_[kernelA][2] contains the isl map
		//! describing the read accesses on argument number two
		map<const Function*, vector<string>> islReads_;
		map<const Function*, vector<string>> islWrite_;
		map<const Function*, vector<vector<string>>> islReadParameters_;
		map<const Function*, vector<vector<string>>> islWriteParameters_;
		map<const Function*, vector<int>> islNumDimArrays_;
		map<const Function*, vector<vector<string>>> islArrayDimSizes_;
};

void DatabaseWriter::setIslReadWriteStrings(const map<const Function*, vector<string>>& islReadMaps,
                                            const map<const Function*, vector<string>>& islWriteMaps) {
	errs() << "[+] CLASS DatabaseWriter, FUNC setIslReadWriteStrings():\n";
	islReads_ = islReadMaps;
	islWrite_ = islWriteMaps;
	errs() << "[-] CLASS DatabaseWriter, FUNC setIslReadWriteStrings()\n";
}

void DatabaseWriter::setIslReadWriteParameterStrings(
                           const map<const Function*, vector<vector<string>>>& islReadParameters,
                           const map<const Function*, vector<vector<string>>>& islWriteParameters) {
	islReadParameters_ = islReadParameters;
	islWriteParameters_ = islWriteParameters;
	errs() << "[-] CLASS DatabaseWriter, FUNC setIslReadWriteStrings()\n";
}

//! This function writes all the data the object contains to a .json file

//! Only reasonable and available information will be written to the
//! specified file. Thus empty read/write access maps will not be listed.
void DatabaseWriter::write(const char* filename) {
	errs() << "[+] CLASS DatabaseWriter, FUNC write():\n";

	// GETS THE POINTER LEVEL OF A TYPE
	// e.g. int** has pointer level two
	//      int   has pointer level zero
	//      int*  has pointer level one
	auto getPointerLVL = [] (const Type* type) {
		unsigned ptrlvl = 0;
		while (auto* cast = dyn_cast<PointerType>(type)) {
			++ptrlvl;
			type = cast->getElementType();
		}
		return ptrlvl;
	};

	// GET FUNDAMENTAL TYPE
	// e.g. int** has fundamental type int
	//      float* has fundamental type float
	auto getFundamentalTy = [] (const Type* type) {
		unsigned ptrlvl = 0;
		while (auto* cast = dyn_cast<PointerType>(type)) {
			++ptrlvl;
			type = cast->getElementType();
		}
		return type;
	};

	auto getTypeChar = [] (const Type* type) {
		return  type->isIntegerTy() ? "i" : // char are also i8 //
		        type->isFloatTy() ? "f" :
		        type->isDoubleTy() ? "d" : "None";	
	};

	auto getTypeString = [] (const Type* type) {
		string dummy;
		raw_string_ostream ss(dummy);
		type->print(ss);
		string argName = ss.str();
		ss.flush();
		return argName;
	};

	// BUILDING THE DASHDB OBJECT //
	unsigned kernelNr = 0;
	dashdb::Butler b;
	for (const Function* kernel : kernels_) {
		errs() << "  * Creating dashdb object for kernel " << kernel->getName().str() << '\n';
		b["kernels"][kernelNr]["partitioning"] = partitioning_.at(kernel);
		b["kernels"][kernelNr]["name"] = kernel->getName().str();

		// WRITE INFORMATION ABOUT KERNEL ARGUMENTS //
		auto argument = kernel->getArgumentList().begin(); // snd iterator in foor loop
		int numArguments = kernel->getArgumentList().size();
		for (unsigned argumentNr = 0; argumentNr < numArguments; ++argumentNr, ++argument) {
			const Type* argType = argTypes_.at(kernel)[argumentNr];
			const string argName = argument->getName().str();
			errs() << "  * Processing function argument nr " << argumentNr;
			errs() << " %" << argName << '\n';
			const Type* fundT = getFundamentalTy(argType);
			b["kernels"][kernelNr]["arguments"][argumentNr]["name"] = argName;
			b["kernels"][kernelNr]["arguments"][argumentNr]["pointer level"] = getPointerLVL(argType);
			b["kernels"][kernelNr]["arguments"][argumentNr]["fundamental type"] = getTypeChar(fundT);
			// size in bits
			b["kernels"][kernelNr]["arguments"][argumentNr]["size"] = argType->getPrimitiveSizeInBits();
			errs() << "  - size = " << argType->getPrimitiveSizeInBits() << "\n";
			b["kernels"][kernelNr]["arguments"][argumentNr]["type name"] = getTypeString(argType);
			// if an argument has a non-pointer type it has no elements to point to,
			// thus we set the size of the elements to zero
			b["kernels"][kernelNr]["arguments"][argumentNr]["element size"] = fundT != argType ? fundT->getPrimitiveSizeInBits() : 0;

			// WRITE ISL READ AND WRITE MAPS
			try { // write only information which is available
				string rmap = islReads_.at(kernel).at(argumentNr);
				string wmap = islWrite_.at(kernel).at(argumentNr);
				if (rmap != "null" && !rmap.empty()) {
					b["kernels"][kernelNr]["arguments"][argumentNr]["isl read map"] = rmap;
				}
				if (wmap != "null" && !wmap.empty()) {
					b["kernels"][kernelNr]["arguments"][argumentNr]["isl write map"] = wmap;
				}
			} catch(...) {}

			// WRITE ISL PARAMS
			try { // write only information which is available
				int numReadParams = islReadParameters_.at(kernel)[argumentNr].size();
				if (numReadParams != 0) {
					for (int i = 0; i < numReadParams; ++i) {
						b["kernels"][kernelNr]["arguments"][argumentNr]["isl read params"][i] = islReadParameters_.at(kernel)[argumentNr][i];
					}
				}
				int numWriteParams = islWriteParameters_.at(kernel)[argumentNr].size();
				if (numWriteParams != 0) {
					for (int i = 0; i < numWriteParams; ++i) {
						b["kernels"][kernelNr]["arguments"][argumentNr]["isl write params"][i] = islWriteParameters_.at(kernel)[argumentNr][i];
					}
				}
			} catch(...) {}

			// WRITE NUM DIM ARRAYS
			try {
				int dimSize = islNumDimArrays_.at(kernel)[argumentNr];
				if (dimSize != 0) { // write dim size only if it is not zero, as this is trivial
					b["kernels"][kernelNr]["arguments"][argumentNr]["num dimensions"] = dimSize;
				}
			}
			catch(...) {}

			// WRITE ARRAY DIM SIZES
			try {
				vector<string> dimSizes = islArrayDimSizes_.at(kernel)[argumentNr];
				if (!dimSizes.empty()) {
					for (int i = 0; i < dimSizes.size(); ++i) {
						b["kernels"][kernelNr]["arguments"][argumentNr]["dim sizes"][i] = dimSizes[i];
					}
				}
			}
			catch(...) {}
		}

		++kernelNr;
	}

	// BUILDING THE .ddb FILE //
	errs() << "  * Building the dashdb file...\n";

	b.write(filename);

	errs() << "[-] CLASS DatabaseWriter, FUNC write()\n";
}

} // namespace end //

#undef DEBUG_TYPE
