/******************************************************************
 *
 * FILENAME    - bsp_analysis.cc
 *
 * DESCRIPTION - 
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-05-23
 *
 ******************************************************************/ 

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/IRBuilder.h"

#include <vector>
#include <stdexcept>
#include <fstream>

#include "polly/ScopInfo.h"
#include "isl/map.h"
#include "isl/union_map.h"
#include "isl/set.h"
#include "isl/space.h"

#include "database_writer.h"

#define DEBUG_TYPE "BSP_ANALYSIS"

namespace Mekong {

using namespace std;
using namespace llvm;

// This enables to give the database filename per command line paramater to llvm's opt
static cl::opt<string> CLOPT_DB("mekong_db", cl::desc("specifies the database file for mekong (format .json)"), cl::value_desc("filename"));

struct bsp_analysis : public ModulePass {
	static char ID; // Pass identification, replacement for typeid
	bsp_analysis() : ModulePass(ID) {}

	bool runOnModule(Module &M) override; 
	
	void printPollyInfo(Function* f);
	static vector<Function*> getKernels(Module& M);
	void getPollyAccessPatterns(Function& kernel, Module& M);
	void getIslParams(Function& kernel);
	void getPollyNumDimPerArray(Function& kernel, Module& M);
	void getPollyArrayDimSizes(Function& kernel, Module& M);

	void getAnalysisUsage(AnalysisUsage &Info) const override {
		Info.addRequiredTransitive<polly::ScopDetection>();
		Info.addRequiredTransitive<polly::ScopInfo>();
		Info.setPreservesAll();
	}

	//! Each kernel has multiple arguments, where each argument can have
	//! more than one array dimension
	map<const Function*, vector<vector<string>>> islArrayDimSizes_;

	/*//! We save the array dimension as a string of a isl piecewise affine function,
	//! which can have multiple isl space parameters. The runtime must know how
	//! to handle these parameters, thus we end up in this ugly map value type.
	map<const Function*, vector<vector<vector<string>>>> islArrayDimSizeParams_;*/

	//! Saves the number of dimensions for all kernel arguments
	map<const Function*, vector<int>> islNumDimArrays_;
	map<const Function*, vector<string>> islReads_;
	map<const Function*, vector<string>> islWrites_;
	map<const Function*, vector<vector<string>>> islReadParams_;
	map<const Function*, vector<vector<string>>> islWriteParams_;
}; 

bool bsp_analysis::runOnModule(Module& M) {
	errs() << "[+] CLASS bsp_analysis, FUNC runOnModule():\n";
	errs() << "  * Function list in Module <function name> : [<arg name>]\n";
	for (Function& f : M.getFunctionList()) {
		errs() << "    * " << f.getName() << ": ";
		for (const auto& arg : f.getArgumentList()) {
			errs() << arg.getName() << " ";
		}
		errs() << '\n';
	}
	
	vector<Function*> kernels = getKernels(M);
	vector<const Function*> constKernels;
	for (const Function* kernel : kernels) {
		constKernels.push_back(kernel);
	}
	
	DatabaseWriter dbWriter;
	dbWriter.setKernels(constKernels);


	for (Function* kernel : kernels) {
		// HACK
		dbWriter.setPartitioning(kernel, "y"); // set every partitioning to x dimension
		// HACK END
		dbWriter.setArgumentTypes(kernel, kernel->getFunctionType()->params().vec()); // set kernel types
		//printPollyInfo(M.getFunction(kernel->getName().str() + "_lwrapped"));
		getPollyAccessPatterns(*kernel, M); // grep access patterns from polly
		getIslParams(*kernel); // link the isl params to kernel args
		getPollyNumDimPerArray(*kernel, M);
		getPollyArrayDimSizes(*kernel, M);
	}

	dbWriter.setIslReadWriteStrings(islReads_, islWrites_); // set the access patterns in the writer object
	dbWriter.setIslReadWriteParameterStrings(islReadParams_, islWriteParams_);
	dbWriter.setIslNumDimArrays(islNumDimArrays_);
	dbWriter.setIslArrayDimSizes(islArrayDimSizes_);

	errs() << "  * Writing database to file " << CLOPT_DB.getValue() << '\n';
	dbWriter.write(CLOPT_DB.getValue().c_str());
	errs() << "[-] CLASS bsp_analysis, FUNC runOnModule()\n";
	return false; // true if module was modificated //
}

void bsp_analysis::getPollyAccessPatterns(Function& kernel, Module& M) {
	errs() << "[+] CLASS bsp_analysis, FUNC getPollyAccessPatterns:\n";

	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS bsp_analysis, FUNC getPollyAccessPatterns():\n" + msg);
	};

	Function* kernel_lw = M.getFunction(kernel.getName().str() + "_lwrapped");
	if (!kernel_lw) {
		throwError("could not find loop wrapped kernel version of '" +
		            kernel.getName().str() + "'. Run the loop wrapping pass first!");
	}

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
	
	// construct polly analysis for loop wrapped kernel version
	auto& scopInfo = getAnalysis<polly::ScopInfo>(*kernel_lw);
	auto* scop = scopInfo.getScop();

	auto reportPollyError = [&] (const string& msg) {
		errs() << "  * Warning: Could not deduce access patterns from polly\n";
		errs() << "    for kernel " << kernel.getName().str() << '\n';
		errs() << "    " << msg << '\n';
		scopInfo.releaseMemory();
		errs() << "[-] CLASS bsp_analysis, FUNC getPollyAccessPatterns\n";
	};

	if (!scop) {
		reportPollyError("No Valid polly SCoP object");
		return;
	}
	if (scop->getSize() < 1) {
		reportPollyError("No Valid polly SCoP Statement");
		return;
	}

	//  COLLECT THE MEMORY ACCESSES OF POLLY's ANALYSIS
	//  1. We have to union all accesses on one array in one polly statement
	//     into one isl_map
	//  2. Intersect that union with the corresponding domain of the statement
	//  3. For all statements union all maps in an isl_union_map object
	auto getAccessMapStr = [&] (Value* arg, bool is_read_access) {
		isl_ctx* ctx = isl_ctx_alloc();
		vector<isl_map*> scop_accesses;
		for (auto stmt_it = scop->begin(); stmt_it != scop->end(); ++stmt_it) {
			isl_set* domain = isl_set_read_from_str(ctx, stmt_it->getDomainStr().c_str());
			isl_map* statement_accesses = nullptr;

			// 1. loop over all memory accesses in this statement
			for (auto acc_it = stmt_it->begin(); acc_it != stmt_it->end(); ++acc_it) {
				if ((*acc_it)->isArrayKind() &&
				   ((*acc_it)->isRead() == is_read_access) &&
					(*acc_it)->getBaseAddr() == arg) {

					const string accMapStr = (*acc_it)->getOriginalAccessRelationStr();
					isl_map* islAccMap = isl_map_read_from_str(ctx, accMapStr.c_str());
					if (!statement_accesses) { // for first iteration we must init our accumulator
						statement_accesses = isl_map_empty(isl_map_get_space(islAccMap));
					}
					statement_accesses = isl_map_union(statement_accesses, islAccMap);
				}
			} // END loop over all memory accesses in a statement
			// 2.
			if (statement_accesses) {
				statement_accesses = isl_map_intersect_domain(statement_accesses, domain);
				scop_accesses.push_back(statement_accesses);
			}
			else {
				isl_set_free(domain);
			}
		} // END loop over all statements

		if (scop_accesses.empty()) {
			isl_ctx_free(ctx);
			return string("null");
		}
		// 3.
		isl_union_map* umap = isl_union_map_from_map(scop_accesses[0]);
		for (auto map_it = scop_accesses.begin() + 1; map_it != scop_accesses.end(); ++map_it) {
			umap = isl_union_map_union(umap, isl_union_map_from_map(*map_it));
		}
		
		string result = isl_union_map_to_str(umap);
		isl_union_map_free(umap);
		isl_ctx_free(ctx);
		return result;
	};

	vector<string> isl_reads(kernel_lw->arg_size(), "null");
	vector<string> isl_writes(kernel_lw->arg_size(), "null");

	// Iterate over the kernels arguments and grep all read and write access
	// patterns for pointer type arguments from polly
	int arg_nr = 0;
	for (auto arg_it = kernel_lw->arg_begin(); arg_it != kernel_lw->arg_end(); ++arg_it, ++arg_nr) {
		// we search only for accesses on simple pointers
		if (getPointerLVL(arg_it->getType()) != 1) {
			continue;
		}
		errs() << "  * grep read access pattern for arg %" << arg_it->getName().str() << " nr " << arg_nr << '\n';
		isl_reads[arg_nr] = getAccessMapStr(&*arg_it, true);
		errs() << "  * grep write access pattern for arg %" << arg_it->getName().str() << " nr " << arg_nr << '\n';
		isl_writes[arg_nr] = getAccessMapStr(&*arg_it, false);
	} 
	
	// erase the additional arguments of the loop wrapped kernel
	isl_reads.erase(isl_reads.end() - (kernel_lw->arg_size() - kernel.arg_size()), isl_reads.end());
	isl_writes.erase(isl_writes.end() - (kernel_lw->arg_size() - kernel.arg_size()), isl_writes.end());

	// save the results
	islReads_[&kernel] = isl_reads;
	islWrites_[&kernel] = isl_writes;

	// free memory
	scopInfo.releaseMemory();

	errs() << "[-] CLASS bsp_analysis, FUNC getPollyAccessPatterns\n";
} // END getPollyAccessPatterns()

//! This function creates the linkage between the isl parameters and the llvm values.

//! E.g. [size_x, size_y, size_z, N] -> {...
//! where size_[x|y|z] = global grid size in [x|y|z]
//!       N = argument number 2 of the kernel
//! TODO up to know we compare strings, because polly does not seem to have any
//! linkage between the isl space parameters and the llvm values. We should find
//! a consistent solution for that.
void bsp_analysis::getIslParams(Function& kernel) {
	errs() << "[+] CLASS bsp_analysis, FUNC getIslParams:\n";
	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS bsp_analysis, FUNC getIslParams():\n" + msg);
	};

	vector<string> readMaps;
	vector<string> writeMaps;
	try {
		readMaps = islReads_.at(&kernel);
	}
	catch(...) {
		throwError("Found no isl read map for kernel " + kernel.getName().str());
	}
	try {
		writeMaps = islWrites_.at(&kernel);
	}
	catch(...) {
		throwError("Found no isl write map for kernel " + kernel.getName().str());
	}

	// TODO find a save way to get the linkage between polly's isl
	// space parameters and the belonging llvm values. Up to know
	// we do a simple string comparison.
	auto getLLVMArgumentNr = [] (const Function* f, string argName) {
		int argNr = 0;
		for (auto& arg : f->getArgumentList()) {
			if (arg.getName().str() == argName) {
				return argNr;
			}
			++argNr;
		}
		errs() << "***ERROR: could not identify argument name!\n";
		return -1;
	};

	auto getParams = [&] (const string& mapStr) {
		vector<string> parameters;
		isl_ctx* ctx = isl_ctx_alloc();
		isl_union_map*  umap = isl_union_map_read_from_str(ctx, mapStr.c_str());
		isl_set* temp_set = isl_union_map_params(umap);
		isl_space* space = isl_set_get_space(temp_set);
		for (int i = 0; i < isl_space_dim(space, isl_dim_param); ++i) {
			string paramName = isl_space_get_dim_name(space, isl_dim_param, i);
			if (paramName == "size_x" || paramName == "size_y" || paramName == "size_z") {
				parameters.push_back(paramName);
			}
			else {
				parameters.push_back(string("arg") + to_string(getLLVMArgumentNr(&kernel, paramName)));
			}
		}
		isl_space_free(space);
		isl_set_free(temp_set);
		isl_ctx_free(ctx);
		return parameters;
	};

	vector<vector<string>> readParams(readMaps.size());
	int map_nr = 0;
	for (string& rmap : readMaps) {
		if (rmap != "null" && !rmap.empty()) {
			readParams[map_nr] = getParams(rmap);
		}
		++map_nr;
	}

	vector<vector<string>> writeParams(writeMaps.size());
	map_nr = 0;
	for (string& wmap : writeMaps) {
		if (wmap != "null" && !wmap.empty()) {
			writeParams[map_nr] = getParams(wmap);
		}
		++map_nr;
	}

	// save the results for this kernel to the class
	islReadParams_[&kernel] = readParams;
	islWriteParams_[&kernel] = writeParams;
	errs() << "[-] CLASS bsp_analysis, FUNC getIslParams\n";
} // END getIslParams

void bsp_analysis::getPollyNumDimPerArray(Function& kernel, Module& M) {
	errs() << "[+] CLASS bsp_analysis, FUNC getPollyNumDimPerArray:\n";
	
	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS bsp_analysis, FUNC getPollyNumDimPerArray():\n" + msg);
	};

	Function* kernel_lw = M.getFunction(kernel.getName().str() + "_lwrapped");
	if (!kernel_lw) {
		throwError("could not find loop wrapped kernel version of '" +
		            kernel.getName().str() + "'. Run the loop wrapping pass first!");
	}

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
	
	// construct polly analysis for loop wrapped kernel version
	auto& scopInfo = getAnalysis<polly::ScopInfo>(*kernel_lw);
	auto* scop = scopInfo.getScop();

	auto reportPollyError = [&] (const string& msg) {
		errs() << "  * Warning: Could not deduce array sizes from polly\n";
		errs() << "    for kernel " << kernel.getName().str() << '\n';
		errs() << "    " << msg << '\n';
		scopInfo.releaseMemory();
		errs() << "[-] CLASS bsp_analysis, FUNC getPollyNumDimPerArray\n";
	};

	if (!scop) {
		reportPollyError("No Valid polly SCoP object");
		return;
	}
	if (scop->getSize() < 1) {
		reportPollyError("No Valid polly SCoP Statement");
		return;
	}

	//  COLLECT THE ARRAY DIM SIZES OF POLLY's ANALYSIS
	vector<int> array_dims(kernel.arg_size(), 0);
	int arg_nr = 0;
	for (auto arg_it = kernel_lw->arg_begin(); arg_it != kernel_lw->arg_end(); ++arg_it, ++arg_nr) {
		// skip kernel args which are additional for the loop wrapped kernel version
		if (arg_nr == kernel.arg_size()) {
			break;
		}
		// we search only for arrays represented by simple pointers
		if (getPointerLVL(arg_it->getType()) != 1) {
			continue;
		}
		// TOFIX: if a pointer kernel argument has no accesses there is no ScopArrayInfo
		// object, which leads to an assertion error if we ask for it. The solution is to
		// iterate over all existing ScopArrayInfo objects and to map them to the kernel
		// arguments
		errs() << "  * Requiring ScopArrayInfo for arg_nr = " << arg_nr << " and arg_name = " << arg_it->getName() << "...\n";
		const polly::ScopArrayInfo* array_info = scop->getScopArrayInfo(&*arg_it, polly::ScopArrayInfo::MK_Array);
		array_dims[arg_nr] = array_info->getNumberOfDimensions();
	} 

	// save the results
	errs() << "  * kernel = " << kernel.getName() << '\n';
	errs() << "  * argument array dimensions =";
	for (auto& dim : array_dims) {
		errs() << " " << dim;
	}
	errs() << '\n';
	islNumDimArrays_[&kernel] = array_dims;

	// free memory
	scopInfo.releaseMemory();

	errs() << "[-] CLASS bsp_analysis, FUNC getPollyNumDimPerArray\n";
} // END getPollyNumDimPerArray

void bsp_analysis::getPollyArrayDimSizes(Function& kernel, Module& M) {

	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS bsp_analysis, FUNC getPollyArrayDimSizes():\n" + msg);
	};

	Function* kernel_lw = M.getFunction(kernel.getName().str() + "_lwrapped");
	if (!kernel_lw) {
		throwError("could not find loop wrapped kernel version of '" +
		            kernel.getName().str() + "'. Run the loop wrapping pass first!");
	}

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
	
	// construct polly analysis for loop wrapped kernel version
	auto& scopInfo = getAnalysis<polly::ScopInfo>(*kernel_lw);
	auto* scop = scopInfo.getScop();

	auto reportPollyError = [&] (const string& msg) {
		errs() << "  * Warning: Could not deduce array sizes from polly\n";
		errs() << "    for kernel " << kernel.getName().str() << '\n';
		errs() << "    " << msg << '\n';
		scopInfo.releaseMemory();
		errs() << "[-] CLASS bsp_analysis, FUNC getPollyArrayDimSizes\n";
	};

	if (!scop) {
		reportPollyError("No Valid polly SCoP object");
		return;
	}
	if (scop->getSize() < 1) {
		reportPollyError("No Valid polly SCoP Statement");
		return;
	}

	// TODO find a save way to get the linkage between polly's isl
	// space parameters and the belonging llvm values. Up to know
	// we do a simple string comparison.
	auto getLLVMArgumentNr = [] (const Function* f, string argName) {
		int argNr = 0;
		for (auto& arg : f->getArgumentList()) {
			if (arg.getName().str() == argName) {
				return argNr;
			}
			++argNr;
		}
		errs() << "***ERROR: could not identify argument name!\n";
		return -1;
	};

	auto getParams = [&] (isl_pw_aff* pwAff) {
		if (!pwAff) {
			return vector<string>();
		}
		vector<string> parameters;
		isl_space* space = isl_pw_aff_get_domain_space(pwAff);
		for (int i = 0; i < isl_space_dim(space, isl_dim_param); ++i) {
			string paramName = isl_space_get_dim_name(space, isl_dim_param, i);
			if (paramName == "size_x" || paramName == "size_y" || paramName == "size_z") {
				parameters.push_back(paramName);
			}
			else {
				parameters.push_back(string("arg") + to_string(getLLVMArgumentNr(&kernel, paramName)));
			}
		}
		isl_space_free(space);
		return parameters;
	};

	//  COLLECT THE ARRAY DIM SIZES OF POLLY's ANALYSIS
	vector<vector<string>> arrayDimSizes(kernel.arg_size());
	int arg_nr = 0;
	for (auto arg_it = kernel_lw->arg_begin(); arg_it != kernel_lw->arg_end(); ++arg_it, ++arg_nr) {
		// skip kernel args which are additional an the loop wrapped kernel version
		if (arg_nr == kernel.arg_size()) {
			break;
		}
		// we search only for arrays represented by simple pointers
		if (getPointerLVL(arg_it->getType()) != 1) {
			continue;
		}
		const polly::ScopArrayInfo* array_info = scop->getScopArrayInfo(&*arg_it, polly::ScopArrayInfo::MK_Array);
		int dimensions = array_info->getNumberOfDimensions();
		if (dimensions > 1) {
			for (int curr_dim = 1; curr_dim < dimensions; ++curr_dim) {
				isl_pw_aff* pwAff = array_info->getDimensionSizePw(curr_dim);
				vector<string> pwAffParams = getParams(pwAff);
				if (pwAffParams.size() != 1) {
					throwError("A multidimensional array access can only be handled on an"
					           "array with a size fixed by a kernel parameter. E.g. you can"
					           "have a kernel 'void kernel(int* arr, int N)', where 'N'"
					           "is the size of the second dimension of 'arr'.");
				}
				arrayDimSizes[arg_nr].push_back(pwAffParams[0]);
				isl_pw_aff_free(pwAff);
			}
		}
	} 

	// save the results
	islArrayDimSizes_[&kernel] = arrayDimSizes;

	// free memory
	scopInfo.releaseMemory();

} // END getPollyArrayDimSizes

void bsp_analysis::printPollyInfo(Function* f) {
	if (!f) {
		errs() << "  * print polly info got nullptr\n";
		return;
	}
	auto& scopinfo = getAnalysis<polly::ScopInfo>(*f);
	auto* scop = scopinfo.getScop();
//	scop->print(errs());
	if (!scop) {
		errs() << "\t* no valid scop\n";
	}
	else {
		errs() << "\t* scop->getNumArrays() = " << scop->getNumArrays() << '\n';
		errs() << "\t* scop->isEmpty() = " << scop->isEmpty() << '\n';
		errs() << "\t* scop->getNameStr = " << scop->getNameStr() << '\n';
		errs() << "\t* scop->getSize() = " << scop->getSize();
		errs() << " (= num of scop statements)\n";
		errs() << "\t* scop->getContextStr() = " << scop->getContextStr() << '\n';
		errs() << "\t* listing the statements:\n";
		for (auto statIt = scop->begin(); statIt != scop->end(); ++statIt) {
			errs() << "\t\t- statement with name " << statIt->getBaseName() << '\n';
			errs() << "\t\t- has the size " << statIt->size() << '\n';
			errs() << "\t\t- has the domain:\n";
			auto* domain = statIt->getDomain();
			if (domain != nullptr) {
//				errs() << "\t\t  " << isl_set_to_str(domain) << '\n';
				errs() << "\t\t  " << statIt->getDomainStr() << '\n';
				isl_set_free(domain);
			}
			else {
				errs() << "\t\t  domain = nullptr\n";
			}
			errs() << "\t\t- listing the memory accesses in this statement:\n";
			for (auto accIt = statIt->begin(); accIt != statIt->end(); ++accIt) {
				auto* accRel = (*accIt)->getAccessRelation();
				auto* addrFunc = (*accIt)->getAddressFunction(); 
				auto* domain = (*accIt)->getInvalidDomain();
				Value* base = (*accIt)->getBaseAddr();
				if (accRel != nullptr) {
					errs() << "\t\t\t-- access has access relation:\n";
					errs() << "\t\t\t   " << isl_map_to_str(accRel) << '\n';
					isl_map_free(accRel);
				}
				else {
					errs() << "\t\t\t-- access has no access relation\n";
				}
				if (addrFunc != nullptr) {
					errs() << "\t\t\t-- access has address function:\n";
					errs() << "\t\t\t   " << isl_map_to_str(addrFunc) << '\n';
					isl_map_free(addrFunc);
				}
				else {
					errs() << "\t\t\t-- access has no address function\n";
				}
				if (domain != nullptr) {
					errs() << "\t\t\t-- access has invalid domain:\n";
					errs() << "\t\t\t   " << isl_set_to_str(domain) << '\n';
					isl_set_free(domain);
				}
				if (base != nullptr) {
					errs() << "\t\t\t-- access has base %" << base->getName() << "\n";
				}
				else {
					errs() << "\t\t\t-- access has no llvm base pointer\n";
				}
			}
		}
	}
	scopinfo.releaseMemory();
}

vector<Function*> bsp_analysis::getKernels(Module& M) {
	errs() << "[+] CLASS bsp_analysis, FUNC getKernels:\n";
    auto* kernel_md = M.getNamedMetadata("opencl.kernels");
    vector<Function*> kernels;
    if (kernel_md) {
        // MDNodes in NamedMDNode
        for (const auto& node : kernel_md->operands()) {
            // MDOperands in MDNode
            for (auto& op : node->operands()) {
                // is actual value?
                if (ValueAsMetadata* v = dyn_cast<ValueAsMetadata>(op)) {
                    // is function?
                    if (Function* f = dyn_cast<Function>(v->getValue())) {
                        kernels.push_back(f);
                    }
                }
            }
        }
    } else {
		errs() << "***ERROR: CLASS bsp_analysis, FUNC getKernels\n";
		errs() << "          Code contains no opencl kernels!\n";
    }
	errs() << "  * found the following opencl kernels:\n";
	for (const auto& f : kernels) {
		errs() << "    * " << f->getName() << '\n';
	}
	errs() << "[-] CLASS bsp_analysis, FUNC getKernels\n";
    return kernels;
}

char bsp_analysis::ID = 0;
static RegisterPass<bsp_analysis> X("bsp_analysis", "Makes opencl kernel analysis");

} // namespace //

#undef DEBUG_TYPE
