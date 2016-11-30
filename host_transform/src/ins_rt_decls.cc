/******************************************************************
 *
 * FILENAME    - ins_rt_decls.cc
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-07-05
 *
 ******************************************************************/ 

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"

#include "ins_rt_decls.h"

#include <string>
#include <map>
#include <vector>

#define DEBUG_TYPE "INS_RT_DECLS"

using namespace llvm;
using namespace std;

namespace Mekong {

char ins_rt_decls::ID = 0;
static RegisterPass<ins_rt_decls> XX("ins-rt-decls", "inserts Mekong's runtime functions' declarations");

ins_rt_decls::ins_rt_decls() : ModulePass(ID) {}

bool ins_rt_decls::runOnModule(Module &M) {

	errs() << "\n# Pass to insert wrapping functions' declaration\n\n";

	// LIST THE FUNCTIONS WE HAVE TO TRANSFORM
	vector<string> startFunctionNames = {
		"cuInit",
		"cuDeviceGetCount",
		"cuDeviceGet",
		"cuDeviceComputeCapability",
		"cuCtxCreate_v2",
		"cuModuleLoad",
		"cuModuleGetFunction",
		"cuMemAlloc_v2",
		"cuMemcpyHtoD_v2",
		"cuLaunchKernel",
		"cuCtxSynchronize",
		"cuMemcpyDtoH_v2",
		"cuMemFree_v2",
		"cuCtxDestroy_v2"
	};

	vector<string> targetFunctionNames = {
		"wrapInit",
		"wrapDeviceGetCount",
		"wrapDeviceGet",
		"wrapDeviceComputeCapability",
		"wrapCtxCreate",
		"wrapModuleLoad",
		"wrapModuleGetFunction",
		"wrapMemAlloc",
		"wrapMemcpyHtoD",
		"wrapLaunchKernel",
		"wrapCtxSynchronize",
		"wrapMemcpyDtoH",
		"wrapMemFree",
		"wrapCtxDestroy"
	};
	
	// CREATE THE MAPPING BETWEEN OLD AND NEW NAMES
	map<string, string> startName2targetName;
	auto sname_it = startFunctionNames.begin();
	auto tname_it = targetFunctionNames.begin();
	for (; sname_it != startFunctionNames.end(); ++sname_it, ++tname_it) {
		startName2targetName[*sname_it] = *tname_it;
	}
	
	// MAP FUNCTION TYPES TO NEW NAMES
	map<FunctionType*, string> ftype2wrapperName;
	for (string& name : startFunctionNames) {
		auto* f = M.getFunction(name);
		if (f) {
			ftype2wrapperName[f->getFunctionType()] = startName2targetName[name];
		}
	}

	// INSERT NEW FUNCTIONS WITH THE SAME TYPE
	for (auto& ftype_targetName : ftype2wrapperName) {
		string targetName = get<1>(ftype_targetName);
		FunctionType* ftype = get<0>(ftype_targetName);
		errs() << "  - Inserting Function with name " << targetName << '\n';
		M.getOrInsertFunction(targetName, ftype);
	}

	// INSERT MEKONG'S REPORT FUNCTION
	errs() << "  - Inserting Function with name MEKONG_report\n";
	LLVMContext& ctx = M.getContext();
	Type* voidTy = Type::getVoidTy(ctx);
	FunctionType* voidFTy = FunctionType::get(voidTy, false);
	M.getOrInsertFunction("MEKONG_report", voidFTy);

	errs() << '\n';
	return true; // true if module was modificated //
}

} // namespace //

#undef DEBUG_TYPE
