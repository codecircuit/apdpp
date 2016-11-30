/******************************************************************
 *
 * FILENAME    - host_transform.cc
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-03-24
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

#include "base_transformer.h"
#include "ins_rt_decls.h"
#include "keepArgFunc_transformer.h"
#include "function_adder.h"

#include <string>

#define DEBUG_TYPE "HOST_TRANSFORM"

using namespace llvm;

namespace Mekong {

static cl::opt<std::string> CLOPT_DB("mekong_db", cl::desc("specifies the database file for alleycat (format .json)"), cl::value_desc("filename"));
	
struct host_transform : public ModulePass {
	static char ID; // Pass identification, replacement for typeid
	host_transform() : ModulePass(ID) {}

	bool runOnModule(Module &M) override {

		// TODO: check used cuda version to check usage of  _v2 suffix
		KeepArgFuncTransformer cuInit(M, "wrapInit", "cuInit");
		KeepArgFuncTransformer cuDeviceGetCount(M, "wrapDeviceGetCount", "cuDeviceGetCount");
		KeepArgFuncTransformer cuDeviceGet(M, "wrapDeviceGet", "cuDeviceGet");
		KeepArgFuncTransformer cuDeviceComputeCapability(M,
														 "wrapDeviceComputeCapability",
														 "cuDeviceComputeCapability");
		KeepArgFuncTransformer cuCtxCreate(M, "wrapCtxCreate", "cuCtxCreate_v2");
		KeepArgFuncTransformer cuModuleLoad(M, "wrapModuleLoad", "cuModuleLoad");
		KeepArgFuncTransformer cuModuleGetFunction(M, "wrapModuleGetFunction", "cuModuleGetFunction");
		KeepArgFuncTransformer cuMemAlloc(M, "wrapMemAlloc", "cuMemAlloc_v2");
		KeepArgFuncTransformer cuMemcpyHtoD(M, "wrapMemcpyHtoD", "cuMemcpyHtoD_v2");
		KeepArgFuncTransformer cuLaunchKernel(M, "wrapLaunchKernel", "cuLaunchKernel");
		KeepArgFuncTransformer cuCtxSynchronize(M, "wrapCtxSynchronize", "cuCtxSynchronize");
		KeepArgFuncTransformer cuMemcpyDtoH(M, "wrapMemcpyDtoH", "cuMemcpyDtoH_v2");
		KeepArgFuncTransformer cuMemFree(M, "wrapMemFree", "cuMemFree_v2");
		KeepArgFuncTransformer cuCtxDestroy(M, "wrapCtxDestroy", "cuCtxDestroy_v2");
		FunctionAdder adder(M, "MEKONG_report");
		
		adder.run();
		cuInit.run();
		cuDeviceGetCount.run();
		cuDeviceGet.run();
		cuDeviceComputeCapability.run();
		cuCtxCreate.run();
		cuModuleLoad.run();
		cuModuleGetFunction.run();
		cuMemAlloc.run();
		cuMemcpyHtoD.run();
		cuLaunchKernel.run();
		cuCtxSynchronize.run();
		cuMemcpyDtoH.run();
		cuMemFree.run();
		cuCtxDestroy.run();

		return true; // true if module was modificated //
	}

	void getAnalysisUsage(AnalysisUsage &Info) const override {
		Info.addRequired<ins_rt_decls>();
	}
}; 


char host_transform::ID = 0;
static RegisterPass<host_transform> X("host_transform", "make function replacements");

}

#undef DEBUG_TYPE
