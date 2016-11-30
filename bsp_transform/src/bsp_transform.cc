/******************************************************************
 *
 * FILENAME    - bsp_transform.cc
 *
 * DESCRIPTION - llvm pass for the Mekong project. This pass
 *               transforms the device code, thus creating
 *               _super versions of functions which use
 *               get_global_id().
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-02-09
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

#include <vector>
#include <string>
#include <fstream>

#include "promoter.h"
#include "propagator.h"
#include "offsetter.h"

#include "dashdb.h"

using namespace llvm;

#define DEBUG_TYPE "DEVICE_CODE"

namespace {

// Here we get the command line argument from opt
// CLOPT_DB.getValue().c_str() gets you the c-string
static cl::opt<string> CLOPT_DB("mekong_db", cl::desc("specifies the database file for mekong (format .ddb)"), cl::value_desc("filename"));

/*! \brief Transforms the device code and splits along a specified dimension.

    The transformation consists of three steps:
      1. Promoting: all device functions and kernels, which contain a get_global_id
         or get_global_size call along the dimension we want to split will be copied.
         The copy will be labeled with another name to distinguish it from the original version.
         Moreover the promoted function has six additional kernel arguments:
         offset_[x|y|z], global_size_[x|y|z].
      2. Propagating: all promoted functions should call only promoted functions. Thus
         this transformation step replaces all function calls of original functions with
         calls of the promoted versions, if there exists one.
      3. Offsetting: Finally, all promoted functions' calls of get_global_id with
         the dimension we want to split will be replaced with get_global_id() + offset.
         Calls of get_global_size will be replaced with the kernel argument global_size.

	\todo Support multiple kernel transformation along different dimensions. E.g.
	      split kernelA along x-dimension and kernelB along y-dimension. Up to now we
	      split every kernel along the dimension the first kernel shall be splitted.
*/
struct bsp_transform : public ModulePass {
	static char ID; // Pass identification, replacement for typeid
	bsp_transform() : ModulePass(ID) {}

	bool runOnModule(Module &M) override {

		auto throwError = [] (const string& msg) {
			throw runtime_error("SPACE Mekong, CLASS bsp_transform, FUNC runOnModule():\n" + msg);
		};

		// promote all functions with calls of get_global_id and get_global_size
		std::vector<std::string> targets({"get_global_id", "get_global_size"});
		Promoter pr(M, targets);
		pr.run();
		Propagator pg(pr.getMapping());
		pg.run();
		
		const string& dbName = CLOPT_DB.getValue().c_str();
		if (dbName.empty()) {
			throwError("I need a path to a database file!");
		}
		string dbPattern;
		string pattern;
		dashdb::Butler b(dbName);

		// HACK: we split always along the split dimension of the first kernel	
		dbPattern = b["kernels"][0]["partitioning"].asString();
		if (dbPattern == "x") {
			pattern = "OneDim_X";
		}
		else if (dbPattern == "y") {
			pattern = "OneDim_Y";
		}
		else {
			throwError(string("I do not know the partitioning pattern: ") + dbPattern);
		}
		Offsetter of(pr.getMapping(), pattern);
		of.run();
		return true; // true if module was modificated //
	}
}; 

} // namespace //

char bsp_transform::ID = 0;
static RegisterPass<bsp_transform> X("bsp_transform", "introduce _super-versions of function, which use get_global_id().");

#undef DEBUG_TYPE
