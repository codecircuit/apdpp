#ifndef LOOP_WRAPPING_PASS
#define LOOP_WRAPPING_PASS

#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h" 
#include "llvm/Transforms/Utils/BasicBlockUtils.h" // ReplaceInstWithInst()

#include <stdexcept>
#include <tuple>
#include <string>

#define DEBUG_TYPE "GEF"

using namespace llvm;
using namespace std;

struct loop_wrapping_pass : public ModulePass {
	static char ID; // Pass identification, replacement for typeid
	loop_wrapping_pass();

	bool runOnModule(Module &M) override;
	
	static tuple<Value*, Value*, Value*> insertLoops(Function* f);
	static Function* cloneAndAddArgs(Function* parent); 
	static void replaceCalls(Function* f, Value* xAddress, Value* yAddress, Value* zAddress);
	static void throwError(const string& functionName, const string& message);
};

#undef DEBUG_TYPE
#endif
