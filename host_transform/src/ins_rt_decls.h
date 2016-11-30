/******************************************************************
 *
 * FILENAME    - ins_rt_decls.h
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-07-05
 *
 ******************************************************************/ 

#ifndef INSERT_RUNTIME_DECLARATIONS_H
#define INSERT_RUNTIME_DECLARATIONS_H

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"

using namespace llvm;	

namespace Mekong {

struct ins_rt_decls : public ModulePass {

	static char ID; // Pass identification, replacement for typeid

	ins_rt_decls();

	bool runOnModule(Module &M) override;
}; 

}

#endif
