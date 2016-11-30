/******************************************************************
 *
 * FILENAME    - propagator.cc
 *
 * DESCRIPTION - source file for Propagator class. Look into
 *               propagator.h for detailed description.
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-05-13
 *
 ******************************************************************/ 

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h" // ReplaceInstWithInst(...) //

#include "propagator.h"

#define DEBUG_TYPE "PROPAGATOR"

Propagator::Propagator(std::shared_ptr<Propagator::MappingT> orig2super) :
					   orig2super_(orig2super),
					   worklist_(getWorkList()) {}

void Propagator::run() {
	DEBUG(errs() << "[+] CLASS Propagator, FUNC run():\n");
	if (worklist_.empty()) {
		DEBUG(errs() << "\t* worklist is empty.\n"
		             << "[-] CLASS Propagator, FUNC run()\n");
		return;
	}
	
	auto getArg = [] (Function* f, size_t argId) {
		auto arg = f->arg_begin();
		for (unsigned short i = 0; i < argId; ++i) {
			++arg;
		}
		return arg;
	};

	auto getArgOfFunc = [&] (CallInst* call, size_t argId) {
		return getArg(call->getParent()->getParent(), argId);
	};
	
	// iterate over all call instructions which called one of the given
	// target functions, e.g. get_global_size, get_global_id
	for (auto* ci : worklist_) {

		// get old argument operands for the function call //
		std::vector<Value*> new_args;
		for (const auto& arg_it : ci->arg_operands()) {
			new_args.push_back(arg_it);
		}

		auto* superFunc = orig2super_->at(ci->getCalledFunction());
		size_t newArgCount = superFunc->arg_size();
		size_t oldArgCount = ci->getCalledFunction()->arg_size();	

		// add new argument operands for call of '_super' function version //
		for (size_t argId = oldArgCount; argId < newArgCount; ++argId) {
			new_args.push_back(&(*getArgOfFunc(ci, argId)));
		}

		// creating the new call instruction without a name, as it can be a void call //
		CallInst* nci = CallInst::Create(superFunc, ArrayRef<Value*>(new_args));
		DEBUG(errs() << "\t* new call instruction\n\t" << *nci << '\n');

		// replacing old instruction with new instruction //
		BasicBlock::iterator ii(ci);
		ReplaceInstWithInst(ci->getParent()->getInstList(), ii, nci);
	}

	DEBUG(errs() << "[-] CLASS Propagator, FUNC run()\n");
}

std::vector<CallInst*> Propagator::getWorkList() const {
	std::vector<CallInst*> result;

	// get '_super' versions of functions //
	std::vector<Function*> superFunctions;
	for (const auto& pair : *orig2super_)	{
		superFunctions.push_back(std::get<1>(pair));
	}
	
	// iterate over all super functions //
	for (Function* f_super : superFunctions) {
		// iterate over all instructions within a super function //
		for (auto it = inst_begin(f_super); it != inst_end(f_super); ++it) {
			// check if the actual instruction is a call instruction //
			if (auto* ci = dyn_cast<CallInst>(&(*it))) {
				auto mapIt = orig2super_->find(ci->getCalledFunction());
				// if the subsequent condition is true, the called function 
				// has a '_super' version
				if (mapIt != orig2super_->end()) {
					result.push_back(ci);
				}
			}
		}
	}
	return result;
}

#undef DEBUG_TYPE
