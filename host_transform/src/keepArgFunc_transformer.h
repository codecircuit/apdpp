/******************************************************************
 *
 * FILENAME    - keepArgFunc.h
 *
 * DESCRIPTION - this class replaces a function call and
 *               keeps the operands
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2015-11-16
 *
 ******************************************************************/ 

#ifndef KEEPARGFUNC_TRANSFORMER_H
#define KEEPARGFUNC_TRANSFORMER_H

#include "base_transformer.h"

#define DEBUG_TYPE "KEEP_ARG_FUNC_TRANSFORMER"

class KeepArgFuncTransformer : public BaseTransformer {
	public:
		KeepArgFuncTransformer(Module& M, const char* targetName, const char* startName) :
		                       BaseTransformer(M, std::vector<std::string>(1, targetName),
		                                       std::vector<std::string>(1, startName)),
		                       targetName_(targetName),
		                       startName_(startName) {}

		void run() override {
			DEBUG(errs() << "[+] CLASS KeepArgFuncTransformer, FUNC run():\n");

			// CHECK PRECONDITIONS //
			if (startName2Addresses_.empty()) {
				DEBUG(errs() << "\t* no calls of " << startName_
				             << "\t  thus nothing replaced with " << targetName_ << '\n');
				return;
			}
			else if (targetName2Address_.empty()) {
				errs() << "***ERROR: CLASS KeepArgFuncTransformer, FUNC run():\n"
				       << "\t* no target function found\n";
				return;
			}

			Value* targetFunc = targetName2Address_.begin()->getValue();
			for (Value* val : startName2Addresses_.begin()->getValue()) {
				if (CallInst* ci = dyn_cast<CallInst>(val)) {
					DEBUG(errs() <<   "\t* Replacing " << *ci);
					ci->setCalledFunction(targetFunc);
					DEBUG(errs() << "\n\t  with " << *ci << '\n');
					++nrTransforms();
				}
				else if (InvokeInst* ii = dyn_cast<InvokeInst>(val)) {
					DEBUG(errs() <<   "\t* Replacing " << *ii);
					ii->setCalledFunction(targetFunc);
					DEBUG(errs() << "\n\t  with " << *ii << '\n');
					++nrTransforms();
				}
			}

			DEBUG(errs() << "[-] CLASS KeepArgFuncTransformer, FUNC run()\n");
		}

	private:
		const std::string targetName_;
		const std::string startName_;
};

#undef DEBUG_TYPE

#endif
