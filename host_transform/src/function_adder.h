/******************************************************************
 *
 * FILENAME    - function_adder.h
 *
 * DESCRIPTION - This class adds a function call before
 *               the first call of another given function or
 *               inserts a call instruction before every
 *               return statement in the main function.
 *               The latter is useful to add Mekong's
 *               report function.
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-04-25
 *
 ******************************************************************/

#ifndef FUNCTION_ADDER_H
#define FUNCTION_ADDER_H

#include "llvm/IR/Instructions.h"
#include "base_transformer.h"

#include <vector>
#include <string>

#define DEBUG_TYPE "FUNCTION_ADDER"

class FunctionAdder : public BaseTransformer {
	public:
		FunctionAdder(Module& M, const char* targetName, const char* successorName) 
		      : BaseTransformer(M, std::vector<std::string>(1, targetName),
		                           std::vector<std::string>(1, successorName)),
		        targetName_(targetName),
		        successorName_(successorName) {}
		
		//! With this constructor the FunctionAdder will add the target function
		//! in front of every return statement
		FunctionAdder(Module& M, const char* targetName)
		      : BaseTransformer(M, std::vector<std::string>(1, targetName),
		                           std::vector<std::string>()),
		        targetName_(targetName),
		        M_(&M),
		        successorName_("") {}

		void run() override {
			DEBUG(errs() << "[+] CLASS FunctionAdder, FUNC run():\n");
			DEBUG(errs() << "\t* target name: " << targetName_ << '\n');
			DEBUG(errs() << "\t* successor name: " << successorName_ << '\n');

			if (startName2Addresses_.empty() && !successorName_.empty()) {
				errs() << "***ERROR: CLASS FunctionAdder, FUNC run():\n"
				       << "\tno start addresses!\n";
					   return;
			}
			if (targetName2Address_.empty()) {
				errs() << "***ERROR: CLASS FunctionAdder, FUNC run():\n"
				       << "\tno target address!\n";
					   return;
			}

			// add in front of every return statement
			std::vector<Instruction*> retInstrs;
			auto* function = targetName2Address_.begin()->getValue();
			if (successorName_.empty()) {
				for (Instruction& instr : instructions(main_)) {
					if (ReturnInst* retIns = dyn_cast<ReturnInst>(&instr)) {
						retInstrs.push_back(retIns);
					}
				}
				for (auto* successor : retInstrs) {
					auto* newCallInst = CallInst::Create(function, ArrayRef<Value*>(),
														 Twine(""),
														 successor);
					DEBUG(errs() << "\t* inserting: " << *newCallInst << '\n'
								 << "\t  before:    " << *successor << '\n');
				}
			}
			// add only one call instruction
			else {
				Instruction* successor = cast<Instruction>(startName2Addresses_.begin()->getValue().at(0));

				auto* newCallInst = CallInst::Create(function, ArrayRef<Value*>(),
													 Twine("functionAdder" + std::to_string(nrTransforms()++)),
													 successor);

				DEBUG(errs() << "\t* inserting: " << *newCallInst << '\n'
							 << "\t  before:    " << *successor << '\n');
			}
			DEBUG(errs() << "[-] CLASS FunctionAdder, FUNC run()\n");
		}

	private:
		Module* M_;
		const std::string targetName_;
		const std::string successorName_;
};

#undef DEBUG_TYPE

#endif
