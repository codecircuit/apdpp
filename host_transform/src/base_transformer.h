/******************************************************************
 *
 * FILENAME    - base_transformer.h
 *
 * DESCRIPTION - Base class for transformation classes.
 *               Every transformer class has some start objects
 *               with names and types and targets with names
 *               and types. The run() function will transform
 *               the start objects to the target objects.
 *               Every Transformations are only done in
 *               the main function of the program up to now.
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2016-05-13
 *
 ******************************************************************/ 

#ifndef BASE_TRANSFORMER_H
#define BASE_TRANSFORMER_H

#include "llvm/IR/Module.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Function.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/InstIterator.h"

#include <string>
#include <utility> // std::pair, std::make_pair
#include <vector>

#define DEBUG_TYPE "BASE_TRANSFORMER"

using namespace llvm;

/*
 *  DESCRIPTION - This function counts the number of transformations done
 *                by all derived classes of all instances of BaseTransformer
 */
unsigned int& nrTransforms() {
	static unsigned int count = 0;
	return count;
}

class BaseTransformer {
	public:
		BaseTransformer(Module& M) : M_(M), main_(M.getFunction("main")) {}
		BaseTransformer(Module& M, const std::vector<std::string>& targetNames,
		                           const std::vector<std::string>& startNames);
		virtual void run() = 0;
		static bool uses(User* user, Value* val);

		void printMain() const;

		void printFunctions() const;

	protected:
		/*
		 *  DESCRIPTION - This function removes all instructions
		 *                which uses 'used' AFTER 'pos' in the
		 *                same function. If some instruction after 'pos'
		 *                depends somehow (not directly) on 'used' it will be
		 *                deleted too. 
		 */
		static void removeUsesAfter(Instruction* pos, Value* used);

		/*
		 *  DESCRIPTION - This function removes the instruction
		 *                and every usage of this instruction.
		 *                The remove process is recursive, thus
		 *                if something depends on 'instr' it
		 *                will be deleted.
		 */
		static void removeUses(Instruction* instr);

		
		/*
		 *  DESCRIPTION - gets you the instruction iterator to the instruction
		 *                and as second return argument the iterator to the
		 *                end of the belonging function.
		 */		
		static std::pair<inst_iterator, inst_iterator> getInstIterator(Instruction* instr);
		
		Module& M_;
		Function* main_;
		StringMap<Value*> targetName2Address_;
		StringMap<std::vector<Value*> > startName2Addresses_;
};

BaseTransformer::BaseTransformer(Module& M, const std::vector<std::string>& targetNames,
                                            const std::vector<std::string>& startNames) 
                                : M_(M), main_(M.getFunction("main")) {

	// get the adresses of the target functions //
	DEBUG(errs() << "[+] CLASS BaseTransformer, FUNC Constructor:\n\t* targetNames:\n");
	for (const auto& tn : targetNames) {
		if (auto* f = M_.getFunction(tn)) { // only insert if f != nullptr //
			targetName2Address_.insert(std::make_pair(tn, f));
			DEBUG(errs() << "\t\t  " << tn << ", " << M_.getFunction(tn) << '\n');
		}
	}

	// get the adresses of the start function calls //
	DEBUG(errs() << "\t* startNames:\n");
	for (const auto& sn : startNames) {
		if (auto* f = M_.getFunction(sn)) {
			std::vector<Value*> addresses;
			if (main_) {
				for (auto& bb : main_->getBasicBlockList()) {
					for (auto& ins : bb.getInstList()) {
						if (auto* ci = dyn_cast<CallInst>(&ins)) {
							if (ci->getCalledFunction() == f) {
								addresses.push_back(ci);
								DEBUG(errs() << "\t\t  " << sn << ", " << ci->getName() << ", " << ci << '\n');
							}

						}
						else if (auto* invInst = dyn_cast<InvokeInst>(&ins)) {
							if (invInst->getCalledFunction() == f) {
								addresses.push_back(invInst);
								DEBUG(errs() << "\t\t  " << sn << ", " << invInst->getName() << ", " << invInst << '\n');
							}
						}
					}
				}
				if (!addresses.empty()) {
					startName2Addresses_.insert(std::make_pair(sn, addresses));
				}
			}
		}
	}
}

std::pair<inst_iterator,inst_iterator> BaseTransformer::getInstIterator(Instruction* instr) {
	auto* func = instr->getParent()->getParent();
	inst_iterator it_instr;

	// TO DO: this first for loop is unnecessary as it goes only to the 
	// correct position of 'pos'
	for (it_instr = inst_begin(func); it_instr != inst_end(func); ++it_instr) {
		if (&*it_instr == instr) {
			return std::make_pair(it_instr, inst_end(func));
		}
	}
	errs() << "FATAL ERROR IN HOST TRANSFORM PASS: could not find instruction iterator\n";
	return std::make_pair(it_instr, inst_end(func));
}

void BaseTransformer::removeUsesAfter(Instruction* pos, Value* used) {
	DEBUG(errs() << "[+] CLASS BaseTransformer, FUNC removeUsesAfter:\n");
	DEBUG(errs() << "\t* pos " << *pos << "\n\t  used " << used->getName() << '\n');

	auto pair = getInstIterator(pos);
	inst_iterator it_instr = std::get<0>(pair);
	inst_iterator end = std::get<1>(pair);
	++it_instr;

	while (it_instr != end) {
		if (uses(&*it_instr, used)) {
			DEBUG(errs() << "\t* going to delete " << *it_instr << '\n');
			BaseTransformer::removeUsesAfter(&*it_instr, &*it_instr);
			auto* toBeRemoved = &*(it_instr++);
			DEBUG(errs() << "\t* now delete " << *toBeRemoved << '\n');
			toBeRemoved->eraseFromParent();
		}
		else {
			++it_instr;
		}
	}
	DEBUG(errs() << "[-] CLASS BaseTransformer, FUNC removeUsesAfter\n");
}

bool BaseTransformer::uses(User* user, Value* val) {
	for (auto* use = user->op_begin(); use != user->op_end(); ++use) {
		if (val == use->get()) {
			return true;
		}
	}
	return false;
}

void BaseTransformer::removeUses(Instruction* instr) {
	DEBUG(errs() << "[+] CLASS BaseTransformer, FUNC removeUses:\n");
	DEBUG(errs() << "\t* instruction: " << *instr << '\n');

	auto pair = getInstIterator(instr);
	inst_iterator it_instr = std::get<0>(pair);
	inst_iterator end = std::get<1>(pair);

	while (it_instr != end) {
		if (uses(&*it_instr, instr) || instr == &*it_instr) {
			DEBUG(errs() << "\t* going to delete " << *it_instr << '\n');
			removeUsesAfter(&*it_instr, &*it_instr);
			auto* toBeRemoved = &*(it_instr++);
			DEBUG(errs() << "\t* now delete " << *toBeRemoved << '\n');
			toBeRemoved->eraseFromParent();
		}
		else {
			++it_instr;
		}
	}

	DEBUG(errs() << "[-] CLASS BaseTransformer, FUNC removeUses\n");
}

void BaseTransformer::printMain() const {
	DEBUG(errs() << "[+] CLASS BaseTransformer, FUNC printMain():\n");
	if (main_) {
		errs() << *main_ << '\n';
	}
	else {
		errs() << "***ERROR: ClASS BaseTransformer, FUNC printMain():\n"
		       << "\tmain_ == nullptr!\n";
	}
	DEBUG(errs() << "[-] CLASS BaseTransformer, FUNC printMain()\n");
}

void BaseTransformer::printFunctions() const {
	DEBUG(errs() << "[+] CLASS BaseTransformer, FUNC printFunctions():\n");
	for (const auto& f : M_.getFunctionList()) {
		errs() << f.getName() << '\n';
	}
	DEBUG(errs() << "[-] CLASS BaseTransformer, FUNC printFunctions()\n");
}


#undef DEBUG_TYPE

#endif
