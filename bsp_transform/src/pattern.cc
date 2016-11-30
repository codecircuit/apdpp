/******************************************************************
 *
 * FILENAME    - pattern.cc
 *
 * DESCRIPTION - source file for pattern.h
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2015-12-15
 *
 ******************************************************************/ 

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "pattern.h"

#define DEBUG_TYPE "PATTERN"

void replaceAndErase(Instruction* o, Value* n) {
	DEBUG(errs() << "[+] GP LLVM, FUNC replaceAndErase():\n");

	if (!o || !n) {
		DEBUG(errs() << "\t* got nullptr\n");
		DEBUG(errs() << "[-] GP LLVM, FUNC replaceAndErase()\n");
		return;
	}

	DEBUG(errs() << "\t* Replacing:  " << *o << '\n'
	             << "\t  with        " << *n << '\n'
	             << "\t  in function " << o->getParent()->getParent()->getName() << '\n');

	// It is mandatory to collect the users first, before
	// changing the underlaying iterator structure
	std::vector<User*> vUsers;
	for (const auto& user : o->users()) {
		vUsers.push_back(user);
	}

	for (auto* user : vUsers) {
		DEBUG(errs() << "\t* user of old instr. " << *user << '\n');
		for (int i = 0; i < user->getNumOperands(); ++i) {
			if (user->getOperand(i) == o) {
				user->getOperandUse(i).set(n);
				DEBUG(errs() << "\t  replaced with      " << *user << '\n');
			}
		}	
	}

	o->eraseFromParent();
	DEBUG(errs() << "[-] GP LLVM, FUNC replaceAndErase()\n");
}

std::vector<CallInst*> Pattern::getWorkList(short int axis, Function* f) {
	DEBUG(errs() << "[+] CLASS Pattern, FUNC getWorkList():\n");
	std::vector<CallInst*> result;
	if (!f) {
		DEBUG(errs() << "\t* got nullptr\n");
		DEBUG(errs() << "[-] CLASS Pattern, FUNC getWorkList()\n");
		return result;
	}
	DEBUG(errs() << "\t* axis = " << axis << '\n' << "\t* function = " << f->getName() << '\n');
	// iterate over all instructions in the function //
	for (auto it = inst_begin(f); it != inst_end(f); ++it) {
		// if you find a call instruction //
		if (auto* ci = dyn_cast<CallInst>(&(*it))) {
			// check if it is a call of opencl 'get_global_id' //
			if (std::string(ci->getCalledFunction()->getName()) == "get_global_id" ||
				std::string(ci->getCalledFunction()->getName()) == "get_global_size") {
				// grep the argument of the call //
				if (auto* cint = dyn_cast<ConstantInt>(ci->getArgOperand(0))) {
					// and check if it is the wanted axis //
					if (cint->getValue().getLimitedValue() == axis) {
						// if so, add it to the worklist //
						DEBUG(errs() << "\t* adding " << *ci << '\n');
						result.push_back(ci);
					}
				}
				else {
					errs() << "***ERROR CLASS Pattern, FUNC getWorkList(" << axis << ", " << f->getName() << "):\n"
					       << "\tfound function named 'get_global_id' without a constant integer as an argument.\n";
				}
			}
		}
	}
	DEBUG(errs() << "[-] CLASS Pattern, FUNC getWorkList()\n");
	return result;
}

void Pattern::addOffsetAndReplace(CallInst* ci) {
	DEBUG(errs() << "[+] CLASS Pattern, FUNC addOffsetAndReplace():\n");
	if (!ci) {
		DEBUG(errs() << "\t* got nullptr\n" << "[-] CLASS Pattern, FUNC addOffsetAndReplace()\n");
		return;
	}
	DEBUG(errs() << "\t* given call Instruction " << *ci << '\n');

	// get last (last) argument of a function //
	auto getLastArg = [] (Function* f) {return --f->arg_end();};
	auto getLastLastArg = [] (Function* f) {return --(--f->arg_end());};
	
	auto getArg = [] (Function* f, size_t argId) {
		auto arg = f->arg_begin();
		for (unsigned short i = 0; i < argId; ++i) {
			++arg;
		}
		return arg;
	};

	// grep the argument of get_global_id or get_global_size call //
	ConstantInt* cint = cast<ConstantInt>(ci->getArgOperand(0));
	size_t axis = cint->getValue().getLimitedValue();

	Function* parent = ci->getParent()->getParent();
	DEBUG(errs() << "\t* parent " << parent->getName() << '\n');

	// if this is a get_global_size call only replace it with the kernel argument
	if (std::string(ci->getCalledFunction()->getName()) == "get_global_size") {
		DEBUG(errs() << "\t* got global size instruction at axis " << axis << '\n');
		// we know that the last three kernel args belong to the global size values
		replaceAndErase(ci, &(*getArg(parent, parent->arg_size() - (3 - axis))));
	}

	// else add to the global id the offset of the partition given as a kernel argument
	else {
		DEBUG(errs() << "\t* got get global id call\n");
		LLVMContext& ctx = parent->getContext();
		BasicBlock* entry = &parent->getEntryBlock();
		BasicBlock* bb = BasicBlock::Create(ctx);
		IRBuilder<> Builder(bb);

		// %a = get_global_id() + partition.offset
		Value *gid_call = Builder.CreateCall(ci->getCalledFunction(), cint);
		Value *a = Builder.CreateAdd(gid_call, &(*getArg(parent, parent->arg_size() - (6 - axis))));
		bb->insertInto(parent, entry);
		Builder.CreateBr(entry);
		DEBUG(errs() << "\t* created the basic block " << *bb << '\n');

		// replacing old instruction with new instruction //
		replaceAndErase(ci, cast<Instruction>(a));
		DEBUG(errs() << "\t* new entry block:\n" << *bb << '\n');
	}
	DEBUG(errs() << "[-] CLASS Pattern, FUNC addOffsetAndRelace()\n");
}

void OneDim_Y::operator()(Function* f) const {
	DEBUG(errs() << "[+] CLASS OneDim_Y, FUNC operator():\n");
	if (!f) {
		DEBUG(errs() << "\t* got nullptr\n");
		DEBUG(errs() << "[-] CLASS OneDim_Y, FUNC operator()\n");
		return;
	}
	DEBUG(errs() << "\t* given function name " << f->getName() << '\n');
	// get the calls of get_global_id(1) //
	std::vector<CallInst*> worklist(getWorkList(1, f));
	for (auto* ci : worklist) {
		addOffsetAndReplace(ci); // TODO if we call get_global_id more than once we
		                         // insert more than one basic block which is not
		                         // neccessary.
	}
	DEBUG(errs() << "[-] CLASS OneDim_Y, FUNC operator()\n");
}

void OneDim_X::operator()(Function* f) const {
	DEBUG(errs() << "[+] CLASS OneDim_X, FUNC operator():\n");
	if (!f) {
		DEBUG(errs() << "\t* got nullptr\n");
		DEBUG(errs() << "[-] CLASS OneDim_X, FUNC operator()\n");
		return;
	}
	DEBUG(errs() << "\t* given function name " << f->getName() << '\n');
	// get the calls of get_global_id(1) //
	std::vector<CallInst*> worklist(getWorkList(0, f));
	for (auto* ci : worklist) {
		addOffsetAndReplace(ci);
	}
	DEBUG(errs() << "[-] CLASS OneDim_X, FUNC operator()\n");
}

#undef DEBUG_TYPE
