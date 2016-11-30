/******************************************************************
 *
 * FILENAME    - promoter.cc
 *
 * DESCRIPTION - Source code. For a more detailed description look
 *               in promoter.h
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2015-12-08
 *
 ******************************************************************/ 

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils/Cloning.h"

#include "promoter.h"

#include <utility> // std::make_pair //
#include <vector>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "PROMOTER"

Promoter::Promoter(Module& M, const char* targetName)
		: M_(&M), targetNames_(1, targetName),
		vOrig_(getOriginalFunctions()),
		orig2super_(new MappingT) {}

Promoter::Promoter(Module& M, const std::vector<std::string>& targetNames)
		: M_(&M), targetNames_(targetNames),
		vOrig_(getOriginalFunctions()),
		orig2super_(new MappingT) {}

std::shared_ptr<Promoter::MappingT> Promoter::getMapping() { return orig2super_; }

std::vector<const Function*> Promoter::getOriginalFunctions() const {
	DEBUG(errs() << "[+] CLASS Promoter, FUNC getOriginalFunctions():\n");
	DEBUG(errs() << "\t* target names: ");
	for (const auto& target : targetNames_) {
		DEBUG(errs() << target << ", ");
	}
	DEBUG(errs() << "\n");

	// function which we have to promote //
	std::vector<const Function*> vF;
	
	std::vector<const Function*> vTargets;

	for (const std::string& target : targetNames_) {
		const Function* ptr = M_->getFunction(target.c_str());
		if (ptr) {
			vTargets.push_back(ptr);
		}
		else {
			DEBUG(errs() << "*** WARNING: could not declaration/definition of target function " << target << '\n');
		}
	}

	if (vTargets.empty()) {
		DEBUG(errs() << "*** WARNING: FUNC getOriginalFunctions:\n"
		             << "\tcould not find any definition/declaration of targets!\n");
		DEBUG(errs() << "[-] CLASS Promoter, FUNC getOriginalFunctions()\n");
		return vF;
	}
	else {
		DEBUG(errs() << "\t* found " << vTargets.size() << "/" << targetNames_.size() << " targets\n");
	}
	
	bool added;
	for (const auto& f : M_->functions()) {
		added = false;
		DEBUG(errs() << "\t* searching for calls in " << f.getName() << '\n');
		const_inst_iterator it;
		for (it = inst_begin(f); it != inst_end(f); ++it) {
			if (const auto* ci = dyn_cast<CallInst>(&(*it))) {
				for (const Function* target : vTargets) {
					if (ci->getCalledFunction() == target) {
						DEBUG(errs() << "\t\t* found target call inst %" << ci->getName() << '\n');
						vF.push_back(&f);
						added = true;
						break;
					}
				}
			}
			if (added) {
				break;
			}
		}
	}

	DEBUG(errs() << "[-] CLASS Promoter, FUNC getOriginalFunctions()\n");
	return vF;
}

void Promoter::createVMap(ValueToValueMapTy& res, Function* newF, const Function* oldF) {
	DEBUG(errs() << "[+] CLASS Promoter, FUNC createVMap:\n");
	if (newF->arg_size() < oldF->arg_size()) {
		errs() << "***ERROR: FUNC createVMap\n"
		       << "\tnew function has less arguments than old function!\n";
		DEBUG(errs() << "[-] CLASS Promoter, FUNC createVMap\n");
		return;
	}
	auto ait = newF->arg_begin();
	for (auto& arg : oldF->args()) {
		// sadly there is no constructor WeakVH(const Value*P), thus we can not program const consistent here.
		// Function* newF should be a 'const' argument.
		res.insert(std::make_pair(&arg, WeakVH(&(*ait))));

		// give the arguments the same name //
		ait->setName(arg.getName());
		++ait;
	}

	// just for better reading: give names to additional arguments //
	if (newF->arg_size() - oldF->arg_size() == 6) {
		DEBUG(errs() << "\t* giving names to last new function's arguments: ");
		DEBUG(errs() << "offset[X|Y|Z] and GlobalSize[X|Y|Z]\n");
		ait->setName("offsetX");
		++ait;
		ait->setName("offsetY");
		++ait;
		ait->setName("offsetZ");
		++ait;
		ait->setName("GlobalSizeX");
		++ait;
		ait->setName("GlobalSizeY");
		++ait;
		ait->setName("GlobalSizeZ");
	}

	DEBUG(errs() << "[-] CLASS Promoter, FUNC createVMap\n");
}

//! Creates the super version of a given function.

//! The returned super function lives in the same module as the
//! given \param startF function. The super function has
//! 6 additional arguments appended at the end of the
//! argument list: offsetX, offsetY, offsetZ, globalSizeX,
//! globalSizeY, globalSizeZ.
Function* Promoter::createSuper(const Function* startF) {
	DEBUG(errs() << "[+] CLASS Promoter, FUNC createSuper():\n");
	if (!startF) {
		DEBUG(errs() << "\t* got nullptr.\n");
		DEBUG(errs() << "[-] CLASS Promoter, FUNC createSuper\n");
		return nullptr;
	}

	// get old function argument types //
	std::vector<Type*> vT = startF->getFunctionType()->params().vec();

	// add i64 types for offset and global size //
	for (unsigned short count = 0; count < 6; ++count) {
		vT.push_back(IntegerType::get(startF->getContext(), 64));
	}
	FunctionType* fT = FunctionType::get(startF->getReturnType(), vT, false);

	// create new function with '_super' suffix //
	Function* ret = Function::Create(fT, startF->getLinkage(), startF->getName() + "_super", M_);

	DEBUG(errs() << "\t* created '_super' function declaration\n\t" << *ret << '\n');

	// linking between new and old function's arguments //
	ValueToValueMapTy vMap;
	createVMap(vMap, ret, startF);

	// clone old content to new function //
	SmallVector<ReturnInst*, 10> dummy;
	CloneFunctionInto(ret, startF, vMap, true, dummy);

	// adding metadata to mark the new function as a kernel function
	if (isKernel(startF)) {
		auto* mdnode = mdNodeKernel(ret);
		// isKernel can only be true if there exists an metadata object
		// named 'nvvm.annotations', thus we do not have to check for nullptr
		auto* nvvmAnno = M_->getNamedMetadata("nvvm.annotations");
		nvvmAnno->addOperand(mdnode);
		DEBUG(errs() << "\t* marking new function '" << ret->getName() << "' as kernel\n");
	}

	DEBUG(errs() << "[-] CLASS Promoter, FUNC createSuper\n");
	return ret;
}

MDNode* Promoter::mdNodeKernel(Function* f) {
	DEBUG(errs() << "[+] CLASS Promoter, FUNC mdNodeKernel():\n");
	if (!f) {
		DEBUG(errs() << "\t* got nullptr\n");
		DEBUG(errs() << "[-] CLASS Promoter, FUNC mdNodeKernel()\n");
		return nullptr;
	}
	DEBUG(errs() << "\t* got function " << f->getName() << '\n');
	auto& ctx = f->getContext();
	auto* constMDFunc = ConstantAsMetadata::get(f);
	auto* constMDNum = ConstantAsMetadata::get(ConstantInt::get(ctx, APInt(32, 1)));
	auto* mdstr = MDString::get(ctx, "kernel");
	std::vector<Metadata*> v = {constMDFunc, mdstr, constMDNum};
	auto* result = MDTuple::get(ctx, v);
	DEBUG(errs() << "\t* returning metadata node\n" << *result);
	DEBUG(errs() << "[-] CLASS Promoter, FUNC mdNodeKernel()\n");
	return result;
}

bool Promoter::isKernel(const Function* f) {
	DEBUG(errs() << "[+] CLASS Promoter, FUNC isKernel():\n");
	if (!f->isUsedByMetadata()) {
		DEBUG(errs() << "\t* is not used by metadata\n");
		DEBUG(errs() << "[-] CLASS Promoter, FUNC isKernel()\n");
		return false;
	}
	if (!f) {
		DEBUG(errs() << "\t* got nullptr\n");
		DEBUG(errs() << "[-] CLASS Promoter, FUNC isKernel()\n");
		return false;
	}
	DEBUG(errs() << "\t* got function " << f->getName() << '\n');
	if (const auto* mdn = f->getParent()->getNamedMetadata("nvvm.annotations")) {
		DEBUG(errs() << "\t* nvvm.annotations: " << mdn << '\n');
		for (const auto& operand : mdn->operands()) {
			if (operand->getNumOperands() != 3) {continue;}
			if (const auto* cam = dyn_cast<ConstantAsMetadata>(operand->getOperand(0))) {
				if (cam->getValue() == f) {
					if (const auto* mdstr = dyn_cast<MDString>(operand->getOperand(1))) {
						if (mdstr->getString().equals("kernel")) {
							if (const auto* camInt = dyn_cast<ConstantAsMetadata>(operand->getOperand(2))) {
								if (const auto* cint = dyn_cast<ConstantInt>(camInt->getValue())) {
									if (cint->getValue() == APInt(32, 1)) {
										DEBUG(errs() << "\t* return true\n");
										DEBUG(errs() << "[-] CLASS Promoter, FUNC isKernel()\n");
										return true;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	DEBUG(errs() << "\t* return false\n");
	DEBUG(errs() << "[-] CLASS Promoter, FUNC isKernel()\n");
	return false;
}

void Promoter::run() {
	DEBUG(errs() << "[+] CLASS Promoter, FUNC run():\n");

	if (vOrig_.empty()) {
		DEBUG(errs() << "*WARNING: could not find any calls of given target functions\n");
		DEBUG(errs() << "[-] CLASS Promoter, FUNC run()\n");
		return;
	}

	for (const auto* f : vOrig_) {
		DEBUG(errs() << "\t* promoting function " << f->getName() << '\n');

		auto* f_super = createSuper(f);
		vSuper_.push_back(f_super);

		// creating the mapping //
		orig2super_->operator[](f) = f_super;
	}

	DEBUG(errs() << "[-] CLASS Promoter, FUNC run()\n");
}

#undef DEBUG_TYPE
