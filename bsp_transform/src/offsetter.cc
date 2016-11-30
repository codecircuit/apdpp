/******************************************************************
 *
 * FILENAME    - offsetter.cc
 *
 * DESCRIPTION - look at offsetter.h for a detailed description
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2015-12-14
 *
 ******************************************************************/ 


#include "llvm/IR/Function.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "offsetter.h"

#define DEBUG_TYPE "OFFSETTER"

Offsetter::Offsetter(std::shared_ptr<Offsetter::MappingT> orig2super, const std::string& patternStr) :
					 patternStr_(patternStr),
					 orig2super_(orig2super),
					 worklist_(getWorkList()),
					 pattern_(getPattern()) {}

std::vector<Function*> Offsetter::getWorkList() const {
	DEBUG(errs() << "[+] CLASS Offsetter, FUNC getWorkList():\n");
	std::vector<Function*> result;
	for (auto it = orig2super_->begin(); it != orig2super_->end(); ++it) {
		result.push_back(it->second); // getting all super version functions
		DEBUG(errs() << "\t* adding " << it->second->getName() << '\n');
	}
	DEBUG(errs() << "[-] CLASS Offsetter, FUNC getWorkList()\n");
	return result;
}

std::unique_ptr<Pattern> Offsetter::getPattern() const {
	if (patternStr_ == std::string("OneDim_Y")) {
		return std::unique_ptr<Pattern>(new OneDim_Y);
	}
	if (patternStr_ == std::string("OneDim_X")) {
		return std::unique_ptr<Pattern>(new OneDim_X);
	}
	// more patterns will be added here //
	else {
		return nullptr;
	}
}

void Offsetter::run() {
	DEBUG(errs() << "[+] CLASS Offsetter, FUNC run():\n");
	for (auto* f : worklist_) {
		DEBUG(errs() << "\t* got function " << f->getName() << " in worklist.\n");
		(*pattern_)(f);
	}
	DEBUG(errs() << "[-] CLASS Offsetter, FUNC run()\n");
}

#undef DEBUG_TYPE
