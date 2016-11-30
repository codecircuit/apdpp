/******************************************************************
 *
 * FILENAME    - propagator.h
 *
 * DESCRIPTION - '_super' versions of functions should call only
 *               other '_super' versions of functions, if there
 *               exists a '_super' version. This class transforms
 *               appropriate function calls in that way. This means
 *               replacing the original function call with the
 *               '_super' version and handing over the six
 *               additional arguments.
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2015-12-10
 *
 ******************************************************************/ 

#ifndef PROPAGATOR_H
#define PROPAGATOR_H

// forward declaration to reduce compile time //
namespace llvm {
	class Function;
}

#include <memory> // smart pointers //
#include <vector>

#include "promoter.h"

#define DEBUG_TYPE "PROPAGATOR"

using namespace llvm;


class Propagator {
	public:
		// type for mapping original functions to their '_super' versions //
		typedef Promoter::MappingT MappingT;
		
		/*
		 *  PARAMETER   - mapping from original functions to their '_super'
		 *                versions. This is created by the promoter.
		 */
		Propagator(std::shared_ptr<MappingT> orig2super);
		
		/*
		 *  DESCRIPTION - Executes the transformations described at the top of this file.
		 */
		void run();

	private:
		/*
		 *  DESCRIPTION - This function greps all call instructions, where
		 *                the called function has a '_super' version.
		 */
		std::vector<CallInst*> getWorkList() const;

		std::shared_ptr<MappingT> orig2super_;
		// this are the call instructions we have to modify //
		std::vector<CallInst*> worklist_;

};

#undef DEBUG_TYPE

#endif
