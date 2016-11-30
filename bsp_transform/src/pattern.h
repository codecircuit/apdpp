/******************************************************************
 *
 * FILENAME    - pattern.h
 *
 * DESCRIPTION - This header provides the classes which execute
 *               the thread id transformation for certain super
 *               patterns. Base Functor class is 'Pattern'.
 *               It is only allowed to call the functors with
 *               the '_super' function versions.
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2015-12-15
 *
 ******************************************************************/ 

#ifndef PATTERN_H
#define PATTERN_H

// forward declaration to reduce compile time //
namespace llvm {
	class Function;
	class Instruction;
	class CallInst;
}

#include <vector>
#include <string>

using namespace llvm;

/*
 *  DESCRIPTION - replaces all occurances of 'o' (old)
 *                with 'n' (new). 'n' must already exist
 *                and be defined before 'o'. 'o' will
 *                be erased from its function.
 */
// THIS IS JUST A HACKY POSITION FOR THIS FUNCTION
// TODO: ADD A SOURCE FILE FOR FREE LLVM FUNCTIONS
void replaceAndErase(Instruction* o, Value* n);

// virtual base class for alleycat patterns //
class Pattern {
	public:
		virtual void operator()(Function* f) const = 0;
	
	protected:
		/*
		 *  DESCRIPTION - This function gets you the call instructions,
		 *                which call get_global_id('axis') and live inside
		 *                Function 'f'.
		 */
		static std::vector<CallInst*> getWorkList(short int axis, Function* f);

		/*
		 *  DESCRIPTION - This function inserts a new basic block at the beginning
		 *                of the function, where 'gid' lives in. In that basic
		 *                block you will find:
		 *                %b = call get_global_id()
		 *                %c = %offset + %b 
		 *                where the call of 'get_global_id' will be done with
		 *                the same argument of 'gid' (thus the same axis).
		 *                %offset is given as a function argument.
		 *
		 *  PARAMETER   - must be a call of 'get_global_id'
		 */
		static void addOffsetAndReplace(CallInst* gid);
	private:
};

/*
 *  DESCRIPTION - This pattern only adds an offset to the y-dimension.
 *                Thus only calls of 'get_global_id(0)' are affected.
 */
class OneDim_Y : public Pattern {
	public:
		void operator()(Function* f) const;
	private:
};

class OneDim_X : public Pattern {
	public:
		void operator()(Function* f) const;
	private:
};

// More patterns can be added here... //

#endif
