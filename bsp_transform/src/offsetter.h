/******************************************************************
 *
 * FILENAME    - offsetter.h
 *
 * DESCRIPTION - This header provides the class which takes the
 *               '_super' function versions created by the promoter
 *               and adds the offset to the calls of get_global_id().
 *               Furthermore the pattern determines which dimension
 *               is changed.
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2015-12-14
 *
 ******************************************************************/ 

#ifndef OFFSETTER_H
#define OFFSETTER_H

// forward declaration to reduce compile time //
namespace llvm {
	class Function;
}

#include <memory> // for smart pointers //
#include <vector>
#include <string>

#include "promoter.h"
#include "pattern.h"

using namespace std;

class Offsetter {
	public:
		// type for mapping the original functions to their '_super' versions //
		typedef Promoter::MappingT MappingT;

		/*
		 *  PARAMETER   - orig2super: mapping of original functions to their
		 *                '_super' versions -- created by an object of class Promoter
		 *              - patternStr: keyword for the underlaying alleycat access pattern.
		 *                This should be determined in the kernel analysis.
		 */
		Offsetter(std::shared_ptr<MappingT> orig2super, const std::string& patternStr);
		
		/*
		 *  DESCRIPTION - this function transforms all the appropriate calls of
		 *                get_global_id with the pattern given in the
		 *                constructor. You can see all available patterns
		 *                in pattern.h.
		 */
		void run();

	private:
		// '_super' versions of functions, determined by the given mapping 'orig2super' //
		std::vector<Function*> getWorkList() const;

		/*
		 *  DESCRIPTION - This function defines the mapping between the 'patternStr'
		 *                and the belonging pattern class. Thus if you add a pattern
		 *                in pattern.h/cc you have to add the mapping in this function.
		 */
		std::unique_ptr<Pattern> getPattern() const;
		
		const std::string patternStr_;
		const std::shared_ptr<MappingT> orig2super_;
		const std::vector<Function*> worklist_;
		const std::unique_ptr<Pattern> pattern_;
};

#endif
