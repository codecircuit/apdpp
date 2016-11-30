/******************************************************************
 *
 * FILENAME    - promoter.h
 *
 * DESCRIPTION - This header provides the class which searches
 *               for calls of function named 'targetName' (given in the constructor)
 *               in the device code. For every function with such calls this
 *               class will create a copy of the function named
 *               like the original + '_super' suffix. Moreover
 *               the '_super' version will have 6 additional
 *               function arguments, globalOffsetX, globalOffsetY, globalOffsetZ,
 *               globalSizeX, globalSizeY, globalSizeZ.
 *
 * AUTHOR      - Christoph Klein
 *
 * LAST CHANGE - 2015-12-15
 *
 ******************************************************************/ 

#ifndef PROMOTER_H
#define PROMOTER_H

// forward declaration to reduce compile time //
namespace llvm {
	class MDNode;
	class Module;
	class Function;
}

#include "llvm/Transforms/Utils/ValueMapper.h" // ValueToValueMapTy //

#include <vector>
#include <map>
#include <string>
#include <memory> // for smart pointers //

#define DEBUG_TYPE "PROMOTER"

using namespace llvm;

class Promoter {
	public:
		// type for mapping old functions to their belonging '_super' version functions //
		typedef std::map<const Function*, Function*> MappingT;

		/*
		 *  PARAMETER   - targetName: is the name of the global index function,
		 *                which takes one i32 argument and has i64 return type.
		 *                e.g. 'get_global_id'
		 */
		Promoter(Module& M, const char* targetName);
		Promoter(Module& M, const std::vector<std::string>& targetNames);

		// do transformations //
		void run();

		std::vector<const Function*> getOriginalFunctions() const;

		/*
		 *  DESCRIPTION - This function creates the linkage between the arguments
		 *                of the original function and the belonging '_super' version
		 *                function arguments. The result is needed for llvm function
		 *                'CloneFunctionInto'.
		 */
		static void createVMap(ValueToValueMapTy& res, Function* newF, const Function* oldF);

		Function* createSuper(const Function* startF);
		
		/*
		 *  DESCRIPTION - get the mapping between the original and their
		 *                belonging '_super' version.
		 */
		std::shared_ptr<MappingT> getMapping();

	private:
		/*
		 *  DESCRIPTION - When you create a function in llvm and you want to mark
		 *                it as a kernel you have to create the appropriate
		 *                metadata object and insert it into the 'NamedMDNode'
		 *                (named metadata node) with name 'nvvm.annotations'.
		 *  PARAMETER   - The function you want to mark as kernel
		 *  RETURN      - Metadata Node
		 */
		static MDNode* mdNodeKernel(Function* f);

		/*
		 *  DESCRIPTION - checks if a function is marked by metadata as an opencl kernel
		 */
		static bool isKernel(const Function* f);

		Module* const M_;
		std::vector<std::string> targetNames_; // given by constructor //
		std::shared_ptr<MappingT> orig2super_; // original funct to associated '_super' function //
		const std::vector<const Function*> vOrig_; // original func with calls of func named 'targetName_' //
		std::vector<Function*> vSuper_; // '_super' versions of the original functions //
};

#undef DEBUG_TYPE

#endif 
