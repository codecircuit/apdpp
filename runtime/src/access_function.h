#ifndef MEKONG_ACCESS_FUNCTION_H
#define MEKONG_ACCESS_FUNCTION_H

#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <array>
#include <cstdint>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/point.h>

namespace Mekong {

class KernelArg;

using namespace std;

/*! \brief Contains information about which thread reads which elements.

    Thus the access function's purpose is to know which elements of *one* kernel
    array are read or written by which threads. This information must be
    calculated from the static kernel analysis' information.
*/
class AccFunc {
	public:
		typedef array<unsigned, 3> Array3;
		AccFunc(const vector<shared_ptr<const string>>& islReadParams,
		        const string& islRead,
		        const vector<shared_ptr<const string>>& islWriteParams,
		        const string& islWrite, int argNr = -1);

		bool isAffine() const;

		const string& getIslRead() const;
		const string& getIslWrite() const;
		const vector<shared_ptr<const string>>& getIslReadParams() const;
		const vector<shared_ptr<const string>>& getIslWriteParams() const;
		int getArgNr() const;

		intmax_t getReadParam(unsigned short paramId,
		                    const vector<shared_ptr<const KernelArg>>* args = nullptr,
		                    const Array3* gridSize = nullptr, const Array3* blockSize = nullptr) const;

		intmax_t getWriteParam(unsigned short paramId,
		                    const vector<shared_ptr<const KernelArg>>* args = nullptr,
		                    const Array3* gridSize = nullptr, const Array3* blockSize = nullptr) const;

		//! returns the isl read acces map with fixed parameters
		__isl_give isl_union_map* getReadIslMap(isl_ctx* ctx,
		                    const vector<shared_ptr<const KernelArg>>* args = nullptr,
		                    const Array3* gridSize = nullptr, const Array3* blockSize = nullptr) const;

		//! returns the isl write acces map with fixed parameters
		__isl_give isl_union_map* getWriteIslMap(isl_ctx* ctx,
		                    const vector<shared_ptr<const KernelArg>>* args = nullptr,
		                    const Array3* gridSize = nullptr, const Array3* blockSize = nullptr) const;

		// these get[Read|Write]Acc function should not be used in the actual
		// wrapping infrastructure due to performance reasons
		//! Returns the indices of accessed pointer elements 
		vector<size_t> getReadAcc(const Array3& threadId,
		                    const vector<shared_ptr<const KernelArg>>* args = nullptr,
		                    const Array3* gridSize = nullptr, const Array3* blockSize = nullptr) const;

		vector<size_t> getWriteAcc(const Array3& threadId,
		                    const vector<shared_ptr<const KernelArg>>* args = nullptr,
		                    const Array3* gridSize = nullptr, const Array3* blockSize = nullptr) const;
	private:
		intmax_t getParam(unsigned short paramId,
		                    const vector<shared_ptr<const KernelArg>>* args,
		                    const Array3* gridSize, const Array3* blockSize,
		                    bool is_read) const; // if is_read = true parse read param

		__isl_give isl_union_map* getIslMap(isl_ctx* ctx,
		                    const vector<shared_ptr<const KernelArg>>* args,
		                    const Array3* gridSize, const Array3* blockSize,
		                    bool is_read) const; // if is_read = true get read islMap

		vector<size_t> getAcc(const Array3& threadId,
		                    const vector<shared_ptr<const KernelArg>>* args,
		                    const Array3* gridSize, const Array3* blockSize,
		                    bool is_read) const; // if is_read = true get read access

		//! hacky functions to extract points from a isl_map
		static isl_stat fixAndGetRange(__isl_take isl_map* map, void* threadId_and_ResVector);
		static isl_stat addPoint(__isl_take isl_point*, void* points);

		static size_t flatPoint(vector<size_t>& point, vector<size_t>& dimSizes);

		//! searches for occurrances of \param substr in all \param strings 
		static bool foundSubStr(const string& substr, const vector<shared_ptr<const string>>& strings);
		
		// TODO this const qualifiers are unnecessary, as we create
		// only const AccFunc objects anyways, thus they should be
		// removed
		const vector<shared_ptr<const string>> islReadParams_;
		const string islRead_;
		const vector<shared_ptr<const string>> islWriteParams_;
		const string islWrite_;
		const int argNr_; ///< arg number this access function belongs to
};

}; // namespace end

#endif
