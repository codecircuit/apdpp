#ifndef MEKONG_ARG_ACCESS_H
#define MEKONG_ARG_ACCESS_H

#include "memory_copy.h"

#include <map>
#include <tuple>
#include <ostream>
#include <vector>

namespace Mekong {

using namespace std;

/*! \brief describes the GPUs accesses on one kernel array

    This object is created if the runtime demands it (e.g. to resolve
    dependencies). One argument access object belongs always to a kernel
    argument and describes either the *read* or the *write* access of the
    GPUs on that particular kernel array. Thus the purpose of this object
    is to say wich GPU accesses certain kernel array elements.
*/
class ArgAccess {
	public:
		//! Every GPU accesses a certain amount of linear aligned memory
		//! locations. This type creates the mapping between a GPU's id and the
		//! accessed ranges.
		typedef map<unsigned short, vector<tuple<size_t, size_t>>>
		GpuToRangesMapping_t;

		// for performance reasons only use this constructor
		ArgAccess(GpuToRangesMapping_t&& gpuToRanges);

		//! Get accessed ranges by a GPU
		const vector<tuple<size_t, size_t>>& operator[](unsigned short gpu) const;

		//! Returns the number of GPUs
		unsigned short size() const;

		const GpuToRangesMapping_t& getMap() const;
		
		//! result[0][1] are the intervals of the intersection between gpu 0 and 1
		map<unsigned short, map<unsigned short, vector<tuple<size_t, size_t>>>>
		intersect(const ArgAccess& other) const;

		//! help function which gives the intersection of two ranges
		static tuple<size_t, size_t>
		intersectIntervals(const tuple<size_t, size_t>& a,
		                   const tuple<size_t, size_t>& b);
	private:
		GpuToRangesMapping_t gpuToRanges_;
};

ostream& operator<<(ostream& out, const ArgAccess& arac);

ostream& operator<<(ostream& out, const tuple<size_t, size_t>& range);

}; // namespace end

#endif
