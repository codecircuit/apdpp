#include "memory_copy.h"
#include "argument_access.h"

#include <map>
#include <tuple>
#include <ostream>
#include <utility>
#include <vector>

namespace Mekong {

using namespace std;

ArgAccess::ArgAccess(GpuToRangesMapping_t&& gpuToRanges) :
		gpuToRanges_(gpuToRanges) {}

unsigned short ArgAccess::size() const {
	return gpuToRanges_.size();
}

const vector<tuple<size_t, size_t>>&
ArgAccess::operator[](unsigned short gpu) const {
	return gpuToRanges_.at(gpu);
}

tuple<size_t, size_t>
ArgAccess::intersectIntervals(const tuple<size_t, size_t>& a,
                              const tuple<size_t, size_t>& b) {
	tuple<size_t, size_t> res;
	if (get<0>(a) <= get<0>(b)) {
		get<0>(res) = get<0>(b);
	}
	else {
		get<0>(res) = get<0>(a);
	}
	if (get<1>(a) >= get<1>(b)) {
		get<1>(res) = get<1>(b);
	}
	else {
		get<1>(res) = get<1>(a);
	}
	if (get<1>(res) <= get<0>(res)) {
		return make_tuple(0, 0);
	}
	return res;
}

const ArgAccess::GpuToRangesMapping_t& ArgAccess::getMap() const {
	return gpuToRanges_;
}

// number of gpus for both argAccesses must be equal
map<unsigned short, map<unsigned short, vector<tuple<size_t, size_t>>>>
ArgAccess::intersect(const ArgAccess& other) const {
	map<unsigned short, map<unsigned short, vector<tuple<size_t, size_t>>>> res;
	for (auto it = gpuToRanges_.begin(); it != gpuToRanges_.end(); ++it) {
		for (auto itt = other.gpuToRanges_.begin();
				  itt != other.gpuToRanges_.end(); ++itt) {
			for (auto thisRange : it->second) {
				for (auto otherRange : itt->second) {
					auto isect = intersectIntervals(thisRange, otherRange);
					if (get<0>(isect) != 0 || get<1>(isect) != 0) {
						res[it->first][itt->first].push_back(isect);
					}
				}
			}
		}
	}
	return res;
}

ostream& operator<<(ostream& out, const ArgAccess& arac) {
	out << "(Size: " << arac.size() << ", Accs: ";
	for (int i = 0; i < arac.size(); ++i) {
		out << "GPU" << i << "{";
		for (int j = arac[i].size() - 1; j > 0; --j) {
			out << "[" << get<0>(arac[i][j]) << ", " << get<1>(arac[i][j]) << "), ";
		}
		if (arac[i].size() > 0) {
			out << "[" << get<0>(arac[i][0]) << ", " << get<1>(arac[i][0]) << ")";
		}
		out << "}" << (i == arac.size() - 1 ? "" : " ");
	}
	out << ")";
	return out;
}

ostream& operator<<(ostream& out, const tuple<size_t, size_t>& range) {
	out << "(" << get<0>(range) << ", " << get<1>(range) << ")";
	return out;
}

}; // namespace end
