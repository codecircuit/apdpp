#ifdef MEKONG_TEST

#include <iomanip>
#include <iostream>
#include <tuple>
#include <set>
#include <vector>

#include "partition.h"
#include "partitioning.h"
#include "alias_handle.h"

using namespace std;
using namespace Mekong;

int main() {
	// The following examples explains how the 2D partitioning should be done.
	// An 'x' represents a thread block.
	// Test Case I (2x6 grid and 6 devices):
	//
	//     x   x | x   x | x   x
	//     ---------------------
	//     x   x | x   x | x   x
	//
	// Test Case II (8x8 grid and 16 devices):
	//
	//     x   x | x   x | x   x | x   x
	//           |       |       |
	//     x   x | x   x | x   x | x   x
	//     -----------------------------
	//     x   x | x   x | x   x | x   x
	//           |       |       |
	//     x   x | x   x | x   x | x   x
	//     -----------------------------
	//     x   x | x   x | x   x | x   x
	//           |       |       |
	//     x   x | x   x | x   x | x   x
	//     -----------------------------
	//     x   x | x   x | x   x | x   x
	//           |       |       |
	//     x   x | x   x | x   x | x   x
	//
	// Test Case III (2 x 16 grid and 16 devices):
	//
	//     x   x | x   x | x   x | x   x | x   x | x   x | x   x | x   x
	//     -------------------------------------------------------------
	//     x   x | x   x | x   x | x   x | x   x | x   x | x   x | x   x
	//
	// Test Case IV (6 x 15 and 15 devices):
	//
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//               |           |           |           |
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//     ---------------------------------------------------------
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//               |           |           |           |
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//     ---------------------------------------------------------
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//               |           |           |           |
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//
	// Test Case V (8 x 15 and 15 devices):
	//
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//               |           |           |           |
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//               |           |           |           |
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//     ---------------------------------------------------------
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//               |           |           |           |
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//               |           |           |           |
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//     ---------------------------------------------------------
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//               |           |           |           |
	//     x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//
	// Test Case VI (7 x 16 and 15 devices):
	//
	//     x   x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//                   |           |           |           |
	//     x   x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//                   |           |           |           |
	//     x   x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//     -------------------------------------------------------------
	//     x   x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//                   |           |           |           |
	//     x   x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//     -------------------------------------------------------------
	//     x   x   x   x | x   x   x | x   x   x | x   x   x | x   x   x
	//                   |           |           |           |
	//     x   x   x   x | x   x   x | x   x   x | x   x   x | x   x   x

	cout << "# Test of Partition Class" << endl;
	cout << endl;

	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	MEdevice dev;
	bool success;

	// Test Case I (2x6 grid and 6 devices):
	success = true;
	vector<MEdevice> vdev(6, dev);
	(*aliasH)[dev] = vdev;
	using T3 =  Partition::Array3;
	unsigned N = 100;
	unsigned tiling = 7;
	T3 grid = { 2, 6, 1 };
	T3 block = { tiling, tiling, 1 };
	shared_ptr<const Partitioning> parting(new Partitioning("xy"));
	auto partitions = Partition::createPartitions(grid, block, aliasH, parting);
	size_t sum = 0;
	set<T3> trueOff; // true offsets
	trueOff.insert({ 0, 0, 0 });
	trueOff.insert({ 14, 28, 0 });
	trueOff.insert({ 28, 56, 0});

	for (auto p : partitions) {
		success = success && (get<0>(p->getSize()) == tiling)
		                  && (get<1>(p->getSize()) == 2 * tiling)
		                  && (get<2>(p->getSize()) == 1);
		if (trueOff.count(p->getOffset()) > 0) {
			trueOff.erase(p->getOffset());
		}

	}
	success = success && trueOff.empty();
	cout << "  - Test Case I" << flush;
	if (success) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}
	return 0;
}

#endif
