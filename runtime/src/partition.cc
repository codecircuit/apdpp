#include "partitioning.h"
#include "alias_handle.h"
#include "partition.h"

#include <tuple>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <algorithm> // std::min, std::max
#include <memory>

#include "isl/val.h"
#include "isl/space.h"

namespace Mekong {

using namespace std;

/*! \brief Creates all partitions with a certain partitioning scheme.

    \param aliasH to determine the number of available gpus and to get all the
           kernel function pointers.

    \todo Ensure that the returned vector is sorted by GPU id.
          This can improve performance, as we do not always have to push
          different contexts if we cycle through the partitions. This can have
          an effect if we have multiple partitions for a GPU.
*/
vector<shared_ptr<const Partition>>
Partition::createPartitions(const Array3& orgGrid, const Array3& orgBlock,
                            shared_ptr<AliasHandle> aliasH,
                            shared_ptr<const Partitioning> parting) {
	vector<shared_ptr<const Partition>> res;
	unsigned short numDev = aliasH->getNumDev();

	// Get the id of the dimensions which should be splitted
	int splitDims[3];
	int tmp = 0;
	if (parting->isSplitAt('x')) {
		splitDims[tmp++] = 0;
	}
	if (parting->isSplitAt('y')) {
		splitDims[tmp++] = 1;
	}
	if (parting->isSplitAt('z')) {
		splitDims[tmp++] = 2;
	}

/*******************
 * 2D PARTITIONING *
 *******************/
	if (parting->getSplitStr().size() == 2) {
		// sizes of the grid dimension we want to split
		unsigned smallGridSize;
		unsigned largeGridSize;
		int smallDim; // 0 = x, 1 = y, 2 = z
		int largeDim;

		if (orgGrid[splitDims[0]] > orgGrid[splitDims[1]]) {
			smallGridSize = orgGrid[splitDims[1]];
			largeGridSize = orgGrid[splitDims[0]];
			smallDim = splitDims[1];
			largeDim = splitDims[0];
		}
		else {
			smallGridSize = orgGrid[splitDims[0]];
			largeGridSize = orgGrid[splitDims[1]];
			smallDim = splitDims[0];
			largeDim = splitDims[1];
		}

		// The following examples explains how the 2D partitioning should be done.
		// An 'x' represents a thread block.
		// Example I (2x6 grid and 6 devices):
		//
		//     x   x | x   x | x   x
		//     ---------------------
		//     x   x | x   x | x   x
		//
		// Example II (8x8 grid and 16 devices):
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
		// Example III (2 x 16 grid and 16 devices):
		//
		//     x   x | x   x | x   x | x   x | x   x | x   x | x   x | x   x
		//     -------------------------------------------------------------
		//     x   x | x   x | x   x | x   x | x   x | x   x | x   x | x   x

		if (smallGridSize < 2) {
			throw invalid_argument(
				"I should split along dimensions " + parting->getSplitStr() +
				", but the smallest of both dimensions has only a size of " +
				to_string(smallGridSize) +
				", wich I can not split in a two dimensional way."
			);
		}

		for (int i = numDev/2; i > 1; --i) {
			if (numDev % i == 0) {
				int ii = numDev / i;
				// now ii * i = numDev
				int small_fac = min(ii, i);
				int large_fac = max(ii, i);
				// If the small factor is too large, search for a smaller one
				if (small_fac > smallGridSize) { continue; }
				else {
					// Calculate how many work is on split dimension small;
					// analog to 1D splitting for small_i gpus

					vector<Array3> work(numDev, orgGrid);

					for (unsigned short gpu = 0; gpu < numDev; ++gpu) {
						work[gpu][smallDim] = orgGrid[smallDim] / small_fac;
					}
					// distribute the rest along the gpus
					for (size_t rest = 0; rest < (orgGrid[smallDim] % small_fac); ++rest) {
						++work[rest % numDev][smallDim];
					}

					// Calculate how many work is on split dimension big;
					// analog to 1D splitting for big_i gpus
					for (unsigned short gpu = 0; gpu < numDev; ++gpu) {
						work[gpu][largeDim] = orgGrid[largeDim] / large_fac;
					}
					// distribute the rest along the gpus
					for (size_t rest = 0; rest < (orgGrid[largeDim] % large_fac); ++rest) {
						++work[rest % numDev][largeDim];
					}

					// Calculate the offsets
					vector<Array3> offset(numDev, {0, 0, 0});
					for (int i = 0; i < 2; ++i) {
						int currSplitDim = splitDims[i];
						unsigned count = work[0][currSplitDim];
						for (unsigned short gpu = 1; gpu < numDev; ++gpu) {
							offset[gpu][currSplitDim] = count * orgBlock[currSplitDim];
							count += work[gpu][currSplitDim];
						}
					}

					for (unsigned short gpu = 0; gpu < numDev; ++gpu) {
						shared_ptr<const Partition> sh(new Partition(work[gpu], orgBlock, offset[gpu], gpu));
						res.push_back(sh);
					}
					return res;
				}
			} // if (numDev % i == 0)
		} // loop over i
		throw invalid_argument(
			"I should split along dimensions " + parting->getSplitStr() + ". "
			"My split algorithm failed on the grid with size " +
			"(" + to_string(orgGrid[0]) + ", " + to_string(orgGrid[1]) +
			", " + to_string(orgGrid[2]) + ")"
		);
	}
/***************************
 * nD PARTITIONING (n > 2) *
 * not supported yet       *
 ***************************/
	if (parting->getSplitStr().size() > 2) {
		throw invalid_argument("Splitting along more than two dimensions is not "
		                       "supported yet!");
	}

/*******************
 * 1D PARTITIONING *
 *******************/
	// CHECK REASONABILITY OF 1D PARTITIONING SCHEME
	int currSplitDim = splitDims[0];
	if (orgGrid[currSplitDim] < numDev) {
		throw invalid_argument("I can not split along " + parting->getSplitStr() +
		                       " axis if the grid size of that dimension " +
		                       "is smaller than number of gpus.");
	}

	// CALCULATE THE WORK FOR EVERY GPU
	// First we calculate how many rows each gpu has work along the splitted dimension
	vector<Array3> work(numDev, orgGrid);
	for (unsigned short gpu = 0; gpu < numDev; ++gpu) {
		work[gpu][currSplitDim] = orgGrid[currSplitDim] / numDev;
	}
	// distribute the rest along the gpus
	for (size_t rest = 0; rest < orgGrid[currSplitDim] % numDev; ++rest) {
		++work[rest % numDev][currSplitDim];
	}

	// CALCULATE THE GPU OFFSET ON THE GRID
	// Calculate the offsets; GPU0 has no offset
	vector<Array3> offset(numDev, {0, 0, 0});
	unsigned count = work[0][currSplitDim];
	for (unsigned short gpu = 1; gpu < numDev; ++gpu) {
		offset[gpu][currSplitDim] = count * orgBlock[currSplitDim];
		count += work[gpu][currSplitDim];
	}

	for (unsigned short gpu = 0; gpu < numDev; ++gpu) {
		shared_ptr<const Partition> sh(new Partition(work[gpu], orgBlock, offset[gpu], gpu));
		res.push_back(sh);
	}
	return res;
}

Partition::Partition(const Array3& grid,
                     const Array3& block,
                     const Array3& offset, int device) :
				grid_(grid),
				block_(block),
				offset_(offset),
				device_(device) {}

//! Returns the grid size (numBlocks.x, numBlocks.y, numBlocks.z)
const Partition::Array3& Partition::getGrid() const {
	return grid_;
}

//! Returns the block size (numThreads.x, numThreads.y, numThreads.z)
const Partition::Array3& Partition::getBlock() const {
	return block_;
}

/*! \brief Returns the partition's offset in number of threads.

     Thus if you have a partition with offset (6,8,0)
     and a block size of (3,2,1), the partition will work
     on the 'x' marked threads.

     -----> y
     |    0  1  2  3  4  5  6  7  8  9  10 11 12 13
    x| 0  .  .  .  .  .  .  .  .  .  .  .  .  .  .
     ∨ 1  .  .  .  .  .  .  .  .  .  .  .  .  .  .
       2  .  .  .  .  .  .  .  .  .  .  .  .  .  .
       3  .  .  .  .  .  .  .  .  .  .  .  .  .  .
       4  .  .  .  .  .  .  .  .  .  .  .  .  .  .
       5  .  .  .  .  .  .  .  .  .  .  .  .  .  .
       6  .  .  .  .  .  .  .  .  .  .  .  .  .  .
       7  .  .  .  .  .  .  .  .  .  .  .  .  .  .
       8  .  .  .  .  .  .  x  x  .  .  .  .  .  .
       9  .  .  .  .  .  .  x  x  .  .  .  .  .  .
       10 .  .  .  .  .  .  x  x  .  .  .  .  .  .
       11 .  .  .  .  .  .  .  .  .  .  .  .  .  .
       12 .  .  .  .  .  .  .  .  .  .  .  .  .  .
       13 .  .  .  .  .  .  .  .  .  .  .  .  .  .

     \todo The offset should be given in number of blocks, because we
           will not split block borders anyway.
*/
const Partition::Array3& Partition::getOffset() const {
	return offset_;
}

//! Returns the size of the partition in threads.
Partition::Array3 Partition::getSize() const {

	auto mult3 = [] (const Array3& a, const Array3& b) {
		return Array3({ a[0] * b[0],
		                            a[1] * b[1],
		                            a[2] * b[2]});
	};
	return mult3(getBlock(), getGrid());
}

//! Returns the device which executes this partition.
int Partition::getDevice() const {
	return device_;
}

ostream& operator<<(ostream& out, const Partition& p) {
	auto print3 = [&] (const Partition::Array3& a) {
		out << "(" << a[0] << ", " << a[1] << ", " << a[2] << ")";
	};

	out << "{ Grid";
	print3(p.getGrid());
	out << " Block";
	print3(p.getBlock());
	out << " Offset";
	print3(p.getOffset());
	out << " Dev(" << p.getDevice() << ") }";
	return out;
}

}; // namespace end
