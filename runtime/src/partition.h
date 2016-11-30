#ifndef MEKONG_PARTITION_H
#define MEKONG_PARTITION_H

#include <tuple>
#include <ostream>
#include <vector>
#include <memory>
#include <array>

namespace Mekong {

class Partitioning;
class AliasHandle;

using namespace std;

class Partition {
	public:
		typedef array<unsigned, 3> Array3;
		static vector<shared_ptr<const Partition>>
		createPartitions(const Array3& orgGrid,
		                 const Array3& orgBlock,
		                 shared_ptr<AliasHandle> aliasH,
		                 shared_ptr<const Partitioning> parting);

		Partition(const Array3& grid,
		          const Array3& block,
		          const Array3& offset, int device);
		const Array3& getGrid() const;
		const Array3& getBlock() const;
		const Array3& getOffset() const;
		Array3 getSize() const;
		int getDevice() const;

	private:
		Array3 grid_;   ///< number of blocks
		Array3 block_;  ///< number of threads per block
		Array3 offset_; ///< offset in threads on the total grid
		int device_;
};

ostream& operator<<(ostream& out, const Partition& p);

}; // namespace end

#endif
