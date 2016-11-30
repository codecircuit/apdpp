#ifndef MEKONG_KERNEL_LAUNCH_H
#define MEKONG_KERNEL_LAUNCH_H

#include "alias_handle.h"
#include "argument_access.h"
#include "memory_copy.h"
#include "argument.h"
#include "kernel_info.h"
#include "mekong-cuda.h"
#include "partitioning.h"
#include "partition.h"

#include <memory>
#include <map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <chrono>
#include <utility> // std::pair
#include <tuple>
#include <ostream>
#include <string>

#include <isl/set.h>
#include <isl/union_set.h>
#include <isl/point.h>

namespace Mekong {

using namespace std;

using Clock = chrono::high_resolution_clock;
using Duration = chrono::duration<double>;

class KernelLaunch {
	public:
		typedef Partition::Array3 Array3;
		// save all instances of this class as we want to
		// calculate the arg accesses only once
		struct equal_to;
		struct hash;
		static unordered_set<shared_ptr<KernelLaunch>, hash, equal_to> all;

		static pair<shared_ptr<KernelLaunch>, bool>
		getOrInsert(MEfunction func, const Array3& grid, const Array3& block,
		            size_t shMem,
		            void** rawArgs, shared_ptr<const bsp_KernelInfo> info,
		            shared_ptr<AliasHandle> aliasH);

		KernelLaunch(MEfunction func, const Array3& grid, const Array3& block,
		             size_t shMem,
		             void** rawArgs, shared_ptr<const bsp_KernelInfo> info,
		             shared_ptr<AliasHandle> aliasH);

		// IS- FUNCTIONS
		bool isArg(MEdeviceptr ptr) const;
		bool isArg(shared_ptr<const KernelArg> arg) const;
		bool hasEqualArgAccess(const KernelLaunch& other) const;
		
		// GET- FUNCTIONS
		shared_ptr<const KernelArg>                getArg(MEdeviceptr ptr) const;
		int                                        getArgId(MEdeviceptr ptr) const;
		int                                        getArgId(shared_ptr<const KernelArg> arg) const;
		vector<MEdeviceptr>                        getWrites() const;
		vector<MEdeviceptr>                        getReads() const;
		vector<MEdeviceptr>                        getPtrs() const;
		shared_ptr<const KernelArg>                getArgFromId(unsigned short nr) const;

		const vector<shared_ptr<const Partition>>& getPartitions() const;
		shared_ptr<const Partitioning>             getPartitioning() const;
		unsigned short                             getGPU(unsigned idx, unsigned idy, unsigned idz) const;
		unsigned short                             getGPU(const Array3& id) const;
		size_t                                     getExecs() const;
		const vector<shared_ptr<const KernelArg>>& getArgs()  const;
		shared_ptr<const bsp_KernelInfo>           getInfo()  const;
		const Array3&                              getGrid()  const;
		const Array3&                              getBlock() const;
		size_t                                     getShMem() const;
		MEfunction                                 getFunc()  const;
		unsigned                                   getNumArgAccessCalls() const;
		unsigned                                   getNumArgAccessCalcs() const;
		double                                     getTime()  const;
		double                                     getArgAccessTime() const;
		double                                     getLinearizationTime() const;

		const vector<shared_ptr<const ArgAccess>>&       getReadArgAccesses() const;
		const vector<shared_ptr<const ArgAccess>>&       getWriteArgAccesses() const;

		shared_ptr<const ArgAccess>                getReadArgAccess(unsigned short argNr);
		shared_ptr<const ArgAccess>                getWriteArgAccess(unsigned short argNr);
		shared_ptr<MemCpyDtoH>                     getWrittenData(MEdeviceptr ptr, void* hptr);

		void depsResolved();

		//! to save equal kernel launches in a std::set we need this functor
		struct equal_to {
			bool operator()(const shared_ptr<KernelLaunch>& a,
			                const shared_ptr<KernelLaunch>& b) const;
		};

		//! to save equal kernel launches in a std::set we need this functor
		struct hash {
			size_t operator()(const shared_ptr<KernelLaunch>& kl) const;
		};

		MEresult exec();

	private:
		static shared_ptr<KernelLaunch>
		initBare(MEfunction func, const Array3& grid,
		         const Array3& block, size_t shMem, void** rawArgs,
		         shared_ptr<const bsp_KernelInfo> info);

		KernelLaunch(MEfunction func, const Array3& grid, const Array3& block,
		             size_t shMem, void** rawArgs,
		             shared_ptr<const bsp_KernelInfo> info);

		void setPartitions(shared_ptr<AliasHandle> aliasH);

		static isl_stat partIntoMap(__isl_take isl_map* map,
		                            void* partition_and_mapVec);

		static __isl_give isl_union_map*
		partIntoUnionMap(shared_ptr<const Partition> part, isl_union_map* umap);

		static isl_stat addMinAndMax_1D(__isl_take isl_basic_set* bset,
		                                void* boundingPointsRaw);

		static isl_stat addPoint(__isl_take isl_point*, void* points);
		static isl_stat bset_2D_to_1D_intervals(__isl_take isl_basic_set* bset,
		                                        void* dimSize_intervals);

		shared_ptr<const ArgAccess> getArgAccess(unsigned short argNr,
		                                         bool getReadArgAccess);

		const Array3 orgGrid_;
		const Array3 orgBlock_;
		const size_t shMem_;
		const MEfunction func_;
		const shared_ptr<const bsp_KernelInfo> info_;
		const vector<shared_ptr<const KernelArg>> args_;
		shared_ptr<AliasHandle> aliasH_;
		vector<shared_ptr<const Partition>> parts_;

		bool depResolved_ = false;
		size_t executions_ = 0;

		double time_ = 0;
		double argAccessTime_ = 0;
		double linearizationTime_ = 0;
		unsigned numArgAccessCalls_ = 0;
		unsigned numArgAccessCalcs_ = 0;

		vector<shared_ptr<const ArgAccess>> readAccs_;
		vector<shared_ptr<const ArgAccess>> writeAccs_;

		map<unsigned short, shared_ptr<MemCpyDtoH>> argId2memcpy_;
};

bool operator==(const KernelLaunch& a, const KernelLaunch& b); 

bool operator!=(const KernelLaunch& a, const KernelLaunch& b);

ostream& operator<<(ostream& out, const KernelLaunch& kl);

}; // namespace end

#endif
