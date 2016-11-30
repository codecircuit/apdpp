#ifndef MEKONG_KERNEL_INFO_H
#define MEKONG_KERNEL_INFO_H

#include <string>
#include <memory>
#include <ostream>
#include <vector>

namespace Mekong {

class bsp_ArgType;
class Partitioning;
class AccFunc;

using namespace std;

//! Provides information from bsp kernel analysis for one kernel

//! Provides information from bsp kernel analysis for one kernel.
//! The information will be read from a .json file at the moment
//! of construction.
class bsp_KernelInfo {
	public:
		static vector<shared_ptr<const bsp_KernelInfo>> createKInfos(const char* bspAnalysis);
		bsp_KernelInfo(const string& name,
		               const vector<shared_ptr<const bsp_ArgType>>& args,
					   shared_ptr<const Partitioning> part,
					   const vector<shared_ptr<const AccFunc>>& accFuncs = {});

		string getName() const;
		unsigned short getNumArgs() const; 
		const vector<shared_ptr<const bsp_ArgType>>& getArgTypes() const;
		shared_ptr<const Partitioning> getPartitioning() const;
		shared_ptr<const AccFunc> getAccFunc(unsigned short argId) const;

	private:
		string name_;
		vector<shared_ptr<const bsp_ArgType>> args_; ///< kernel argument types
		vector<shared_ptr<const AccFunc>> accFuncs_; ///< kernel argument memory access functions
		shared_ptr<const Partitioning> part_;        ///< Partitioning scheme for this kernel
};

ostream& operator<<(ostream& out, const bsp_KernelInfo& kinfo);

}; // namespace end

#endif
