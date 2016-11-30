#include "alias_handle.h"
#include "argument_access.h"
#include "memory_copy.h"
#include "argument.h"
#include "access_function.h"
#include "kernel_info.h"
#include "mekong-cuda.h"
#include "partitioning.h"
#include "partition.h"
#include "kernel_launch.h"

#include <memory>     // smart pointer
#include <algorithm>  // std::sort
#include <new>        // std::bad_alloc
#include <chrono>     // time measurements
#include <thread>     // for parallel linearization
#include <utility>    // std::move
#include <limits>     // needed for overflow check
#include <functional> // std::hash
#include <ostream>
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>

#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/printer.h>
#include <isl/constraint.h>
#include <isl/val.h>
#include <isl/aff.h>
#include <isl/union_set.h>
#include <isl/point.h>
#include <isl/local_space.h>
#include <isl/space.h>

namespace Mekong {

using namespace std;


/*! \brief Avoids redundundant Object creating using a static set.
    \return the object and if it was successfully inserted
    \sa KernelLaunch::all
*/
pair<shared_ptr<KernelLaunch>, bool>
KernelLaunch::getOrInsert(MEfunction func, const Array3& grid,
                          const Array3& block,
                          size_t shMem,
                          void** rawArgs,
                          shared_ptr<const bsp_KernelInfo> info,
                          shared_ptr<AliasHandle> aliasH) {
	auto kl = initBare(func, grid, block, shMem, rawArgs, info);

	// add only different kernel launches
	auto it_and_bool = all.insert(kl);
	if (get<1>(it_and_bool)) { // element was inserted successfully
		kl->setPartitions(aliasH); // calculate partitions
	}
	else { // element was not inserted
		kl = *get<0>(it_and_bool); // take already existing kernel launch object
	}
	return make_pair(kl, get<1>(it_and_bool));
}

KernelLaunch::KernelLaunch(MEfunction func, const Array3& grid,
                           const Array3& block,
                           size_t shMem,
                           void** rawArgs,
                           shared_ptr<const bsp_KernelInfo> info,
                           shared_ptr<AliasHandle> aliasH) :
			 orgGrid_(grid),
			 orgBlock_(block),
			 func_(func),
			 shMem_(shMem),
			 info_(info), 
			 aliasH_(aliasH),
			 args_(KernelArg::createArgs(info->getArgTypes(), rawArgs)),
			 parts_(Partition::createPartitions(grid,
			                                    block,
			                                    aliasH,
			                                    info->getPartitioning())),
			 readAccs_(args_.size(), nullptr),
			 writeAccs_(args_.size(), nullptr) {}

//! Checks if \param ptr is a given argument of this launch.
bool KernelLaunch::isArg(MEdeviceptr ptr) const {
	for (auto arg : args_) {
		if (*arg == ptr) {
			return true;
		}
	}
	return false;
}

//! Checks if \param arg is a given argument of this launch.
bool KernelLaunch::isArg(shared_ptr<const KernelArg> arg) const {
	for (auto thisarg : args_) {
		if (thisarg == arg) {
			return true;
		}
		if (*thisarg == *arg) {
			return true;
		}
	}
	return false;
}

//! Only the partitioning, grid size, block size, non-pointer kernel arguments and
//! the kernel function  affect the argument access.
bool KernelLaunch::hasEqualArgAccess(const KernelLaunch& other) const {

	auto isEqArr = [] (const KernelLaunch::Array3& t0, const KernelLaunch::Array3& t1) {
		return (t0[0] == t1[0])
			&& (t0[1] == t1[1])
			&& (t0[2] == t1[2]);
	};

	if (this == &other) {
		return true;
	}

	if (!isEqArr(getGrid(), other.getGrid())) {
		return false;
	}
	if (!isEqArr(getBlock(), other.getBlock())) {
		return false;
	}
	// If functions are equal, the number of kernel args is also equal.
	// If functions, grid size, and block size are equal, the partitioning
	// is also equal.
	if (getFunc() != other.getFunc()) {
		return false;
	}

	auto it_A = getArgs().begin();
	auto it_B = other.getArgs().begin();
	for (; it_A != getArgs().end(); ++it_A, ++it_B) {
		// only non pointer kernel arguments affect the argument
		// access of a kernel launch
		if ((*it_A)->getType()->getPtrlvl() != 1) {
			if (**it_A != **it_B) { // we call the operator!= for class KernelArg
				return false;
			}
		}
	}

	return true;
}

//! Returns the belonging KernelArg object for the given device pointer \param ptr.

//! If no argument is equal to \param ptr the function returns a nullptr.
shared_ptr<const KernelArg> KernelLaunch::getArg(MEdeviceptr ptr) const {
	for (auto arg : args_) {
		if (*arg == ptr) {
			return arg;
		}
	}
	return nullptr;
}

//! Returns -1 if \param ptr is not one of the arguments
int KernelLaunch::getArgId(MEdeviceptr ptr) const {
	int res = 0;
	for (auto arg : args_) {
		if (*arg == ptr) {
			return res;
		}
		++res;
	}
	return -1;
}

//! Returns the argument id in this launch for the given KernelArg object.

//! If the given KernelArg object is not an argument in this launch the
//! function will return -1.
int KernelLaunch::getArgId(shared_ptr<const KernelArg> arg) const {
	int res = 0;
	for (auto thisarg : args_) {
		if (thisarg == arg) { // pointer comparison
			return res;
		}
		if (*thisarg == *arg) { // object comparison
			return res;
		}
		++res;
	}
	return -1;
}

//! returns the device pointers, which will be written in the kernel launch
vector<MEdeviceptr> KernelLaunch::getWrites() const {
	vector<MEdeviceptr> res;
	for (auto arg : args_) {
		if (arg->getType()->isModified() && arg->getType()->getPtrlvl() == 1) {
			res.push_back(arg->asDevPtr());
		}
	}
	return res;
}

//! returns the device pointers, which will be read in the kernel launch
vector<MEdeviceptr> KernelLaunch::getReads() const {
	vector<MEdeviceptr> res;
	for (auto arg : args_) {
		if (arg->getType()->isRead() && arg->getType()->getPtrlvl() == 1) {
			res.push_back(arg->asDevPtr());
		}
	}
	return res;
}

//! returns all device pointers, given as kernel arguments in this launch.
vector<MEdeviceptr> KernelLaunch::getPtrs() const {
	vector<MEdeviceptr> res;
	for (auto arg : args_) {
		if (arg->getType()->getPtrlvl() == 1) {
			res.push_back(arg->asDevPtr());
		}
	}
	return res;
}

//! Marks the dependencies for this kernel launch as solved
void KernelLaunch::depsResolved() {
	depResolved_ = true;
}

//! Comparison functor to save KernelLaunch objects in a std::set.
bool KernelLaunch::equal_to::operator()(const shared_ptr<KernelLaunch>& a,
                                        const shared_ptr<KernelLaunch>& b) const {
	return *a == *b; // call free operator==
}

//! Hash function based on kernel, grid and block size
size_t KernelLaunch::hash::operator()(const shared_ptr<KernelLaunch>& kl) const {
	std::hash<unsigned> h;
	std::hash<unsigned int> h_dummy;
	return ((((h(kl->orgGrid_[0]) ^ (h(kl->orgGrid_[1]) << 1)) >> 1 )
	         ^ h(kl->orgGrid_[2]))
	       ^ (((h(kl->orgBlock_[0]) ^ (h(kl->orgBlock_[1]) >> 1)) << 1 )
	         ^ h(kl->orgBlock_[2]))) ^ (((size_t) kl->func_));

}

//! Given a global thread id this function returns the gpu's id, which executes that thread.

//! If the global thread id exceeds the kernel launch limits, this function will throw
//! an invalid_argument exception.
unsigned short KernelLaunch::getGPU(const Array3& id) const {

	auto mult3 = [] (const Array3& a, const Array3& b) {
		return Array3({ a[0] * b[0],
		                a[1] * b[1],
		                a[2] * b[2] });
	};

	auto add3 = [] (const Array3& a, const Array3& b) {
		return Array3({ a[0] + b[0],
		                a[1] + b[1],
		                a[2] + b[2] });
	};

	auto sub3 = [] (const Array3& a, const Array3& b) {
		return Array3({ a[0] - b[0],
		                a[1] - b[1],
		                a[2] - b[2] });
	};

	for (auto part : parts_) {

		// get the partition size in threads
		auto size = mult3(part->getGrid(), part->getBlock());
		
		// get the maximum indexes which are still inside the partition
		auto max  = sub3(add3(part->getOffset(), size), { 1, 1, 1 });

		isl_ctx* ctx = isl_ctx_alloc();
		isl_space* space;
		isl_basic_set* bset;
		isl_basic_set* sing;
		isl_basic_set* inter;
		isl_point* pmin;
		isl_point* pmax;
		isl_point* p;
		isl_val* val;

		space = isl_space_set_alloc(ctx, 0, 3);
		pmin = isl_point_zero(isl_space_copy(space));
		pmax = isl_point_zero(isl_space_copy(space));
		p    = isl_point_zero(space);

		// MIN POINT
		// x
		val = isl_val_int_from_ui(ctx, part->getOffset()[0]);
		pmin = isl_point_set_coordinate_val(pmin, isl_dim_set, 0, val);
		// y
		val = isl_val_int_from_ui(ctx, part->getOffset()[1]);
		pmin = isl_point_set_coordinate_val(pmin, isl_dim_set, 1, val);
		// z
		val = isl_val_int_from_ui(ctx, part->getOffset()[2]);
		pmin = isl_point_set_coordinate_val(pmin, isl_dim_set, 2, val);

		// MAX POINT
		// x
		val = isl_val_int_from_ui(ctx, max[0]);
		pmax = isl_point_set_coordinate_val(pmax, isl_dim_set, 0, val);
		// y
		val = isl_val_int_from_ui(ctx, max[1]);
		pmax = isl_point_set_coordinate_val(pmax, isl_dim_set, 1, val);
		// z
		val = isl_val_int_from_ui(ctx, max[2]);
		pmax = isl_point_set_coordinate_val(pmax, isl_dim_set, 2, val);	

		// CREATING THE BOX
		bset = isl_basic_set_box_from_points(pmin, pmax);
		
		// POINT TO CHECK IF IT IS IN THE BOX
		// x
		val = isl_val_int_from_ui(ctx, id[0]);
		p = isl_point_set_coordinate_val(p, isl_dim_set, 0, val);
		// y
		val = isl_val_int_from_ui(ctx, id[1]);
		p = isl_point_set_coordinate_val(p, isl_dim_set, 1, val);
		// z
		val = isl_val_int_from_ui(ctx, id[2]);
		p = isl_point_set_coordinate_val(p, isl_dim_set, 2, val);
		
		// CREATING SINGLETON
		sing = isl_basic_set_from_point(p);
		
		// INTERSECT
		inter = isl_basic_set_intersect(bset, sing);
		
		if (!isl_basic_set_is_empty(inter)) {
			return part->getDevice();
		}
		isl_basic_set_free(inter);
		isl_ctx_free(ctx);
	}
	throw invalid_argument("could not find thread id in this kernel launch!");
}

//! Given a global thread id this function returns the gpu's id, which executes that thread.

//! If the global thread id exceeds the kernel launch limits, this function will throw
//! an invalid_argument exception.
unsigned short KernelLaunch::getGPU(unsigned idx, unsigned idy, unsigned idz) const {
	return getGPU({idx, idy, idz});
}

//! Returns a vector with the partitions, which will be launched in this KernelLaunch object.
const vector<shared_ptr<const Partition>>& KernelLaunch::getPartitions() const {
	return parts_;
}

//! Returns the partitioning used for this kernel launch, e.g. splitted at X-Dim, splitted at XY-Dim.
shared_ptr<const Partitioning> KernelLaunch::getPartitioning() const {
	return info_->getPartitioning();
}

/*! \brief Executes the kernel launch.

    For security reasons you have to mark the dependency resolution regarding
    this kernel launch by calling depsResolved. Otherwise this function will
    throw an error. After a successful execution the depsResolved flag will be
    set to false again, thus you always have to mark a correct dependency
    resolution before calling this function. For each partition one kernel
    will be launched.
    \todo Support shared memory and streams.
*/
MEresult KernelLaunch::exec() {
	// TODO as we can launch a kernel launch more than once
	// we can create all needed objects at the first kernel
	// launch, and save them for further executions

	auto timestamp = Clock::now();

	auto mult3 = [] (const Array3& a, const Array3& b) {
		return Array3({ a[0] * b[0],
		                a[1] * b[1],
		                a[2] * b[2] });
	};

	if (!depResolved_) {
		throw runtime_error("dependencies for kernel launch are not resolved!");
	}

	MEresult res;
	// We add 6 new kernel arguments
	// offsetX, offsetY, offsetZ, globalSizeX, globalSizeY, globalSizeZ
	void** rawArgs = nullptr;
	try {
		rawArgs = new void*[args_.size() + 6];
	}
	catch (bad_alloc& ba) {
		throw runtime_error(string(string("SPACE Mekong, CLASS KernelLaunch, FUNC exec(): "
		                    "could not allocate memory for kernel args, ") + ba.what()).c_str());
	}

	// GET ORIGINAL ARGUMENTS //
	unsigned short i = 0;
	vector<unique_ptr<charPack>> rawData;
	for (auto arg : args_) {
		// kernel launch function expects a void* and not a const void*, thus
		// we need to copy the data and hold it in memory until the
		// kernel launch is done.
		rawData.push_back(arg->cpyCharPack()); // CAUTION: removing this line will lead to a seg fault
		rawArgs[i++] = rawData.back()->getRaw();
	}

	// get devptr args //
	vector<tuple<unsigned short, shared_ptr<const KernelArg>>> devptrArgs;
	unsigned short argid = 0;
	for (auto arg : args_) {
		if (arg->getType()->getPtrlvl() > 0) {
			devptrArgs.push_back(make_tuple(argid, arg));
		}
		++argid;
	}

	// CALCULATE AND SET GLOBAL SIZE //
	auto globalSize = mult3(orgGrid_, orgBlock_);
	rawArgs[args_.size() + 3] = &globalSize[0];
	rawArgs[args_.size() + 4] = &globalSize[1];
	rawArgs[args_.size() + 5] = &globalSize[2];

	// ITERATE OVER EVERY PARTITION AND LAUNCH IT //
	for (auto part : parts_) {
		for (auto devptrArg : devptrArgs) {
			// in the mekong context there exists one device ptr per memory
			// buffer, which can be accessed by many gpus. On hardware level we
			// have to allocate one dev ptr on every gpu refering that memory
			// buffer. Thus we get the appropriate dev ptr on the device the
			// partition belongs to from the alias handle object.
			rawArgs[get<0>(devptrArg)] = &(*aliasH_)[get<1>(devptrArg)->asDevPtr()].at(part->getDevice());
		}

		Array3 offCpy = part->getOffset(); // we need a void* not a const void*, thus copy temporarily
		rawArgs[args_.size()]     = &offCpy[0];
		rawArgs[args_.size() + 1] = &offCpy[1];
		rawArgs[args_.size() + 2] = &offCpy[2];

		res &= meCtxPushCurrent(aliasH_->getCtx().at(part->getDevice()));
		res &= meLaunchKernel((*aliasH_)[func_].at(part->getDevice()), 
		                        part->getGrid()[0],
		                        part->getGrid()[1],
		                        part->getGrid()[2],
		                        part->getBlock()[0],
		                        part->getBlock()[1],
		                        part->getBlock()[2],
		                        shMem_, 0, rawArgs, 0); // TODO support streams and extra args
		res &= meCtxPopCurrent(0);
	}

	delete[] rawArgs;
	++executions_;
	Duration time_exec = Clock::now() - timestamp;
	depResolved_ = false;
	time_ += time_exec.count();
	return res;
}

//! For debugging purposes: print all points to a visual grid
void printGrid(const vector<vector<size_t>>& points, size_t N) {

	auto pointIsGiven = [&] (size_t x, size_t y) {
		for (auto& p : points) {
			if (p.size() != 2) {
				throw runtime_error("got point with numDims != 2!");
			}
			if (p[0] == y && p[1] == x) {
				return true;
			}
		}
		return false;
	};

	for (size_t index = 0; index < N * N; ++index) {
		size_t x = index % N;
		size_t y = index / N;
		if (x == 0) {
			cout << endl;
		}
		cout << (pointIsGiven(x, y) ? 'x' : '_') << "  ";
	}
	cout << endl;
}

//! Given a kernel argument id this function returns the read behaviour on that argument.

//! If the given argument id does not belong to an argument with a device pointer type
//! This function will throw an std::invalid_argument exception. Before calculating
//! the argument access by itself, the function looks for an equal kernel launch which has
//! already calculated the argument write access for the wanted kernel argument.
//! The calculation of the argument access object is done with the integer set library.
//! The ArgAccess represents a mapping between gpuId and accessed linear intervals on
//! the corresponding kernel argument with \param argNr.
shared_ptr<const ArgAccess>
KernelLaunch::getReadArgAccess(unsigned short argNr) {
	return getArgAccess(argNr, true);
}

//! Given a kernel argument id this function returns the write behaviour on that argument.

//! If the given argument id does not belong to an argument with a device pointer type
//! This function will throw an std::invalid_argument exception. Before calculating
//! the argument access by itself, the function looks for an equal kernel launch which has
//! already calculated the argument write access for the wanted kernel argument.
//! The calculation of the argument access object is done with the integer set library.
//! The ArgAccess represents a mapping between gpuId and accessed linear intervals on
//! the corresponding kernel argument with \param argNr.
shared_ptr<const ArgAccess>
KernelLaunch::getWriteArgAccess(unsigned short argNr) {
	return getArgAccess(argNr, false);
}

//! Given a kernel argument id this function returns the write or read behaviour on that argument.

//! If \param getReadArgAccess is true the function returns the read behaviour and vice versa.
//! If the given argument id does not belong to an argument with a device pointer type
//! This function will throw an std::invalid_argument exception. Before calculating
//! the argument access by itself, the function looks for an equal kernel launch which has
//! already calculated the argument write access for the wanted kernel argument.
//! The calculation of the argument access object is done with the integer set library.
//! The ArgAccess represents a mapping between gpuId and accessed linear intervals on
//! the corresponding kernel argument with \param argNr. This function can get called from
//! the wrapKernelLaunch *and* wrapDtoH function, thus the total time spent in this function
//! can be greater as the total time spent in the wrapKernelLaunch function.
shared_ptr<const ArgAccess> KernelLaunch::getArgAccess(unsigned short argNr, bool getReadArgAccess) {
	++numArgAccessCalls_;
	auto time_argAcc_begin = Clock::now();

	auto throwError = [&getReadArgAccess] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS KernelLaunch, FUNC getArgAccess()\n" + msg);
	};

	if (args_[argNr]->getType()->getPtrlvl() != 1) {
		throwError("can not calculate argument access"
		           "of non pointer kernel argument");
	}

	vector<shared_ptr<const ArgAccess>>& accs = (getReadArgAccess ? readAccs_ : writeAccs_);
	
	// did I already calculated the argument access object?
	if (accs[argNr]) {
		Duration time_argAcc = Clock::now() - time_argAcc_begin;
		argAccessTime_ += time_argAcc.count();
		return accs[argNr];
	}
	
	// search for an equal kernel launch which already calculated the argument
	// access object
	for (auto kernel_it = all.begin(); kernel_it != all.end(); ++kernel_it) {
		// if there was an equal kernel launch before
		// check if the arg access was already calculated
		if (&**kernel_it == this) { // skip yourself
			continue;
		}
		if (hasEqualArgAccess(**kernel_it)) {
			if (getReadArgAccess) {
				if ((*kernel_it)->readAccs_[argNr] != nullptr) {
					accs[argNr] = (*kernel_it)->readAccs_[argNr];
					Duration time_argAcc = Clock::now() - time_argAcc_begin;
					argAccessTime_ += time_argAcc.count();
					return accs[argNr];
				}
			}
			else {
				if ((*kernel_it)->writeAccs_[argNr] != nullptr) {
					accs[argNr] = (*kernel_it)->writeAccs_[argNr];
					Duration time_argAcc = Clock::now() - time_argAcc_begin;
					argAccessTime_ += time_argAcc.count();
					return accs[argNr];
				}
			}
		}
	}

	++numArgAccessCalcs_;
	auto numDims = args_[argNr]->getType()->getNumDims();
	if (numDims > 2) {
		throwError("polly arrays with num dimensions > 2 are not supported.");
	}

	// If no equal kernel launch was found calculate the arg access here
	// 1. For every partition create the range set, which represents the accessed
	//    indices by that partition.
	// 2. Look at every gpu and union all range sets of every partition living
	//    on that gpu, thus we get all indices accessed by a certain gpu.
	// 3. Make that union disjoint and simplify its representation.
	// 4. For every gpu calculate the linear intervals from the union set.
	//    This can be more expensive for 2D access maps
	
	// Here we have to initialize the map on the main thread.
	// Otherwise the threads will create the interval vectors,
	// which leads to undefined runtime behaviour
	map<unsigned short, vector<tuple<size_t, size_t>>> gpuToRanges;
	vector<tuple<size_t, size_t>> empty_vector;
	for (int gpuId = 0; gpuId < aliasH_->getNumDev(); ++gpuId) {
		gpuToRanges[gpuId] = empty_vector;
	}

	// This lambda represents the calculations, which must be done for
	// every GPU. This encapsulation enables us easy parallelization.
	auto calcIndices = [&] (unsigned short gpuId) {
		// As you must use one isl_ctx per thread we have to do some work
		// for every thread
		isl_ctx* ctx = isl_ctx_alloc();
		isl_union_map* accFuncMap;
		if (getReadArgAccess) {
			accFuncMap = getInfo()->getAccFunc(argNr)->getReadIslMap(ctx,
			                                                         &args_,
			                                                         &orgGrid_,
			                                                         &orgBlock_);
		}
		else {
			accFuncMap = getInfo()->getAccFunc(argNr)->getWriteIslMap(ctx,
			                                                          &args_,
			                                                          &orgGrid_,
			                                                          &orgBlock_);
		}
		// end of additional work

		// 1.
		// Collect all sets of current GPU
		vector<isl_set*> setVector;
		for (auto part : parts_) { // iterate over the partitions
			if (part->getDevice() == gpuId) { // only work on current device
				isl_union_map* umap = isl_union_map_copy(accFuncMap);
				umap = partIntoUnionMap(part, umap); // __isl_take
				isl_union_set* range_uset = isl_union_map_range(umap); // __isl_take
				// As all accesses in an isl_union_map MUST have the same array as
				// the target space, the range_uset MUST be trivially convertible
				// to an isl_set
				isl_set* rangeSet = isl_set_from_union_set(range_uset); // __isl_take
				setVector.push_back(rangeSet);
			}
		}
		if (setVector.empty()) {
			isl_union_map_free(accFuncMap);
			isl_ctx_free(ctx);
			return;
		}

		// 2. Union all sets of current GPU
		// __isl_give isl_space* isl_set_get_space(__isl_keep isl_set*)
		// __isl_give isl_set* isl_set_empty(__isl_take isl_space*)
		isl_set* currPoints = isl_set_empty(isl_set_get_space(setVector[0]));
		for (auto* set : setVector) {
			currPoints = isl_set_union(currPoints, set); // __isl_take, __isl_take
		}

		// 3. Simplify the representation of a set, relation or functions by trying to combine
		//    pairs of basic sets or relations into a single basic set or relation.
		currPoints = isl_set_make_disjoint(currPoints); // __isl_take
		currPoints = isl_set_coalesce(currPoints); // __isl_take
		currPoints = isl_set_remove_redundancies(currPoints); // __isl_take
		// 4.
		if (numDims == 1) {
			isl_set_foreach_basic_set(currPoints, addMinAndMax_1D, &gpuToRanges[gpuId]);
		}
		else if (numDims == 2) {
			// Now we collect the intervals contained in the basic set
			vector<tuple<size_t, size_t>>& intervals = gpuToRanges[gpuId];
			size_t dimSize = args_[argNr]->getDimSize(0);
			auto dimSize_intervals = make_tuple(dimSize, &intervals);
			isl_set_foreach_basic_set(currPoints, bset_2D_to_1D_intervals, &dimSize_intervals);

			// Now consider the following accessed points on a 2D array
			// ('O' denotes accessed elements):
			//
			// y    Array                basic sets
			// 0    * * * * * * * * *
			// 1    * O O O O O O O O    <- bset1
			// 2    * * O O O O O O O    <- bset1
			// 3    * * * O O O O O O    <- bset1
			// 4    O O O O O O * * *    <- bset0
			// 5    O O O O O O O O O    <- bset0
			// 6    O O O O O O O O O    <- bset0
			// 7    * * * * * * * * *
			//
			// It could happen that the lower basic will be processed first,
			// which will result in a seperation of interval y = 3 and y = 4.
			// As we want to have the minimum amount of memcpys we want to have
			// one interval for y = 3,4. Thus we have to sort the calculated
			// intervals at this point of the program

			// FOR PERFORMANCE REASONS WE IGNORE THIS. THE SORT IS VERY EXPENSIVE
			// AND MIGHT BE NOT AMORTIZED BY A LOWER NUMBER OF MEMCPYS 
			/*sort(intervals.begin(), intervals.end(), [] (const tuple<size_t, size_t>& a,
														 const tuple<size_t, size_t>& b) {
															   return get<0>(a) < get<0>(b);
													 }
			);*/

			// Now we can try to concatenate the intervals where it is possible
			if (!intervals.empty()) {
				for (auto interval_it = intervals.begin() + 1;
				     interval_it != intervals.end();
				     ++interval_it) {
					auto predecessor = interval_it - 1;
					if (get<1>(*predecessor) == get<0>(*interval_it)) {
						get<1>(*predecessor) = get<1>(*interval_it);
						intervals.erase(interval_it);
						--interval_it;
					}
				}
			}
			// 5. save the results
			//gpuToRanges.emplace(gpuId, move(intervals));
			gpuToRanges.insert(make_pair(gpuId, intervals));
		}

		isl_set_free(currPoints);
		isl_union_map_free(accFuncMap);
		isl_ctx_free(ctx);
	}; // end of lambda function

	// start a thread for every gpu to calculate the accessed indices
	auto time_linearization_begin = Clock::now();
	vector<thread> threads;
	for (unsigned short gpuId = 0; gpuId < aliasH_->getNumDev(); ++gpuId) {
		//calcIndices(gpuId);
		threads.push_back(thread(calcIndices, gpuId));
	}
	for (auto& t : threads) {
		t.join();
	}
	Duration time_linearization = Clock::now() - time_linearization_begin;
	linearizationTime_ += time_linearization.count();
	accs[argNr] = shared_ptr<const ArgAccess>(new ArgAccess(move(gpuToRanges)));

	Duration time_argAcc = Clock::now() - time_argAcc_begin;
	argAccessTime_ += time_argAcc.count();
	return accs[argNr];
}

//! Returns a memcpy object, to get written data.

//! If \param ptr is not an argument this function throws an std::invalid_argument exception.
shared_ptr<MemCpyDtoH> KernelLaunch::getWrittenData(MEdeviceptr ptr, void* hptr) {

	int argId = getArgId(ptr);

	if (argId == -1) {
		throw invalid_argument("SPACE Mekong, CLASS KernelLaunch, FUNC getWrittenData(): "
		                       "could not find device pointer in this kernel arguments!");
	}
	
	// SEARCH FOR ALREADY EXISTING MEMCPY OBJECT
	// this should be efficient for iterative calls of memcpyDtoH
	// TODO write two different versions of the wrapMemcpyDtoH() function.
	// One optimized for iterative calls with the for loop below included
	// and another one without the foor-loop
	/*for (auto it = all.rbegin(); it != all.rend(); ++it) {
		if (hasEqualArgAccess(**it)) {
			if ((*it)->argId2memcpy_.find(argId) != (*it)->argId2memcpy_.end()) {
				argId2memcpy_[argId] = (*it)->argId2memcpy_[argId];
				break;
			}
		}
	}*/

	// IF IT ALREADY EXISTS, RETURN IT
	if (argId2memcpy_.find(argId) != argId2memcpy_.end()) {
		argId2memcpy_[argId]->setDst(hptr);
		return argId2memcpy_[argId];
	}
	// ELSE CREATE THE OBJECT
	auto argAcc = getWriteArgAccess(argId);
	vector<MemSubCopy> subcpys;
	for (auto it = argAcc->getMap().begin(); it != argAcc->getMap().end(); ++it) { // loop over gpus
		auto gpuId = it->first;
		auto& intervals = it->second;
		for (auto& interval : intervals) {
			MemSubCopy subcpy;
			subcpy.src = gpuId;
			subcpy.dst = -1; // always host as aim
			size_t elSize = args_[argId]->getType()->getElSize();
			size_t offset = get<0>(interval) * elSize;  
			subcpy.from = offset; 
			subcpy.to = offset;
			subcpy.size = get<1>(interval) * elSize - offset;
			subcpys.push_back(move(subcpy));
		}
	}
	auto pattern = shared_ptr<const vector<MemSubCopy>>(new vector<MemSubCopy>(move(subcpys)));
	auto cpy = shared_ptr<MemCpyDtoH>(new MemCpyDtoH(hptr, ptr, pattern, aliasH_));
	argId2memcpy_[argId] = cpy;

	return cpy;
}

//! Returns how often this launch was executed
size_t KernelLaunch::getExecs() const {
	return executions_;
}

//! Returns the KernelArg object for the argument id \param nr.

//! The function will throw if \param nr exceeds the limits
shared_ptr<const KernelArg> KernelLaunch::getArgFromId(unsigned short nr) const {
	return args_[nr];
}

//! Returns a vector with the kernel arguments.
const vector<shared_ptr<const KernelArg>>& KernelLaunch::getArgs() const {
	return args_;
}

const KernelLaunch::Array3& KernelLaunch::getGrid() const {
	return orgGrid_;
}

const KernelLaunch::Array3& KernelLaunch::getBlock() const {
	return orgBlock_;
}

size_t KernelLaunch::getShMem() const {
	return shMem_;
}

//! Returns the raw function pointer belonging to the kernel.
MEfunction KernelLaunch::getFunc() const {
	return func_;
}

//! Returns the number of getArgAccess function calls
unsigned KernelLaunch::getNumArgAccessCalls() const {
	return numArgAccessCalls_;
}

//! Returns how often the arg access object was calculated.

//! The getArgAccess function tries to find an appropriate arg
//! access object first. If that is not found the function starts
//! to calculate a new object, which is an rather expensive operation.
unsigned KernelLaunch::getNumArgAccessCalcs() const {
	return numArgAccessCalcs_;
}

//! Returns the kernel information object, which is determined by the static kernel analysis.
shared_ptr<const bsp_KernelInfo> KernelLaunch::getInfo() const {
	return info_;
}

//! Returns the total time consumed by the exec() member function of this launch in seconds.

//! Keep in mind that this is not equal to the actual kernel time.
double KernelLaunch::getTime() const {
	if (time_ == 0) {
		throw runtime_error("kernel launch is not executed yet, thus has no measured time");
	}
	return time_;
}

//! Returns the time needed to calculate or find the appropriate argument access object

//! An argument access object describes the intervals accessed by a certain GPU. This
//! function returns the total time needed by this object to calculate or to lookup the
//! wanted argument access objects
double KernelLaunch::getArgAccessTime() const {
	return argAccessTime_;
}

//! Returns the time needed to extract the linear intervals out of the ISL maps.

//! For memory accesses like arr[x + N * y] polly's analysis emits a two dimensional isl map.
//! When we want to calculate the argument access objects we have to linearize this two
//! dimensional objects again, which costs additional time. Moreover time is also needed to calculate
//! the ArgAccess object for one dimensional access patterns, which will be returned by this function.
double KernelLaunch::getLinearizationTime() const {
	return linearizationTime_;
}

//! For debugging purposes only! Use getReadArgAccess(unsigned short argNr) instead!
const vector<shared_ptr<const ArgAccess>>& KernelLaunch::getReadArgAccesses() const {
	return readAccs_;
}

//! For debugging purposes only! Use getWriteArgAccess(unsigned short argNr) instead!
const vector<shared_ptr<const ArgAccess>>& KernelLaunch::getWriteArgAccesses() const {
	return writeAccs_;
}

/*! \brief Initializes a kernel launch without partition creation.
    \sa getOrInsert uses this function
*/
shared_ptr<KernelLaunch>
KernelLaunch::initBare(MEfunction func, const Array3& grid, 
                       const Array3& block, size_t shMem, void** rawArgs,
                       shared_ptr<const bsp_KernelInfo> info) {
	return shared_ptr<KernelLaunch>(new KernelLaunch(func, grid, block, shMem,
	                                                 rawArgs, info));
}

/*! \brief Calculate the partitions for a bare initialized kernel launch object.
    \sa initBare
*/
void KernelLaunch::setPartitions(shared_ptr<AliasHandle> aliasH) {
	this->aliasH_ = aliasH;
	parts_ = Partition::createPartitions(orgGrid_,
	                                     orgBlock_,
	                                     aliasH,
	                                     info_->getPartitioning());
}

//! Slim constructor without partition creation.

//! If a kernel launch object is created with this constructor you can not
//! launch the kernel until you complete the necessary information by calling
//! \sa setPartitions
KernelLaunch::KernelLaunch(MEfunction func, const Array3& grid,
                           const Array3& block,
                           size_t shMem,
                           void** rawArgs,
                           shared_ptr<const bsp_KernelInfo> info) :
			 orgGrid_(grid),
			 orgBlock_(block),
			 func_(func),
			 shMem_(shMem),
			 info_(info), 
			 args_(KernelArg::createArgs(info->getArgTypes(), rawArgs)),
			 readAccs_(args_.size(), nullptr),
			 writeAccs_(args_.size(), nullptr) {}

//! Set the partition's boundaries to the given isl_map

//! This function is used by function partIntoUnionMap. We have to integrate
//! the partition's boundaries to every map in the given isl_union_map.
isl_stat KernelLaunch::partIntoMap(__isl_take isl_map* map, void* partition_and_mapVec) {

	auto part= get<0>(*((pair<shared_ptr<const Partition>, vector<isl_map*>*>*) partition_and_mapVec));
	auto mapVec = get<1>(*((pair<shared_ptr<const Partition>, vector<isl_map*>*>*) partition_and_mapVec));

	isl_space* space = isl_map_get_space(map);
	isl_local_space* ls = isl_local_space_from_space(space);

	auto checkOverflow = [] (size_t s) {
		if (s > numeric_limits<int>::max()) {
			throw overflow_error("isl cant handle such big numbers");
		}
		return (int) s;
	};

	int v;

	// MINIMUM CONSTRAINT
	// x
	isl_constraint* c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
	v = (-1) * checkOverflow(part->getOffset()[0]);
	c = isl_constraint_set_constant_si(c, v);
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 0, 1);
	map = isl_map_add_constraint(map, c);
	// y
	c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
	v = (-1) * checkOverflow(part->getOffset()[1]);
	c = isl_constraint_set_constant_si(c, v);
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 1, 1);
	map = isl_map_add_constraint(map, c);
	// z
	c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
	v = (-1) * checkOverflow(part->getOffset()[2]);
	c = isl_constraint_set_constant_si(c, v);
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 2, 1);
	map = isl_map_add_constraint(map, c);

	// MAXIMUM CONSTRAINT
	// x
	c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
	v = checkOverflow(part->getOffset()[0] + part->getSize()[0] - 1);
	c = isl_constraint_set_constant_si(c, v);
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 0, -1);
	map = isl_map_add_constraint(map, c);
	// y
	c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
	v = checkOverflow(part->getOffset()[1] + part->getSize()[1] - 1);
	c = isl_constraint_set_constant_si(c, v);
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 1, -1);
	map = isl_map_add_constraint(map, c);
	// z
	c = isl_constraint_alloc_inequality(ls);
	v = checkOverflow(part->getOffset()[2] + part->getSize()[2] - 1);
	c = isl_constraint_set_constant_si(c, v);
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 2, -1);
	map = isl_map_add_constraint(map, c);
	mapVec->push_back(map);
	return isl_stat_ok;
}

//! Integrates the boundaries of a partition into an isl union map.

//! \param part can be any partition object, but \param map must be
//! an isl union map with maps, which have at least a three dimensional domain.
//! [n, nn] -> { [x, y, z, w] -> [x + 2 + w] : x, y, z >= 0 and 0 <= w <= 9 }
//! and a partition with offset(10, 20, 30) and size(5, 15, 25) you get 
//! [n, nn] -> { [x, y, z, w] -> [x + 2] : x, y, z >= 0 and x >= 10 and y >= 20 and
//! z >= 30 and x <= 15 and y <= 35 and z <= 55 and 0 <= w <= 9 } as a result.
__isl_give isl_union_map* KernelLaunch::partIntoUnionMap(shared_ptr<const Partition> part,
                                                         __isl_take isl_union_map* umap) {

	vector<isl_map*> mapVec;
	auto partition_and_mapVec = make_pair(part, &mapVec);
	isl_union_map_foreach_map(umap, partIntoMap, &partition_and_mapVec);

	if (mapVec.empty()) {
		return umap;
	}

	// UNION THE RESULTS IN AN isl_union_map OBJECT
	isl_union_map* umap_bounded = isl_union_map_from_map(mapVec[0]);
	for (int i = 1; i < mapVec.size(); ++i) {
		umap_bounded = isl_union_map_union(umap_bounded, isl_union_map_from_map(mapVec[i]));
	}

	isl_union_map_free(umap);
	return umap_bounded;
}

//! Hacky function to grep the minimun and maximum value out of a isl basic set.

//! As a basic set always has to be convex you can represent the complete 1D basic set
//! by the interval [min, max]
isl_stat KernelLaunch::addMinAndMax_1D(__isl_take isl_basic_set* bset, void* boundingPointsRaw) {
	
	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS KernelLaunch, FUNC addMinAndMax_1D():\n" + msg);	
	};

	unsigned numDims = isl_basic_set_dim(bset, isl_dim_out); // __isl_keep
	
	if (numDims != 1) {
		throwError("I got a basic set with " + to_string(numDims) + " dimensions");
	}

	isl_set* minset = isl_basic_set_lexmin(isl_basic_set_copy(bset)); // __isl_take
	isl_set* maxset = isl_basic_set_lexmax(isl_basic_set_copy(bset));
	isl_point* minpt = isl_set_sample_point(minset); // __isl_take
	isl_point* maxpt = isl_set_sample_point(maxset);

	isl_val* minv = isl_point_get_coordinate_val(minpt, isl_dim_out, 0);
	isl_val* maxv = isl_point_get_coordinate_val(maxpt, isl_dim_out, 0);

	long min = isl_val_get_num_si(minv);
	long max = isl_val_get_num_si(maxv) + 1; // + 1, as maximum is exclusive
	
	isl_val_free(minv);
	isl_val_free(maxv);

	auto boundingPointsVec = (vector<tuple<size_t, size_t>>*) boundingPointsRaw;
	boundingPointsVec->push_back(make_tuple(min, max));

	isl_point_free(minpt);
	isl_point_free(maxpt);
	isl_basic_set_free(bset);

	return isl_stat_ok;
}

//! For debug purposes: grep all points from a N-dimensional set
isl_stat KernelLaunch::addPoint(__isl_take isl_point* point, void* points) {

	auto pointVec = (vector<vector<size_t>>*) points;
	vector<size_t> coords;
	isl_space* space = isl_point_get_space(point);
	unsigned numDim = isl_space_dim(space, isl_dim_out);
	if (numDim > 2) {
		throw runtime_error("SPACE Mekong, CLASS AccFunc, FUNC addPoint():\n"
		                    "Polly array with dim > 2 are not supported");
	}

	for (int dim = 0; dim < numDim; ++dim) {
		isl_val* val = isl_point_get_coordinate_val(point, isl_dim_out, dim);
		coords.push_back(isl_val_get_num_si(val));
		isl_val_free(val);
	}
	isl_point_free(point);
	isl_space_free(space);
	pointVec->push_back(coords);
	return isl_stat_ok;
}

//! Extract the intervals contained in a two dimensional isl basic set

//! E.g. assume we have the following 2D array with an x-axis
//! size of dimSize = 17. In this array lives an isl basic set
//! (shown by 'O') illustrated below.
//!
//!    Input                                Output
//!    * * * * * * * * * * * * * * * * * 
//!    * * * * * * * * * * * * * * * * * 
//!    * * * * O O O O O O O * * * * * *    [21, 28)
//!    * * * O O O O O O O O O * * * * *    [54, 60)
//!    * * O O O O O O O O O * * * * * *    [70, 79)
//!    * * * * * O O O O O * * * * * * *    [90, 95)
//!    * * * * * * * * * * * * * * * * * 
//!    * * * * * * * * * * * * * * * * * 
//!    * * * * * * * * * * * * * * * * * 
//!
//!
isl_stat KernelLaunch::bset_2D_to_1D_intervals(__isl_take isl_basic_set* bset, void* dimSize_intervals) {

	size_t dimSize = get<0>(*((tuple<size_t, vector<tuple<size_t, size_t>>*>*) dimSize_intervals));
	auto* intervals = get<1>(*((tuple<size_t, vector<tuple<size_t, size_t>>*>*) dimSize_intervals));

	auto bsetFix_Y_Val = [] (size_t val, __isl_keep isl_basic_set* bs) {
		isl_val* islVal = isl_val_int_from_ui(isl_basic_set_get_ctx(bs), val);
		return isl_basic_set_fix_val(isl_basic_set_copy(bs), isl_dim_out, 0, islVal);
	};

	isl_set* minset = isl_basic_set_lexmin(isl_basic_set_copy(bset));
	isl_set* maxset = isl_basic_set_lexmax(isl_basic_set_copy(bset));
	isl_point* minpt = isl_set_sample_point(minset);
	isl_point* maxpt = isl_set_sample_point(maxset);

	isl_val* minv_y = isl_point_get_coordinate_val(minpt, isl_dim_out, 0);
	isl_val* maxv_y = isl_point_get_coordinate_val(maxpt, isl_dim_out, 0);

	long min_y = isl_val_get_num_si(minv_y);
	long max_y = isl_val_get_num_si(maxv_y) + 1; // + 1, as maximum is exclusive

	isl_point_free(minpt);
	isl_point_free(maxpt);
	isl_val_free(minv_y);
	isl_val_free(maxv_y);

	for (size_t y = min_y; y < max_y; ++y) {
		isl_basic_set* fbset = bsetFix_Y_Val(y, bset);
		minset = isl_basic_set_lexmin(isl_basic_set_copy(fbset));
		maxset = isl_basic_set_lexmax(isl_basic_set_copy(fbset));
		minpt = isl_set_sample_point(minset);
		maxpt = isl_set_sample_point(maxset);

		isl_val* minv_x = isl_point_get_coordinate_val(minpt, isl_dim_out, 1);
		isl_val* maxv_x = isl_point_get_coordinate_val(maxpt, isl_dim_out, 1);

		long min_x = isl_val_get_num_si(minv_x);
		long max_x = isl_val_get_num_si(maxv_x) + 1; // + 1, as maximum is exclusive

		intervals->push_back(make_tuple(min_x + y * dimSize, max_x + y * dimSize));

		isl_point_free(minpt);
		isl_point_free(maxpt);
		isl_val_free(minv_x);
		isl_val_free(maxv_x);
		isl_basic_set_free(fbset);
	}
	
	isl_basic_set_free(bset);

	return isl_stat_ok;
}

//! Both launches are equal if they have the same grid size, block size, 
//! kernel function, shared memory size, and kernel arguments. An argument is
//! equal to another  argument if it has an equal value. We do not need to check
//! for the same  kernel argument type, because the position of the kernel
//! argument implicitly  specifies its type.
bool operator==(const KernelLaunch& a, const KernelLaunch& b) {

	auto isEqArr = [] (const KernelLaunch::Array3& t0,
	                   const KernelLaunch::Array3& t1) {
		return (t0[0] == t1[0])
			&& (t0[1] == t1[1])
			&& (t0[2] == t1[2]);
	};

	if (&a == &b) {
		return true;
	}

	if (!isEqArr(a.getGrid(), b.getGrid())) {
		return false;
	}
	if (!isEqArr(a.getBlock(), b.getBlock())) {
		return false;
	}
	// If functions are equal, the number of kernel args is also equal.
	// If functions, grid size, and block size are equal, the partitioning
	// is also equal.
	if (a.getFunc() != b.getFunc()) {
		return false;
	}
	if (a.getShMem() != b.getShMem()) {
		return false;
	}

	auto it_A = a.getArgs().begin();
	auto it_B = b.getArgs().begin();
	int hackcount = 0; // HACK REMOVE ME
	for (; it_A != a.getArgs().end(); ++it_A, ++it_B) {
		// KernelArg::isEqualInBits
		if (!(*it_A)->isEqualInBits(**it_B)) {
			return false;
		}
		++hackcount; // HACK REMOVE ME
	}

	return true;
}

bool operator!=(const KernelLaunch& a, const KernelLaunch& b) {
	return !(a == b);
}

ostream& operator<<(ostream& out, const KernelLaunch& kl) {
	auto print3 = [&] (const KernelLaunch::Array3& a) {
		out << "(" << a[0] << ", " << a[1] << ", " << a[2] << ")";
	};
	out << "{ Name(" << kl.getInfo()->getName() << ")";
	out << " Grid";
	print3(kl.getGrid());
	out << " Block";
	print3(kl.getBlock());
	out << " shMem(" << kl.getShMem() << ")";
	out << " numPartition(" << kl.getPartitions().size() << ")";
	out << " execs(" << kl.getExecs() << ") }";
	return out;
}

}; // namespace end
