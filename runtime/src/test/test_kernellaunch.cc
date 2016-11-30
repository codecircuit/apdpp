#ifdef MEKONG_TEST

#include <iostream>
#include <sstream>
#include <algorithm>
#include <new> // bad_alloc
#include <iomanip>
#include <memory>
#include <functional> // hashing
#include <unordered_set>
#include <stdexcept>

#include "kernel_launch.h"
#include "kernel_info.h"
#include "argument_type.h"
#include "argument.h"

using namespace std;
using namespace Mekong;

// Here we init a set, which saves only different kernel launches.
std::unordered_set<std::shared_ptr<Mekong::KernelLaunch>,
                   Mekong::KernelLaunch::hash,
                   Mekong::KernelLaunch::equal_to>
Mekong::KernelLaunch::all(0, Mekong::KernelLaunch::hash(),
                          Mekong::KernelLaunch::equal_to());

// stencil 5p as test case
const char* bspAnalysisStr_TEST =
"kernels-0-arguments-0-dim sizes-0=arg2\n"
"kernels-0-arguments-0-element size=32\n"
"kernels-0-arguments-0-fundamental type=f\n"
"kernels-0-arguments-0-isl read map=[size_x, size_y, size_z, N] -> { Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, 1 + i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[1 + i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[-1 + i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, -1 + i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z }\n"
"kernels-0-arguments-0-isl read params-0=size_x\n"
"kernels-0-arguments-0-isl read params-1=size_y\n"
"kernels-0-arguments-0-isl read params-2=size_z\n"
"kernels-0-arguments-0-isl read params-3=arg2\n"
"kernels-0-arguments-0-name=in\n"
"kernels-0-arguments-0-num dimensions=2\n"
"kernels-0-arguments-0-pointer level=1\n"
"kernels-0-arguments-0-size=0\n"
"kernels-0-arguments-0-type name=float addrspace(1)*\n"
"kernels-0-arguments-1-dim sizes-0=arg2\n"
"kernels-0-arguments-1-element size=32\n"
"kernels-0-arguments-1-fundamental type=f\n"
"kernels-0-arguments-1-isl write map=[size_x, size_y, size_z, N] -> { Stmt_if_then[i0, i1, i2] -> MemRef_out[i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z }\n"
"kernels-0-arguments-1-isl write params-0=size_x\n"
"kernels-0-arguments-1-isl write params-1=size_y\n"
"kernels-0-arguments-1-isl write params-2=size_z\n"
"kernels-0-arguments-1-isl write params-3=arg2\n"
"kernels-0-arguments-1-name=out\n"
"kernels-0-arguments-1-num dimensions=2\n"
"kernels-0-arguments-1-pointer level=1\n"
"kernels-0-arguments-1-size=0\n"
"kernels-0-arguments-1-type name=float addrspace(1)*\n"
"kernels-0-arguments-2-element size=0\n"
"kernels-0-arguments-2-fundamental type=i\n"
"kernels-0-arguments-2-name=N\n"
"kernels-0-arguments-2-pointer level=0\n"
"kernels-0-arguments-2-size=32\n"
"kernels-0-arguments-2-type name=i32\n"
"kernels-0-name=stencil5p_2D\n"
"kernels-0-partitioning=y\n";

bool test0() {

	shared_ptr<AliasHandle> aliasH(new AliasHandle);

	// INIT ALIAS HANDLE GLOBAL VAR
	MEdevice dev;
	vector<MEdevice> vdev(2, dev); // set 2 gpus
	(*aliasH)[dev] = vdev;
	
	// READ DATABASE
	shared_ptr<const bsp_KernelInfo> kinfo = bsp_KernelInfo::createKInfos(bspAnalysisStr_TEST)[0];

	// SET UP KERNEL ARGUMENTS
	MEdeviceptr input = (MEdeviceptr) 0;
	MEdeviceptr output = (MEdeviceptr) 1;
	MEfunction kernel = (MEfunction) 2;
	int N = 8;
	void* rawArgs[] = {&input, &output, &N};

	// SET UP KERNEL LAUNCH CONFIGURATION
	using T3 = Partition::Array3;
	T3 gridSize = { 2, 2, 1 };
	T3 blockSize = { 4, 4, 1 };
	
	// CREATE KERNEL LAUNCH OBJECT
	shared_ptr<KernelLaunch> kl0(new KernelLaunch(kernel,
	                                              gridSize,
	                                              blockSize,
	                                              0,
	                                              rawArgs,
	                                              kinfo,
	                                              aliasH));
	shared_ptr<KernelLaunch> kl1(new KernelLaunch(kernel,
	                                              gridSize,
	                                              blockSize,
	                                              0,
	                                              rawArgs,
	                                              kinfo,
	                                              aliasH));
	KernelLaunch::all.insert(kl0);
	KernelLaunch::all.insert(kl1);
	cout << "  - insert two equal launches in set " << flush;
	if (KernelLaunch::all.size() == 1) {
		cout << "[OK]" << endl;
	}
	else {
		cout << "[FALSE] set has size of " << KernelLaunch::all.size() << endl; 
	}
	rawArgs[0] = &output;
	rawArgs[1] = &input;
	shared_ptr<KernelLaunch> kl2(new KernelLaunch(kernel,
	                                              gridSize,
	                                              blockSize,
	                                              0,
	                                              rawArgs,
	                                              kinfo,
	                                              aliasH));
	KernelLaunch::all.insert(kl2);
	cout << "  - insert different launch in set " << flush;
	if (KernelLaunch::all.size() == 2) {
		cout << "[OK]" << endl;
	}
	else {
		cout << "[FALSE] set has size of " << KernelLaunch::all.size() << endl; 
	}

	vector<shared_ptr<KernelLaunch>> v;
	for (int i = 0; i < 50; ++i) {
		rawArgs[i % 2] = &output;
		rawArgs[(i + 1) % 2] = &input;
		auto p = shared_ptr<KernelLaunch>(new KernelLaunch(kernel, gridSize,
		                                                   blockSize, 0,
		                                                   rawArgs, kinfo,
		                                                   aliasH));
		v.push_back(p);
		KernelLaunch::all.insert(p);
	}

	cout << "  - insert a lot equal launches into set " << flush;
	if (KernelLaunch::all.size() == 2) {
		cout << "[OK]" << endl;
	}
	else {
		cout << "[FALSE] set has size of " << KernelLaunch::all.size() << endl; 
	}

	cout << endl;
	return true;
}

bool test1() {
	shared_ptr<AliasHandle> aliasH(new AliasHandle);

	// INIT ALIAS HANDLE GLOBAL VAR
	MEdevice dev;
	vector<MEdevice> vdev(2, dev); // set 2 gpus
	(*aliasH)[dev] = vdev;
	
	// READ DATABASE
	shared_ptr<const bsp_KernelInfo> kinfo = bsp_KernelInfo::createKInfos(bspAnalysisStr_TEST)[0];

	// SET UP KERNEL ARGUMENTS
	MEdeviceptr input = (MEdeviceptr) 0;
	MEdeviceptr output = (MEdeviceptr) 1;
	MEfunction kernel = (MEfunction) 2;
	int N = 8;
	void* rawArgs[] = {&input, &output, &N};

	// SET UP KERNEL LAUNCH CONFIGURATION
	using T3 = Partition::Array3;
	T3 gridSize  = { 2, 2, 1 };
	T3 blockSize = { 4, 4, 1 };
	
	// CREATE KERNEL LAUNCH OBJECT
	KernelLaunch kl(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH);

	// Now we should have the following configuration:
	//
	//   * GPU0 writes indices [i * 8 + 1, i * 8 + 5] for i =  1,..,3
	//   * GPU1 writes indices [i * 8 + 1, i * 8 + 5] for i = 4,..,6
	//
	//   * GPU0 reads indices [1, 6] and [8, 31] and [33, 38]
	//   * GPU1 reads indices [25, 30] and [32, 55] and [57, 62]
	//
	// Our array has a size of N * N = 64

	cout << endl;
	cout << "PARTITIONS:" << endl;
	for (auto part : kl.getPartitions()) {
		cout << *part << endl;
	}
	cout << endl;
	cout << "CALLING getReadArgAccess ON KERNEL OBJECT:" << endl;
	cout << endl;
	cout << "Kernel reads argument 0:" << endl;
	auto rac = kl.getReadArgAccess(0);
	cout << "Kernel writes argument 1:" << endl;
	auto wac = kl.getWriteArgAccess(1);
	cout << *rac << endl;
	cout << *wac << endl;

	void* dummy;
	shared_ptr<MemCpyDtoH> memcpy = kl.getWrittenData(output, dummy);
	auto pattern = memcpy->getPattern();
	for (auto subcpy : *pattern) {
		cout << subcpy << endl;
	}
	return true;
}

int main() {

	cout << endl;
	cout << "# Testing Kernel Launch Class" << endl;
	cout << endl;

	test0();
	test1();
	return 0;
}

#endif
