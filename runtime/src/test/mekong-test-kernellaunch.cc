#ifdef MEKONG_TEST

#include <iostream>
#include <sstream>
#include <algorithm>
#include <new> // bad_alloc
#include <iomanip>
#include <memory>
#include <stdexcept>

#include "kernel_launch.h"
#include "kernel_info.h"
#include "argument_type.h"
#include "argument.h"

using namespace std;
using namespace Mekong;

const char* bspAnalysisStrTEST =
"\n"
"{\n"
" \"kernels\" : \n"
" [\n"
"  \n"
"  {\n"
"   \"arguments\" : \n"
"   [\n"
"    \n"
"    {\n"
"     \"dim sizes\" : [ \"arg2\" ],\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z, N] -> { Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, 1 + i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[1 + i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[-1 + i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, -1 + i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg2\" ],\n"
"     \"name\" : \"in\",\n"
"     \"num dimensions\" : 2,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"dim sizes\" : [ \"arg2\" ],\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl write map\" : \"[size_x, size_y, size_z, N] -> { Stmt_if_then[i0, i1, i2] -> MemRef_out[i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl write params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg2\" ],\n"
"     \"name\" : \"out\",\n"
"     \"num dimensions\" : 2,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 0,\n"
"     \"fundamental type\" : \"i\",\n"
"     \"name\" : \"N\",\n"
"     \"pointer level\" : 0,\n"
"     \"size\" : 32,\n"
"     \"type name\" : \"i32\"\n"
"    }\n"
"   ],\n"
"   \"name\" : \"stencil5p_2D\",\n"
"   \"partitioning\" : \"y\"\n"
"  }\n"
" ]\n"
"}\n"
"\n";

bool test0() {
	shared_ptr<AliasHandle> aliasH(new AliasHandle);

	// INIT ALIAS HANDLE GLOBAL VAR
	MEdevice dev;
	vector<MEdevice> vdev(2, dev); // set 2 gpus
	(*aliasH)[dev] = vdev;
	
	// READ DATABASE
	shared_ptr<const bsp_KernelInfo> kinfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0];

	// SET UP KERNEL ARGUMENTS
	MEdeviceptr input = (MEdeviceptr) 0;
	MEdeviceptr output = (MEdeviceptr) 1;
	MEfunction kernel = (MEfunction) 2;
	int N = 8;
	void* rawArgs[] = {&input, &output, &N};

	// SET UP KERNEL LAUNCH CONFIGURATION
	using T3 = AccFunc::Tuple3;
	T3 gridSize = make_tuple(2, 2, 1);
	T3 blockSize = make_tuple(4, 4, 1);
	
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
	auto rac = kl.getReadArgAccess(0);
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

void test1() {
	shared_ptr<AliasHandle> aliasH(new AliasHandle);

	// INIT ALIAS HANDLE GLOBAL VAR
	MEdevice dev;
	vector<MEdevice> vdev(3, dev); // set 3 gpus
	(*aliasH)[dev] = vdev;
	
	// READ DATABASE
	shared_ptr<const bsp_KernelInfo> kinfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0];

	// SET UP KERNEL ARGUMENTS
	MEdeviceptr input = (MEdeviceptr) 0;
	MEdeviceptr output = (MEdeviceptr) 1;
	MEfunction kernel = (MEfunction) 2;
	int N = 24;
	void* rawArgs[] = {&input, &output, &N};

	// SET UP KERNEL LAUNCH CONFIGURATION
	using T3 = AccFunc::Tuple3;
	T3 gridSize = make_tuple(4, 4, 1);
	T3 blockSize = make_tuple(6, 6, 1);
	
	// CREATE KERNEL LAUNCH OBJECT
	KernelLaunch kl(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH);

	// Now we should have the following configuration:
	//
	//   * GPU0 writes indices [i * 24 + 1, i * 24 + 22] for i =  1,..,11
	//   * GPU1 writes indices [i * 24 + 1, i * 24 + 22] for i = 12,..,17
	//   * GPU2 writes indices [i * 24 + 1, i * 24 + 22] for i = 18,..,22
	//
	//   * GPU0 reads indices [1, 22] and [24, 287] and [289, 310]
	//   * GPU1 reads indices [288, 431] and [265, 286] and [433, 454]
	//   * GPU2 reads indices [432, 551] and [553, 574] and [409, 430]
	//
	// Our array has a size of N * N = 576

	cout << "ARGS:" << endl;
	for (auto arg : kl.getArgs()) {
		cout << *arg << endl;
	}
	cout << endl;
	cout << "PARTITIONS:" << endl;
	for (auto part : kl.getPartitions()) {
		cout << *part << endl;
	}
	cout << endl;
	cout << "CALLING getReadArgAccess ON KERNEL OBJECT:" << endl;
	cout << endl;
	auto rac = kl.getReadArgAccess(0);
	auto wac = kl.getWriteArgAccess(1);
	cout << *rac << endl;
	cout << *wac << endl;

}

void test2() {
	shared_ptr<AliasHandle> aliasH(new AliasHandle);

	// INIT ALIAS HANDLE GLOBAL VAR
	MEdevice dev;
	vector<MEdevice> vdev(3, dev); // set 3 gpus
	(*aliasH)[dev] = vdev;
	
	// READ DATABASE
	shared_ptr<const bsp_KernelInfo> kinfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0];

	// SET UP KERNEL ARGUMENTS
	MEdeviceptr input = (MEdeviceptr) 0;
	MEdeviceptr output = (MEdeviceptr) 1;
	MEfunction kernel = (MEfunction) 2;
	int N = 24;
	void* rawArgs[] = {&input, &output, &N};

	// SET UP KERNEL LAUNCH CONFIGURATION
	using T3 = AccFunc::Tuple3;
	T3 gridSize = make_tuple(4, 4, 1);
	T3 blockSize = make_tuple(6, 6, 1);
	
	// CREATE KERNEL LAUNCH OBJECT
	auto kl0 = shared_ptr<KernelLaunch>(new KernelLaunch(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH));
	swap(input, output);
	KernelLaunch::all.push_back(kl0);
	cout << endl;
	cout << "CALLING getReadArgAccess ON KERNEL OBJECT the fist time:" << endl;
	cout << endl;
	auto rac = kl0->getReadArgAccess(0);
	auto wac = kl0->getWriteArgAccess(1);

	auto& racs = kl0->getReadArgAccesses();
	auto& wacs = kl0->getWriteArgAccesses();
	cout << "Looking at the " << racs.size() << " read arg accesses of the first kernel launch" << endl;
	int i = 0;
	for (auto& ptr : racs) {
		if (ptr == nullptr) {
			cout << "arg " << i << " is has nullptr arg access object" << endl;
		}
		++i;
	}
	cout << "Looking at the " << wacs.size() << " write arg accesses of the first kernel launch" << endl;
	i = 0;
	for (auto& ptr : wacs) {
		if (ptr == nullptr) {
			cout << "arg " << i << " is has nullptr arg access object" << endl;
		}
		++i;
	}
	

	auto kl1 = shared_ptr<KernelLaunch>(new KernelLaunch(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH));
	cout << "CALLING getReadArgAccess ON KERNEL OBJECT the second time:" << endl;
	auto rac1 = kl1->getReadArgAccess(0);
	auto wac1 = kl1->getWriteArgAccess(1);
	KernelLaunch::all.push_back(kl1);
	cout << "*kl0 == *kl1 = " << (*kl0 == *kl1) << endl;
	cout << "kl1->hasEqualArgAccess(*kl0) = " << kl1->hasEqualArgAccess(*kl0) << endl;
	cout << endl;
	cout << "kl0->getArgAccessCalcs() = " << kl0->getNumArgAccessCalcs() << endl;
	cout << "kl1->getArgAccessCalcs() = " << kl1->getNumArgAccessCalcs() << endl;

}

int main() {
	test0();
	//test1();
	//test2();
	return 0;
}

#endif
