#ifdef MEKONG_TEST

#include <iostream>
#include <algorithm>
#include <sstream>
#include <new> // bad_alloc
#include <iomanip>
#include <memory>
#include <stdexcept>

#include "dependency_resolution.h"
#include "kernel_launch.h"
#include "access_function.h"
#include "kernel_info.h"
#include "argument_type.h"
#include "argument.h"

using namespace std;
using namespace Mekong;

const char* bspAnalysisStr_nbody =
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
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_x[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"isl write map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_x[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl write params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"name\" : \"pos_x\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_y[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"isl write map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_y[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl write params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"name\" : \"pos_y\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_z[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"isl write map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_z[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl write params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"name\" : \"pos_z\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_vel_x[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"name\" : \"vel_x\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_vel_y[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"name\" : \"vel_y\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_vel_z[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\" ],\n"
"     \"name\" : \"vel_z\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 0,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"name\" : \"dt\",\n"
"     \"pointer level\" : 0,\n"
"     \"size\" : 32,\n"
"     \"type name\" : \"float\"\n"
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
"   \"name\" : \"updatePositions\",\n"
"   \"partitioning\" : \"x\"\n"
"  },\n"
"  \n"
"  {\n"
"   \"arguments\" : \n"
"   [\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z, N] -> { Stmt_cond_end[i0, i1, i2, i3] -> MemRef_masses[i3] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z and 0 <= i3 < N }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"name\" : \"masses\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_body_lr_ph[i0, i1, i2] -> MemRef_pos_x[i0] : N > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z; Stmt_for_body__TO__cond_end[i0, i1, i2, i3] -> MemRef_pos_x[i3] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z and 0 <= i3 < N }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"name\" : \"pos_x\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_body__TO__cond_end[i0, i1, i2, i3] -> MemRef_pos_y[i3] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z and 0 <= i3 < N; Stmt_for_body_lr_ph[i0, i1, i2] -> MemRef_pos_y[i0] : N > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"name\" : \"pos_y\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_body__TO__cond_end[i0, i1, i2, i3] -> MemRef_pos_z[i3] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z and 0 <= i3 < N; Stmt_for_body_lr_ph[i0, i1, i2] -> MemRef_pos_z[i0] : N > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"name\" : \"pos_z\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_end[i0, i1, i2] -> MemRef_vel_x[i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"isl write map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_end[i0, i1, i2] -> MemRef_vel_x[i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl write params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"name\" : \"vel_x\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_end[i0, i1, i2] -> MemRef_vel_y[i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"isl write map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_end[i0, i1, i2] -> MemRef_vel_y[i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl write params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"name\" : \"vel_y\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 32,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"isl read map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_end[i0, i1, i2] -> MemRef_vel_z[i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl read params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"isl write map\" : \"[size_x, size_y, size_z, N] -> { Stmt_for_end[i0, i1, i2] -> MemRef_vel_z[i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\",\n"
"     \"isl write params\" : [ \"size_x\", \"size_y\", \"size_z\", \"arg9\" ],\n"
"     \"name\" : \"vel_z\",\n"
"     \"num dimensions\" : 1,\n"
"     \"pointer level\" : 1,\n"
"     \"size\" : 0,\n"
"     \"type name\" : \"float addrspace(1)*\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 0,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"name\" : \"dt\",\n"
"     \"pointer level\" : 0,\n"
"     \"size\" : 32,\n"
"     \"type name\" : \"float\"\n"
"    },\n"
"    \n"
"    {\n"
"     \"element size\" : 0,\n"
"     \"fundamental type\" : \"f\",\n"
"     \"name\" : \"epsilon\",\n"
"     \"pointer level\" : 0,\n"
"     \"size\" : 32,\n"
"     \"type name\" : \"float\"\n"
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
"   \"name\" : \"updateSpeed\",\n"
"   \"partitioning\" : \"x\"\n"
"  }\n"
" ]\n"
"}\n"
"\n";

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

vector<shared_ptr<KernelLaunch>> KernelLaunch::all;

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
	shared_ptr<KernelLaunch> kl0(new KernelLaunch(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH));
	swap(input, output);
	shared_ptr<KernelLaunch> kl1(new KernelLaunch(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH));
	swap(input, output);
	shared_ptr<KernelLaunch> kl2(new KernelLaunch(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH));

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
	for (auto part : kl0->getPartitions()) {
		cout << *part << endl;
	}
	cout << endl;
	cout << "CALLING getReadArgAccess ON KERNEL OBJECT:" << endl;
	cout << endl;
	auto rac = kl0->getReadArgAccess(0);
	auto wac = kl0->getWriteArgAccess(1);
	cout << *rac << endl;
	cout << *wac << endl;
	
	cout << endl << endl;
	cout << "Going to create the depres object" << endl;
	DepResolution depres0(kl0, kl1, aliasH);
	cout << depres0 << endl;
	DepResolution depres1(kl0, kl2, aliasH);
	cout << depres1 << endl;
	return true;
}

bool test1() {
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
	int N = 1024;
	void* rawArgs[] = {&input, &output, &N};

	// SET UP KERNEL LAUNCH CONFIGURATION
	using T3 = AccFunc::Tuple3;
	T3 gridSize = make_tuple(64, 64, 1);
	T3 blockSize = make_tuple(16, 16, 1);
	
	// CREATE KERNEL LAUNCH OBJECT
	shared_ptr<KernelLaunch> kl0(new KernelLaunch(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH));
	swap(input, output);
	shared_ptr<KernelLaunch> kl1(new KernelLaunch(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH));
	swap(input, output);
	shared_ptr<KernelLaunch> kl2(new KernelLaunch(kernel, gridSize, blockSize, 0, rawArgs, kinfo, aliasH));

	cout << endl;
	cout << "PARTITIONS:" << endl;
	for (auto part : kl0->getPartitions()) {
		cout << *part << endl;
	}
	cout << endl;
	cout << "CALLING getReadArgAccess ON KERNEL OBJECT:" << endl;
	cout << endl;
	auto rac = kl0->getReadArgAccess(0);
	cout << *rac << endl;
	
	cout << endl << endl;
	cout << "Going to create the depres object" << endl;
	DepResolution depres0(kl0, kl1, aliasH);
	cout << depres0 << endl;
	DepResolution depres1(kl0, kl2, aliasH);
	cout << depres1 << endl;
	return true;
}

bool test_nbody() {

	cout << "# Testing n body dep res objects" << endl;
	cout << endl;

	shared_ptr<AliasHandle> aliasH(new AliasHandle);

	// INIT ALIAS HANDLE GLOBAL VAR
	MEdevice dev;
	vector<MEdevice> vdev(2, dev); // set 2 gpus
	(*aliasH)[dev] = vdev;
	
	// READ DATABASE
	auto kinfos = bsp_KernelInfo::createKInfos(bspAnalysisStr_nbody);

	for (auto kinfo : kinfos) {
		cout << "  - kernel name = " << kinfo->getName() << endl;
	}

	// INIT KERNEL FUNCTION
	MEfunction kernel_us = (MEfunction) 11;
	MEfunction kernel_up = (MEfunction) 12;

	// SET UP KERNEL ARGUMENTS
	int i = 0;
	vector<MEdeviceptr> m_pos_vel(7);
	for (auto& devptr : m_pos_vel) {
		devptr = (MEdeviceptr) i++;
	}

	int N = 12;
	double dt = 0.001;
	double epsilon = 0.001;

	void* rawArgs_us[] = {&m_pos_vel[0], // raw arguments for update speed kernel
	                      &m_pos_vel[1],
	                      &m_pos_vel[2],
	                      &m_pos_vel[3],
	                      &m_pos_vel[4],
	                      &m_pos_vel[5],
	                      &m_pos_vel[6],
	                      &dt,
	                      &epsilon,
	                      &N};

	void* rawArgs_up[] = {&m_pos_vel[1], // raw arguments for update positions kernel
	                      &m_pos_vel[2],
	                      &m_pos_vel[3],
	                      &m_pos_vel[4],
	                      &m_pos_vel[5],
	                      &m_pos_vel[6],
	                      &dt,
	                      &N};

	// SET UP KERNEL LAUNCH CONFIGURATION
	using T3 = AccFunc::Tuple3;
	T3 gridSize = make_tuple(4, 1, 1);   // 4 blocks with 3 threads -> total 12 threads
	T3 blockSize = make_tuple(3, 1, 1);  // if 2 gpus active each should get 2 blocks

	// CREATE KERNEL LAUNCH OBJECT
	shared_ptr<KernelLaunch> kl_us(new KernelLaunch(kernel_us, gridSize, blockSize, 0, rawArgs_us, kinfos[1], aliasH));
	shared_ptr<KernelLaunch> kl_up(new KernelLaunch(kernel_up, gridSize, blockSize, 0, rawArgs_up, kinfos[0], aliasH));
	cout << endl;
	cout << "## Kernel launches" << endl;
	cout << endl;
	cout << "  1. " << *kl_us << endl;
	cout << "  2. " << *kl_up << endl;
	cout << endl;
	cout << "## Partitions" << endl;
	cout << endl;
	cout << "1." << endl;
	for (auto part : kl_us->getPartitions()) {
		cout << *part << endl;
	}
	cout << endl;
	cout << "2." << endl;
	for (auto part : kl_up->getPartitions()) {
		cout << *part << endl;
	}
	cout << endl;
	cout << "## Arg Access objects" << endl;
	cout << endl;
	cout << "### Read accesses" << endl;
	cout << endl;

	auto r_us_pos_x = kl_us->getReadArgAccess(1);
	auto r_us_pos_y = kl_us->getReadArgAccess(2);
	auto r_us_pos_z = kl_us->getReadArgAccess(3);
	auto r_us_vel_x = kl_us->getReadArgAccess(4);
	auto r_us_vel_y = kl_us->getReadArgAccess(5);
	auto r_us_vel_z = kl_us->getReadArgAccess(6);

	cout << "r_us_pos_x = " << *r_us_pos_x << endl;
	cout << "r_us_pos_y = " << *r_us_pos_y << endl;
	cout << "r_us_pos_z = " << *r_us_pos_z << endl;
	cout << "r_us_vel_x = " << *r_us_vel_x << endl;
	cout << "r_us_vel_y = " << *r_us_vel_y << endl;
	cout << "r_us_vel_z = " << *r_us_vel_z << endl;
	cout << endl;

	auto r_up_pos_x = kl_up->getReadArgAccess(0);
	auto r_up_pos_y = kl_up->getReadArgAccess(1);
	auto r_up_pos_z = kl_up->getReadArgAccess(2);
	auto r_up_vel_x = kl_up->getReadArgAccess(3);
	auto r_up_vel_y = kl_up->getReadArgAccess(4);
	auto r_up_vel_z = kl_up->getReadArgAccess(5);

	cout << "r_up_pos_x = " << *r_up_pos_x << endl;
	cout << "r_up_pos_y = " << *r_up_pos_y << endl;
	cout << "r_up_pos_z = " << *r_up_pos_z << endl;
	cout << "r_up_vel_x = " << *r_up_vel_x << endl;
	cout << "r_up_vel_y = " << *r_up_vel_y << endl;
	cout << "r_up_vel_z = " << *r_up_vel_z << endl;

	cout << endl;
	cout << "### Write accesses" << endl;
	cout << endl;

	auto w_us_vel_x = kl_us->getWriteArgAccess(4);
	cout << "w_us_vel_x = " << *w_us_vel_x << endl;
	auto w_us_vel_y = kl_us->getWriteArgAccess(5);
	cout << "w_us_vel_y = " << *w_us_vel_y << endl;
	auto w_us_vel_z = kl_us->getWriteArgAccess(6);
	cout << "w_us_vel_z = " << *w_us_vel_z << endl;
	cout << endl;

	auto w_up_pos_x = kl_up->getWriteArgAccess(0);
	cout << "w_up_pos_x = " << *w_up_pos_x << endl;
	auto w_up_pos_y = kl_up->getWriteArgAccess(1);
	cout << "w_up_pos_y = " << *w_up_pos_y << endl;
	auto w_up_pos_z = kl_up->getWriteArgAccess(2);
	cout << "w_up_pos_z = " << *w_up_pos_z << endl;

	cout << endl;
	cout << "## Dep Res Objects" << endl;
	cout << endl;
	DepResolution depres(kl_up, kl_us, aliasH);
	cout << depres << endl;
	return true;
}

int main() {
//	test0();
//	test1();
	test_nbody();
	return 0;
}

#endif
