//
//  PURPOSE OF THIS FILE: This file provides several test cases
//                        of Mekong's runtime part. You can compile
//                        this file with the -DMEKONG_TEST command
//                        line flag and execute the binary to see
//                        the test results.
//

#ifdef MEKONG_TEST

#include <iomanip>
#include <iostream>
#include <memory>
#include <new> // bad_alloc
#include <sstream>
#include <stdexcept>

#include "access_function.h"
#include "alias_handle.h"
#include "argument.h"
#include "argument_type.h"
#include "dependency_resolution.h"
#include "information.h"
#include "kernel_info.h"
#include "kernel_launch.h"
#include "mekong-cuda.h"
#include "mekong-wrapping.h"
#include "memory_copy.h"
#include "partition.h"
#include "virtual_buffer.h"

using namespace std;
using namespace Mekong;

const char* userConfigTEST =
"LOG_LEVEL = 3\n"
"PARTITIONING = \n"
"CHECK_DEVICE_LIMITS = \n"
"MAKE_REPORT = true\n"
"USER_OPTION = this_is_an_option\n";

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

bool test_bsp_KernelInfo() {
	
	// Kernel: stencil5p_2D
	
	auto kInfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0];
	auto argTypes = kInfo->getArgTypes();

	int N = 16;
	MEdeviceptr d_ptr0 = (MEdeviceptr) 0;
	MEdeviceptr d_ptr1 = (MEdeviceptr) 1;
	const void* rawArgs[] = { &d_ptr0, &d_ptr1, &N};

	vector<shared_ptr<const KernelArg>> args;
	vector<size_t> dimSizes(1, N);
	args.push_back(shared_ptr<const KernelArg>(new KernelArg(argTypes[0], rawArgs[0], dimSizes)));
	args.push_back(shared_ptr<const KernelArg>(new KernelArg(argTypes[1], rawArgs[1], dimSizes)));
	args.push_back(shared_ptr<const KernelArg>(new KernelArg(argTypes[2], rawArgs[2])));

	shared_ptr<const AccFunc> af(kInfo->getAccFunc(0));

	// number refers to the index in the isl space
	// [size_x, size_y, size_z, N] -> {...}
	//    0       1       2     3
	if (af->getReadParam(3, &args) != N) { 
		cout << "Got wrong read param" << endl;
		return false;
	}
	
	auto tid = make_tuple(2,2,0);
	Partition::Tuple3 grid = make_tuple(8, 8, 1);
	Partition::Tuple3 block = make_tuple(2, 2, 1);
	vector<size_t> reads = af->getReadAcc(tid, &args, &grid, &block);
	vector<size_t> writes = af->getWriteAcc(tid, &args, &grid, &block);

	if (reads.size() != 5) {
		cout << "reads.size = " << reads.size() << endl;
		cout << "but it should be 5" << endl;
		return false;
	}
	if (!writes.empty()) {
		return false;
	}

	bool check[] = {false, false, false, false, false};
	for (size_t id : reads) {
		switch (id) {
			case 18: // element above
				check[0] = true;
				break;
			case 34: // element on
				check[1] = true;
				break;
			case 50: // element below
				check[2] = true;
				break;
			case 33: // element left
				check[3] = true;
				break;
			case 35: // element right
				check[4] = true;
				break;
		}
	}
	if (!check[0] || !check[1] || !check[2] || !check[3] || !check[4]) {
		return false;
	}
	return true;
}

bool test_KernelArg() {

	auto kInfos(bsp_KernelInfo::createKInfos(bspAnalysisStrTEST));

	auto argTypes = kInfos[0]->getArgTypes();
	int N = 100;
	int notN = 101;
	MEdeviceptr d_ptr0;
	MEdeviceptr d_ptr1;
	MEdeviceptr d_ptr00 = d_ptr0;
	MEdeviceptr d_ptr2;
	const void* rawArgs[] = { &d_ptr0, &d_ptr1, &N};
	vector<size_t> dimSizes(1, N);
	KernelArg arg0(argTypes[0], rawArgs[0], dimSizes);
	KernelArg arg00(argTypes[0], rawArgs[0], dimSizes);
	KernelArg arg1(argTypes[1], rawArgs[1], dimSizes);
	KernelArg arg2(argTypes[2], rawArgs[2]);

	if (arg0 == arg2) {
		return false;
	}
	if (arg0 != arg00) {
		return false;
	}
	if (arg0 != d_ptr0 || arg0 != d_ptr00 || arg2 != N) {
		return false;
	}
	if (arg2 == notN || arg0 == arg1) {
		return false;
	}
	if (arg0.asDevPtr() != d_ptr0) {
		return false;
	}
	if (arg1.asDevPtr() != d_ptr1) {
		return false;
	}
	
	return true;
}

bool test_Partition() {

	shared_ptr<AliasHandle> aliasH(new AliasHandle);

	// INIT ALIAS HANDLE GLOBAL VAR
	MEdevice dev;
	vector<MEdevice> vdev(7, dev);
	(*aliasH)[dev] = vdev;

	using T3 =  Partition::Tuple3;
	size_t N = 100;
	size_t tiling = 10;
	T3 grid = make_tuple(N, N, 1);
	T3 block = make_tuple(tiling, tiling, 1);
	shared_ptr<const Partitioning> parting(new Partitioning("x"));
	auto partitions = Partition::createPartitions(grid, block, aliasH, parting);
	size_t sum = 0;
	
	for (auto p : partitions) {
		sum += get<0>(p->getGrid());
	}
	if (sum != N) {
		return false;
	}
	return true;
}

bool test_MemCpyHtoD() {

	// INIT DEVICE ENVIRONMENT
	MEresult err = meInit(0);
	int devCount;
	err &= meDeviceGetCount(&devCount);
	if (devCount < 1) {
		return false;
	}
	vector<MEdevice> devs(devCount);
	vector<MEcontext> ctxs(devCount);
	
	unsigned short gpu = 0;
	for (auto& dev : devs) {
		err &= meDeviceGet(&dev, 0);
		err &= meCtxCreate(&ctxs[gpu], 0, dev);
		err &= meCtxPushCurrent(ctxs[gpu++]);
		err &= meCtxPopCurrent(0);
	}

	// ALLOC MEMORY ON DEVICE
	vector<MEdeviceptr> dptrs(devCount);
	size_t N = 1000;
	size_t size = N * sizeof(int);
	gpu = 0;
	for (auto& dptr : dptrs) {
		err &= meCtxPushCurrent(ctxs[gpu++]);
		err &= meMemAlloc(&dptr, size);
		err &= meCtxPopCurrent(0);
	}

	// INIT DATA ON HOST
	int* data;
	int* fromGpu;
	try {
		data = new int[N];
		fromGpu = new int[N];
	}
	catch (bad_alloc& ba) {
		cerr << "bad_alloc caught: " << ba.what() << endl;
	}
	for (size_t i = 0; i < N; i++) {
		data[i] = i;
	}

	// INIT GLOBAL VARIABLES
	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	(*aliasH)[devs[0]] = devs;
	(*aliasH)[ctxs[0]] = ctxs;
	(*aliasH)[dptrs[0]] = dptrs;

	// CREATE A BROADCAST AND EXECUTE IT
	auto broadC = MemCpyHtoD::createBroadcast(dptrs[0], data, size, aliasH);
	err &= broadC->exec();

	// VERIFY DATA
	gpu = 0;
	for (auto& ctx : ctxs) {
		err &= meCtxPushCurrent(ctx);
		err &= meMemcpyDtoH(fromGpu, dptrs[gpu], size);
		for (size_t i = 0; i < N; ++i) {
			if (fromGpu[i] != data[i]) {
				delete[] data;
				delete[] fromGpu;
				return false;
			}
		}
		err &= meCtxPopCurrent(nullptr);
		++gpu;
	}

	for (auto& ctx : ctxs) {
		err &= meCtxDestroy(ctx);
	}

	if (!err.isSuccess()) {
		delete[] data;
		delete[] fromGpu;
		return false;
	}

	delete[] data;
	delete[] fromGpu;
	return true;
}

bool test_MemCpyDtoH() {

	// INIT DEVICE ENVIRONMENT
	MEresult err = meInit(0);
	int devCount;
	err &= meDeviceGetCount(&devCount);
	if (devCount < 1) {
		return false;
	}
	vector<MEdevice> devs(devCount);
	vector<MEcontext> ctxs(devCount);
	
	unsigned short gpu = 0;
	for (auto& dev : devs) {
		err &= meDeviceGet(&dev, 0);
		err &= meCtxCreate(&ctxs[gpu], 0, dev);
		err &= meCtxPushCurrent(ctxs[gpu++]);
		err &= meCtxPopCurrent(0);
	}

	// ALLOC MEMORY ON DEVICE
	vector<MEdeviceptr> dptrs(devCount);
	size_t N = 50;
	size_t size = N * sizeof(int);
	gpu = 0;
	for (auto& dptr : dptrs) {
		err &= meCtxPushCurrent(ctxs[gpu++]);
		err &= meMemAlloc(&dptr, size);
		err &= meCtxPopCurrent(0);
	}

	// INIT DATA ON HOST
	int* data;
	int* fromGpu;
	try {
		data = new int[N];
		fromGpu = new int[N];
	}
	catch (bad_alloc& ba) {
		cerr << "bad_alloc caught: " << ba.what() << endl;
	}
	for (size_t i = 0; i < N; i++) {
		data[i] = N - i;
	}

	// INIT GLOBAL VARIABLES
	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	(*aliasH)[devs[0]] = devs;
	(*aliasH)[ctxs[0]] = ctxs;
	(*aliasH)[dptrs[0]] = dptrs;

	// CREATE A BROADCAST AND EXECUTE IT
	auto broadC = MemCpyHtoD::createBroadcast(dptrs[0], data, size, aliasH);
	err &= broadC->exec();

	// VERIFY DATA
	for (gpu = 0; gpu < aliasH->getNumDev(); ++gpu) {
		vector<MemSubCopy> subcpys;
		MemSubCopy sc;
		sc.size = size;
		sc.src = gpu;
		sc.dst = -1;
		sc.to = 0;
		sc.from = 0;
		subcpys.push_back(sc);
		shared_ptr<const vector<MemSubCopy>> pattern(new vector<MemSubCopy>(move(subcpys)));
		MemCpyDtoH cpy(fromGpu, dptrs[0], pattern, aliasH);
		for (size_t i = 0; i < N; ++i) {
			fromGpu[i] = 0;
		}

		err &= cpy.exec();

		for (size_t i = 0; i < N; ++i) {
			if (fromGpu[i] != data[i]) {
				delete[] data;
				delete[] fromGpu;
				for (auto& ctx : ctxs) {
					meCtxDestroy(ctx);
				}
				return false;
			}
		}
	}
	
	for (auto& ctx : ctxs) {
		err &= meCtxDestroy(ctx);
	}

	if (!err.isSuccess()) {
		delete[] data;
		delete[] fromGpu;
		return false;
	}

	delete[] data;
	delete[] fromGpu;
	return true;
}

bool test_MemCpyDtoD() {

	// INIT DEVICE ENVIRONMENT
	MEresult err = meInit(0);
	int devCount;
	err &= meDeviceGetCount(&devCount);
	if (devCount < 2) {
		return false;
	}
	vector<MEdevice> devs(devCount);
	vector<MEcontext> ctxs(devCount);
	
	unsigned short gpu = 0;
	for (auto& dev : devs) {
		err &= meDeviceGet(&dev, 0);
		err &= meCtxCreate(&ctxs[gpu], 0, dev);
		err &= meCtxPushCurrent(ctxs[gpu++]);
		err &= meCtxPopCurrent(0);
	}

	// ALLOC MEMORY ON DEVICE
	vector<MEdeviceptr> dptrs(devCount);
	size_t N = 50;
	size_t size = N * sizeof(int);
	gpu = 0;
	for (auto& dptr : dptrs) {
		err &= meCtxPushCurrent(ctxs[gpu++]);
		err &= meMemAlloc(&dptr, size);
		err &= meCtxPopCurrent(0);
	}

	// INIT DATA ON HOST
	int* data;
	int* fromGpu;
	try {
		data = new int[N];
		fromGpu = new int[N];
	}
	catch (bad_alloc& ba) {
		cerr << "bad_alloc caught: " << ba.what() << endl;
	}

	// INIT GLOBAL VARIABLES
	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	(*aliasH)[devs[0]] = devs;
	(*aliasH)[ctxs[0]] = ctxs;
	(*aliasH)[dptrs[0]] = dptrs;
	
	// VERIFY DATA
	for (gpu = 0; gpu < aliasH->getNumDev(); ++gpu) {
		for (size_t i = 0; i < N; i++) {
			data[i] = gpu;
		}
		vector<MemSubCopy> subc;
		MemSubCopy msc;
		msc.size = size;
		msc.src = -1;
		msc.dst = gpu;
		msc.to = 0;
		msc.from = 0;
		subc.push_back(msc);
		shared_ptr<const vector<MemSubCopy>> pat(new vector<MemSubCopy>(move(subc)));
		MemCpyHtoD giveGpuData(dptrs[0], data, pat, aliasH);
		err &= giveGpuData.exec();

		// CREATE A BROADCAST FROM ONE GPU TO THE OTHERS
		auto broadC = MemCpyDtoD::createBroadcast(dptrs[0], size, aliasH, gpu);
		err &= broadC->exec();
		
		// VERIFY RESULTS ON EVERY OTHER GPU
		for (unsigned short otherGpu = 0; otherGpu < aliasH->getNumDev(); ++otherGpu) {
			if (otherGpu != gpu) {
				vector<MemSubCopy> subcpys;
				MemSubCopy sc;
				sc.size = size;
				sc.src = otherGpu;
				sc.dst = -1;
				sc.to = 0;
				sc.from = 0;
				subcpys.push_back(sc);
				shared_ptr<const vector<MemSubCopy>> pattern(new vector<MemSubCopy>(move(subcpys)));
				MemCpyDtoH cpy(fromGpu, dptrs[0], pattern, aliasH);

				// RESET DESTINATION ARRAY
				for (size_t i = 0; i < N; ++i) {
					fromGpu[i] = 99;
				}

				err &= cpy.exec();

				for (size_t i = 0; i < N; ++i) {
					if (fromGpu[i] != gpu) {
						delete[] data;
						delete[] fromGpu;
						for (auto& ctx : ctxs) {
							meCtxDestroy(ctx);
						}
						return false;
					}
				}
			}
		} // other gpu loop
	} // gpu loop

	delete[] data;
	delete[] fromGpu;
	for (auto& ctx : ctxs) {
		meCtxDestroy(ctx);
	}
	return true;
}

bool test_DepResolution() {
	{ // scope begin

	// INIT GLOBAL VARIABLES
	auto kInfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0]; // take the 2D stencil as test case
	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	auto info = shared_ptr<Info>(new Info(userConfigTEST));
	MEdevice dev;
	vector<MEdevice> vdev(3, dev);
	(*aliasH)[dev] = vdev;

	// INIT KERNEL ARGS
	MEdeviceptr devptr0 = (MEdeviceptr) 0;
	MEdeviceptr devptr1 = (MEdeviceptr) 1;
	int N = 16;
	void* rawArgs[] = { &devptr0, &devptr1, &N };
	
	// INTIT GRID
	MEfunction f = (MEfunction) 2;
	// we create a grid 16x16 = 256 threads in total
	Partition::Tuple3 grid = make_tuple(8, 8, 1);
	Partition::Tuple3 block = make_tuple(2, 2, 1);
	size_t shMem = 1024;

	shared_ptr<KernelLaunch> launch0(new KernelLaunch(f, grid, block, shMem, rawArgs, kInfo, aliasH));
	rawArgs[0] = &devptr1;
	rawArgs[1] = &devptr0;
	shared_ptr<KernelLaunch> launch1(new KernelLaunch(f, grid, block, shMem, rawArgs, kInfo, aliasH));

	DepResolution depRes(launch0, launch1, aliasH);

	// CREATE A PRIORI MEM SUB COPY OBJECTS
	// ====================================
	//
	// As we know that we access two float arrays we know a priori the Bytes
	// which are accessed. Moreover the data will lay on the same starting
	// Byte on all GPUs.

	vector<MemSubCopy> correctCopies;
	MemSubCopy scpy;
	scpy.src= 0; scpy.dst= 1; scpy.from= 324; scpy.to= 324; scpy.size= 56 ; correctCopies.push_back(scpy);
	scpy.src= 1; scpy.dst= 0; scpy.from= 388; scpy.to= 388; scpy.size= 56 ; correctCopies.push_back(scpy);
	scpy.src= 1; scpy.dst= 2; scpy.from= 708; scpy.to= 708; scpy.size= 56 ; correctCopies.push_back(scpy);
	scpy.src= 2; scpy.dst= 1; scpy.from= 772; scpy.to= 772; scpy.size= 56 ; correctCopies.push_back(scpy);


	const auto& memcpys = depRes.getMemCpys();
	bool found = false;

	for (const auto& subcpy : *memcpys[0]->getPattern()) {
		for (auto& corSubCpy : correctCopies) {
			if (subcpy == corSubCpy) {
				found = true;
			}
		}
		if (!found) {
			cout << "could not find memsubcpy " << subcpy << endl;
			return false;
		}
		else {
			found = false;
		}
	}
	} // scope end
	{ // scope begin

	// INIT GLOBAL VARIABLES
	auto kInfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0]; // take the stencil4p_2D as test case
	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	auto info = shared_ptr<Info>(new Info(userConfigTEST));
	MEdevice dev;
	vector<MEdevice> vdev(4, dev); // add 4 devices
	(*aliasH)[dev] = vdev;

	// INIT KERNEL ARGS
	MEdeviceptr devptr0 = (MEdeviceptr) 0;
	MEdeviceptr devptr1 = (MEdeviceptr) 1;
	int N = 16;
	void* rawArgs[] = { &devptr0, &devptr1, &N };
	
	// INTIT GRID
	MEfunction f = (MEfunction) 2;
	Partition::Tuple3 grid = make_tuple(8, 8, 1);
	Partition::Tuple3 block = make_tuple(2, 2, 1);
	size_t shMem = 1024;

	shared_ptr<KernelLaunch> launch0(new KernelLaunch(f, grid, block, shMem, rawArgs, kInfo, aliasH));
	rawArgs[0] = &devptr1;
	rawArgs[1] = &devptr0;
	shared_ptr<KernelLaunch> launch1(new KernelLaunch(f, grid, block, shMem, rawArgs, kInfo, aliasH));

	DepResolution depRes(launch0, launch1, aliasH);

	vector<vector<bool>> pairs(4, vector<bool>(4, false));
	for (const auto& memcpy : depRes.getMemCpys()) {
		for (const auto& subcpy : *memcpy->getPattern()) {
			pairs[subcpy.src][subcpy.dst] = true;
		}
	}

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (pairs[i][j] && !pairs[j][i]) {
				return false;
			}
		}
	}

	} // scope end

	return true;
}

bool test_KernelLaunch() {

	// INIT GLOBAL VARIABLES
	auto kInfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0]; // stencil5p_2D
	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	auto info = shared_ptr<Info>(new Info(userConfigTEST));
	MEdevice dev;
	vector<MEdevice> vdev(7, dev);
	(*aliasH)[dev] = vdev;

	// INIT KERNEL ARGS
	MEdeviceptr devptr0;
	MEdeviceptr devptr1;
	MEdeviceptr devptr2;
	int N = 20;
	void* rawArgs[] = { &devptr0, &devptr1, &N };
	
	// INTIT GRID
	MEfunction f = (MEfunction) 2;
	MEfunction ff = (MEfunction) 4;
	Partition::Tuple3 grid = make_tuple(120, 100, 1);
	Partition::Tuple3 block = make_tuple(32, 32, 1);
	size_t shMem = 1024;

	KernelLaunch launch0(f, grid, block, shMem, rawArgs, kInfo, aliasH);
	KernelLaunch launch1(f, grid, block, shMem, rawArgs, kInfo, aliasH);
	KernelLaunch launch2(ff, grid, block, shMem, rawArgs, kInfo, aliasH);
	N = 30;
	KernelLaunch launch3(f, grid, block, shMem, rawArgs, kInfo, aliasH);
	if (launch0 == launch2) {
		cout << "kernel launch inequality failed" << endl;
		return false;
	}
	if (launch0 != launch1) {
		cout << "kernel launch inequality failed" << endl;
		return false;
	}
	if (launch3 == launch0) {
		cout << "kernel launch inequality failed" << endl;
		return false;
	}
	
	// CHECK THREAD ID TO GPU MAPPING
	Partition::Tuple3 p0 = make_tuple(0, 0, 0);
	if (launch0.getGPU(p0) != 0) {
		return false;
	}
	Partition::Tuple3 p1 = make_tuple(120*32 - 1, 100*32 - 1, 0);
	if (launch0.getGPU(p1) != 6) {
		return false;
	}
	
	return true;
}

bool test_ArgAccess() {
	// INIT GLOBAL VARIABLES
	auto kInfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0]; // take the stencil4p_2D as test case
	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	auto info = shared_ptr<Info>(new Info(userConfigTEST));
	MEdevice dev;
	vector<MEdevice> vdev(3, dev);
	(*aliasH)[dev] = vdev;

	// INIT KERNEL ARGS
	MEdeviceptr devptr0 = (MEdeviceptr) 0;
	MEdeviceptr devptr1 = (MEdeviceptr) 1;
	int N = 16;
	void* rawArgs[] = { &devptr0, &devptr1, &N };
	
	// INTIT GRID
	MEfunction f = (MEfunction) 2;
	auto grid = make_tuple(8, 8, 1);
	auto block = make_tuple(2, 2, 1);
	size_t shMem = 1024;

	KernelLaunch launch(f, grid, block, shMem, rawArgs, kInfo, aliasH);
	auto parts = launch.getPartitions();

	// CHECK GET ARGUMENT ACCESS FUNCTION

	auto readMap0 = launch.getReadArgAccess(0)->getMap();
	//auto writeMap0 = launch.getWriteArgAccess(0)->getMap();
	//auto readMap1 = launch.getReadArgAccess(1)->getMap();
	auto writeMap1 = launch.getWriteArgAccess(1)->getMap();
	
	// TODO add reasonable tests here

	return true;
}

bool test_createPartitions() {
	// INIT GLOBAL VARIABLES
	shared_ptr<AliasHandle> aliasH(new AliasHandle);
	MEdevice dev;
	vector<MEdevice> vdev(2, dev);
	(*aliasH)[dev] = vdev;
	auto orgGrid = make_tuple(1024, 1, 1);
	auto orgBlock = make_tuple(1024, 1, 1);
	auto partitioning = shared_ptr<const Partitioning>(new
			Partitioning("x"));
	auto parts = Partition::createPartitions(orgGrid, orgBlock, aliasH, partitioning);

	if (parts.size() != 2) {
		return false;
	}
	if (get<0>(parts[0]->getGrid()) != 512 ||
	    get<1>(parts[0]->getGrid()) != 1 ||
		get<2>(parts[0]->getGrid()) != 1) {
		return false;
	}
	if (get<0>(parts[0]->getBlock()) != 1024 ||
	    get<1>(parts[0]->getBlock()) != 1 ||
		get<2>(parts[0]->getBlock()) != 1) {
		return false;
	}
	return true;
}

int main() {

	cout << endl;
	cout << "Mekong Wrapping Infrastructure Test" << endl;
	cout << "===================================" << endl;
	cout << endl;

	// TEST KERNEL ARGUMENT CLASS
	cout << "  * Testing kernel argument class";
	if (test_KernelArg()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	// TEST KERNEL LAUNCH CLASS
	cout << "  * Testing kernel launch class";
	if (test_KernelLaunch()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	// TEST PARTITION CLASS
	cout << "  * Testing partition class";
	if (test_Partition()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}
	
	// TEST BSP KERNEL INFO CLASS
	cout << "  * Testing bsp kernel info class";
	if (test_bsp_KernelInfo()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	// TEST DEPENDENCY RESOLVE CLASS
	cout << "  * Testing dependency resolve class";
	if (test_DepResolution()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	// TEST ARGUMENT ACCESS
	cout << "  * Testing argument access class";
	if (test_ArgAccess()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	// TEST HOST TO DEVICE MEMORY COPY
	cout << "  * Testing memory copy HtoD class";
	bool last;
	if ((last = test_MemCpyHtoD())) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	// TEST DEVICE TO HOST MEMORY COPY
	cout << "  * Testing memory copy DtoH class";
	if (last && test_MemCpyDtoH()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	// TEST DEVICE TO DEVICE MEMORY COPY
	cout << "  * Testing memory copy DtoD class";
	if (last && test_MemCpyDtoD()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	// TEST PARTITION CREATION
	cout << "  * Testing partition creation";
	if (test_createPartitions()) {
		cout << " [OK]" << endl;
	}
	else {
		cout << " [FAILED]" << endl;
	}

	cout << endl;
	return 0;
}

#endif
