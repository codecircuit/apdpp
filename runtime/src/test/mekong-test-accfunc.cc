#ifdef MEKONG_TEST

#include <iostream>
#include <sstream>
#include <new> // bad_alloc
#include <iomanip>
#include <memory>
#include <stdexcept>

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
"\n"
;


const char islReadMapStr[] = "[size_x, size_y, size_z, N] -> { Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, 1 + i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[1 + i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[-1 + i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z; Stmt_if_then[i0, i1, i2] -> MemRef_in[i1, -1 + i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z }";
const char* islReadParams[] = { "size_x", "size_y", "size_z", "arg2" };
const char islWriteMapStr[] = "[size_x, size_y, size_z, N] -> { Stmt_if_then[i0, i1, i2] -> MemRef_out[i1, i0] : size_x > 0 and size_y > 0 and size_z > 0 and 0 < i0 <= -2 + N and i0 < size_x and 0 < i1 <= -2 + N and i1 < size_y and 0 <= i2 < size_z }";
const char* islWriteParams[] = { "size_x", "size_y", "size_z", "arg2" };

bool test_bsp_KernelInfo() {

	shared_ptr<const bsp_KernelInfo> kinfo = bsp_KernelInfo::createKInfos(bspAnalysisStrTEST)[0];
	auto af = kinfo->getAccFunc(0);
	MEdeviceptr input = (MEdeviceptr) 0;
	MEdeviceptr output = (MEdeviceptr) 1;
	int N = 24;
	void* rawArgs[] = {&input, &output, &N};
	auto args = KernelArg::createArgs(kinfo->getArgTypes(), rawArgs);

	using T3 = AccFunc::Tuple3;
	T3 gridSize = make_tuple(4, 4, 1);
	T3 blockSize = make_tuple(6, 6, 1);
	isl_ctx* ctx = isl_ctx_alloc();
	isl_union_map* umap = af->getReadIslMap(ctx, &args, &gridSize, &blockSize);
//	isl_union_map* wmap = af->getWriteIslMap(ctx, &args, &gridSize, &blockSize); // error as arg is not written
	auto indices = af->getReadAcc(make_tuple(16, 16, 0), &args, &gridSize, &blockSize);
	for (auto index : indices) {
		cout << "index = " << index << endl;
	}
	cout << endl;
	indices = af->getReadAcc(make_tuple(1, 1, 0), &args, &gridSize, &blockSize);
	for (auto index : indices) {
		cout << "index = " << index << endl;
	}
	isl_union_map_free(umap);
	isl_ctx_free(ctx);
	return true;
}

int main() {
	cout << endl;
	// TEST BSP KERNEL INFO CLASS
	//cout << "  * Testing bsp kernel info class";
	if (test_bsp_KernelInfo()) {
	//	cout << " [OK]" << endl;
	}
	else {
	//	cout << " [FAILED]" << endl;
	}

	cout << endl;
	return 0;
}

#endif
