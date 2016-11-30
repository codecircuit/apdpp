// In PTX representation there are
// 4 multiplications
// 10 additions
// 1 subtraction
// --------------
// 15 in total
//
// and 5 ld.global instructions
__kernel void stencil5p_2D(__global float* in, __global float* out, int N) {
	float res;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int id;
	if (x > 0 && y > 0 && x < N - 1 && y < N - 1) {
		id = x + y * N;

		// the order of execution affects the result, as the operator
		// precedence is not clearly defined in opencl
		// res = in[id] + 0.24f * (-4.f * in[id] + in[id + 1] + in[id - 1] + in[id + N] + in[id - N]);
		res = in[id - N];
		res += in[id + N];
		res += in[id - 1];
		res += in[id + 1];
		res += -4.0f * in[id];
		res *= 0.24f;
		res += in[id];

		res = res > 127 ? 127 : res;
		res = res < 0 ? 0 : res;
		out[id] = res;
	}
}

/* 25 point stencil
 *
 *    0  1  2  3  4
 *  0 x  x  x  x  x
 *  1 x  x  x  x  x
 *  2 x  x  x  x  x
 *  3 x  x  x  x  x
 *  4 x  x  x  x  x
 * 
 * In PTX representation there are
 * 30 additions
 * 4 multiplications
 * 2 subtractions
 * -----------------
 * 36 in total
 * 
 * And 25 ld.global instructions.
 * If one block has a size of 32x32, it will load 12% more elements from
 * global memory
 */
__kernel void stencil25p_2D(__global float* in, __global float* out, int N) {
	float res;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int id;
	if (x > 1 && y > 1 && x < N - 2 && y < N - 2) {
		id = x + y * N;

		// the order of execution affects the result, as the operator
		// precedence is not clearly defined in opencl
		res = in[id] + 0.05f * (-4.f * in[id]
		                             + in[id + 1]
		                             + in[id - 1]
		                             + in[id + N]
		                             + in[id - N]
		                             + in[id + 2]
		                             + in[id - 2]
		                             + in[id - N - N]
		                             + in[id - N - 1]
		                             + in[id - N + 1]
		                             + in[id - N + 2]
		                             + in[id - N - 2]
		                             + in[id + N - 1]
		                             + in[id + N + 1]
		                             + in[id + N + 2]
		                             + in[id + N - 2]
		                             + in[id + N + N - 1]
		                             + in[id + N + N + 1]
		                             + in[id + N + N + 2]
		                             + in[id + N + N - 2]
		                             + in[id - N - N - 1]
		                             + in[id - N - N + 1]
		                             + in[id - N - N + 2]
		                             + in[id - N - N - 2]
		                             + in[id + N + N]);

		res = res > 127 ? 127 : res;
		res = res < 0 ? 0 : res;
		out[id] = res;
	}
}

void f(__global int* acc, int N) {
	if (N > 5) {
		*acc = 7;
	}
	else {
		*acc = 2;
	}
}
