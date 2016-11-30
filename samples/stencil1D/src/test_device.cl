__kernel void stencil5p_1D(__global float* in, __global float* out, int N) {
	int id = get_global_id(0);
	float res;
	res = 0;

	if (id > N && id % N != 0 && id % (N - 1) != 0 && id < (N - 1) * N) {

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
