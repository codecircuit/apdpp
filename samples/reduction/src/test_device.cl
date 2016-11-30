__kernel void reduceKernel(__global int* in,
                           __global int* out) {

	int localId = get_local_id(0);
	int id = get_global_id(0);
	int blockSize = get_local_size(0);
	int s;
	for (s = 1; s < blockSize; s *= 2) {
		if (id % (2 * s) == 0) {
			in[id] += in[id + s];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (localId == 0) {
		out[id / blockSize] = in[id];
	}
}
