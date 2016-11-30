#define TILESIZE 32

/*__kernel void mat_mul_blocked(__global float* C, 
                              __global float* A, 
                              __global float* B,
                              int N) {
	int tx = get_global_id(0); 
	int ty = get_global_id(1);
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	__local float Al[TILESIZE][TILESIZE];
	__local float Bl[TILESIZE][TILESIZE];

	int m;
	float acc = 0;
	for (m = 0; m < N/TILESIZE; ++m) {
		Al[ly][lx] = A[m*TILESIZE+lx + ty*N];
		Bl[ly][lx] = B[tx + (m*TILESIZE+ly)*N];
		barrier(CLK_LOCAL_MEM_FENCE);
		int j;
		for (j = 0; j < TILESIZE; j++) {
			acc += Al[ly][j] * Bl[j][lx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	C[ty * N + tx] = acc;
}*/

__kernel void mat_mul(__global float* C, 
                      __global float* A, 
                      __global float* B,
                      int N) {
	int tx = get_global_id(0); 
	int ty = get_global_id(1);

	int m;
	float acc = 0;
	for (m = 0; m < N; ++m) {
		acc += A[tx * N + m] * B[m * N + ty];
	}

	C[tx * N + ty] = acc;
}
