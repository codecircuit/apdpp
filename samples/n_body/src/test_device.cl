__kernel void updatePositions(__global float* pos_x,
                              __global float* pos_y,
                              __global float* pos_z,
                              __global const float* vel_x,
                              __global const float* vel_y,
                              __global const float* vel_z,
                              float dt,
                              int N) {

	int id = get_global_id(0);

	pos_x[id] += vel_x[id] * dt;
	pos_y[id] += vel_y[id] * dt;
	pos_z[id] += vel_z[id] * dt;
}

__kernel void updateSpeed(__global const float* masses,
                          __global const float* pos_x,
                          __global const float* pos_y,
                          __global const float* pos_z,
                          __global float* vel_x,
                          __global float* vel_y,
                          __global float* vel_z,
                          float dt,
                          float epsilon,
                          int N) {

	// Newton's gravitational constant
	// const float G = 1.0f;

	int id = get_global_id(0);

	// the body's acceleration
	float acc_x = 0;
	float acc_y = 0;
	float acc_z = 0;

	// temporary register
	float diff_x;
	float diff_y;
	float diff_z;
	float norm;
	int j;

	for (j = 0; j < N; ++j) {
		diff_x = pos_x[j] - pos_x[id];
		diff_y = pos_y[j] - pos_y[id];
		diff_z = pos_z[j] - pos_z[id];

		// to ensure a certain order of execution we write
		// the calculations in seperate lines. Keep in mind
		// that opencl does not define an operator precedence,
		// thus we have to ensure this by ourselves.
		norm = diff_x * diff_x;
		norm += diff_y * diff_y;
		norm += diff_z * diff_z;
		norm = sqrt(norm);
		norm = norm * norm * norm;
		norm = norm == 0 ? 0 : 1.0f / norm + epsilon;
		norm *= masses[j];

		acc_x += norm * diff_x;
		acc_y += norm * diff_y;
		acc_z += norm * diff_z;
	}

//	acc_x *= G;
//	acc_y *= G;
//	acc_z *= G;

	vel_x[id] += acc_x * dt;
	vel_y[id] += acc_y * dt;
	vel_z[id] += acc_z * dt;
}

/*#define BLOCKSIZE 512
__kernel void updateSpeed_shared(__global const float* masses,
                                 __global const float* pos_x,
                                 __global const float* pos_y,
                                 __global const float* pos_z,
                                 __global float* vel_x,
                                 __global float* vel_y,
                                 __global float* vel_z,
                                 float dt,
                                 float epsilon,
                                 int N,
                                 int blocksize) {

	// Newton's gravitational constant
	// const float G = 1.0f;

	__local float masses_sh[BLOCKSIZE];
	__local float pos_x_sh[BLOCKSIZE];
	__local float pos_y_sh[BLOCKSIZE];
	__local float pos_z_sh[BLOCKSIZE];

	int id = get_global_id(0);
	int id_sh = get_local_id(0);

	// the body's acceleration
	float acc_x = 0;
	float acc_y = 0;
	float acc_z = 0;

	// own position
	float pos_x_self = pos_x[id];
	float pos_y_self = pos_y[id];
	float pos_z_self = pos_z[id];

	// temporary register
	float diff_x;
	float diff_y;
	float diff_z;
	float norm;
	int j;
	int i;

	for (i = 0; i < N; i += blocksize) {
		// load data into shared memory
		masses_sh[id_sh] = masses[i + id_sh];
		pos_x_sh[id_sh] = pos_x[i + id_sh];
		pos_y_sh[id_sh] = pos_y[i + id_sh];
		pos_z_sh[id_sh] = pos_z[i + id_sh];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (j = 0; j < blocksize; ++j) {
			diff_x = pos_x_sh[j] - pos_x_self;
			diff_y = pos_y_sh[j] - pos_y_self;
			diff_z = pos_z_sh[j] - pos_z_self;

			// to ensure a certain order of execution we write
			// the calculations in seperate lines. Keep in mind
			// that opencl does not define an operator precedence,
			// thus we have to ensure this by ourselves.
			norm = diff_x * diff_x;
			norm += diff_y * diff_y;
			norm += diff_z * diff_z;
			norm = sqrt(norm);
			norm = norm * norm * norm;
			norm = norm == 0 ? 0 : 1.0f / norm + epsilon;
			norm *= masses_sh[j];

			acc_x += norm * diff_x;
			acc_y += norm * diff_y;
			acc_z += norm * diff_z;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

//	acc_x *= G;
//	acc_y *= G;
//	acc_z *= G;

	vel_x[id] += acc_x * dt;
	vel_y[id] += acc_y * dt;
	vel_z[id] += acc_z * dt;
}*/
