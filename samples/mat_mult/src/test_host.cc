//
// MAT MUL CODE
//

#include <iostream>
#include <algorithm> // std::max_element
#include <cmath>     // std::abs
#include <iomanip>
#include <cuda.h>
#include <string>
#include <cmath>
#include <chrono>

using namespace std;
using Clock = chrono::high_resolution_clock;
using Duration = chrono::duration<double>;

template<class IntType>
void commandLineGetInt(IntType* p, const std::string& key, int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == key) {
			*p = std::stoll(argv[i+1]);
			break;
		}
	}
}

void commandLineGetFloat(float* p, const std::string& key, int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == key) {
			*p = std::stof(argv[i+1]);
			break;
		}
	}
}

void commandLineGetString(string* s, const std::string& key, int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == key) {
			s->assign(argv[i + 1]);
			break;
		}
	}
}

void commandLineGetBool(bool* p, const std::string& key, int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == key) {
			*p = true;
			break;
		}
	}
}

bool commandLineGetBool(const std::string& key, int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == key) {
			return true;
			break;
		}
	}
	return false;
}

//! Prints an array of length \param N with numerical \param precision and
//! a row length of \param linebreak. If \param linebreak is smaller 1 the
//! array is assumed to be one dimensional.
template<class ArrayT>
void printArray(ArrayT p, int N, int linebreak = -1, int precision = 4) {
	auto maximum = *max_element(p, p + N);
	auto minimum = *min_element(p, p + N);
	maximum = max(abs(maximum), abs(minimum));
	unsigned leadingDig = 1;
	while (maximum / pow(10.0, leadingDig) >= 1) {
		++leadingDig;
	}
	for (int i = 0; i < N - 1; ++i) {
		if (linebreak == 1) {
			cout << setprecision(precision) << setw(precision + leadingDig + 2) << left << p[i] << endl;
		}
		else if ((i + 1) % linebreak == 0 && i != 0 && linebreak >= 1) {
			cout << setprecision(precision) << setw(precision + leadingDig + 2) << left << p[i] << endl;
		}
		else {
			cout << setprecision(precision) << setw(precision + leadingDig + 2) << left << p[i] << ' ';
		}
	}
	if (linebreak >= 1) {
		cout << p[N - 1] << endl;
	}
	else {
		cout << p[N - 1];
	}
}

void mat_mul_cpu(float *C, float *A, float *B, int N) {
	float *Bt = new float [N * N];
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			Bt[j*N + i] = B[i*N + j];
		}
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			float acc = 0;
			for (int k = 0; k < N; ++k) {
				acc += A[i*N+k] * Bt[j*N+k];
			}
			C[i*N + j] = acc;
		}
	}
	delete[] Bt;
}

// gives back the relative number of different elements,
// thus the result will be between 0 and 1.
template<class ArrayA_T, class ArrayB_T>
double relNumDiffEl(ArrayA_T a, ArrayB_T b, int N, double diff = 0) {
	double res = 0;
	for (int i = 0; i < N; ++i) {
		if (fabs(a[i] - b[i]) > diff) {
			res += 1. / (float) N;
		}
	}
	return res;
}

// gives back the mean of the relative differences
// between the elements.
template<class ArrayA_T, class ArrayB_T>
double relDiffEl(ArrayA_T a, ArrayB_T b, int N) {
	double res = 0;
	for (int i = 0; i < N; ++i) {
		res += abs(a[i] - b[i]) / b[i];
	}
	return res;
}

void printHelp(char* progname) {
	cout << endl;
	cout << "Usage: " << progname << " [Options]" << endl;
	cout << "Options:" << endl;
	cout << "  -N <matrix width and height>" << endl;
	cout << "  -bs <block size>" << endl;
	cout << "  -check" << endl;
	cout << "    Verify GPU result with CPU result" << endl;
	cout << "  -v" << endl;
	cout << "    Be verbose and print result matrix" << endl;
	cout << "  -precision <num digits>" << endl;
	cout << "    Set the cout precision in verbose mode" << endl;
	cout << "  -fkernel <.ptx file path>" << endl;
	cout << "    File path to ptx module" << endl;
	cout << endl;
}

int main(int argc, char **argv) {

	// DEFAULT VALUES //
	int N = 1024;
	bool checkResult = false;
	int blockSize = 32;
	bool verbose = false;
	int printPrecision = 4;
	string fkernel = "mat-mul-kernel.ptx";

	// READ VALUES FROM COMMAND LINE //
	commandLineGetInt(&N, "-N", argc, argv);
	commandLineGetInt(&blockSize, "-bs", argc, argv);
	checkResult = commandLineGetBool("-check", argc, argv);
	verbose = commandLineGetBool("-v", argc, argv);
	commandLineGetInt(&printPrecision, "-precision", argc, argv);
	commandLineGetString(&fkernel, "-fkernel", argc, argv);

	if (commandLineGetBool("-h", argc, argv) || commandLineGetBool("--help", argc, argv)) {
		printHelp(argv[0]);
		return 0;
	}

	unsigned pd = 35; // padding for output
	cout << left << boolalpha;
	cout << endl;
	cout << "# Matrix Multiplication" << endl;
	cout << endl;
	cout << "## Input Arguments" << endl;
	cout << endl;
	cout << "  - N  = " << setw(pd) << N << "(N bodies in total)" << endl;
	cout << "  - bs = " << setw(pd) << blockSize << "(number of threads per block)" << endl;
	cout << "  - checkResult = " << setw(pd-9) << checkResult << "(execute cpu n-body verification)" << endl;
	cout << "  - verbose     = " << setw(pd-9) << verbose << "(print result matrix)" << endl;
	cout << "  - fkernel     = " << setw(pd-9) << fkernel << "(.ptx kernel file)" << endl;
	cout << endl;
	cout << "## Execution Progress" << endl;
	cout << endl;

	auto checkCudaErrors = [] (CUresult err, bool print = true) {
		if (err != CUDA_SUCCESS) {
			const char* msg;
			cuGetErrorName(err, &msg);
			cout << "[FAILED] " << msg << endl;
			exit(1);
		}
		else if (print) {
			cout << "[OK]" << endl;
		}
	};

	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction function;
	CUresult err;

	cout << "  - Initializing CUDA..." << flush;
	checkCudaErrors(cuInit(0));
	cout << "  - Getting CUDA device..." << flush;
	checkCudaErrors(cuDeviceGet(&device, 0));
	cout << "  - Getting Compute Capability of CUDA device..." << flush;
	int major = 0, minor = 0;
	checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));
	cout << "    Architecture: sm_" << major << minor << endl;

	const char* kernel_name = "mat_mul";

	cout << "  - Initializing CUDA context..." << flush;
	checkCudaErrors(cuCtxCreate(&context, 0, device));

	cout << "  - Loading module " << fkernel << "..." << flush;
	checkCudaErrors(cuModuleLoad(&module, fkernel.c_str()));
	cout << "  - Acquiring kernel function " << kernel_name << flush;
	checkCudaErrors(cuModuleGetFunction(&function, module, kernel_name));

	cout << "  - Allocating and initializing host memory..." << flush;

	float* a_h;
	float* b_h;
	float* c_h;
	try {
		a_h = new float [N * N];
		b_h = new float [N * N];
		c_h = new float [N * N];
	}
	catch (...) {
		cout << "[FAILED]" << endl;
		return -1;
	}
	cout << "[OK]" << endl;

	cout << "  - Initializing Data..." << flush; 
	for (int i = 0; i < N * N; i++) {
		c_h[i] = -90;
		a_h[i] = (float) i / (N * N);
		b_h[i] = (float) ((N * N) - i) / (N * N);
	}
	cout << "[OK]" << endl;

	double dev_MB = (double) (3 * N * N * sizeof(float)) / 1e6;
	cout << "  - Allocating " << dev_MB << " MB device memory..." << flush;
	CUdeviceptr a_d, b_d, c_d;
	checkCudaErrors(cuMemAlloc(&a_d, sizeof(float) * N * N), false);
	checkCudaErrors(cuMemAlloc(&b_d, sizeof(float) * N * N), false);
	checkCudaErrors(cuMemAlloc(&c_d, sizeof(float) * N * N));
	void *args[] = { &c_d, &a_d, &b_d, &N };

	int blocks = N / blockSize;

	if (N % blockSize != 0) {
		cout << "***ERROR: N % blockSize != 0" << endl;
		cout << "          N = " << N << endl;
		cout << "  blockSize = " << blockSize << endl;
		return -1;
	}

	cout << "  - Going to copy data to GPU..." << flush;
	auto timestamp = Clock::now();
	checkCudaErrors(cuMemcpyHtoD(a_d, a_h, sizeof(float) * N * N), false);
	checkCudaErrors(cuMemcpyHtoD(b_d, b_h, sizeof(float) * N * N), false);
	Duration t_htod = Clock::now() - timestamp;
	cout << "[OK]" << endl;

	cout << "  - Going to launch kernel with configuration:" << endl;
	cout << "    {" << blocks << ", " << blocks << ", 1} {" << blockSize << ", " << blockSize << ", 1}..." << flush;
	timestamp = Clock::now();
	checkCudaErrors(cuLaunchKernel(function,
	                               blocks, blocks, 1,       // Grid
	                               blockSize, blockSize, 1, // Block 
	                               0, 0, args, 0), false);
	checkCudaErrors(cuCtxSynchronize(), false);
	Duration t_kernel = Clock::now() - timestamp;
	cout << "[OK]" << endl;

	cout << "  - Going to copy data to CPU..." << flush;
	timestamp = Clock::now();
	checkCudaErrors(cuMemcpyDtoH(c_h, c_d, sizeof(float) * N * N), false);
	Duration t_dtoh = Clock::now() - timestamp;
	cout << "[OK]" << endl;

	if (verbose) {
		cout << "    - printing GPU result:" << endl;
		printArray(c_h, N * N, N, printPrecision);
	}

	if (checkResult) {
		cout << "  - Verifying with CPU results:" << endl;
		cout << "    - Try to alloc memory for CPU result..." << flush;
		float* cref;
		try {
			cref = new float [N * N];
		}
		catch (...) {
			cout << "[FAILED]" << endl;
			return -1;
		}
		cout << "[OK]" << endl;
		mat_mul_cpu(cref, a_h, b_h, N);
		cout << "    - num rel diff el = " << relDiffEl(cref, c_h, N * N) << endl;
		if (verbose) {
			cout << "    - printing CPU Result:" << endl;
			printArray(cref, N * N, N, printPrecision);
		}
		cout << "    - freeing additional host memory for CPU result..." << flush;
		delete[] cref;
		cout << "[OK]" << endl;
	}
	
	cout << "  - Realeasing GPU memory..." << flush;
	checkCudaErrors(cuMemFree(a_d), false);
	checkCudaErrors(cuMemFree(b_d), false);
	checkCudaErrors(cuMemFree(c_d));

	cout << "  - Realeasing CPU memory..." << flush;
	delete[] a_h;
	delete[] b_h;
	delete[] c_h;
	cout << "[OK]" << endl;

	cout << "  - Destroying CUDA context..." << flush;
	checkCudaErrors(cuCtxDestroy(context));
	cout << endl;

	// REPORT //
	cout << "## Program Report" << endl;
	cout << endl;
	cout << "  - host to device copy time = " << t_htod.count() << " s" << endl
	     << "  - device to host copy time = " << t_dtoh.count() << " s" << endl
	     << "  - kernel time = " << t_kernel.count() << " s" << endl;

	cout << endl;

	return 0;
}
