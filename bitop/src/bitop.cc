/*! \brief BInary TO character Pack (bitop) Library

    Cplusplus lacks dynamic type casting, which represents a difficulty when
    handling the unknown kernel argument values. The kernel arguments in CUDA
    driver API are given by an `void* args[]` object. At some point in Mekong's
    runtime we need to evaluate the kernel arguments, which expresses the
    necessity for this libray.
*/
#include <string>
#include <limits>
#include <iostream>
#include <iomanip>
#include <utility>
#include <math.h>
#include <new>
#include <cstdint>

#include "bitop.h"

namespace bitop {

using namespace std;

charPack::charPack(const void* vptr, size_t bits) : vector<unsigned char>(bits/(sizeof(unsigned char)*8)) {
	if (bits % (sizeof(unsigned char) * 8) != 0) {
		cerr << "***ERROR: CLASS charPack, FUNC charPack(const void* vptr, size_t bits):" << endl
		     << "          bits % (sizeof(unsigned char) * 8) must be equal to zero!" << endl;
		throw;
	}
	
	if (!vptr) {
		clear();
	}
	else {
		const unsigned char* cptr = reinterpret_cast<const unsigned char*>(vptr);
		for (int j = 0; j < size(); ++j) {
			this->operator[](j) = cptr[j];
		}
	}
}

charPack::charPack(const charPack& other) {
	for (auto el : other) {
		this->push_back(el);
	}
}

bool charPack::getBit(size_t pos) const {
	if (pos >= bits()) {
		cerr << "***ERROR: CLASS charPack, FUNC getBit(unsigned int pos):" << endl
		     << "          pos is greater as the highest bit" << endl
		     << "          pos = " << pos << ", bits = " << bits() << endl;
		throw;
	}
	size_t arrayPos = pos/8;
	size_t offset = pos % 8;
	unsigned char res = this->operator[](arrayPos) & (1 << offset);
	return (int) res;
}

void charPack::print() const {
	cout << "*** Printing charPack:" << endl;
	if (size()) {
		cout << "bits = " << bits() << endl
			 << "size = " << size() << endl
			 << "Binary Representation:" << endl
			 << noboolalpha;
		for (int i = bits() - 1; i >= 0; --i) {
			cout << "| ";
			cout << setw(2) << i;
		}
		cout << "|" << endl;
		for (int i = bits() - 1; i >= 0; --i) {
			cout << "| ";
			cout << getBit(i) << " ";
		}
		cout << "|" << endl;
		cout << boolalpha << "*** End of Printing" << endl;
	}
	else {
		cout << "    (no content, charPack is empty)" << endl;
	}
}

float charPack::asFloat_IEEE754() const {
	if (bits() != 32) {
		cerr << "***ERROR: CLASS charPack, FUNC asFloat_IEEE754():" << endl
		     << "          bit size must be equal to 32" << endl;
		throw;
	}

	float res = 0;

	// GETTING THE UNBIASED EXPONENT //
	unsigned char expBits = this->operator[](size() - 1);
	expBits = (expBits << 1);
	expBits += ((this->operator[](size() - 2) & 128) ? 1 : 0);
	int E = expBits;

	// GETTING THE BIASED EXPONENT //
	int exp = E;
	if (E == 0) {
		exp = -126;
	}
	else {
		exp -= 127;
		res =  pow(2, exp);
	}
	
	// GETTING THE MANTISSA //
	int divExp = exp - 23;
	unsigned counter = 0;
	for (int i = 0; i < 23; ++i) {
		if (getBit(i)) {
			res += pow(2, i + divExp);
			++counter;
		}
	}

	// HANDLING SPECIAL VALUES //
	if (counter == 0 && E == 0) {
		res = 0;
	}
	else if (E == 255 && counter == 0) {
		res = numeric_limits<float>::infinity();
	}
	else if (E == 255) {
		res = numeric_limits<float>::signaling_NaN();
	}

	// GETTING THE SIGN //
	if (getBit(31)) {
		res *= (-1.);
	}

	return res;
}

double charPack::asDouble_IEEE754() const {
	if (bits() != 64) {
		cerr << "***ERROR: CLASS charPack, FUNC asDouble_IEEE754():" << endl
		     << "          bit size must be equal to 64" << endl;
		throw;
	}
		
	double res = 0;

	// GETTING THE UNBIASED EXPONENT //
	int E = 0;
	unsigned char expBits = this->operator[](size() - 1);
	expBits = expBits & 127;
	E += expBits;
	E = (E << 4);
	expBits = this->operator[](size() - 2) & 240;
	expBits = expBits >> 4;
	E += expBits;

	// GETTING THE BIASED EXPONTENT //
	int exp = E;
	if (E == 0) {
		exp = -1022;
	}
	else {
		exp -= 1023;
		res =  pow(2, exp);
	}
	
	// GETTING THE MANTISSA //
	int divExp = exp - 52;
	unsigned int counter = 0; // counter for setted bits in the mantissa //
	for (int i = 0; i < 52; ++i) {
		if (getBit(i)) {
			res += pow(2, i + divExp);
			++counter;
		}
	}
	
	// HANDLING SPECIAL VALUES //
	if (counter == 0 && E == 0) {
		res = 0;
	}
	else if (E == 2047 && counter == 0) {
		res = numeric_limits<double>::infinity();
	}
	else if (E == 2047) {
		res = numeric_limits<double>::signaling_NaN();
	}

	// GETTING THE SIGN //
	if (getBit(63)) {
		res *= (-1.);
	}

	return res;
}

void* charPack::getRaw() {
	return &(this->operator[](0));
}

const void* charPack::getRaw() const {
	return &(this->at(0));
}

charPack& charPack::operator=(const charPack& other) {
	this->clear();
	for (auto el : other) {
		this->push_back(el);
	}
	return *this;
}


ostream& operator<<(ostream& out, const charPack& cp) {
	for (int i = cp.bits() - 1; i >= 0; --i) {
		out << (cp.getBit(i) ? 1 : 0);
	}
	return out;
}

// gives back a string with Big-Endian order
// from the read memory location
string readBits(const void* vptr, int size) {
	string rev;
	string res;

	if (size % sizeof(const unsigned char) != 0) {
		cerr << "***ERROR: FUNC readBits:\n" << endl
		     << "          size % sizeof(unsigned char) == 0 is not true" << endl;
		throw;
	}

	const unsigned char* val = reinterpret_cast<const unsigned char*>(vptr);

	for (int j = 0; j < (size/8)/sizeof(const unsigned char); ++j) {
		char copy = val[j];
		for (int i = 0; i < 8; ++i) {
			if ((1 & copy) == 1) {
				rev += "1";
			}
			else {
				rev += "0";
			}
			copy = copy >> 1;
		}
	}
	for (int i = 0; i < size; ++i) {
		res += rev[size - 1 - i];
	}
	return res;
}

bool test_float_conversion(float val) {
	charPack cp(&val, sizeof(float) * 8);
	return cp.asFloat_IEEE754() == val;
}

bool test_double_conversion(double val) {
	charPack cp(&val, sizeof(double) * 8);
	return cp.asDouble_IEEE754() == val;
}

} // namespace end

#ifdef BITOP

#include <random>

using namespace bitop;

int main() {

	cout << "*** Starting bitop test case..." << endl;
	cout << "*** sizeof(int) = " << sizeof(int) << endl;
	cout << "*** sizeof(intmax_t) = " << sizeof(intmax_t) << endl;
	cout << "*** sizeof(double) = " << sizeof(double) << endl;
	cout << "*** sizeof(float) = " << sizeof(float) << endl;
	
	int N = 100000;
	unsigned wrong;
	default_random_engine gen;

	// Testing int8_t //
	cout << "*** Going to test " << N << " conversions of int8_t->charPack->intmax_t->int8_t... [";
	uniform_int_distribution<int8_t> dist_i8(INT8_MIN, INT8_MAX);
	wrong = 0;
	for (int i = 0; i < N; ++i) {
		wrong += (test_int_conversion(dist_i8(gen)) ? 0 : 1);
	}
	cout << (wrong == 0 ? "OK]" : (to_string(wrong) + " FAILED]")) << endl;

	
	// Testing int16_t //
	cout << "*** Going to test " << N << " conversions of int16_t->charPack->intmax_t->int16_t... [";
	uniform_int_distribution<int16_t> dist_i16(INT16_MIN, INT16_MAX);
	wrong = 0;
	for (int i = 0; i < N; ++i) {
		wrong += (test_int_conversion(dist_i16(gen)) ? 0 : 1);
	}
	cout << (wrong == 0 ? "OK]" : (to_string(wrong) + " FAILED]")) << endl;

	// Testing int32_t //
	cout << "*** Going to test " << N << " conversions of int32_t->charPack->intmax_t->int32_t... [";
	uniform_int_distribution<int32_t> dist_i32(INT32_MIN, INT32_MAX);
	wrong = 0;
	for (int i = 0; i < N; ++i) {
		wrong += (test_int_conversion(dist_i32(gen)) ? 0 : 1);
	}
	cout << (wrong == 0 ? "OK]" : (to_string(wrong) + " FAILED]")) << endl;

	// Testing int64_t //
	cout << "*** Going to test " << N << " conversions of int64_t->charPack->intmax_t->int64_t... [";
	uniform_int_distribution<int64_t> dist_i64(INT64_MIN, INT64_MAX);
	wrong = 0;
	for (int i = 0; i < N; ++i) {
		wrong += (test_int_conversion(dist_i64(gen)) ? 0 : 1);
	}
	cout << (wrong == 0 ? "OK]" : (to_string(wrong) + " FAILED]")) << endl;

	// Testing float IEEE754 //
	cout << "*** Going to test " << N << " conversions of float->charPack->float... [";
	uniform_real_distribution<float> dist_float(numeric_limits<float>::lowest(), numeric_limits<float>::max());
	wrong = 0;
	for (int i = 0; i < N; ++i) {
		wrong += (test_float_conversion(dist_float(gen)) ? 0 : 1); 
	}
	cout << (wrong == 0 ? "OK]" : (to_string(wrong) + " FAILED]")) << endl;

	// Testing double IEEE754 //
	cout << "*** Going to test " << N << " conversions of double->charPack->double... [";
	uniform_real_distribution<double> dist_double(numeric_limits<double>::lowest(), numeric_limits<double>::max());
	wrong = 0;
	for (int i = 0; i < N; ++i) {
		wrong += (test_double_conversion(dist_double(gen)) ? 0 : 1); 
	}
	cout << (wrong == 0 ? "OK]" : (to_string(wrong) + " FAILED]")) << endl;
	
	// Testing templated constructor //
	{
	float f = 234.214;
	int i = 123;
	int im = 123;
	charPack cp0(f);
	charPack cp1(f);
	charPack cp2(i);
	charPack cp3(im);
	}; // end anonymous namespace
	
	// Testing getRaw() function //
	{
	double D;
	int N = 20;
	void* arr[] = {&N, &D};
	charPack cp0(arr[0], sizeof(int) * 8);
	charPack cp1(arr[1], sizeof(double) * 8);
	void* raw = cp0.getRaw();
	cout << "*** Testing getRaw() function...";
	if (*((int*) cp0.getRaw()) != N) {
		cout << " [FAILED]" << endl;
	}
	else {
		cout << " [OK]" << endl;
	}
	}; // end of scope

	// Testing copy operator
	{
	const charPack con(8);
	charPack con1 = con;
	};

	return 0;
}

#endif
