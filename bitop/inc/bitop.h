/*! \file bitop.h
    \brief Library for dynamic type handling

    this class can save bits from a certain memory location and convert the bits
    to a specific type. You can access the bits by a vector<unsigned char>,
    where the highest index denotes the highest bytes, e.g.
 
    memory         |0   1   0   1   0   0   0   1 |  1   0   0   0   0   1   0   1|
    vector element |             [1]              |               [0]             |
    if converted   |  2**14 + 2**12      +     2**8+2**7       +       2**2 + 2**0|
    
    \author Christoph Klein
*/

#ifndef BITOP_H
#define BITOP_H
#include <string>
#include <utility>
#include <cstdint>
#include <vector>
#include <iostream>

namespace bitop {

using namespace std;

class charPack : public vector<unsigned char> {
	public:
		charPack() = default;
		charPack(const charPack& other);

		//! Reads the bits from a memory location
		charPack(const void* vptr, size_t bits);
		//! Reads bits from a value
		template<class T> charPack(T object);
		//! True if bit at position `pos` is 1
		bool getBit(size_t pos) const;
		//! Prints the saved bits in a human readable format
		void print() const;
		//! Prints the number of contained bits
		size_t bits() const { return this->size() * 8; }
		//! Converts bits to an integer
		template<class int_t> int_t asInt() const;
		//! Converts bits to a floating point number
		float asFloat_IEEE754() const;
		//! Converts to a double precision number
		double asDouble_IEEE754() const;
		//! Gets raw pointer on lowest bit
		void* getRaw();
		const void* getRaw() const;

		charPack& operator=(const charPack& other);

		friend ostream& operator<<(ostream &out, const charPack& cp);
};

template<class T> charPack::charPack(T object) : charPack(&object, sizeof(T) * 8) {}

template<class int_t>
int_t charPack::asInt() const {
	if ( bits() > sizeof(int_t) * 8) {
		cerr << "***ERROR: CLASS charPack, FUNC asInt():"      << endl
			 << "          number of bits is too high for the" << endl
			 << "          wanted integer type!"    << endl;
		throw;
	}
	int_t res = 0;
	for (int i = size() - 1; i >= 0; --i) {
		res = (res << 8) + this->operator[](i);
	}
	return res;
}

string readBits(const void* vptr, int size);

template<class int_t>
bool test_int_conversion(int_t val) {
	charPack cp(&val, sizeof(int_t) * 8);
	return cp.asInt<int_t>() == val;
}

bool test_float_conversion(float val);

bool test_double_conversion(double val);

}
#endif
