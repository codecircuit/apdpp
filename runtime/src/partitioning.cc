#include "partitioning.h"

#include <tuple>
#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip>

namespace Mekong {

using namespace std;

Partitioning::Partitioning(bool x, bool y, bool z) : x_(x), y_(y), z_(z) {} 

Partitioning::Partitioning(bool3_t b3) : x_(get<0>(b3)), y_(get<1>(b3)), z_(get<2>(b3)) {}

Partitioning::Partitioning(const string& dims) :
			x_(dims.find('x') == string::npos ? (dims.find('X') == string::npos ? false : true) : true),
			y_(dims.find('y') == string::npos ? (dims.find('Y') == string::npos ? false : true) : true),
			z_(dims.find('z') == string::npos ? (dims.find('Z') == string::npos ? false : true) : true) {}

bool Partitioning::isSplitAt(char axis) const {
	if (axis == 'x' || axis == 'X') {
		return x_;
	}
	else if (axis == 'y' || axis == 'Y') {
		return y_;
	}
	else if (axis == 'z' || axis == 'Z') {
		return z_;
	}
	else {
		throw invalid_argument("Namespace Mekong, Class Partitioning, Func isSplitAt(char axis)\n");
	}
}

bool Partitioning::isSplitAt(short unsigned axis) const {
	if (axis == 0) {
		return x_;
	}
	else if (axis == 1) {
		return y_;
	}
	else if (axis == 2) {
		return z_;
	}
	else {
		throw invalid_argument("Namespace Mekong, Class Partitioning, Func isSplitAt(unsigned short axis)\n");
	}
}

Partitioning::bool3_t Partitioning::getSplit() const {
	return make_tuple(x_, y_, z_);
}

string Partitioning::getSplitStr() const {
	string out = "";
	if (x_) {
		out += "x";
	}
	if (y_) {
		out+="y";
	}
	if (z_) {
		out+="z";
	}
	return out;
}

ostream& operator<<(ostream& out, const Partitioning& p) {
	out << boolalpha << "(" << p.x_ << ", " << p.y_ << ", " << p.z_ << ")";
	return out;
}

bool operator==(const Partitioning& a, const Partitioning& b) {
	return a.getSplit() == b.getSplit();
}

bool operator!=(const Partitioning& a, const Partitioning& b) {
	return !(a == b);
}

}; // namespace end
