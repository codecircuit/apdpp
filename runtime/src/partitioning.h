#ifndef MEKONG_PARTITIONING_H
#define MEKONG_PARTITIONING_H

#include <tuple>
#include <string>
#include <ostream>

namespace Mekong {

using namespace std;

class Partitioning {
	public:
		typedef tuple<bool, bool, bool> bool3_t;
		Partitioning(bool x, bool y, bool z); 
		Partitioning(bool3_t b3);
		Partitioning(const string& dims);

		bool isSplitAt(char axis) const;
		bool isSplitAt(short unsigned axis) const;
		bool3_t getSplit() const;
		string getSplitStr() const;

		friend ostream& operator<<(ostream& out, const Partitioning& p);
	private:
		bool x_;
		bool y_;
		bool z_;
};

bool operator==(const Partitioning& a, const Partitioning& b);
bool operator!=(const Partitioning& a, const Partitioning& b);

}; // namespace end

#endif
