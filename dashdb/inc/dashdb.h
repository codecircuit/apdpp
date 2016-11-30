/*! \brief Dash Database Library

    Mekong's runtime wants to be independent of any heavy database libraries
    (e.g. jsoncpp). Therefore our light weight database is used. The key
    concept is to have concatenated keys, followed by an equal sign, followed
    by the value. Moreover there is a simple transformation rule from json
    databases to dashdb files. Imagine the following json file:
	{
    	"kernels" : [
			{
				"name" : "update_positions",
				"partitioning" : "x"
			},
			{
				"name" : "update_speed",
				"partitioning" : "x"
			}
		],
		"meta information" : "None"
	}

	Would become in dashdb representation:

	kernels-0-name=update_positions
	kernels-0-partitioning=x
	kernels-1-name=update_speed
	kernels-1-partitioning=x
	meta information=None

	Reserved symbols: You can not use a '-', '=' or a key beginning with a
	number.
*/
#ifndef DASHDB_h
#define DASHDB_h

#include <unordered_map>
#include <string>
#include <vector>

namespace dashdb {

/*! \brief He does everything for you.

    You can use the butler class to read a dashdb string:

	Butler b;
	b.read(str);
	string name = b["kernels"][0]["name"].asString();

	When accessing elements you have must always refer to the base
    object:

	Butler child = b["kernels"][0]; // WON'T WORK
	string name = child["name"]     // WON'T WORK
*/
class Butler {
	public:
		Butler() = default;
		//! Reads a database from a file
		Butler(const char* filename);
		//! Reads a database from a file
		Butler(const std::string& filename);
		//! Reads database from a string
		void read(const char* content);

		//! Delete to prevent wrong usage
		Butler& operator=(const Butler& other) = delete;
		//! Delete to prevent wrong usage
		Butler(const Butler& other) = delete;

		//! Writes database to a file
		void write(const char* filename) const;

		//! Prints each entry to one standard output line.
		void dump() const;

		//! Adds a current key
		Butler& operator[](const char* key);
		//! Adds a current key and introduces a list
		Butler& operator[](int index);

		//! Get the length of current list (0 if no list)
		int len();

		//! Sets value and resets current key
		template<class Numeric_t>
		Butler& operator=(Numeric_t number);
		Butler& operator=(bool b);
		Butler& operator=(const std::string& str);
		Butler& operator=(const char* str);

		double asFloat();
		long long asInt();
		//! If the conversion fails, the function returns the alternative value
		long long asInt(long long alternative);
		bool asBool();
		std::string asString();

	private:
		//! Current key
		std::string key_;
		//! The original DB string
		std::vector<std::string> lines_;
		//! Underlaying container
		std::unordered_map<std::string, std::string> db_;
		//! To save a list length
		std::unordered_map<std::string, int> lengths_;
};

template<class Numeric_t>
Butler& Butler::operator=(Numeric_t number) {
	db_[key_] = std::to_string(number);
	key_.clear();
	return *this;
}

} // end of namespace

#endif
