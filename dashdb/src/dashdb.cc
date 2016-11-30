#include "dashdb.h"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <utility> // std::pair
#include <sstream>
#include <algorithm>

namespace dashdb {

Butler::Butler(const char* filename) {
	std::fstream fs(filename, std::fstream::in);
	std::string line;

	auto getKeyValue = [] (const std::string& line) {
		int i = 0;
		for (char c : line) {
			++i;
			if (c == '=') {
				break;
			}
		}
		return std::make_pair(line.substr(0, i - 1), line.substr(i));
	};

	while (std::getline(fs, line)) {
		lines_.push_back(line);
		auto keyVal = getKeyValue(line);
		db_[keyVal.first] = keyVal.second;
	}
	fs.close();
}

Butler::Butler(const std::string& filename) : Butler(filename.c_str()) {}

void Butler::read(const char* content) {
	db_.clear();
	lengths_.clear();
	std::stringstream ss(content);
	std::string line;

	auto getKeyValue = [] (const std::string& line) {
		int i = 0;
		for (char c : line) {
			++i;
			if (c == '=') {
				break;
			}
		}
		return std::make_pair(line.substr(0, i - 1), line.substr(i));
	};

	while (std::getline(ss, line)) {
		lines_.push_back(line);
		auto keyVal = getKeyValue(line);
		db_[keyVal.first] = keyVal.second;
	}
}

void Butler::write(const char* filename) const {
	std::fstream fs(filename, std::fstream::out | std::fstream::trunc);
	std::vector<std::string> keys;
	for (const auto& el : db_) {
		keys.push_back(el.first);
	}
	std::sort(keys.begin(), keys.end());
	for (const std::string& key : keys) {
		fs << key << "=" << db_.at(key) << std::endl;
	}
	fs.close();
}

void Butler::dump() const {
	for (const auto& el : db_) {
		std::cout << el.first << "=" << el.second << std::endl;
	}
}

Butler& Butler::operator[](const char * key) {
	if (!key_.empty()) {
		key_ += "-";
	}
	key_ += key;
	return *this;
}

Butler& Butler::operator[](int index) {
	if (lengths_[key_] <= index) {
		lengths_[key_] = index + 1;
	}
	if (!key_.empty()) {
		key_ += "-";
	}
	key_ += std::to_string(index);
	return *this;
}

/*! \brief Checks if c = [0-9]
*/
bool isCipher(char c) {
	// the numbers are sorted for their natural
	// probability (Benford's law)
	switch (c) {
		case '1':
			return true;
		case '2':
			return true;
		case '3':
			return true;
		case '4':
			return true;
		case '5':
			return true;
		case '6':
			return true;
		case '7':
			return true;
		case '8':
			return true;
		case '9':
			return true;
		case '0':
			return true;
	}
	return false;
}

int Butler::len() {
	int res = lengths_[key_];
	// if len is unset, we must check for it in the database string
	if (res == 0) {
		for (const auto& line : lines_) {
			int ksize = std::string(key_).size();
			if (line.compare(0, ksize, key_) == 0) {
				if (isCipher(line[ksize + 1])) {
					int num_end = 1;
					while (line[ksize + num_end + 1] != '-') {
						++num_end;
					}
					int index = std::stoi(line.substr(ksize + 1, num_end));
					if (res <= index) {
						res = index + 1;
						lengths_[key_] = res;
					}
				}
			}
		}
	}
	key_.clear();
	return res;
}

Butler& Butler::operator=(bool b) {
	db_[key_] = std::to_string(b);
	key_.clear();
	return *this;
}

Butler& Butler::operator=(const std::string& str) {
	db_[key_] = str;
	key_.clear();
	return *this;
}

Butler& Butler::operator=(const char* str) {
	return this->operator=(std::string(str));
}

double Butler::asFloat() {
	std::string res = db_[key_];
	key_.clear();
	return std::stod(res);
}

long long Butler::asInt() {
	std::string res = db_[key_];
	long long res_num;
	if (res.empty()) {
		throw std::invalid_argument(
			"SPACE Dashdb, FUNC Butler::asInt(): "
			"could not convert empty value belonging "
			"to key '" + std::string(key_) + "' to an int."
		);
	}
	try {
		res_num = stoll(res);
	}
	catch(...) {
		throw std::invalid_argument(
			"SPACE Dashdb, FUNC Butler::asInt(): "
			"could not convert value '" + res + "' belonging "
			"to key '" + std::string(key_) + "' to an int."
		);
	}
	key_.clear();
	return res_num;
}

long long Butler::asInt(long long alternative) {
	std::string res = db_[key_];
	long long res_num;
	try {
		res_num = stoll(res);
	}
	catch(...) {
		res_num = alternative;
	}
	key_.clear();
	return res_num;
}

bool Butler::asBool() {
	std::string res = db_[key_];
	key_.clear();
	return res == "True" || res == "true" || res == "1";
}

std::string Butler::asString() {
	std::string res = db_[key_];
	key_.clear();
	return res;
}

} // namespace end

#ifdef TEST

#define ISEQUAL(l, r) cout << "  - " << left << setw(w) << #l << flush; if (l == r) { cout << "[OK]" << endl; } else { cout << "[FAILED]" << endl; }

#include <iostream>
#include <iomanip>
#include "test_database.h"

using namespace std;
using namespace dashdb;

const char* db_str =
"kernels-0-name=update_positions\n"
"kernels-0-partitioning=x\n"
"kernels-0-arguments-0-element size=32\n"
"kernels-0-arguments-0-fundamental type=f\n"
"kernels-0-arguments-0-isl read params=[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_x[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\n"
"kernels-0-arguments-1-element size=16\n"
"kernels-0-arguments-1-fundamental type=d\n"
"kernels-0-arguments-1-isl write params=[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_x[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }\n"
"kernels-1-name=update_speed\n"
"kernels-1-partitioning=x\n"
"kernels-1-arguments-0-element size=64\n"
"kernels-1-arguments-0-fundamental type=i\n"
"meta information-0=None\n";

int main() {

	Butler b;
	b.read(db_str);
	int w = 70;

	cout << "# Test of DashDB" << endl;
	cout << endl;
	cout << "## Input Database" << endl;
	cout << endl;
	cout << db_str << endl;
	cout << "## Test Cases" << endl;
	cout << endl;
	cout << "### Query Entries" << endl;
	cout << endl;

	ISEQUAL(b["kernels"][0]["name"].asString(), "update_positions")
	ISEQUAL(b["kernels"][0]["partitioning"].asString(), "x")
	ISEQUAL(b["kernels"][0]["arguments"][0]["element size"].asInt(), 32)
	ISEQUAL(b["kernels"][0]["arguments"][1]["element size"].asInt(), 16)
	ISEQUAL(b["kernels"][0]["arguments"][1]["isl write params"].asString(), "[size_x, size_y, size_z] -> { Stmt_entry[i0, i1, i2] -> MemRef_pos_x[i0] : 0 <= i0 < size_x and 0 <= i1 < size_y and 0 <= i2 < size_z }")
	ISEQUAL(b["kernels"][1]["arguments"][0]["element size"].asInt(), 64)
	ISEQUAL(b["kernels"][1]["name"].asString(), "update_speed")

	cout << endl;
	cout << "### Test List Lengths" << endl;
	cout << endl;

	ISEQUAL(b["kernels"].len(), 2)
	ISEQUAL(b["kernels"][0]["arguments"].len(), 2)
	ISEQUAL(b["kernels"][1]["arguments"].len(), 1)
	ISEQUAL(b["meta information"].len(), 1)
	ISEQUAL(b["meta information"].len(), 1)
	ISEQUAL(b["not in database"].len(), 0)
	ISEQUAL(b["not in database"].len(), 0)
	b["kernels"][0]["name"].asString();
	ISEQUAL(b["kernels"].len(), 2)

	Butler bb;
	bb.read(test_database);
	ISEQUAL(bb["kernels"].len(), 2);
	cout << "  - calling bb[\"kernels\"][0][\"name\"].asString()..." << endl;
	bb["kernels"][0]["name"].asString();
	ISEQUAL(bb["kernels"].len(), 2);

	cout << endl;
	cout << "### Test Insertions" << endl;
	cout << endl;

	b["kernels"][2]["name"] = "foo_kernel";
	ISEQUAL(b["kernels"][2]["name"].asString(), "foo_kernel");
	ISEQUAL(b["kernels"].len(), 3)

	b["kernels"][0]["arguments"][2]["element size"] = 20;
	ISEQUAL(b["kernels"][0]["arguments"][2]["element size"].asInt(), 20)
	ISEQUAL(b["kernels"][0]["arguments"].len(), 3)

	b["kernels"][1]["arguments"][0]["is read"] = true;
	ISEQUAL(b["kernels"][1]["arguments"][0]["is read"].asBool(), true);

	cout << endl;
	return 0;
}

#endif
