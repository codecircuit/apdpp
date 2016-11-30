#ifndef MEKONG_INFO_H
#define MEKONG_INFO_H


#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>

#include "memory_copy.h"
#include "user_config.h"

namespace Mekong {

class DepResolution;
class Partitioning;
class KernelLaunch;

using namespace std;

//! You can simply use this class as an std output stream
template<class charT, class traits = char_traits<charT>>
class Log {
	public:
		Log() = default;
		const stringstream& getLog() const;

		template<class T>
		Log& operator<<(const T& in);
		Log& operator<<(basic_ostream<charT,traits>&
		                    (*pf)(basic_ostream<charT,traits>&));
		~Log();

	private:
		stringstream log_;
};

//! Collects runtime statistics, which can be shown on a final report
class Statistics {
	public:
		Statistics() = default;

		double getMemCpyTime() const;
		double getMemCpyTime(MemCpyKind kind) const;
		size_t getMemCpySize() const;
		size_t getMemCpySize(MemCpyKind kind) const;
		size_t getDepResCpySize() const;
		size_t getNumMemCpy() const;
		size_t getNumMemCpy(MemCpyKind kind) const;
		unsigned getNumArgAccessCalls() const;
		unsigned getNumArgAccessCalcs() const;
		unsigned getNumDepResExecs() const;
		unsigned getNumDepResObjects() const;
		unsigned getNumLaunchExecs() const;
		unsigned getNumLaunchObjects() const;
		double getMemBW() const;
		double getMemBW(MemCpyKind kind) const;
		double getDepResCreationTime() const;
		double getArgAccessTime() const;
		double getLinearizationTime() const;
		double getDepResExecTime() const;
		double getLaunchCreationTime() const;

		void setNumDev(unsigned numDev);
		void addResolution(shared_ptr<DepResolution> dres);
		void addLaunch(shared_ptr<KernelLaunch> kl);
		void addDepResCreationTime(double t);
		void addKernelLaunchCreationTime(double t);
		void addCpyDtoH(shared_ptr<const MemCpyDtoH> mc);
		void addCpyHtoD(shared_ptr<const MemCpyHtoD> mc);
		void addCpyDtoD(shared_ptr<const MemCpyDtoD> mc);

	private:

		unsigned numDev_ = 0;
		double depResCreationTime_ = 0;
		double kernelLaunchCreationTime_ = 0;
		unordered_set<shared_ptr<DepResolution>> resolutions_;
		unordered_set<shared_ptr<KernelLaunch>> launches_;
		unordered_set<shared_ptr<const MemCpyDtoH>> dev2host_;
		unordered_set<shared_ptr<const MemCpyHtoD>> host2dev_;
		unordered_set<shared_ptr<const MemCpyDtoD>> dev2dev_;
};

// LOGINFO CLASS FUNCTION DEFINITIONS //
template<class charT, class traits>
const stringstream& Log<charT, traits>::getLog() const {
	return log_;
}

//! Behaves like an std ostream object.
template<class charT, class traits>
template<class T>
Log<charT, traits>& Log<charT, traits>::operator<<(const T& in) {
	if (USER_OPTION_LOG_FILE != "") {
		log_ << in;
	}
	else {
		cout << in << flush;
	}
	return *this;
}

//! This enables the streaming of std::endl to a Log class object.
template<class charT, class traits>
Log<charT, traits>&
Log<charT, traits>::operator<<(basic_ostream<charT,traits>&
                                   (*pf)(basic_ostream<charT,traits>&)) {
	if (USER_OPTION_LOG_FILE != "") {
		log_ << pf;
	}
	else {
		cout << pf;
	}
	return *this;
}

//! Will write the log to a file if the user specifies that in the config.
template<class charT, class traits>
Log<charT, traits>::~Log() {
	if (USER_OPTION_LOG_FILE != "") {
		fstream fs;
		fs.open(USER_OPTION_LOG_FILE, fstream::out | fstream::trunc);
		fs << log_.str();
		fs.close();
	}
}

}; // namespace end

#endif
