#include <string>
#include <memory>
#include <sstream>
#include <ostream>
#include <stdexcept>
#include <map>

#include "dashdb.h"
#include "argument_type.h"
#include "partitioning.h"
#include "access_function.h"
#include "kernel_info.h"

namespace Mekong {

using namespace std;

vector<shared_ptr<const bsp_KernelInfo>>
bsp_KernelInfo::createKInfos(const char* bspAnalysis) {

	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS bsp_KernelInfo, FUNC createKInfos()\n" + msg);
	};

	if (!bspAnalysis) {
		throwError("I did not get a string to the analysis results");
	}

	// PARSE THE DASHDB STRING
	dashdb::Butler b;
	b.read(bspAnalysis);

	vector<shared_ptr<const bsp_KernelInfo>> res;
	// as we only want to create argument type and
	// partitioning objects once, we need the following maps
	map<string, shared_ptr<const Partitioning>> parts;
	for (int kernelNr = 0; kernelNr < b["kernels"].len(); ++kernelNr) {
		
		// GET ACTUAL KERNEL NAME
		string name = b["kernels"][kernelNr]["name"].asString();

		// COLLECT KERNEL ARGUMENT TYPE OBJECTS
		vector<shared_ptr<const bsp_ArgType>> thisArgs;
		for (int argNr = 0; argNr < b["kernels"][kernelNr]["arguments"].len(); ++argNr) {
			string argt_name = b["kernels"][kernelNr]["arguments"][argNr]["type name"].asString();
			unsigned argt_ptrlevel = b["kernels"][kernelNr]["arguments"][argNr]["pointer level"].asInt();
			string argt_fundT = b["kernels"][kernelNr]["arguments"][argNr]["fundamental type"].asString();

			// Read the size of the argument type and convert it to Bytes
			unsigned argt_size = b["kernels"][kernelNr]["arguments"][argNr]["size"].asInt() / 8;
			// if the we have a pointer type LLVM's function
			// 'getPrimitiveSizeInBits()' returns a size of zero.
			// Thus we set the size to the default pointer size of
			// the system.
			argt_size = argt_size == 0 && argt_ptrlevel == 1 ?
			            sizeof(void*) : argt_size;
			// Read the size of the pointed type and convert it to Bytes
			unsigned argt_elsize = b["kernels"][kernelNr]["arguments"][argNr]["element size"].asInt() / 8;
			bool argt_isModified = !b["kernels"][kernelNr]["arguments"][argNr]["isl write map"].asString().empty();
			bool argt_isRead = !b["kernels"][kernelNr]["arguments"][argNr]["isl read map"].asString().empty();

			// Get the number of array dimensions with 0 as alternative value
			// e.g. if there is no entry in the database
			unsigned argt_numDims = b["kernels"][kernelNr]["arguments"][argNr]["num dimensions"].asInt(0);
			vector<string> dimSizePatterns;
			int dsp_size = b["kernels"][kernelNr]["arguments"][argNr]["dim sizes"].len();
			for (int dsp = 0; dsp < dsp_size; ++dsp) {
				dimSizePatterns.push_back(b["kernels"][kernelNr]["arguments"][argNr]["dim sizes"][dsp].asString());
			}
			thisArgs.emplace_back(shared_ptr<const bsp_ArgType>(
					 new bsp_ArgType(argt_name, argt_ptrlevel,
					                 argt_fundT[0], argt_size,
					                 argt_elsize, argt_isModified,
					                 argt_isRead, argt_numDims,
					                 dimSizePatterns)));
		}
		// CREATE THE PARTITIONING OBJECT IF NECESSARY
		const string partStr = b["kernels"][kernelNr]["partitioning"].asString();
		shared_ptr<const Partitioning> thisPart = nullptr;

		if (partStr.size() == 0 || partStr == "None") {
			throw invalid_argument("Namespace Mekong, Class bsp_KernelInfo,"
			                       "Func createKInfos():\n"
			                       "could not find partitioning in analysis database");
		}

		if (parts.find(partStr) == parts.end()) {
			thisPart = shared_ptr<const Partitioning>(new Partitioning(partStr));
			parts[partStr] = thisPart;
		}
		else {
			thisPart = parts[partStr];
		}

		// CREATE ACCESS FUNCTIONS
		vector<shared_ptr<const AccFunc>> accFuncs;
		for (int argNr = 0; argNr < b["kernels"][kernelNr]["arguments"].len(); ++argNr) {
			vector<shared_ptr<const string>> readParams;
			int numIslParams = b["kernels"][kernelNr]["arguments"][argNr]["isl read params"].len();
			// collect isl parameter description
			for (int islParamNr = 0; islParamNr < numIslParams; ++islParamNr) { 
				string islParamStr = b["kernels"][kernelNr]["arguments"][argNr]["isl read params"][islParamNr].asString();
				readParams.push_back(shared_ptr<const string>(new string(islParamStr)));

			}

			vector<shared_ptr<const string>> writeParams;
			// collect isl parameter description
			numIslParams = b["kernels"][kernelNr]["arguments"][argNr]["isl write params"].len();
			for (int islParamNr = 0; islParamNr < numIslParams; ++islParamNr) { 
				string islParamStr = b["kernels"][kernelNr]["arguments"][argNr]["isl write params"][islParamNr].asString();
				writeParams.push_back(shared_ptr<const string>(new string(islParamStr)));
			}

			accFuncs.push_back(shared_ptr<const AccFunc>(
				new AccFunc(readParams, b["kernels"][kernelNr]["arguments"][argNr]["isl read map"].asString(),
				            writeParams, b["kernels"][kernelNr]["arguments"][argNr]["isl write map"].asString(), argNr)));
		}
		
		// FINALLY CREATE THE KERNEL INFO OBJECT
		res.push_back(shared_ptr<const bsp_KernelInfo>(
		              new bsp_KernelInfo(name, thisArgs, thisPart, accFuncs)));
	} // loop over all kernels
	if (res.size() == 0) {
		throwError("There is no kernel analysis!");
	}
	return res;
}

bsp_KernelInfo::bsp_KernelInfo(const string& name,
                               const vector<shared_ptr<const bsp_ArgType>>& args,
                               shared_ptr<const Partitioning> part,
                               const vector<shared_ptr<const AccFunc>>& accFuncs) :
			name_(name),
			args_(args),
			part_(part),
			accFuncs_(accFuncs) {}

unsigned short bsp_KernelInfo::getNumArgs() const {
	return args_.size();
}

string bsp_KernelInfo::getName() const {
	return name_;
}

shared_ptr<const AccFunc> bsp_KernelInfo::getAccFunc(unsigned short argId) const {
	return accFuncs_[argId];
}

const vector<shared_ptr<const bsp_ArgType>>&
bsp_KernelInfo::getArgTypes() const {
	return args_;
}

shared_ptr<const Partitioning> bsp_KernelInfo::getPartitioning() const {
	return part_;
}

ostream& operator<<(ostream& out, const bsp_KernelInfo& kinfo) {
	out << "(name: " << kinfo.getName() << "; args: ";
	for (auto arg : kinfo.getArgTypes()) {
		out << *arg << ", ";
	}
	out << "; partitioning: " << *kinfo.getPartitioning() << ")";
	return out;
}

}; // namespace end
