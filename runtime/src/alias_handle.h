#ifndef MEKONG_ALIAS_HANDLER_H
#define MEKONG_ALIAS_HANDLER_H

#include <map>
#include <vector>
#include <stdexcept>
#include <string>

#include "mekong-cuda.h"

namespace Mekong {

using namespace std;

//! Handles the mapping between single- and multiple-GPU pointers

//! In the Mekong context we have to promote single GPU pointers
//! e.g. CUdeviceptr to multiple CUdeviceptr's, which are
//! allocated on different devices. This class provides the
//! mapping between the single GPU pointer used by the
//! programmer and the belonging pointers on other GPUs. 
//! Moreover we save information about number of used GPUs, created
//! contexts, modules, functions, etc.
class AliasHandle {
	public:
		typedef map<MEdevice, vector<MEdevice>> devMap_t;
		typedef map<MEcontext, vector<MEcontext>> ctxMap_t;
		typedef map<MEmodule, vector<MEmodule>> modMap_t;
		typedef map<MEfunction, vector<MEfunction>> funcMap_t;
		typedef map<MEdeviceptr, vector<MEdeviceptr>> ptrMap_t;

		vector<MEdevice>& operator[](const MEdevice& dev);
		vector<MEcontext>& operator[](const MEcontext& ctx);
		vector<MEmodule>& operator[](const MEmodule& mod);
		vector<MEfunction>& operator[](const MEfunction& func);
		vector<MEdeviceptr>& operator[](const MEdeviceptr& ptr);

		void erase(const MEcontext& ctx);
		void erase(const MEdeviceptr& ptr);
	
		string& atName(const MEfunction& func);
		const vector<MEcontext>& getCtx() const;
		unsigned short getNumDev() const;
		const vector<MEdevice>& getDevs() const;
		const ptrMap_t& getDevPtrMap() const;
		const funcMap_t& getFuncMap() const;

	private:

		devMap_t devMap_;                     ///< device mapping
		ctxMap_t ctxMap_;                     ///< context mapping
		modMap_t modMap_;                     ///< module mapping
		funcMap_t funcMap_;                   ///< kernel function mapping
		ptrMap_t ptrMap_;                     ///< device buffer mapping
		map<MEfunction, string> nameMap_;     ///< kernel function to name
};

};

#endif
