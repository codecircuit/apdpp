#ifndef MEKONG_VIRTUAL_BUFFER_H
#define MEKONG_VIRTUAL_BUFFER_H

#include "mekong-cuda.h"

#include <unordered_set>
#include <map>
#include <memory>

namespace Mekong {

class KernelLaunch;

using namespace std;

class Buffer {
	public:
		shared_ptr<KernelLaunch> operator[](MEdeviceptr ptr) const;
		bool isWritten(MEdeviceptr ptr) const;
		void setWritten(MEdeviceptr ptr, shared_ptr<KernelLaunch> kl);
		void setBroadcast(MEdeviceptr ptr);
		bool isBroadcast(MEdeviceptr ptr) const;
		void erase(MEdeviceptr ptr);
	private:
		map<MEdeviceptr, shared_ptr<KernelLaunch>> ptr2launch_;

		// here we save if there was a cuMemcpyHtoD which was casted
		// to a broadcast on a certain buffer. If there is a kernel
		// launch, which writes to that buffer we remove the buffer
		unordered_set<MEdeviceptr> broadcastPtrs_;
};

}; // namespace end

#endif
