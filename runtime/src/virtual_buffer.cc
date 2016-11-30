#include "mekong-cuda.h"
#include "virtual_buffer.h"

#include <unordered_set>
#include <map>
#include <memory>

namespace Mekong {

using namespace std;

shared_ptr<KernelLaunch> Buffer::operator[](MEdeviceptr ptr) const {
	return ptr2launch_.at(ptr);
}

bool Buffer::isWritten(MEdeviceptr ptr) const {
	return ptr2launch_.find(ptr) != ptr2launch_.end();
}

void Buffer::setWritten(MEdeviceptr ptr, shared_ptr<KernelLaunch> kl) {
	broadcastPtrs_.erase(ptr);
	ptr2launch_[ptr] = kl;
}

void Buffer::setBroadcast(MEdeviceptr ptr) {
	broadcastPtrs_.insert(ptr);
}

bool Buffer::isBroadcast(MEdeviceptr ptr) const {
	return broadcastPtrs_.count(ptr) != 0;
}

void Buffer::erase(MEdeviceptr ptr) {
	ptr2launch_.erase(ptr);
}

}; // namespace end
