#ifndef MEKONG_WRAPPING_H
#define MEKONG_WRAPPING_H

#include "mekong-cuda.h"

extern "C" { // to prevent name mangling

// IF YOU CHANGE THE DECLARATIONS  HERE YOU HAVE TO ADJUST THIS IN THE
// PASS WHICH INSERTS THE DECLARATIONS INTO THE LLVM IR.  THIS INSERTION
// PASS LIVES IN THE HOST TRANSFORM PASS DIRECTORY.

Mekong::MErawresult wrapInit(unsigned flags);
Mekong::MErawresult wrapDeviceGetCount(int* num);
Mekong::MErawresult wrapDeviceGet(Mekong::MEdevice* dev, int dummy);
Mekong::MErawresult wrapDeviceComputeCapability(int* major, int* minor,
                                                Mekong::MEdevice dev);
Mekong::MErawresult wrapCtxCreate(Mekong::MEcontext* ctx,
                                  unsigned flags,
                                  Mekong::MEdevice dev);
Mekong::MErawresult wrapModuleLoad(Mekong::MEmodule* module,
                                   const char* name);
Mekong::MErawresult wrapModuleGetFunction(Mekong::MEfunction* func,
                                          Mekong::MEmodule mod,
                                          const char* fname);

Mekong::MErawresult wrapMemAlloc(Mekong::MEdeviceptr* ptr, size_t size);
Mekong::MErawresult wrapMemcpyHtoD(Mekong::MEdeviceptr dstDevPtr,
                                   void* srcHostPtr, size_t size);
Mekong::MErawresult wrapLaunchKernel(Mekong::MEfunction func,
                                     unsigned blocksX,
                                     unsigned blocksY,
                                     unsigned blocksZ,
                                     unsigned threadsX,
                                     unsigned threadsY,
                                     unsigned threadsZ,
                                     unsigned shmem,
                                     Mekong::MEstream stream,
                                     void** args, void** extra);
Mekong::MErawresult wrapCtxSynchronize(); 
Mekong::MErawresult wrapMemcpyDtoH(void* dstHostPtr,
                                   Mekong::MEdeviceptr srcDevPtr,
                                   size_t size);
Mekong::MErawresult wrapMemFree(Mekong::MEdeviceptr ptr);
Mekong::MErawresult wrapCtxDestroy(Mekong::MEcontext ctx);
void MEKONG_report();

}; // end of extern

#endif
