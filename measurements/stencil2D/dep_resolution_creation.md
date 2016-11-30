# Dependency Resolution

Mekong's runtime handles inter kernel dependencies by
creating appropriate Dependency-Resolution-Objects (DROs). This document
explains why the runtime creates exactly two DROs for the stencil code
as a test case.


## Writes invalidate other Writes

You have to keep in mind that we assume that every write to a GPU buffer
invalidates all previous writes on that buffer. Imagine the following
pseudo host code:

    ...
    launchKernel(kernelA, gpuBuffer, ...)
    ...
    launchKernel(kernelB, gpuBuffer, ...)
    ...
    launchKernel(kernelC, gpuBuffer, ...)

Let's say `kernelA` writes to `gpuBuffer` and `kernelB`
first reads and second writes to `gpuBuffer`. Last `kernelC`
reads `gpuBuffer`.

Mekong's runtime recognizes at launch of `kernelB`
that there is a dependency to the launch of `kernelA` and resolves
the dependencies. When `kernelC` is launched the runtime did not
save the write of `kernelA` to `gpuBuffer`, as we assume that
the write of `kernelB` invalidated the write of `kernelA`. Thus
`kernelC` depends not on the write of `kernelA`.


## Example: Stencil Code

The core of a stencil code is the sequential call of the kernel after
swapping the two gpu buffers. We take a look at the first four
kernel launches and assume that the kernel will write only to the
second given buffer:

  1. `launchKernel(stencilKernel, gpuBufferA, gpuBufferB, ...)`
  2. `launchKernel(stencilKernel, gpuBufferB, gpuBufferA, ...)`
  3. `launchKernel(stencilKernel, gpuBufferA, gpuBufferB, ...)`
  4. `launchKernel(stencilKernel, gpuBufferB, gpuBufferA, ...)`

@1: No previous kernel launches, thus no dependency resolution necessary.

@2: This launch reads gpuBufferB, which was written by the first launch, thus
we have to create a DRO with launch1 as master and launch2 as slave.

@3: This launch reads gpuBufferA, which was written by the second lanuch, thus
we have to create a DRO with launch2 as master and launch3 as slave. Although
the kernel arguments are not equal, we know that the accessed indices on a
buffer depend on the grid size and on non-pointer kernel arguments, thus
we can take the already calculated indices of kernel launch two.

@4: Here we take the DRO, which was already calculated in step two.
