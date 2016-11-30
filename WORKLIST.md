Worklist for Mekong's runtime part
==================================

Features
--------

  * Cost calculator for different partitioning schemes
    * based on runtime code
  * Support shared memory kernels
    * workload on runtime: small
    * workload on kernel transformation: low
    * workload on kernel analysis: medium
      polly has problems with non polyhedral accesses.
      This affects also shared memory accesses. Simply removing
      this accesses in LLVM IR does not necessarily enable
      polly to analyze the kernel code (e.g. shared mem mat mul kernel).
  * Array reshaping, thus a GPU only allocates
    data which is in fact needed
  * Delay initial host to device copy (MANDATORY FOR ARRAY RESHAPING)
    if we can analyse statically that the buffer
    remains unchanged until the corresponding kernel
    is launched. If we delay the copy process we know
    which data is needed by the kernel and copy exactly
    the necessary data.
      * workload on runtime: high
      * workload on host analysis: high
      * workload on host transformation: maybe
  * For single non iterative kernel applications we can substitute
    the cuLaunchKernel() function with a specialized version, which
    starts configuring the DtoH memcpy with an additional host thread
    at kernel launch time. This allows an overlap between kernel
    computation and memory copy configuration, thus when the
    cuDtoHMemcpy() is called the memcpy object is ready.
  * We can implement a general dependency resolution without the
    assumption that a writing process of one kernel launch invalidates all
    previous writing processes of other kernel launches. Up to now The
    consequence of this assumption is that our runtime does not suport mesh
    refinement codes. Up to now we identify one dependency resolution object with
    two kernel launch objects. In future one dep res object should belong to a
    state of consecutive executed kernel launches, saved by a global queue
    variable. If there is no dep res object for the current queue state, we must
    create one.  This is done by using an advanced virtual buffer object which
    saves for every buffer which indices were written by a certain GPU last. For
    dep resolution we have to intersect read indices of the current kernel launch
    with the state of the advanced virtual buffer object for all read buffers, to
    calculate the resulting dep resolving memory copies. The advantage of a queue
    is that if we have a repeating order of kernel launches (which is often the
    case) we can delete the last kernel launch if it is equal to the one we add in
    the front.  This is valid because previous equal kernel launches does not
    affect the state of the queue.
    For e.g. a stencil code we have two different launches, because the buffers
    are swapped. A queue of {A, B, A} results in the same virtual buffer state
    as the equivalent queue {B, A}. Thus we have two queue states in total:
    {A, B} and {B, A}, which results in two different dependency resolution
    objects.

TODOS
-----
  * write encapsulated test for memory copy class, which deploys `sofire`
    check bare data movements
  * implement good encapsulated test cases
    * encapsulated means that you have test cases for each class
      which are compilable without sources the class does not depend on
  * Add check for injective access maps
  * enable peer to peer access for inter kernel dependency resoultion.
    A simple CUDA memcpy between two GPUs results per default a copy
    operation into host memory, and subsquently a copy operation to the
    target GPU. Thus the peer to peer access must be enabled explicitly.
    Check if this is possible for all 16 GPUs on victoria.

CODE REVIEW
-----------

  * base class in host transform is not neccessary
  * keep argument transformer to static function
  * function adder to static function
  * loop wrapping pass helper function do not need to be part of the
    class
  * uneccessary exceptions in analysis pass. Use ReportFatalError of
    LLVM
  * analysis pass: getKernel to external helper function
  * difficult names in device code transform pass
  * bsp transform offsetter is helper function
  * bsp transform overloaded call operator difficult to understand
  * propagator to function
  * promoter if clause overload; use return statements instead
  * promoter has many helper function which can be independent
  * createvMap has no error report ( bsp transform )
  * general: names in transformation passes should have good name
  * runtime: partitioning can be replaced by a tuple
