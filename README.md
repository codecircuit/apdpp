# Automated partitioning of data-parallel programs

This repository contains the software which has been written by myself
during my master thesis in physics at the Institute of Computer
Engineering of the University of Heidelberg (ZITI).

This software embodies the static part of project Mekong, which intends
to simplify multi GPU programming by simulating a one GPU environment
to the programmer. For a detailed introduction to the project please
read my master thesis contained in the top level directory of this
repository.

## Installation requirements

  - CUDA 7.5 Toolkit
  - LLVM 3.9.0. Even better with git hash
    314a1b6403bc971a9d68d7eed3a0ce1359d03656
  - Clang compiler of version 3.9.0. Even better with git hash
    f847bad18bcc46b5b47d32274a9d53a50209e555
  - libclc to enable Clang's OpenCL frontend: http://libclc.llvm.org
  - Polly library of version 3.9. Even better with git hash
    48731f9f119b03d7dc8b73aa7dd24720c4859645
  - ISL library of version 0.16.

## Building the project

Building the software consists of three parts: (1) building the LLVM
custom passes, (2) building the test case and its LLVM IR, and (3)
building Mekong's runtime library.

### Building the passes
In the top level directory do:
```
mkdir build
cd build
cmake ..
make
```
If this does not work for you, because e.g. you did not install the
excact listed requirements, you might want to take a look into the
`CMakeLists.txt` file located in the top level directory.

### Building the test case
Three test cases are located in `./samples`, which you can copy into
`./test_orig` to speed up your working progress with the scripts
given in `./sh`. Each test case consists of a host code written in CUDA
Driver API, and a device code written in OpenCL. If the test case is
used as a single GPU application, the device code must be compiled to
the PTX format deploying Clang's OpenCL frontend and its PTX backend.
The PTX device code can be included in the host code at application
runtime with NVIDIA's Just-In-Time (JIT) PTX compiler. You can take a
look into the code of the test cases to understand how this can be
implemented. Once you copied the test case into `./test_orig` you can
compile it with the provided Makefile if you have set your environment
variables:

  - `CLC_INC`: include directory of libclc intallation
  - `CLC_LIB`: `$CLC_LIB/built_libs/` must contain `nvptx64--nvidiacl.bc`
  - `CUDA_LIB`: must contain `libcuda.so`
  - `CUDA_INC`: must contain `cuda.h`

Afterwards do
```
cd ./test_orig/
mkdir llvm ptx bin
make
```

### Building the runtime library
Before building the runtime library you can adjust your configuration
in `./CONFIG.txt`. Moreover the information of the analysis pass is
linked statically into the runtime library, which means you have to run
the analysis pass first, before compiling the runtime library. You
should build the runtime library like:
```
mkdir ./runtime/build
cd ./runtime/build
cmake ..
make
```

## Using the software

I recommend to test the functionality of the software on your system
with one of the provided test cases in `./samples`. For further
development of the software I suggest to use the bash scripts
contained in `./sh`:

  - `bsp_analysis.sh` executes the device code analysis pass of
    `./test_orig/llvm/test_device.ll`. The result will be saved in a
    file with `dashdb` format (see `./dashdb` for the database
    implementation). The content file is located in
    `./bsp_analysis/dbs`.
  - `bsp_transform.sh` executes the device code transformation and
    outputs into `./test_trans/llvm` and `./test_trans/ptx`. The device
    code transformation requires the database file produced by the
    analysis pass.
  - `host_transform.sh` executes the host code transformation pass,
    which basically consists of function call replacements und function
    declaration insertions (see `host_transform/src` for the host code
    pass implementation)
  - `trans_and_link_on.sh` can be used to send the transferred device
    and host code with `scp` onto a remote machine. With the also
    transferred runtime library the object files will be linked against
    the runtime library and the CUDA library on the remote machine to
    produce a executable. You have to adust the paths within this
    script to fit to your installations.

After you got the scripts running you should be able to produce a
running binary.

## Modifying the software
You can build a doxygen documentation of most parts of the software.  To
understand the runtime library I recommend the `wrapLaunchKernel`
function in `./runtime/src/mekong-wrapping.cc` as a starting point.
This function replaces the `cuLaunchKernel` function after the host
transformation pass has been run.

## Performance measurements

The key performance measurements are contained in the master thesis.
All benchmarks, which have been done, are located in ipython notebook
files in directory `./measurements`. To understand the measured times
in detail you should read the master thesis, or at least take a look at
the bare measured data:

  - Matrix Multiplication: https://gist.github.com/codecircuit/d8d58eab0a0a591b5c625b0b5eae7459
  - Stencil Code: https://gist.github.com/codecircuit/d1067257e453f2500be68c24c06799ac
  - N Body Code: https://gist.github.com/codecircuit/ea60dccff3ee2643fcddfdf889ce7ac7

## Comment

The software is in a experimental state and can be used for a few test
cases. It represents a modular written base for Mekong's static part.
If you want to introduce a new test case, you should read chapter
_Usage_ in the master thesis.

## License

The provided software is published under the terms of the GNU General
Public License.
