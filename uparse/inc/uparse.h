#ifndef UPARSE_H
#define UPARSE_H
#include <cstdint>

namespace Uparse {

// underlaying integer container
using Int_t = uint64_t;

enum Error_t { Success, Fail, InputExceedsLim };

enum Element_t { Number, BracketLeft, BracketRight,
                 Multiplication, Plus, Minus, Division, None };

struct Element {
	Element_t et_;
	Int_t val_;
	Int_t successor_;
};

Int_t parse(const char* in); 

} // end namespace Uparse

#endif
