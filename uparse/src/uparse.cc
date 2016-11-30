/* \brief Library to parse unsigned integer operations.

   The parser will execute '*' and '/' with the same predecence.  Furthemore '+'
   and '-' have the same precedence too. Operators with the same precedence will
   be executed from the left to the right. In the following example the first
   line represents the parser input:

     a) 3 * 4 / 5 * 2
     b) 12 / 5 * 2
     c) 2 * 2
     d) 4

   Parsing steps:
     1. Transform string to internal representation
        <Element> ::= <Number> | <Bracket> | <Operator>
     2. Get deepest bracket
     3. Solve Bracket until there is one number left.
       3.1 Solve next dot operation until there is no next dot operation
       3.2 Solve next '+' or '-' operation until there is no operation left

   CAUTION: Keep in mind that the underlaying data type is an unsigned integer.
            Thus the behaviour of the parser is undefined if any temporary
            result is smaller than zero.
*/
#include "uparse.h"
#include <cstdlib>

#define MAXIMUM_NUM_ELEMENTS 50

namespace Uparse {

// internal representation
Element IR[MAXIMUM_NUM_ELEMENTS];

/*! \brief Checks if c = [0-9]
*/
bool isCipher(char c) {
	// the numbers are sorted for their natural
	// probability (Benford's law)
	switch (c) {
		case '1':
			return true;
		case '2':
			return true;
		case '3':
			return true;
		case '4':
			return true;
		case '5':
			return true;
		case '6':
			return true;
		case '7':
			return true;
		case '8':
			return true;
		case '9':
			return true;
		case '0':
			return true;
	}
	return false;
}

//! Checks if a is any kind of space ('\t', ' ')
bool isSpace(char a) {
	switch (a) {
		case '\t':
			return true;
		case ' ':
			return true;
	}
	return false;
}

//! Returns Uparse::None if a is not an operator
Element_t getOp(char a) {
	switch (a) {
		case '*':
			return Multiplication;
		case '/':
			return Division;
		case '-':
			return Minus;
		case '+':
			return Plus;
	}
	return None;
}

//! Returns Uparse::None if a is not a bracket
Element_t getBracket(char a) {
	switch (a) {
		case '(':
			return BracketLeft;
		case ')':
			return BracketRight;
	}
	return None;
}

/*! \brief transforms the input to the internal representation

    If the input string exceeds the MAXIMUM_NUM_ELEMENTS limit the function
    throws an error.
*/
void transform(const char* input) {
	for (int i = 0; i < MAXIMUM_NUM_ELEMENTS; ++i) {
		IR[i].successor_ = i + 1;
	}
	const char* curr = input;
	int currEl = 0;
	while (*curr != '\0') {
		if (currEl == MAXIMUM_NUM_ELEMENTS) {
			throw InputExceedsLim;
		}
		if (isCipher(*curr)) {
			const char* number_start = curr;
			while (isCipher(*curr)) {
				++curr;
			}
			IR[currEl].et_ = Number;
			IR[currEl].val_ = std::strtoull(number_start, NULL, 10);
			++currEl;
			continue;
		}
		if (getOp(*curr) != None) {
			IR[currEl].et_ = getOp(*curr);
			++currEl;
			++curr;
			continue;
		}
		if (getBracket(*curr) != None) {
			IR[currEl].et_ = getBracket(*curr);
			++currEl;
			++curr;
			continue;
		}
		++curr;
	} // end of string loop
	IR[currEl - 1].successor_ = MAXIMUM_NUM_ELEMENTS;
}

/*! \brief Returns the index of the deepest beginning bracket.

    Index:         0   1  2  3  4  5  6  7  8  9  10
    Element Type:  (   3  +  4  *  (  2  -  1  )  )

    The function will return 5 in the example shown above.
*/
int getDeepestBracket() {
	int res = 0;
	int curr_lvl = 0; // bracket level
	int max_lvl = 0;
	int curr = 0;
	while (IR[curr].successor_ != MAXIMUM_NUM_ELEMENTS) {
		if (IR[curr].et_ == BracketLeft) {
			++curr_lvl;
			if (curr_lvl > max_lvl) {
				max_lvl = curr_lvl;
				res = curr;
			}
		}
		if (IR[curr].et_ == BracketRight) {
			--curr_lvl;
		}
		curr = IR[curr].successor_;
	}
	return res;
}

/*! \brief Solves another '/' or '*' operation in the bracket.

    \param pos must point to an open bracket, which does not contain
               another open bracket. If pos points not to a bracket
               it is assumed that there are no brackets in IR anymore.
    \returns false if there is no dot operation left.
*/
bool solveNextDot(int pos) {
	int next; // next element
	int prev; // previous element
	while (IR[pos].et_ != BracketRight &&
	       IR[pos].successor_ != MAXIMUM_NUM_ELEMENTS) {
		next = IR[pos].successor_;
		if (IR[pos].et_ == Division) {
			IR[prev].val_ = IR[prev].val_ / IR[next].val_;
			IR[prev].successor_ = IR[next].successor_;
			return true;
		}
		if (IR[pos].et_ == Multiplication) {
			IR[prev].val_ = IR[prev].val_ * IR[next].val_;
			IR[prev].successor_ = IR[next].successor_;
			return true;
		}
		prev = pos;
		pos = next;
	}
	return false;
}

/*! \brief Solves another '-' or '+' operation in the bracket.

    \param pos must point to an open bracket, which does not contain
               another open bracket. If pos points not to a bracket
               it is assumed that there are no brackets in IR anymore.
    \returns false if there is no dash operation left.
*/
bool solveNextDash(int pos) {
	int next; // next element
	int prev; // previous element
	while (IR[pos].et_ != BracketRight &&
	       IR[pos].successor_ != MAXIMUM_NUM_ELEMENTS) {
		next = IR[pos].successor_;
		if (IR[pos].et_ == Plus) {
			IR[prev].val_ = IR[prev].val_ + IR[next].val_;
			IR[prev].successor_ = IR[next].successor_;
			return true;
		}
		if (IR[pos].et_ == Minus) {
			IR[prev].val_ = IR[prev].val_ - IR[next].val_;
			IR[prev].successor_ = IR[next].successor_;
			return true;
		}
		prev = pos;
		pos = next;
	}
	return false;
}

/*! \brief Solves the calculations in one bracket.

    The bracket must not have any other bracket inside it, thus something
    like (61-(6+1)) would not be a valid input. The result of the bracket
    will be placed in the IR where the opening bracket was. E.g. if you get
    an input bracket in Uparser IR like:

    Position:     pos   pos+1   pos+2   pos+3   pos+4   pos+5   pos+6
    Element Type: (      11       -       4       *       2       )

    it will be solved to

    Position:     pos   pos+1   pos+2   pos+3   pos+4   pos+5   pos+6
    Element Type:  3    None    None    None    None    None    None

    Writing the result always on the first position will result in one single
    Number at IR[0] in the end.
*/
Error_t solveBracket(int pos) {
	while (solveNextDot(pos)) {}
	while (solveNextDash(pos)) {}

	// delete the bracket from IR
	if (IR[pos].et_ == BracketLeft) {
		int next = IR[pos].successor_;
		IR[pos].successor_ = IR[IR[next].successor_].successor_;
		// get the result of the bracket
		IR[pos].val_ = IR[next].val_;
		IR[pos].et_ = Number;
	}

	return Success;
}

Int_t parse(const char* in) {
	transform(in);
	while (IR[0].successor_ != MAXIMUM_NUM_ELEMENTS) {
		solveBracket(getDeepestBracket());
	}
	return IR[0].val_;
}

} // end namespace Uparse

#ifdef TEST
#include <iomanip>
#include <iostream>
using namespace Uparse;
using namespace std;

ostream& operator<<(ostream& os, const Element& el) {
	if (el.et_ == BracketLeft) {
		os << "BracketLeft";
	}
	else if (el.et_ == BracketRight) {
		os << "BracketRight";
	}
	else if (el.et_ == Plus) {
		os << "Plus";
	}
	else if (el.et_ == Minus) {
		os << "Minus";
	}
	else if (el.et_ == Division) {
		os << "Division";
	}
	else if (el.et_ == Multiplication) {
		os << "Multiplication";
	}
	else {
		os << el.val_;
	}
	os << ", Successor = " << el.successor_;
	return os;
}

void printIR() {
	int i = 0;
	while (i != MAXIMUM_NUM_ELEMENTS) {
		cout << "index = " << i << ", " << IR[i] << endl;
		i = IR[i].successor_;
	}
}

int main() {
	cout << "# Uparse Library Test" << endl;
	cout << endl;
	const char* in[] ={ "3 * 4 * 7 / 2 + 2 - 1 + 8",
	                    "((654))",
	                    "815358816",
	                    "123 + (456 + (789) +33)    ",
	                    "((12 + 4) * 3  ) * 2 - 12 * 3",
	                    "31 - 2 + 4 * (2 + 30 / 2 / 3)"
	};
	Int_t res[] = { 51,
	                654,
	                815358816,
	                1401,
	                60,
	                57
	};
	for (int i = 0; i < sizeof(res)/sizeof(Int_t); ++i) {
		cout << "  - " << left << setw(35) << in[i];
		if (parse(in[i]) == res[i]) {
			cout << " [OK]" << endl;
		}
		else {
			cout << " [FAILED]" << endl;
			cout << "  - Expected: " << res[i] << ", but got: "
			     << parse(in[i]) << endl;
		}
	}
	cout << endl;
	return 0;
}
#endif

