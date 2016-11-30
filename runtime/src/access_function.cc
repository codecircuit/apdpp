#include "argument.h"
#include "access_function.h"
#include "uparse.h"

#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <cstdint>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/space.h>
#include <isl/point.h>
#include <isl/set.h>

namespace Mekong {

using namespace std;

/*! \param islReadParams list of strings to calculate the space parameters of
           the ISL read map. E.g. ['arg1', '30+40', 'arg3 + 10', 'arg4 * 3']
           can belong to [A,B,C,D] -> { [x,y,z,i] -> [A*x,B*y,C*z,D*i] }.
    \param islRead ISL read map string
    \param islWriteParams analog to islReadParams
    \param islWrite ISL write map string
    \param argNr Kernel argument number this access function object belongs to.
*/
AccFunc::AccFunc(const vector<shared_ptr<const string>>& islReadParams,
                 const string& islRead,
                 const vector<shared_ptr<const string>>& islWriteParams,
                 const string& islWrite, int argNr)
		: islReadParams_(islReadParams),
		  islRead_(islRead),
		  islWriteParams_(islWriteParams),
		  islWrite_(islWrite),
		  argNr_(argNr) {}

bool AccFunc::isAffine() const {
	throw runtime_error("SPACE Mekong, CLASS AccFunc, FUNC isAffine():\n"
	                    "not implemented yet!\n");
}

//! Return string of ISL read map
const string& AccFunc::getIslRead() const {
	return islRead_;
}

//! Return string of ISL write map
const string& AccFunc::getIslWrite() const {
	return islWrite_;
}

//! Return strings of ISL read map space parameters
const vector<shared_ptr<const string>>& AccFunc::getIslReadParams() const {
	return islReadParams_;
}

//! Return strings of ISL write map space parameters
const vector<shared_ptr<const string>>& AccFunc::getIslWriteParams() const {
	return islWriteParams_;
}

//! Return the kernel argument number this object belongs to.

//! This is the linking between the access behavior and the kernel array.
int AccFunc::getArgNr() const {
	if (argNr_ == -1) {
		throw runtime_error("SPACE Mekong, CLASS AccFunc, FUNC getArgNr():\n"
		                    "I do not know the argument nr I am associated with.\n"
		                    "You forgot to set the arg nr in my constructor!\n");
	}
	return argNr_;
}

/*! \brief Parse one ISL read space parameter value.
    \param paramId E.g. [A,B,C,D] if you want to parse C you have to set this
           value to 2.
    \param args the kernel launch arguments. This is neccessary, because some
           kernel arrays are multi dimensional and have a limited size in their
           dimensions. The size can affect the accessed indices, as the size can
           be a ISL map space parameter. To get the concrete accessed indices
           we have to know about a kernel array's size.
    \param gridSize original grid size
    \param blockSize original block size
*/
intmax_t AccFunc::getReadParam(unsigned short paramId,
                const vector<shared_ptr<const KernelArg>>* args,
                const Array3* gridSize, const Array3* blockSize) const {
	return getParam(paramId, args, gridSize, blockSize, true);
}

/*! \sa getReadParam analog for the read behaviour
*/
intmax_t AccFunc::getWriteParam(unsigned short paramId,
                const vector<shared_ptr<const KernelArg>>* args,
                const Array3* gridSize, const Array3* blockSize) const {
	return getParam(paramId, args, gridSize, blockSize, false);
}

/*! \brief Parse all space parameters and return the ISL map
*/
__isl_give isl_union_map* AccFunc::getReadIslMap(isl_ctx* ctx,
                    const vector<shared_ptr<const KernelArg>>* args,
                    const Array3* gridSize, const Array3* blockSize) const {
	return getIslMap(ctx, args, gridSize, blockSize, true);
}

/*! \brief Parse all space parameters and return the ISL map
*/
__isl_give isl_union_map* AccFunc::getWriteIslMap(isl_ctx* ctx,
                    const vector<shared_ptr<const KernelArg>>* args,
                    const Array3* gridSize, const Array3* blockSize) const {
	return getIslMap(ctx, args, gridSize, blockSize, false);
}

vector<size_t> AccFunc::getReadAcc(const Array3& threadId,
                    const vector<shared_ptr<const KernelArg>>* args,
                    const Array3* gridSize, const Array3* blockSize) const {
	return getAcc(threadId, args, gridSize, blockSize, true);
}

vector<size_t> AccFunc::getWriteAcc(const Array3& threadId,
                    const vector<shared_ptr<const KernelArg>>* args,
                    const Array3* gridSize, const Array3* blockSize) const {
	return getAcc(threadId, args, gridSize, blockSize, false);
}

/*! \todo use a self written parser for positive integer types.
*/
intmax_t AccFunc::getParam(unsigned short paramId,
                const vector<shared_ptr<const KernelArg>>* args,
                const Array3* gridSize, const Array3* blockSize,
                bool is_read) const {
	
	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS AccFunc, FUNC getParam():\n" + msg);
	};

	const vector<shared_ptr<const string>>* islParams;
	if (is_read) { // if true parse the read params
		islParams = &islReadParams_;
	}
	else {
		islParams = &islWriteParams_;
	}

	if (paramId >= islParams->size()) {
		throwError(string("paramId = ") + to_string(paramId) + string(" exceeds the limits of\n") + 
		           string("the given parameter descriptions (max = ") + to_string(islParams->size()) +
		           string(")\n"));
	}

	// replaces occurrences of 'key' in 'str' with 'subst'
	auto replace = [] (string& str, const string& key, const string& subst) {
		size_t pos;
		while (str.find(key) != string::npos) {
			pos = str.find(key);
			str.replace(pos, key.size(), subst);
		}
		return str;
	};

	auto foundSubStr = [] (const string& substr, const string& str) {
		if (str.find(substr) != string::npos) {
			return true;
		}
		else {
			return false;
		}
	};

	// PREPARE PARSER INPUT
	// 1. Replace the arg<i> placeholder
	//    with the value from the kernel argument
	// 2. Replace the size_[x|y|z] placeholeder with
	//    the value from the kernel launch grid size

	string prsInput = *(*islParams)[paramId];
	if (foundSubStr("arg", prsInput) && !args) {
		throwError("Parameter depends on kernel arguments, but I did not get it!\n");
	}
	if (foundSubStr("size_", prsInput) && (!gridSize || !blockSize)) {
		throwError("Parameter depends on kernel launch size, but I did not get it!\n");
	}

	// 1.
	for (unsigned short id = 0; id < args->size(); ++id) {
		auto arg = args->operator[](id);
		// if the argument is an double, int, float convert it correctly
		if (arg->getType()->isFundType() && arg->getType()->getPtrlvl() == 0) {
			string key = "arg" + to_string(id);
			string subst;
			// get the correct value of the argument
			if (arg->getType()->isFloat()) {
				subst = to_string(arg->asFloat());
			}
			else if (arg->getType()->isDouble()) {
				subst = to_string(arg->asDouble()); 
			}
			else if (arg->getType()->isInt()) {
				subst = to_string(arg->asInt());
			}
			replace(prsInput, key, subst);
		}
	}
	// 2.
	const string xKey = "size_x";
	if (foundSubStr(xKey, prsInput)) {
		size_t size_x = (*gridSize)[0] * (*blockSize)[0];
		replace(prsInput, xKey, to_string(size_x));
	}
	const string yKey = "size_y";
	if (foundSubStr(yKey, prsInput)) {
		size_t size_y = (*gridSize)[1] * (*blockSize)[1];
		replace(prsInput, yKey, to_string(size_y));
	}
	const string zKey = "size_z";
	if (foundSubStr(zKey, prsInput)) {
		size_t size_z = (*gridSize)[2] * (*blockSize)[2];
		replace(prsInput, zKey, to_string(size_z));
	}

	uintmax_t parsRes;
	try {
		parsRes = Uparse::parse(prsInput.c_str());
	}
	catch(...) {
		throwError("Could not parse isl read/write parameter. Maybe "
		           "you introduced a new keyword and I do not know how "
		           "to handle that.");
	}
	return parsRes;
}

__isl_give isl_union_map* AccFunc::getIslMap(isl_ctx* ctx,
                            const vector<shared_ptr<const KernelArg>>* args,
                            const Array3* gridSize, const Array3* blockSize,
                            bool is_read) const {

	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS AccFunc, FUNC getIslMap():\n" + msg);
	};

	// GET THE WANTED ISL STRING [READ | WRITE]
	const string* islStr;
	islStr = is_read ? &islRead_ : &islWrite_;
	if (*islStr == "None" || islStr->empty() || *islStr == "null") {
		// TODO this might not be constistent here
		// we should ensure that every pointer kernel argument
		// got an isl read and write map.
		throwError("I did not have a string to build the isl access map");
	}

	// ISL INITIALIZATION
	isl_union_map* umap = isl_union_map_read_from_str(ctx, islStr->c_str());
	vector<intmax_t> params;
	isl_set* param_set = isl_union_map_params(isl_union_map_copy(umap));
	
	// GET THE ISL MAP PARAMETER VALUES
	for (int i = 0; i < isl_set_dim(param_set, isl_dim_param); ++i) {
		params.push_back(getParam(i, args, gridSize, blockSize, is_read));
	}

	// CHECK THE VALUES FOR OVERFLOW
	for (intmax_t param : params) {
		if (param > numeric_limits<int>::max() ||
			param < numeric_limits<int>::lowest()) {
			throw overflow_error("wanted parameter cannot be handled"
								 "by isl. Maybe you should choose smaller"
								 "arguments for the kernel");
		}
	}

	// SET THE VALUES IN THE PARAM SET
	for (int i = 0; i < params.size(); ++i) {
		param_set = isl_set_fix_si(param_set, isl_dim_param, i, params[i]); // __isl_take
	}
	
	// SET THE PARAMETERS IN THE UNION MAP
	umap = isl_union_map_intersect_params(umap, param_set); // __isl_take, __isl_take

	// SIMPLIFY THE REPRESENTATION BY REMOVING THE SPACE PARAMETERS
	umap = isl_union_map_project_out(umap, isl_dim_param, 0, params.size()); // __isl_take

	return umap;
}

// flatten all the points
size_t AccFunc::flatPoint (vector<size_t>& point, vector<size_t>& dimSizes) {
	if (point.size() == 1) {
		return point[0];
	}
	else {
		size_t mul = 1;
		for (size_t dimSize : dimSizes) {
			mul *= dimSize;
		}
		size_t add = mul * point.back();
		point.pop_back();
		dimSizes.pop_back();
		return add + flatPoint(point, dimSizes);
	}
}

vector<size_t> AccFunc::getAcc(const Array3& threadId,
                               const vector<shared_ptr<const KernelArg>>* args,
                               const Array3* gridSize, const Array3* blockSize,
                               bool is_read) const { // if is_read = true get read access

	auto throwError = [] (const string& msg) {
		throw runtime_error("SPACE Mekong, CLASS AccFunc, FUNC getAcc():\n" + msg);
	};
	
	if (!args) {
		throwError("I need the kernel arguments to check array dimension sizes");
	}

	vector<size_t> res;

	// CHECK IF THERE IS A ISL MAP STRING AVAILABLE
	const string* islStr;
	islStr = is_read ? &islRead_ : &islWrite_;
	if (*islStr == "None" || islStr->empty() || *islStr == "null") {
		return res;
	}

	isl_ctx* ctx = isl_ctx_alloc();

	// SET THE THREAD ID IN THE ISL MAP
	if (threadId[0] > numeric_limits<int>::max() ||
		threadId[1] > numeric_limits<int>::max() ||
		threadId[2] > numeric_limits<int>::max()) {
		throw overflow_error("thread id cannot be handled"
		                     "by isl. Isl can only handle int");
	}
	
	vector<isl_set*> ranges;
	auto threadId_and_ResVector = make_pair(&threadId, &ranges);
	isl_union_map* umap = getIslMap(ctx, args, gridSize, blockSize, is_read);

	// now we collect the ranges of the accesses in variable 'ranges'
	isl_union_map_foreach_map(umap, fixAndGetRange, &threadId_and_ResVector);

	if (ranges.empty()) {
		return res;
	}
	
	// as the accesses in the union map must all target the same array with
	// the same dimensions we can union all these ranges
	isl_set* range = ranges[0];
	for (int i = 1; i < ranges.size(); ++i) {
		range = isl_set_union(range, ranges[i]);
	}
	range = isl_set_coalesce(range);
	
	// now we collect all the points
	vector<vector<size_t>> points;
	isl_set_foreach_point(range, addPoint, &points);

	// flat all the points to a linear array access
	auto& dimSizes = args->operator[](argNr_)->getDimSizes();
	
	for (vector<size_t> point : points) {
		vector<size_t> tmp_dimSizes = dimSizes;
		res.push_back(flatPoint(point, tmp_dimSizes));
	}

	isl_set_free(range);
	isl_union_map_free(umap);
	isl_ctx_free(ctx);
	return res;
}

isl_stat AccFunc::fixAndGetRange(__isl_take isl_map* map, void* threadId_and_ResVector) {
	pair<Array3*, vector<isl_set*>*>& threadId_and_ResVector_pair = 
		*((pair<Array3*, vector<isl_set*>*>*) threadId_and_ResVector); // ugly type ... :(
	Array3 threadId = *get<0>(threadId_and_ResVector_pair);
	map = isl_map_fix_si(map, isl_dim_in, 0, threadId[0]);
	map = isl_map_fix_si(map, isl_dim_in, 1, threadId[1]);
	map = isl_map_fix_si(map, isl_dim_in, 2, threadId[2]);

	get<1>(threadId_and_ResVector_pair)->push_back(isl_map_range(map));
	return isl_stat_ok;
}

isl_stat AccFunc::addPoint(__isl_take isl_point* point, void* points) {

	auto pointVec = (vector<vector<size_t>>*) points;
	vector<size_t> coords;
	isl_space* space = isl_point_get_space(point);
	unsigned numDim = isl_space_dim(space, isl_dim_out);
	if (numDim > 2) {
		throw runtime_error("SPACE Mekong, CLASS AccFunc, FUNC addPoint():\n"
		                    "Polly array with dim > 2 are not supported");
	}
	// here we have to add in reverse order because of polly's internal array
	// representation. E.g. an access arr[x + N * y] leads to an internal
	// representation of arr[y, x] with dimSize(0) = inf and dimSize(1) = N
	for (int dim = numDim - 1; dim >= 0; --dim) {
		isl_val* val = isl_point_get_coordinate_val(point, isl_dim_out, dim);
		coords.push_back(isl_val_get_num_si(val));
		isl_val_free(val);
	}
	isl_point_free(point);
	isl_space_free(space);
	pointVec->push_back(coords);
	return isl_stat_ok;
}

bool AccFunc::foundSubStr(const string& substr, const vector<shared_ptr<const string>>& strings) {
	for (const auto& str : strings) {
		if (str->find(substr) != string::npos) {
			return true;
		}
	}
	return false;
}

}; // namespace end
