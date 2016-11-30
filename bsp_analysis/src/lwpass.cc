#include "lwpass.h"

#define DEBUG_TYPE "GEF"

loop_wrapping_pass::loop_wrapping_pass() : ModulePass(this->ID) {}

bool loop_wrapping_pass::runOnModule(Module &M) {
	
	// if we add a function while iterating
	// through the function list, we end up
	// in an non terminating loop
	vector<Function*> worklist;
	for (auto& f : M.getFunctionList()) {
		if (!f.isDeclaration()) {
			worklist.push_back(&f);
		}
	}

	for (auto* f : worklist) {
		// add three new arguments size_x, size_y, size_z //
		auto* child = cloneAndAddArgs(f);
		// insert the three wrapping loops //
		auto triple = insertLoops(child); // triple = (Value* x.addr, Value* y.addr, Value* z.addr)
		// replace get_global_id() calls with loop variable values //
		replaceCalls(child, get<0>(triple), get<1>(triple), get<0>(triple));
	}
	return true; // true if module was modificated //
}

//! Wraps the function's body with three for loops.

//! The limits of the loops are the last three arguments
//! of function \param f. The return value are the three
//! pointers to the loop variables memory location, which
//! are allocated in the first basic block of function \param f.
//! \param f must have void as the return type.
//! The function inserts the following concept into \param f:
//!
//!                      --------
//!                      | head |   init x to zero
//!                      --------
//!                          |
//!                          ∨
//!                    --------------
//!     ---------------| for.cond.x |<----------------
//!     |              --------------                |
//!     |                     |                      |
//!     |                     ∨                      |
//!     |              --------------                |
//!     |              | for.body.x |  init y = 0    |
//!     |              --------------                |
//!     |                     |                      |
//!     |                     ∨                      |
//!     |              --------------                |
//!     |    ----------| for.cond.y |<----------     |
//!     |    |         --------------          |     |
//!     |    |                |                |     |
//!     |    |                ∨                |     |
//!     |    |         --------------          |     |
//!     |    |         | for.body.y |  z = 0   |     |
//!     |    |         --------------          |     |
//!     |    |                |                |     |
//!     |    |                ∨                |     |
//!     |    |         --------------          |     |
//!     |    |    -----| for.cond.z |<----     |     |
//!     |    |    |    --------------    |     |     |
//!     |    |    |           |          |     |     |
//!     |    |    |           ∨          |     |     |
//!     |    |    |    ---------------   |     |     |
//!     |    |    |    |function body|   |     |     |
//!     |    |    |    ---------------   |     |     |
//!     |    |    |           |          |     |     |
//!     |    |    |           ∨          |     |     |
//!     |    |    |    -------------     |     |     |
//!     |    |    |    | for.inc.z |------     |     |
//!     |    |    |    -------------           |     |
//!     |    |    |                            |     |
//!     |    |    |    -------------           |     |
//!     |    |    ---->| for.inc.y |------------     |
//!     |    |         -------------                 |
//!     |    |                                       |
//!     |    |         -------------                 |
//!     |    |-------->| for.inc.x |------------------
//!     |              -------------
//!     |
//!     |              -------------
//!     |------------->| for.end.x | contains new return statement
//!                    -------------
//!     
//!
tuple<Value*, Value*, Value*> loop_wrapping_pass::insertLoops(Function* f) {

	auto error = [] (const string& message) { throwError("insertLoops", message); };

	// up to now we can only handle void functions,
	// as it is not trivial to move the return
	// instruction if it is not void
	auto& ctx = f->getContext();
	if (f->getReturnType() != Type::getVoidTy(ctx)) {
		error("function return type is not void!");
	}

	if (f->getArgumentList().size() < 3) {
		error("expected a function with at least 3 arguments!");
	}

	// get the type of the last argument //
	auto* argType = dyn_cast<IntegerType>(f->getArgumentList().rbegin()->getType());
	if (!argType) {
		error("expected the last function argument's type to be an llvm::IntegerType!");
	}

	// save the body start and end of the for loops //
	BasicBlock* loopBodyStart = &*f->getBasicBlockList().begin();
	BasicBlock* loopBodyLast  = &*f->getBasicBlockList().rbegin();

	// create a constant int to init loop variables and to increase them //
	ConstantInt* zeroInt = ConstantInt::get(argType, 0); 
	ConstantInt* oneInt = ConstantInt::get(argType, 1); 

	// get the sizes //
	Value* zsize = &*f->getArgumentList().rbegin();
	Value* ysize = &*(++f->getArgumentList().rbegin());
	Value* xsize = &*(++(++f->getArgumentList().rbegin()));

	// insert the basic blocks //
	auto* forCondZ = BasicBlock::Create(ctx, "for.cond.z", f, &*f->begin());
	auto* forBodyY = BasicBlock::Create(ctx, "for.body.y", f, &*f->begin());
	auto* forCondY = BasicBlock::Create(ctx, "for.cond.y", f, &*f->begin());
	auto* forBodyX = BasicBlock::Create(ctx, "for.body.x", f, &*f->begin());
	auto* forCondX = BasicBlock::Create(ctx, "for.cond.x", f, &*f->begin());
	auto* initLoops = BasicBlock::Create(ctx, "init.loops", f, &*f->begin());
	auto* forIncZ = BasicBlock::Create(ctx, "for.inc.z", f);
	auto* forIncY = BasicBlock::Create(ctx, "for.inc.y", f);
	auto* forIncX = BasicBlock::Create(ctx, "for.inc.x", f);
	auto* forEndX = BasicBlock::Create(ctx, "for.end.x", f);

	// building the loops initialization //
	IRBuilder<> initBuilder(initLoops);
	auto* xAddress = initBuilder.CreateAlloca(argType, nullptr, "x.address");
	auto* yAddress = initBuilder.CreateAlloca(argType, nullptr, "y.address");
	auto* zAddress = initBuilder.CreateAlloca(argType, nullptr, "z.address");
	initBuilder.CreateStore(zeroInt, xAddress);
	initBuilder.CreateBr(forCondX);

	// building the x condition //
	{
	IRBuilder<> builder(forCondX);
	auto* x = builder.CreateLoad(xAddress, "x");
	auto* isSmaller = builder.CreateICmpSLT(x, xsize);
	builder.CreateCondBr(isSmaller, forBodyX, forEndX);
	}

	// building the x body //
	{
	IRBuilder<> builder(forBodyX);
	builder.CreateStore(zeroInt, yAddress);
	builder.CreateBr(forCondY);
	}

	// building the y condition //
	{
	IRBuilder<> builder(forCondY);
	auto* y = builder.CreateLoad(yAddress, "y");
	auto* isSmaller = builder.CreateICmpSLT(y, ysize);
	builder.CreateCondBr(isSmaller, forBodyY, forIncX);
	}

	// building the y body //
	{
	IRBuilder<> builder(forBodyY);
	builder.CreateStore(zeroInt, zAddress);
	builder.CreateBr(forCondZ);
	}

	// building the z condition //
	{
	IRBuilder<> builder(forCondZ);
	auto* z = builder.CreateLoad(zAddress, "z");
	auto* isSmaller = builder.CreateICmpSLT(z, zsize);
	builder.CreateCondBr(isSmaller, loopBodyStart, forIncY);
	}

	// building the z increase //
	{
	IRBuilder<> builder(forIncZ);
	auto* z = builder.CreateLoad(zAddress, "z");
	auto* plusOne = builder.CreateAdd(z, oneInt);
	builder.CreateStore(plusOne, zAddress);
	builder.CreateBr(forCondZ);
	}

	// building the y increase //
	{
	IRBuilder<> builder(forIncY);
	auto* y = builder.CreateLoad(yAddress, "y");
	auto* plusOne = builder.CreateAdd(y, oneInt);
	builder.CreateStore(plusOne, yAddress);
	builder.CreateBr(forCondY);
	}

	// building the x increase //
	{
	IRBuilder<> builder(forIncX);
	auto* x = builder.CreateLoad(xAddress, "x");
	auto* plusOne = builder.CreateAdd(x, oneInt);
	builder.CreateStore(plusOne, xAddress);
	builder.CreateBr(forCondX);
	}
	
	// move return instruction to the new end //
	for (auto& instr : instructions(*f)) {
		if (auto* retInstr = dyn_cast<ReturnInst>(&instr)) {
			IRBuilder<> builder(retInstr->getParent());
			builder.CreateBr(forIncZ);
			IRBuilder<> endBuilder(forEndX);
			endBuilder.CreateRetVoid();
			retInstr->eraseFromParent();
			break;
		}
	}

	// move function init instructions to the new
	// head of the function. We assume that we find
	// first allocate instructions and second
	// store instructions, which initialize the
	// function's arguments
	vector<Instruction*> instToMove; // instructions that will be moved //
	auto instIt = loopBodyStart->begin();
	while (dyn_cast<AllocaInst>(&*instIt)) {
		instToMove.push_back(&*instIt);
		++instIt;
	}
	while (dyn_cast<StoreInst>(&*instIt)) {
		instToMove.push_back(&*instIt);
		++instIt;
	}
	if (instToMove.size() == 0 && f->getArgumentList().size() != 0) {
		errs() << "***WARNING: loop wrapping pass found no alloca & store instructions, which ";
		errs() << "belong to the original function head, although the function ";
		errs() << "has arguments.\n";
	}

	// move the instructions //
	for (auto inst = instToMove.rbegin(); inst != instToMove.rend(); ++inst) {
		(*inst)->moveBefore(&*f->begin()->begin());
	}

	return make_tuple(xAddress, yAddress, zAddress);
}

//! Replaces the calls of get_global_id() with the appropriate loop variable.

//! The function \param f must be processed by function insertLoops().
//! \param xAddress, \param yAddress, \param zAddress must be the return
//! values by insertLoops(). The function searches for calls of get_global_id
//! and replaces that values with the appropriate loop variable.
void loop_wrapping_pass::replaceCalls(Function* f, Value* xAddress, Value* yAddress, Value* zAddress) {
	auto error = [] (const string& message) { throwError("replaceCalls", message); };

	auto insertLoad = [] (BasicBlock* parent, Value* loadAddress) {
		IRBuilder<> builder(parent);
		return builder.CreateLoad(loadAddress);
	};
	
	Function* ggid = f->getParent()->getFunction("get_global_id");
	if (!ggid) { // get global id is not called in this module //
		return;
	}
	
	// we need to collect the callInst of get_global_id()
	// first, as changing the underlying iterator data,
	// while iterating, results in seg faults
	vector<CallInst*> worklist;
	for (auto& inst : instructions(f)) {
		if (auto* callInst = dyn_cast<CallInst>(&inst)) {
			if (callInst->getCalledFunction() == ggid) {
				errs() << "Found ggid call: " << *callInst << '\n';
				worklist.push_back(callInst);
			}
		}
	}
	
	// now exchange the call instructions with loads from the
	// loop variables
	for (CallInst* callInst : worklist) { // iter over the get_global_id calls //
		auto* axis = dyn_cast<ConstantInt>(callInst->arg_begin()->get());
		if (axis) {
			LoadInst* load;
			ZExtInst* extend;
			switch (axis->getValue().getLimitedValue()) {
				case 0:
					load = new LoadInst(xAddress, "x", callInst);
					extend = new ZExtInst(load, callInst->getCalledFunction()->getReturnType(), "x.ext");
					ReplaceInstWithInst(callInst, extend);
					break;
				case 1:
					load = new LoadInst(yAddress, "y", callInst);
					extend = new ZExtInst(load, callInst->getCalledFunction()->getReturnType(), "y.ext");
					ReplaceInstWithInst(callInst, extend);
					break;
				case 2:
					load = new LoadInst(zAddress, "z", callInst);
					extend = new ZExtInst(load, callInst->getCalledFunction()->getReturnType(), "z.ext");
					ReplaceInstWithInst(callInst, extend);
					break;
				default:
					error("unexpected value in get_global_id call");
			}
		}
		else {
			error("unexpected value in get_global_id call");
		}	
	}
}

//! Returns the clone of \param parent with three new arguments size_x, size_y, size_z.

//! The clone's name is equal to the parent's name with "_lwrapped" suffix.
Function* loop_wrapping_pass::cloneAndAddArgs(Function* parent) {
	auto error = [] (const string& message) { throwError("cloneAndAddArgs", message); };
	if (!parent) {
		error("Got nullptr as an argument");
	}

	if (parent->isDeclaration()) {
		error("Got a declaration function to copy from");
	}

	// get old function argument types //
	std::vector<Type*> vT = parent->getFunctionType()->params().vec();

	// add i64 types for size_x, size_y, size_z //
	for (unsigned short count = 0; count < 3; ++count) {
		vT.push_back(IntegerType::get(parent->getContext(), 32));
	}
	FunctionType* fT = FunctionType::get(parent->getReturnType(), vT, false);

	// create new function with '_lwrapped' suffix //
	Function* child = Function::Create(fT, parent->getLinkage(), parent->getName() + "_lwrapped", parent->getParent());

	// linking between child and parent's arguments //
	ValueToValueMapTy vMap;

	auto ait = child->arg_begin();
	for (auto& arg : parent->args()) {
		vMap.insert(std::make_pair(&arg, WeakVH(&(*ait))));
		// give the arguments the same name //
		ait->setName(arg.getName());
		++ait;
	}

	// just for better reading: give names to additional arguments //
	if (child->arg_size() - parent->arg_size() == 3) {
		ait->setName("size_x");
		++ait;
		ait->setName("size_y");
		++ait;
		ait->setName("size_z");
	}

	// clone old content to new function //
	SmallVector<ReturnInst*, 10> dummy;
	CloneFunctionInto(child, parent, vMap, true, dummy);

	return child;
}

void loop_wrapping_pass::throwError(const string& functionName, const string& message) {
	throw runtime_error("PASS loop_wrapping_pass (create loop wrapped function), "
	                    "FUNC " + functionName + ":\n" + message);
}

char loop_wrapping_pass::ID = 0;
static RegisterPass<loop_wrapping_pass> X("lwpass", "Creates a loop wrapped function of each "
                                       "function in the module");
