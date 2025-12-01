// TARJAN ALGO
#include "CallGraphAnalysis.hpp"
// ORC JIT + thread safe module
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

// legacy function pass manager (small and simple to plug in)
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <map> //std::map for binop precedence
#include <vector> 
#include <memory> //for unique ptr
//====================================TOKENIZER===================================
//
enum Token { tok_eof= -1, //end of file
	tok_def= -2, //fun def 
	tok_identifier=-3, // assign function names like 
	tok_number= -4,	// type number
	tok_extern = -5
	};

//static means they persist between fun calls live in program!!
//global for this file
static std::string IdentifierStr; // save name
static double NumVal; // save value  

//function to buffer characters to make tokens
static int getTok(){
    // make lastChar save between calls
    static int LastChar= ' ';
    
    //start by skipping white space ie ' ' OR '\n' ...
    while(isspace(LastChar))
	LastChar= getchar(); //isspace and getChar are CPP functions getchar reads char from buffer

    //now we're checking for a string which would be identifiers in this case!
    if(isalpha(LastChar)){
	IdentifierStr = LastChar; // build IdentifierStr
	
	//loop through entire string aka identifier
	while(isalnum((LastChar = getchar())))
	    IdentifierStr += LastChar; //continue Building
	    
	    //check if str literall says this bc we need to identify key words
	if(IdentifierStr == "def")
	    return tok_def;
	if(IdentifierStr == "extern")
		return tok_extern;
	else //otherwise return function
	    return tok_identifier;
    }
    
    //now check for number values
    if(isdigit(LastChar) || LastChar == '.'){
	std::string NumStr;
	do{                                          
	    NumStr += LastChar;
	    LastChar = getchar();
        } while (isdigit(LastChar) || LastChar == '.'); // allow for decimals
	
	NumVal = strtod(NumStr.c_str(), nullptr);// string to double strtod using  
	return tok_number;
    }
    if(LastChar == '#'){
	//skip line or hit end of file
	do{
	    LastChar = getchar();
	} while(LastChar != EOF && LastChar != '\r' && LastChar!= '\n');
	
	//keep iterating until next token
	if(LastChar != EOF)
	    return getTok();
    }
    //get the end of the file
    if(LastChar == EOF)
	return tok_eof;
    
    //now we gotta process single character tokens
    //fallthough case
    int ThisChar = LastChar;
    LastChar = getchar();
    return ThisChar;
}

//=======================================LEXER=============================================================
namespace {
//
// Blueprint for all expressions in the language, the base, 
// I think all other tress will extend this one
class ExprAST{
public: // must make function in calss public bc they're private by default

	virtual ~ExprAST()=default; //virtual makes a function that can be over ridden in during runtime to achieve polymorphism
								//the ~ signifies that the function is the destructor for the class
								//default simply just means to use compiler generated fn for the destructor
	
	virtual llvm::Value*codegen() = 0; 	// the * means it is a pointer to the data somewhere in mem
										// the = 0 means that is a pure virtual function meaning no fn body; 
										// for the java brains its an interface function  
	
};
class NumberExprAST : public ExprAST{
	double Val;

public:
	//constructor - the Val(Val){} is the initializer list
	//the : Val(Val) means {this->Val = Val} just works more efficiently the fn Name matches the private attributes, and the Param is jsut a random value to be referenced inside of the func
	NumberExprAST(double Val) : Val(Val){}

	//just declaring that the function is going to be an overwritten virtual function
	//the * means its reference to an object
	llvm::Value *codegen() override;
};
/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(const std::string &Name) : Name(Name) {}
  llvm::Value *codegen() override;
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
    : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
  llvm::Value *codegen() override;
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
    : Callee(Callee), Args(std::move(Args)) {}
  llvm::Value *codegen() override;
};
/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;

public:
  PrototypeAST(const std::string &Name, std::vector<std::string> Args)
    : Name(Name), Args(std::move(Args)) {}

  const std::string &getName() const { return Name; }
  llvm::Function *codegen();
};

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<ExprAST> Body)
	: Proto(std::move(Proto)), Body(std::move(Body)) {}
  llvm::Function *codegen();
};
}//namespace
//=======================PARSER====================================
//foward declarations and helpers
static int CurTok; 

//get next token from lexer and store iti n CurrTok
static int getNextToken(){
	return CurTok = getTok();
}

//error handling
std::unique_ptr<ExprAST> LogError(const char *Str) {
	fprintf(stderr, "Error %s\n", Str);
	return nullptr; //return null to show failure
}

//binop precendence 
static std::map<char, int> BinopPrecedence;

//get token precendence of current char
static int GetTokPrecedence(){
	if(!isascii(CurTok))
		return -1;
	
	//now ensure its a real binop
	int TokPrec = BinopPrecedence[CurTok];
	if (TokPrec <= 0)
		return -1;
	return TokPrec;
}


std::unique_ptr<PrototypeAST> LogErrorP(const char* Str){
	LogError(Str);
	return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

static std::unique_ptr<ExprAST> ParseNumberExpr(){
	auto Result = std::make_unique<NumberExprAST>(NumVal);
	getNextToken();
	return std::move(Result);
}

//get expr w parenthesis
static std::unique_ptr<ExprAST>ParseParenExpr() {
	getNextToken(); // move on from '(' which is currTok
	auto V = ParseExpression();
	if(!V)
		return nullptr;

	if(CurTok != ')')
		return LogError("expected ')'");
	getNextToken();// move on from ')'
	return V;
}

//identifier Expr
//identifier
//identifier '(' expression ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
	std::string IdName = IdentifierStr;

	getNextToken();// move on from the id name
	
	if (CurTok != '(') //simple cariable ref;
		return std::make_unique<VariableExprAST>(IdName);

	getNextToken();
	
	std::vector<std::unique_ptr<ExprAST>> Args;
	if (CurTok != ')') {// if not closed immediately
		while(true) {
			//parse each argument exp
			if(auto Arg = ParseExpression()) 
				Args.push_back(std::move(Arg));
					//push_back is a vector fn 
			else // error in parsing
				return nullptr;
			if (CurTok == ')') // it ended
				break;
			if (CurTok != ',') //mishap in user 
				return LogError("Expected ')' or ',' in arg List");
			getNextToken();
		}
	}

	getNextToken(); // move on from ')' which completes the func
	return std::make_unique<CallExprAST>(IdName, std::move(Args));
}
//primary
//switch case for everything
//identifier expressions
//numberexpressions
//parenexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
	switch(CurTok) {
	default:
		return LogError("unknown token when expecting an expression");
	case tok_identifier:
		return ParseIdentifierExpr();
	case tok_number:
		return ParseNumberExpr();
	case '(':
		return ParseParenExpr();
	}
}

//binary operator Right hand side
// ( '+' primary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, 
									 std::unique_ptr<ExprAST> LHS) {
	//if token is binop get precedence
	while(true) {
		//check the value of the operator
		int TokPrec = GetTokPrecedence();
		
		//ensure the oprerator follows Pemdas
		if(TokPrec < ExprPrec)
			return LHS;
		
		//move on and get the other expr/nums/identifiers
		int BinOp = CurTok;
		getNextToken();

		//get other bs
		auto RHS = ParsePrimary();
		if(!RHS)
			return nullptr;

		// swap order if left is lower precedence like + is to *
		int NextPrec = GetTokPrecedence();
		if(TokPrec < NextPrec) {
			RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
			if(!RHS)
				return nullptr;
		}

		//merge LHS & RHS
		LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
	}
}

static std::unique_ptr<ExprAST> ParseExpression() {
	auto LHS = ParsePrimary();
	if(!LHS)	
		return nullptr;
	return ParseBinOpRHS(0, std::move(LHS));
}

//prototype
static std::unique_ptr<PrototypeAST> ParsePrototype() {
	if(CurTok != tok_identifier)
		return LogErrorP("Expected Function in proto");
	
	std::string FnName = IdentifierStr;
	getNextToken();

	if(CurTok != '(')
		return LogErrorP("Expected '(' in prototype");

	std::vector<std::string> ArgNames;
	while(getNextToken() == tok_identifier)
		ArgNames.push_back(IdentifierStr);
	if(CurTok != ')')
		return LogErrorP("Expected ')' in prototype");
	
	//if we made it here its goood
	getNextToken();

	return std::make_unique<PrototypeAST> (FnName, std::move(ArgNames));
}

static std::unique_ptr<FunctionAST> ParseDefinition() {
	getNextToken(); // move on from 'def' token
	auto Proto = ParsePrototype();
	if(!Proto)
		return nullptr;

	if(auto E = ParseExpression())
		return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
	return nullptr;
}

static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
	if (auto E = ParseExpression()){
//make anonymous proto
		auto Proto = std::make_unique<PrototypeAST>("__anon_expr", std::vector<std::string>());
		
		return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
	}
	return nullptr;
}

//external extern prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
	getNextToken(); //move on from 'extern'
	return ParsePrototype();
}


//===============================================CODE GEN MAKE IT REAL ========================================

//So every AST node gets a codegen() methos that genereates LLVM IR
//It builds itself poco a poco 

//The global environment for types, constants, and internals ie threading
static std::unique_ptr<llvm::LLVMContext> TheContext;  // Owns all LLVM core data structures
 //the .ll file we're building
static std::unique_ptr<llvm::Module> TheModule;         // Contains all functions and globals for this program
// helper to make IR instruction (pseudo assembly)
static std::unique_ptr<llvm::IRBuilder<>> Builder;      // Helper that knows where to insert instructions
//symbol table
static std::map<std::string, llvm::Value *> NamedValues; // Symbol table: maps names -> LLVM values

// this JIT has 
// a compile layer that makes IR into machine code
// a link layer for machine code and memory
// a symbol table for looking up alr compiled funcs
static std::unique_ptr<llvm::orc::LLJIT> TheJit;

//Resource Tracker For THIS module
static llvm::orc::ResourceTrackerSP RT;

//optimization pass manager
//legacy bc of simplicity
static std::unique_ptr<llvm::legacy::FunctionPassManager> TheFPM;

//For Tarjans Cycle Detection
//CGA is the object that will be able to call the functions needed
static CallGraphAnalyzer CGAnalyzer;
// just a pointer helper
static std::string CurrentFunction = "";

//ExitOnError helps print errors and pass them back up
static llvm::ExitOnError ExitOnErr;

//return nullptr when codegen fails

llvm::Value *LogErrorV(const char *Str) {
	LogError(Str);
	return nullptr;  // nullptr propagates up to stop code generation
}

// Creates a floating-point constant that are immutable, Uniqued, and shared across entire context
llvm::Value *NumberExprAST::codegen() {
	// Create an LLVM constant. Constants are uniqued and immutable.
	return llvm::ConstantFP::get(*TheContext, llvm::APFloat(Val));
}

//Looks up variables(Currently only function parameters can be varaibles)
llvm::Value *VariableExprAST::codegen() {
	// Look this variable up in the function's symbol table
	llvm::Value *V = NamedValues[Name];
	if(!V)
		return LogErrorV("Unknown Variable Name");
	return V;  // Return the LLVM Value for this variable
}

// Grabs operators and makes them parent nodes of another set of parent nodes or numbers.

//        [+]           <- BinaryExprAST (Op: '+')
//       /   \
//      2    [*]        <- BinaryExprAST (Op: '*')
//          /   \
//         3     4      <- NumberExprAST nodes
//this function builds expression recursively and executes bottom up bc it recurses down to the numbers and starts executing bin ops
llvm::Value *BinaryExprAST::codegen() {
	// Recursively generate code for left operand
	llvm::Value *L = LHS->codegen();
	// Recursively generate code for right operand
	llvm::Value *R = RHS->codegen();
	if(!L || !R)
		return nullptr;  // Error in child, propagate up

	switch (Op) {
		case '+':
			// Create floating-point add instruction. "addtmp" is optional name hint.
			return Builder->CreateFAdd(L, R, "addtmp");
		case '-':
			return Builder->CreateFSub(L, R, "subtmp");
		case '*' :
			return Builder->CreateFMul(L, R, "multmp");
		case '<':
			// Comparison returns i1 (1-bit int), must convert to double
			L = Builder->CreateFCmpULT(L, R, "cmptmp");
			// Convert bool 0/1 to double 0.0 or 1.0
			return Builder->CreateUIToFP(L, llvm::Type::getDoubleTy(*TheContext), "booltmp");
		default:
			return LogErrorV("invalid binary operator");
	}
}
//handles 3 things
//user defined functions
//external libraries LLVM has alr x-compiled. ie sin cos from C-lang
//foward declarations ie:
//extern foo(x) #declare
//def bar() foo(3) #use b4 defining
//def foo(x) x * 2 #define
llvm::Value *CallExprAST::codegen() {
	// Look up the function in module's symbol table
	llvm::Function *CalleeF = TheModule->getFunction(Callee);
	//Record Call in Tarjan Graph
	if (!CurrentFunction.empty()) {
		CGAnalyzer.recordCall(CurrentFunction, Callee);
	}
	if(!CalleeF)
		return LogErrorV("Unknown function referenced");

	// Check argument count matches
	if(CalleeF->arg_size() != Args.size())
		return LogErrorV("Incorrect # arguments passed");

	std::vector<llvm::Value *> ArgsV;
	for (unsigned i = 0, e = Args.size(); i != e; ++i) {
		// Generate code for each argument
		ArgsV.push_back(Args[i]->codegen());
		if(!ArgsV.back())
			return nullptr;  // Error in argument, propagate
	}

	// Create the function call instruction
	return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

// Generates code for function prototypes (declarations)
llvm::Function *PrototypeAST::codegen() {
	// Make the function type: double(double,double) etc.
	std::vector<llvm::Type *> Doubles(Args.size(), llvm::Type::getDoubleTy(*TheContext));
	llvm::FunctionType *FT =
		llvm::FunctionType::get(llvm::Type::getDoubleTy(*TheContext),  // Return type
						  Doubles,                          // Argument types
						  false);                           // Not vararg

	// Create function with external linkage so it can be called from outside
	llvm::Function *F =
		llvm::Function::Create(FT, llvm::Function::ExternalLinkage, Name, TheModule.get());

	// Set names for all arguments (makes IR readable)
	unsigned Idx = 0;
	for (auto &Arg : F->args())
		Arg.setName(Args[Idx++]);

	return F;
}

llvm::Function *FunctionAST::codegen() {
	// Check for existing function from previous 'extern' declaration
	llvm::Function *TheFunction = TheModule->getFunction(Proto->getName());

	if(!TheFunction)
		TheFunction = Proto->codegen();  // Generate prototype if needed

	if(!TheFunction)
		return nullptr;

	// Error if function already has a body
	if(!TheFunction->empty())
		return (llvm::Function*)LogErrorV("Function cannot be redefined.");
	CurrentFunction = Proto->getName();
	CGAnalyzer.registerFunction(CurrentFunction);

	// Create a new basic block to start insertion into
	llvm::BasicBlock *BB = llvm::BasicBlock::Create(*TheContext, "entry", TheFunction);
	Builder->SetInsertPoint(BB);  // All new instructions go here

	// Record the function arguments in the NamedValues map
	NamedValues.clear();  // Clear any previous function's args
	for(auto &Arg : TheFunction->args())
		NamedValues[std::string(Arg.getName())] = &Arg;  // Add args to symbol table

	// Generate code for function body
	if (llvm::Value *RetVal = Body->codegen()) {
		// Finish off the function with return instruction
		Builder->CreateRet(RetVal);

		// Validate the generated code, checking for consistency
		verifyFunction(*TheFunction);

		// Run the optimization passes (instruction combining, reassociation, GVN, CFG simplification)
		if (TheFPM)
			TheFPM->run(*TheFunction);

		CurrentFunction = "";
		return TheFunction;
	}
// Error reading body, remove function from module
	CurrentFunction = "";
	TheFunction->eraseFromParent();
	return nullptr;
}


//===============================================iTOP LEVEL and JIT Driver============
static void InitializeModuleAndPassManager(){
	//create new llvmContext for each IR batch bc we move ownership of context into JIT, don't want to lose it completely
	TheContext = std::make_unique<llvm::LLVMContext>();

	//make the module w stack and heap data to be root of IR Container
	TheModule = std::make_unique<llvm::Module>("my cool Jit", *TheContext);

	//Set layout from Jit
	TheModule->setDataLayout(TheJit->getDataLayout());

	//Create a new resource Tracker for this module
	RT = TheJit->getMainJITDylib().createResourceTracker();

	//Procedurally build instructions and remember current insertion point
	Builder = std::make_unique<llvm::IRBuilder<>>(*TheContext);

	//create the func optimizer
	// runs per compiled function, efficient for small langs
	TheFPM = std::make_unique<llvm::legacy::FunctionPassManager>(TheModule.get());

	//Add passes
	//combine redundant ops
	TheFPM->add(llvm::createInstructionCombiningPass()); 
	//reorder expressions for better optimiztion
	TheFPM->add(llvm::createReassociatePass());
	//global value numbering (CSE)
	TheFPM->add(llvm::createGVNPass());
	//merge blocks and remove dead branches
	TheFPM->add(llvm::createCFGSimplificationPass());

	//Call before running passes
	TheFPM->doInitialization();
}

static void InitializeJit(){
	//register CPU with LLVM backend, key to emitting native code
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();
	llvm::InitializeNativeTargetAsmParser();

	//The .create func returns llvm::Expected<LLJIT>
	//the ExitOnErr unwraps it or aborts w an err msg
	TheJit = ExitOnErr(llvm::orc::LLJITBuilder().create());
}

static void HandleDefinition() {
    if (auto FnAST = ParseDefinition()) {
        if (auto *FnIR = FnAST->codegen()) {
            fprintf(stderr, "Read function definition:");
            FnIR->print(llvm::errs());
            fprintf(stderr, "\n");
            
            // DON'T execute it - just keep it in the module for now
            // We'll add it to JIT when needed
        }
    } else {
        getNextToken();
    }
}

static void HandleExtern() {
	if (auto ProtoAST = ParseExtern()) {
		if (auto *FnIR = ProtoAST->codegen()) {  // Generate function declaration
			fprintf(stderr, "Read extern: ");
			FnIR->print(llvm::errs());
			fprintf(stderr, "\n");
		}
	} else {
		// Skip token for error recovery
		getNextToken();
	}
}

static void HandleTopLevelExpression() {
    if (auto FnAST = ParseTopLevelExpr()) {
        if (auto *FnIR = FnAST->codegen()) {
            fprintf(stderr, "Read top-level expression:");
            FnIR->print(llvm::errs());
            fprintf(stderr, "\n");

            // Create ThreadSafeModule (simpler constructor in newer LLVM)
            llvm::orc::ThreadSafeModule TSM(std::move(TheModule), std::move(TheContext));
            
            ExitOnErr(TheJit->addIRModule(RT, std::move(TSM)));
            // Look up and execute the anonymous expression
            auto Sym = ExitOnErr(TheJit->lookup("__anon_expr"));
            
            // Cast to function pointer and call
            auto *FP = Sym.toPtr<double()>();
            fprintf(stderr, "Evaluated to %f\n", FP());
            
			//Remove the anonymous Expression to avoid duplicate symbols
			ExitOnErr(RT->remove());
			
            // Recreate module for next expression
            InitializeModuleAndPassManager();
        }
    } else {
        getNextToken();
    }
}

static void MainLoop() {
	while(true){
		fprintf(stderr, "ready> ");
		switch (CurTok) {
		case tok_eof:
			CGAnalyzer.analyze();
			CGAnalyzer.printResults();
			return;
		case ';' : //ignore top-lvl semicolons
			getNextToken();
			break;
		case tok_def:
			HandleDefinition();
			break;
		case tok_extern:
			HandleExtern();
			break;
		default:
			HandleTopLevelExpression();
			break;
		}
	}
}
//link extern libraries
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - print a single character (takes double because language uses doubles)
extern "C" DLLEXPORT double putchard(double X) {
    fputc((char)X, stderr);  // print character to stderr
    return 0;                 // return 0 to match our double-returning convention
}

/// printd - print a double
extern "C" DLLEXPORT double printd(double X) {
    fprintf(stderr, "%f\n", X); // print the double value
    return 0;
}


int main(){
	// Install standard binary operators
	// 1 is lowest precedence
	BinopPrecedence['<'] = 10;
	BinopPrecedence['+'] = 20;
	BinopPrecedence['-'] = 20;
	BinopPrecedence['*'] = 40;  // highest
	
	InitializeJit();

	{
		//Get the Symbol table
		auto &JD = TheJit->getMainJITDylib();
		//Mangle transforms function names to match the OS's calling convention
		llvm::orc::MangleAndInterner Mangle(TheJit->getExecutionSession(), 
											TheJit->getDataLayout());
		//means of holding symbol definitions (getting the data a space in memory)
		llvm::orc::SymbolMap Symbols;
		
		//next 2 statements are getting the two specified functions
		//ExecutorAddr... gets the mem addr of the defined C++ fn above
		//JITSymbolFlags... marks the func as callable from JIT code
		Symbols[Mangle("putchard")] = 
			llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(&putchard),
										 llvm::JITSymbolFlags::Exported);
		Symbols[Mangle("printd")] = 
			llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(&printd),
										 llvm::JITSymbolFlags::Exported);
		
		//register all symbols into JIT's sym table
		//means of making it globally available
		ExitOnErr(JD.define(llvm::orc::absoluteSymbols(Symbols)));
	}

	InitializeModuleAndPassManager();

	// Prime the first token
	fprintf(stderr, "ready> ");
	getNextToken();

	// Run the main interpreter loop
	MainLoop();

	// Print out all of the generated code
	TheModule-> print(llvm::errs(), nullptr);

	return 0;
}
