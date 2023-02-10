#include "feature_matchers.h"
#include "feature_vector.h"

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include "Globals.h"
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

using std::ofstream;
using std::map;
using std::string;


namespace json = rapidjson;

using namespace clang::tooling;
using namespace llvm;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

static cl::extrahelp MoreHelp("\nf1x-transform is a tool used internally by f1x\n");

static llvm::cl::OptionCategory ReteCategory("rete options");

const std::string RETE_INCLUDE = "-I/llvm-3.8.1/lib/clang/3.8.1/include/";

const std::string RETE_CXX_INCLUDE = "-I/llvm-3.8.1/src/libcxx/include/";


static cl::opt<std::string>
    OutputFile("output", cl::desc("output file"), cl::cat(ReteCategory));

static cl::opt<bool>
    DUChainData("get-chain-data", cl::desc("DU chain information"), cl::cat(ReteCategory));



int main(int argc, const char **argv) {
CommonOptionsParser OptionsParser(argc, argv, ReteCategory);
ClangTool Tool(OptionsParser.getCompilations(),
                OptionsParser.getSourcePathList());

ArgumentsAdjuster ardj1 = getInsertArgumentAdjuster(RETE_INCLUDE.c_str());
Tool.appendArgumentsAdjuster(ardj1);

LoopPrinter Printer;
AssignmentsFeatureCalc Assignments;
ASTIfCalc IfMatcherVar;
RValueFeatureCalc rval;
UnaryLValFeatureCalc ulval;
MatchFinder Finder;
Finder.addMatcher(LoopMatcher, &Printer);
Finder.addMatcher(LValueMatcher, &Assignments);
Finder.addMatcher(IfMatcher, &IfMatcherVar);
Finder.addMatcher(RValueMatcher, &rval);
Finder.addMatcher(UnaryLValMatcher, &ulval);

Tool.run(newFrontendActionFactory(&Finder).get());

json::Document document;
const char* json = "{}";
document.Parse(json);
json::Document::AllocatorType& allocator = document.GetAllocator();
if(DUChainData) {
    for(auto it: du_chains) {
        json::Value node(json::kArrayType);
        for (auto it2: it.second) {
            json::Value chainData(json::kObjectType);
            ChainNodeData data = it2;
            
            chainData.AddMember("node", json::Value().SetString(data.node.c_str(), allocator), allocator);
            
            json::Value lvals(json::kArrayType);
            for(auto vars: data.l_vals) {
                lvals.PushBack(json::Value().SetString(vars.c_str(), allocator), allocator);
            }
            
            json::Value rvals(json::kArrayType);
            for(auto vars: data.r_vals) {
                rvals.PushBack(json::Value().SetString(vars.c_str(), allocator), allocator);
            }

            chainData.AddMember("lvals", lvals, allocator);
            chainData.AddMember("rvals", rvals, allocator);

            chainData.AddMember("startColumn", json::Value().SetInt(data.startCol), allocator);
            chainData.AddMember("endColumn", json::Value().SetInt(data.endCol), allocator);

            node.PushBack(chainData, allocator);
        
        }
        document.AddMember(json::Value().SetString(std::to_string(it.first).c_str(), allocator), node, allocator);
    }
}
else {
    for(auto it: features) {
        json::Value node(json::kArrayType);
    
        for (auto it2: it.second) {
            json::Value varData(json::kObjectType);
            VarData data = it2.second;
            
            varData.AddMember("varName", json::Value().SetString(data.varName.c_str(), allocator), allocator);
            varData.AddMember("varType", json::Value().SetString(data.varType.c_str(), allocator), allocator);
            varData.AddMember("isGlobal", json::Value().SetBool(data.isGlobal), allocator);
            
            json::Value operators(json::kArrayType);
            for(auto op: data.surroundingVars) {
                operators.PushBack(json::Value().SetString(op.c_str(), allocator), allocator);
            }
            varData.AddMember("operators", operators, allocator);
            
            varData.AddMember("declLocation", json::Value().SetInt(data.declLocation), allocator);
            varData.AddMember("startColumn", json::Value().SetInt(data.startCol), allocator);
            varData.AddMember("endColumn", json::Value().SetInt(data.endCol), allocator);
            varData.AddMember("startScope", json::Value().SetInt(data.startScope), allocator);
            varData.AddMember("endScope", json::Value().SetInt(data.endScope), allocator);

            json::Value nodeData(json::kArrayType);
            nodeData.PushBack(json::Value().SetString(it2.first.c_str(), allocator), allocator);
            nodeData.PushBack(varData, allocator);
            node.PushBack(nodeData, allocator);
        }
        document.AddMember(json::Value().SetString(std::to_string(it.first).c_str(), allocator), node, allocator);
    }
}
json::StringBuffer sb;
json::PrettyWriter<json::StringBuffer> writer(sb);

document.Accept(writer);    // Accept() traverses the DOM and generates Handler events.
//puts(sb.GetString());

ofstream outdata;
outdata.open(OutputFile);
outdata<<sb.GetString();
outdata.close();
}
