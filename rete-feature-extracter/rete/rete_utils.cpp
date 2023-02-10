#include <algorithm>
#include <map>
#include <stack>
#include <sstream>
#include <iostream>
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/Type.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/ASTContext.h"
#include "rete_utils.h"
#include "Config.h"


namespace json = rapidjson;

using namespace clang;
using namespace llvm;
using std::string;
using std::pair;
using std::stack;
using std::vector;
using std::map;
using std::make_pair;

const BuiltinType::Kind DEFAULT_NUMERIC_TYPE = BuiltinType::Long;
const string DEFAULT_POINTEE_TYPE = "void";


std::shared_ptr<std::vector<clang::SourceRange>> globalConditionalsPP = std::shared_ptr<std::vector<clang::SourceRange>>(new std::vector<clang::SourceRange>());

const unsigned F1XAPP_WIDTH = 32;
const unsigned F1XAPP_VALUE_BITS = 10;

string toString(const Stmt *stmt) {
	/* Special case for break and continue statement
		Reason: There were semicolon ; and newline found
		after break/continue statement was converted to string
	*/
	if (dyn_cast<BreakStmt>(stmt))
	return "break";

	if (dyn_cast<ContinueStmt>(stmt))
	return "continue";

	LangOptions LangOpts;
	PrintingPolicy Policy(LangOpts);
	string str;
	raw_string_ostream rso(str);

	stmt->printPretty(rso, nullptr, Policy);

	string stmtStr = rso.str();
	return stmtStr;
}

string toString(const Expr *expr) {
	/* Special case for break and continue statement
		Reason: There were semicolon ; and newline found
		after break/continue statement was converted to string
	*/
	LangOptions LangOpts;
	PrintingPolicy Policy(LangOpts);
	string str;
	raw_string_ostream rso(str);

	expr->printPretty(rso, nullptr, Policy);

	string stmtStr = rso.str();
	return stmtStr;
}

int getEndScope(const NamedDecl* node, SourceManager& srcMgr, ASTContext* context) {
	int endScope = -1;
	if(node) {
		ASTContext::DynTypedNodeList  parents  = context->getParents(*node);
		int depth = 6;
		while(depth--) {
			if(parents.empty()) break;

			for(int i=parents.size()-1;i>=0;i--) {
			
			const CompoundStmt* compoundStmt = parents[i].get<CompoundStmt>();
			if(compoundStmt) {
				depth = 0;
				endScope = srcMgr.getPresumedLoc(compoundStmt->getLocEnd()).getLine();
				break;
			}
			
			const ForStmt* forStmt = parents[i].get<ForStmt>();
			if(forStmt) {
				depth = 0;
				endScope = srcMgr.getPresumedLoc(forStmt->getLocEnd()).getLine();
				break;
			}
			
			const WhileStmt* whileStmt = parents[i].get<WhileStmt>();
			if(whileStmt) {
				depth = 0;
				endScope = srcMgr.getPresumedLoc(whileStmt->getLocEnd()).getLine();
				break;
			}

			const IfStmt* ifStmt = parents[i].get<IfStmt>();
			if(ifStmt) {
				depth = 0;
				endScope = srcMgr.getPresumedLoc(ifStmt->getLocEnd()).getLine();
				break;
			}

			const FunctionDecl* functionDecl = parents[i].get<FunctionDecl>();
			if(functionDecl) {
				depth = 0;
				endScope = srcMgr.getPresumedLoc(functionDecl->getLocEnd()).getLine();
				break;
			}

			}
			parents = context->getParents(parents[0]);
			
		}
	}

	return endScope;
}


class CollectVariables : public StmtVisitor<CollectVariables> {

public:
	std::vector< std::pair<std::string, VarData> > &feat;
	string nodeType;
	SourceManager& srcMgr;
	ASTContext* context;
	CollectVariables(std::vector< std::pair<std::string, VarData> > & feats, string _nodeType, ASTContext* _context ):
	feat(feats), nodeType(_nodeType), context(_context),srcMgr(_context->getSourceManager()) {}


	void Visit(Stmt* S) {
		StmtVisitor<CollectVariables>::Visit(S);
	}

	void VisitBinaryOperator(BinaryOperator *Node) {
		Visit(Node->getLHS());
		Visit(Node->getRHS());
	}

	void VisitUnaryOperator(UnaryOperator *Node) {
		Visit(Node->getSubExpr());
	}

	void VisitArraySubscriptExpr(ArraySubscriptExpr* Node) {
		Visit(Node->getBase());


	}

	void VisitMemberExpr(MemberExpr* Node) {
		VarData v;
		const NamedDecl* defn = Node->getFoundDecl();
		int declLine = -1, startCol, endCol;
		if(defn) {
			declLine = srcMgr.getPresumedLoc(defn->getSourceRange().getBegin()).getLine();
		}
		defn->dump();
		v.startCol = srcMgr.getPresumedLoc(Node->getSourceRange().getBegin()).getColumn();
		v.endCol = srcMgr.getPresumedLoc(Node->getSourceRange().getEnd()).getColumn();
		std::string defFile = srcMgr.getPresumedLoc(defn->getSourceRange().getBegin()).getFilename();
		std::string sourceFile = srcMgr.getPresumedLoc(Node->getSourceRange().getBegin()).getFilename();
		if(defFile == sourceFile) {
			v.endScope = -1;
			v.startScope = -1;
		}
		else {
			v.endScope = getEndScope(defn, srcMgr, context);
		}
		v.startScope = declLine;
		std::cout<<v.startCol<<" "<<v.endCol<<" "<<v.startScope<<" "<<v.endScope<<" "<<defn->getNameAsString()<<"\n";
		feat.push_back(make_pair(nodeType, v));
	}

	void VisitImplicitCastExpr(ImplicitCastExpr *Node) {
		Visit(Node->getSubExpr());
	}

	void VisitDeclStmt(DeclStmt *Node) {
		for(auto it=Node->decl_begin(); it!=Node->decl_end(); it++) {
			if(VarDecl* var = dyn_cast<VarDecl>(*it)) {
				addNodeToFeat(var, NULL, feat, nodeType, srcMgr, context);
			}
		}
	}

	void VisitVarDecl(VarDecl *Node) {
		addNodeToFeat(Node, NULL, feat, nodeType, srcMgr, context);
	}

	void VisitParenExpr(ParenExpr *Node) {
		Visit(Node->getSubExpr());
	}
	void VisitIntegerLiteral(IntegerLiteral *node) {
		VarData v = {"constant", "constant", false, vector<string>(), -1, -1, -1};
		pair<string, VarData> n = make_pair(nodeType, v);
		for(auto it: feat) {
			if(it.first == n.first && it.second.varName == n.second.varName) 
				return ;
			}
		
		feat.push_back(n);
	}
	void VisitOpaqueValueExpr(OpaqueValueExpr *node) {
		Expr *SE = node->getSourceExpr()->IgnoreImpCasts();
		if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(SE)) {
			Visit(DRE);
		}
		Visit(node->getSourceExpr());
	}

	void VisitCharacterLiteral(CharacterLiteral *Node) {}


	void VisitDeclRefExpr(DeclRefExpr *node) {

		if (clang::VarDecl* VD = dyn_cast<clang::VarDecl>(node->getDecl()))
			addNodeToFeat(VD, node, feat, nodeType, srcMgr, context);

		}
	};


	std::vector< std::pair<std::string, VarData> >  collectFromExpr(const Expr *expr,
								std::vector< std::pair<std::string, VarData> > &feat, string nodeType, ASTContext* context) {
		CollectVariables T(feat, nodeType, context);
		T.Visit(const_cast<Expr*>(expr));
		return T.feat;
	}

	std::vector< std::pair<std::string, VarData> >  collectFromStmt(const Stmt *stmt,
								std::vector< std::pair<std::string, VarData> > &feat, string nodeType, ASTContext* context) {
		CollectVariables T(feat, nodeType, context);
		T.Visit(const_cast<Stmt*>(stmt));
		return T.feat;
}


void addNodeToFeat(
	const VarDecl* node, 
	DeclRefExpr * refNode,
	std::vector< std::pair<std::string, VarData> > &feat,
	string nodeType,
	SourceManager& srcMgr,
	ASTContext* context
) {
	string varName = node->getName();    

	const VarDecl* defn = node->getDefinition();
	int declLine = -1, startCol, endCol;
	if(defn) {
		declLine = srcMgr.getPresumedLoc(defn->getSourceRange().getBegin()).getLine();
	}
	else {
	declLine = srcMgr.getPresumedLoc(node->getCanonicalDecl()->getSourceRange().getBegin()).getLine();
	}
	vector<string> operators;
	int startScope = declLine, endScope = -1;
	if(!node->hasGlobalStorage())
		endScope = getEndScope(node, srcMgr, context);
	if(refNode) {
		ASTContext::DynTypedNodeList  parents  = context->getParents(*refNode);
		int depth = 6;
		while(depth--) {
			if(parents.empty()) break;

			for(int i=parents.size()-1;i>=0;i--) {
			const BinaryOperator* binaryParent = parents[i].get<BinaryOperator>();
			if (binaryParent) {
				operators.push_back(binaryParent->getOpcodeStr().str());
				continue;
			}
			
			const UnaryOperator* unaryParent = parents[i].get<UnaryOperator>();
			if(unaryParent) {
				if(unaryParent->isPrefix()) {
					if(unaryParent->isIncrementOp()) {
						operators.push_back("++_pre");
					}
					if(unaryParent->isDecrementOp()){
						operators.push_back("--_pre");
					}
				}
				else{
					if(unaryParent->isIncrementOp()) {
						operators.push_back("++_post");
					}
					if(unaryParent->isDecrementOp()){
						operators.push_back("--_post");
					}
				}
				
			}  

			}
			parents = context->getParents(parents[0]);
			
		}
	}
	if(refNode) {
		startCol = srcMgr.getPresumedLoc(refNode->getSourceRange().getBegin()).getColumn();
		endCol = srcMgr.getPresumedLoc(refNode->getSourceRange().getEnd()).getColumn();
	}
	else {
		startCol = srcMgr.getPresumedLoc(node->getSourceRange().getBegin()).getColumn();
		endCol = srcMgr.getPresumedLoc(node->getSourceRange().getEnd()).getColumn();
	}
	string type = node->getType().getAsString();
	VarData v = {varName, type, node->hasGlobalStorage(), operators, declLine, startCol, endCol, startScope, endScope};

	feat.push_back(make_pair(nodeType, v));
}

