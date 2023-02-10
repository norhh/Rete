#include "feature_matchers.h"
#include "Globals.h"
#include "feature_vector.h"
#include "rete_utils.h"

#include <set>
#include<map>
#include<string>
#include <iostream>

using std::string;
using std::map;

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang;
using namespace ast_matchers;

StatementMatcher LoopMatcher = 
    anyOf(forStmt().bind("ForLoop"), 
        whileStmt().bind("WhileLoop"));

StatementMatcher LValueMatcher = 
    anyOf(binaryOperator(anyOf(hasOperatorName("="),
                       hasOperatorName("+="),
                       hasOperatorName("-="),
                       hasOperatorName("*="),
                       hasOperatorName("/="),
                       hasOperatorName("%="),
                       hasOperatorName("&="),
                       hasOperatorName("|="),
                       hasOperatorName("^="),
                       hasOperatorName("<<="),
                       hasOperatorName(">>="))).bind("assignment"), 
					   declStmt(has(varDecl(anyOf(hasDescendant(binaryOperator()), hasDescendant(integerLiteral()))).bind("assignment"))));

StatementMatcher RValueMatcher = binaryOperator(unless(anyOf(hasOperatorName("="),
                       hasOperatorName("+="),
                       hasOperatorName("-="),
                       hasOperatorName("*="),
                       hasOperatorName("/="),
                       hasOperatorName("%="),
                       hasOperatorName("&="),
                       hasOperatorName("|="),
                       hasOperatorName("^="),
                       hasOperatorName("<<="),
                       hasOperatorName(">>=")))).bind("rvalue");

StatementMatcher UnaryLValMatcher = unaryOperator(anyOf(hasOperatorName("++"), hasOperatorName("--"))).bind("unaryLValue");

StatementMatcher IfMatcher =  ifStmt().bind("if");

LoopPrinter::LoopPrinter() {}

AssignmentsFeatureCalc::AssignmentsFeatureCalc() {}

ASTIfCalc::ASTIfCalc() {}

RValueFeatureCalc::RValueFeatureCalc() {}

UnaryLValFeatureCalc::UnaryLValFeatureCalc() {}

void push_du_chain_data(vector<pair<std::string, VarData> > feat, int line, int startCol, int endCol, string type) {
	ChainNodeData data;
	bool is_for = (type == FOR_LOOP);
	std::set<string>s {FOR_LOOP_CONDITION, FOR_LOOP_INIT, FOR_LOOP_INCREMENT};
	for(auto it: feat) {
			//std::cout<<(startCol <= it.second.startCol)<<" "<<(endCol >= it.second.endCol)<<" "<<it.second.varName<<" "<<it.second.varType<<" "<<type<<std::endl;
			if( (it.first == R_VALUE)  && startCol <= it.second.startCol && endCol >= it.second.endCol) {
					data.r_vals.push_back(it.second.varName);
			}
			if( (it.first == L_VALUE)  && startCol <= it.second.startCol && endCol >= it.second.endCol) {
					data.l_vals.push_back(it.second.varName);
			}

	}
	data.node = type;
	data.startCol = startCol;
	data.endCol = endCol;
	/*if(type == IF_CONDITION) {
		for(auto &it: du_chains[line]) {
			if(it.startCol == startCol && it.endCol == endCol && (it.node == "Assignment" || it.node == "R_VALUE" || it.node == "L_VALUE")) {
				it.node = type;
				return ;
			}
		}
	}*/
	if(type == "For" || type == "WhileStmt" || type == IF_CONDITION) {
		for(auto it=du_chains[line].begin(); it!=du_chains[line].end(); it++) {
			if(it->startCol >= startCol && it->endCol <= endCol && (it->node == "Assignment" || it->node == "R_VALUE" || it->node == "L_VALUE")) {
				data.l_vals.insert(data.l_vals.end(), it->l_vals.begin(), it->l_vals.end());
				data.r_vals.insert(data.r_vals.end(), it->r_vals.begin(), it->r_vals.end());
				du_chains[line].erase(it);
			}
		}
	}
	else {
		for(auto &it: du_chains[line]) {
			if(it.startCol <= startCol && it.endCol >= endCol && (it.node == "For" || it.node == "WhileStmt" || it.node == IF_CONDITION)) {
				it.l_vals.insert(it.l_vals.end(), data.l_vals.begin(), data.l_vals.end());
				it.r_vals.insert(it.r_vals.end(), data.r_vals.begin(), data.r_vals.end());
				return ;
			}
		}
	}
	du_chains[line].push_back(data);
}



void LoopPrinter::run(const MatchFinder::MatchResult &Result) {
	ASTContext *Context = Result.Context;
	SourceManager &srcMgr = Context->getSourceManager();
	const ForStmt *FS = Result.Nodes.getNodeAs<ForStmt>("ForLoop");
	if(FS) {

		if (!Context->getSourceManager().isWrittenInMainFile(FS->getForLoc()))
			return;
		
		int line = srcMgr.getPresumedLoc(FS->getSourceRange().getBegin()).getLine();
		int startCol, endCol = -1;
		startCol = srcMgr.getPresumedLoc(FS->getSourceRange().getBegin()).getColumn();

		const Expr* inc = FS->getInc();
		if(inc) {
			endCol = srcMgr.getPresumedLoc(inc->getSourceRange().getEnd()).getColumn();
			collectFromStmt(inc, features[line], FOR_LOOP_INCREMENT, Context);
		}
		const Expr* cond = FS->getCond();
		if(cond) {
			if(endCol == -1)
				endCol = srcMgr.getPresumedLoc(cond->getSourceRange().getEnd()).getColumn();

			collectFromStmt(cond, features[line], FOR_LOOP_CONDITION, Context);
		}

		const Stmt* initStmt = FS->getInit();

		if(initStmt) {
				
			if(endCol == -1)
				endCol = srcMgr.getPresumedLoc(initStmt->getSourceRange().getEnd()).getColumn();

			collectFromStmt(initStmt, features[line], FOR_LOOP_INIT, Context);
		}
		push_du_chain_data(features[line], line, startCol, endCol, "For");

	}
	else {
		const WhileStmt *WS = Result.Nodes.getNodeAs<WhileStmt>("WhileLoop");
		if(!WS || !Context->getSourceManager().isWrittenInMainFile(WS->getWhileLoc()))
			return;
		

		int line = srcMgr.getPresumedLoc(WS->getSourceRange().getBegin()).getLine();
		const Expr* vd = WS->getCond();
		
		int startCol = srcMgr.getPresumedLoc(vd->getSourceRange().getBegin()).getColumn();
		int endCol = srcMgr.getPresumedLoc(vd->getSourceRange().getEnd()).getColumn();
		
		collectFromExpr(vd, features[line], LOOP_CONDITION, Context);
		push_du_chain_data(features[line], line, startCol, endCol, "WhileStmt");

	}
}

void AssignmentsFeatureCalc::run(const MatchFinder::MatchResult &Result) {
	ASTContext *Context = Result.Context;
	SourceManager &srcMgr = Context->getSourceManager();
	const BinaryOperator *BO = Result.Nodes.getNodeAs<BinaryOperator>("assignment");
	int line, startCol, endCol;
	if (BO) {
		if (!Context->getSourceManager().isWrittenInMainFile(BO->getExprLoc()))
			return;
		
		line = srcMgr.getPresumedLoc(BO->getSourceRange().getBegin()).getLine();
		startCol = srcMgr.getPresumedLoc(BO->getSourceRange().getBegin()).getColumn();
		endCol = srcMgr.getPresumedLoc(BO->getSourceRange().getEnd()).getColumn();
		Expr* l_value = BO->getLHS();
		collectFromExpr(l_value, features[line], L_VALUE, Context);
		Expr* r_value = BO->getRHS();
		collectFromExpr(r_value, features[line], R_VALUE, Context);
	}
	else {
		const VarDecl *decl = Result.Nodes.getNodeAs<VarDecl>("assignment");
		if (!Context->getSourceManager().isWrittenInMainFile(decl->getSourceRange().getBegin()))
				return;

		line = srcMgr.getPresumedLoc(decl->getSourceRange().getBegin()).getLine();
		startCol = srcMgr.getPresumedLoc(decl->getSourceRange().getBegin()).getColumn();
		endCol = srcMgr.getPresumedLoc(decl->getSourceRange().getEnd()).getColumn();
		const Expr* assignment = decl->getInit();
		if(assignment)
				collectFromExpr(assignment, features[line], R_VALUE, Context);
		addNodeToFeat(decl, NULL, features[line], L_VALUE, srcMgr, Context);
		std::cout<<line<<std::endl;
	}
	
	push_du_chain_data(features[line], line, startCol, endCol, "Assignment");

}


void ASTIfCalc::run(const MatchFinder::MatchResult &Result) {
	ASTContext *Context = Result.Context;
	SourceManager &srcMgr = Context->getSourceManager();
	const IfStmt *If = Result.Nodes.getNodeAs<IfStmt>("if");
	if (!If)
		return;
	if (!Context->getSourceManager().isWrittenInMainFile(If->getIfLoc()))
		return;
	int line = srcMgr.getPresumedLoc(If->getSourceRange().getBegin()).getLine();
	int line_end = srcMgr.getPresumedLoc(If->getSourceRange().getEnd()).getLine();
	const Expr* cond = If->getCond();
	map<string, int> vars;
	collectFromExpr(cond, features[line], IF_CONDITION, Context);
	int startCol = srcMgr.getPresumedLoc(If->getSourceRange().getBegin()).getColumn();
	int endCol = srcMgr.getPresumedLoc(cond->getSourceRange().getEnd()).getColumn();
	std::cout<<line<<" "<<line_end<<" "<<startCol<<" "<<endCol<<std::endl;
	push_du_chain_data(features[line],line, startCol, endCol, IF_CONDITION);
}

void RValueFeatureCalc::run(const MatchFinder::MatchResult &Result) {
	ASTContext *Context = Result.Context;
	SourceManager &srcMgr = Context->getSourceManager();
	const BinaryOperator *BO = Result.Nodes.getNodeAs<BinaryOperator>("rvalue");
	int line, startCol, endCol;
	if (!Context->getSourceManager().isWrittenInMainFile(BO->getExprLoc()))
		return;
	vector<pair<std::string, VarData> > feat;
	collectFromExpr(BO, feat, R_VALUE, Context);
	
	line = srcMgr.getPresumedLoc(BO->getSourceRange().getBegin()).getLine();
	startCol = srcMgr.getPresumedLoc(BO->getSourceRange().getBegin()).getColumn();
	endCol = srcMgr.getPresumedLoc(BO->getSourceRange().getEnd()).getColumn();

	push_du_chain_data(feat, line, startCol, endCol, "R_VALUE");
}


void UnaryLValFeatureCalc::run(const MatchFinder::MatchResult &Result) {
	ASTContext *Context = Result.Context;
	SourceManager &srcMgr = Context->getSourceManager();
	const UnaryOperator *UO = Result.Nodes.getNodeAs<UnaryOperator>("unaryLValue");
	int line, startCol, endCol;
	if (!Context->getSourceManager().isWrittenInMainFile(UO->getExprLoc()))
		return;
	vector<pair<std::string, VarData> > feat;
	collectFromExpr(UO, feat, L_VALUE, Context);
	line = srcMgr.getPresumedLoc(UO->getSourceRange().getBegin()).getLine();
	startCol = srcMgr.getPresumedLoc(UO->getSourceRange().getBegin()).getColumn();
	endCol = srcMgr.getPresumedLoc(UO->getSourceRange().getEnd()).getColumn();
	push_du_chain_data(feat, line, startCol, endCol, "L_VALUE");
}

