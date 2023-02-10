#pragma once


#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"


using namespace clang;
using namespace ast_matchers;

extern clang::ast_matchers::StatementMatcher LoopMatcher;

class LoopPrinter : public MatchFinder::MatchCallback {
public:
LoopPrinter();

virtual void run(const MatchFinder::MatchResult &Result);

};


extern clang::ast_matchers::StatementMatcher LValueMatcher;


class AssignmentsFeatureCalc : public MatchFinder::MatchCallback {
	public:
	AssignmentsFeatureCalc();

	virtual void run(const MatchFinder::MatchResult &Result);

};

extern clang::ast_matchers::StatementMatcher IfMatcher;

class ASTIfCalc : public MatchFinder::MatchCallback {
public:
ASTIfCalc();

virtual void run(const MatchFinder::MatchResult &Result);

};

extern clang::ast_matchers::StatementMatcher DeclMatcher;


class RValueFeatureCalc : public MatchFinder::MatchCallback {
	public:
	RValueFeatureCalc();

	virtual void run(const MatchFinder::MatchResult &Result);

};

extern clang::ast_matchers::StatementMatcher RValueMatcher;

class UnaryLValFeatureCalc : public MatchFinder::MatchCallback {
	public:
	UnaryLValFeatureCalc();

	virtual void run(const MatchFinder::MatchResult &Result);

};

extern clang::ast_matchers::StatementMatcher UnaryLValMatcher;
