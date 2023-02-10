

#pragma once

#include <memory>
#include <vector>
#include <utility>
#include "feature_vector.h"

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/Type.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace llvm;


#include <rapidjson/document.h>


std::string toString(const clang::Stmt *stmt);
std::string toString(const clang::Expr *expr);


std::vector< std::pair<std::string, VarData> >  collectFromExpr(const Expr *expr,
                                          std::vector< std::pair<std::string, VarData> >&feat, std::string nodeType, ASTContext* context);

std::vector< std::pair<std::string, VarData> >  collectFromStmt(const Stmt* stmt, 
                                          std::vector< std::pair<std::string, VarData> >&feat, std::string nodeType, ASTContext* context);
                                          
void addNodeToFeat(
  const VarDecl* node, 
  DeclRefExpr * refNode,
  std::vector< std::pair<std::string, VarData> > &feat,
  string nodeType,
  SourceManager& srcMgr,
  ASTContext* context
);
