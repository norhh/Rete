#pragma once


#include <map>
#include <string>
#include <vector>
#include <utility>

using std::pair;
using std::vector;
using std::map;
using std::string;

typedef struct VarData {
    string varName;
    string varType;
    bool isGlobal;
    vector<string> surroundingVars;
    int declLocation;
    int startCol;
    int endCol;
    int startScope;
    int endScope;
} VarData;

typedef struct ChainNodeData {
    string node;
    vector<string> l_vals;
    vector<string> r_vals;
    int startCol, endCol;
} ChainNodeData;


map<int,vector<pair<std::string, VarData> > > &get_feature_vector();
extern map<int,vector<pair<std::string, VarData> > > features;
extern map<int,vector<ChainNodeData> >du_chains;
