#include "feature_vector.h"
#include <iostream>
map<int,vector<pair<std::string, VarData> > >features;
map<int,vector<ChainNodeData> >du_chains;

map<int,vector<pair<std::string, VarData> > > &get_feature_vector()
{
  return features;
}
