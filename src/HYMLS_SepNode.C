#include "HYMLS_SepNode.H"
#include <algorithm>

namespace HYMLS
  {

  // default constructor
  SepNode::SepNode() : gid_(-1) {;}
  
  // destructor
  SepNode::~SepNode(){;}

   //! advanced constructor
   SepNode::SepNode(int gid, const Teuchos::Array<int> connectedSubs, int varType)
     : gid_(gid), varType_(varType)
     {
     Teuchos::Array<int> tmpArray=connectedSubs;
     
     std::sort(tmpArray.begin(),tmpArray.end());

     Teuchos::Array<int>::iterator end =
       std::unique(tmpArray.begin(), tmpArray.end());
       
     connectedSubs_.resize(std::distance(tmpArray.begin(),end));
     
     std::copy(tmpArray.begin(),end,connectedSubs_.begin());     
     }

std::ostream& operator<<(std::ostream& os, const SepNode& S)
  {
  os << S.GID()<<" ("<<S.type()<<"): [";
  for (int i=0;i<S.level()-1;i++)
    {
    os << S(i) << " ";
    }
  os << S(S.level()-1)<<"]";
  return os;
  }

  
  }
