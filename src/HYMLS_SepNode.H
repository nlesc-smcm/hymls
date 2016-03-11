#ifndef HYMLS_SEPNODE_H
#define HYMLS_SEPNODE_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include <algorithm>


namespace HYMLS {

//! a class to store info about a separator node:        
//! what is the node's GID, and which subdomains does it 
//! separate? This class implements comparison operators 
//! so that we can conveniently group nodes belonging to 
//! the same separator. The sorting criteria defined here
//! are crucial for the functioning of the preconditioner
//! as they determine how the Schur-complement is sorted 
//! and to which groups of variables we apply our trans- 
//! forms and dropping.                                  
class SepNode
  {
  public:

    //! default constructor
    SepNode();
    
    //! advanced constructor
    SepNode(int gid, const Teuchos::Array<int> connectedSubs, int varType=0);
      
   //! destructor
   virtual ~SepNode();
   
   //! get the GID of this node
   inline int GID() const {return gid_;}
   
   //! get the number of connected subdomains
   inline int level() const {return connectedSubs_.size();}
   
   //! get variable type
   inline int type() const {return varType_;}
   
   //! get the k'th subdomain id this node connects to:
   inline int& operator()(int k){return connectedSubs_[k];}

   //! get the k'th subdomain id this node connects to:
   inline const int& operator()(int k) const {return connectedSubs_[k];}
   
   //! assignment operator
   SepNode& operator=( const SepNode &t ) 
     {
     gid_=t.gid_;
     connectedSubs_=t.connectedSubs_;
     varType_=t.varType_;
     return *this;
     }
   
   
   //! equality operator - returns true if the two SepNodes connect
   //! to the same subdomains
   inline bool operator==( const SepNode &a) const
     {
     bool t = (connectedSubs_.size()==a.connectedSubs_.size());
     if (t) t = varType_==a.varType_;
     if (t) t = (connectedSubs_==a.connectedSubs_);
     return t;
     }


   //! inequality operator
   inline bool operator!=( const SepNode &a) const
     {
     return !(this->operator==(a));
     }

   //! less-than operator - returns true if a1 connects to fewer
   //! subdomains or to the same number of subdomains but with  
   //! lower first ranks
   inline bool operator<( const SepNode a) const
     {
     int s1=connectedSubs_.size();
     int s2=a.connectedSubs_.size();
     if (s1>s2)
       {
       return false;
       }       
     else if (s1<s2)
       {
       return true;
       }
     else // both separators connect to the same number of subdomains
       {
       // do we connect to the same subdomains?
       if (connectedSubs_==a.connectedSubs_)
         {
         // decide based on variable type
         return (varType_<a.varType_);
         }
       else 
         {
         // default: decide based on subdomains we connect to
         return (connectedSubs_<a.connectedSubs_);
         }
       } 
     return true;// never reached
     }
   

   //! less or equal operator 
   inline bool operator<=( const SepNode &a) const
     {     
     bool t1=this->operator==(a);
     bool t2=this->operator<(a);
     return t1||t2;
     }

   //! greater-than operator
   inline bool operator>( const SepNode &a ) const
     {
     bool t1=this->operator<=(a);
     return !t1;
     }

   //! greater or equal operator 
   inline bool operator>=( const SepNode &a) const
     {
     bool t1=this->operator<(a);
     return !t1;
     }


  protected:
    
    //! what is my global index?
    int gid_;
    
    //! which subdomains do I connect to?
    Teuchos::Array<int> connectedSubs_;
    
    //! for multi-dof problems: variable type (i.e. U, V, P etc)
    int varType_;
  };

std::ostream& operator<<(std::ostream& os, const SepNode& S);  

}

#endif
