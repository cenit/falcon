// -*- C++ -*-                                                                  
////////////////////////////////////////////////////////////////////////////////
///                                                                             
/// \file   utils/inc/bintree.h                                                 
/// \warning not yet fully tested                                               
///                                                                             
/// \brief  provides classes BinaryTree<D,X> and MutualBinaryTreeWalker<>       
///                                                                             
/// \author Walter Dehnen                                                       
/// \date   2006-2008                                                           
///                                                                             
/// \todo   full testing                                                        
///                                                                             
/// \version 23-05-2007 WD created based on code from walter/enbid.cc           
/// \version 19-10-2007 WD adapted from btree.h: allow Ncrit>1, no Lmin,Lmax    
/// \version 19-11-2007 WD fudged number of boxes allocated to ~5% accurate     
/// \version 03-09-2008 WD improved & debugged MutualBinaryTreeWalker<>         
///                                                                             
////////////////////////////////////////////////////////////////////////////////
//                                                                              
// Copyright (C) 2006-2008 Walter Dehnen                                        
//                                                                              
// This program is free software; you can redistribute it and/or modify         
// it under the terms of the GNU General Public License as published by         
// the Free Software Foundation; either version 2 of the License, or (at        
// your option) any later version.                                              
//                                                                              
// This program is distributed in the hope that it will be useful, but          
// WITHOUT ANY WARRANTY; without even the implied warranty of                   
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU            
// General Public License for more details.                                     
//                                                                              
// You should have received a copy of the GNU General Public License            
// along with this program; if not, write to the Free Software                  
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                    
//                                                                              
////////////////////////////////////////////////////////////////////////////////
#ifndef WDutils_included_bintree_h
#define WDutils_included_bintree_h

#ifndef WDutils_included_iomanip
#  include <iomanip>
#  define WDutils_included_iomanip
#endif
#ifndef WDutils_included_exception_h
#  include <exception.h>
#endif
#if __cplusplus < 201103L
# ifndef WDutils_included_tupel_h
#  include <utils/tupel.h>
# endif
#else
# ifndef WDutils_included_vector_h
#  include <utils/vector.h>
# endif
#endif
#ifndef WDutils_included_memory_h
#  include <memory.h>
#endif
#ifndef WDutils_included_utility
#  include <utility>
#  define WDutils_included_utility
#endif

// #define DEBUG
// #define TESTING
namespace WDutils {
  template<typename BinTree> class MutualBinaryTreeWalker;
  // ///////////////////////////////////////////////////////////////////////////
  //
  //  class BinaryTree<Dim,X>
  //
  /// A binary tree of rectangular boxes in Dim dimensions.
  ///
  /// On construction, a binary tree is build, consisting of sub-types Dot and
  /// Box. Only Boxes with more than Ncrit points are splitted.
  ///
  // ///////////////////////////////////////////////////////////////////////////
  template<int Dim, typename _X=float>
  class BinaryTree {
  public :
    /// \name public types
    //@{
    typedef _X               real;    ///< type for scalars
#if __cplusplus < 201103L
    typedef tupel<Dim,real>  point;   ///< type for positions
#else
    typedef vector<Dim,real>  point;  ///< type for positions
#endif
    /// represents a particle in Dim-dimensional space.
    /// \note This type is designed to be memory minimal to reduce the costs
    ///       of splitting a list of dots (during box splitting).
    struct Dot {
      unsigned I;                     ///< index
      point    X;                     ///< position
    };
    /// represents a Box in Dim-dimensional space.
    /// The auxiliary data (K/S, R, and S) are not used during tree build, but
    /// may be used afterwards to hold/point to satellite data.\n
    /// During tree build, Xmin & Xmax refer to the box boundaries generated by
    /// splitting. After tree build, function BinaryTree::SetBoxLimits() can be
    /// used to make them refer to the minimum and maximum positions of dots
    /// contained.
    struct Box {
      Dot     *D0;                    ///< dots
      unsigned N;                     ///< # dots
      Box     *P;                     ///< parent box
      Box     *C;                     ///< first child box; 0 for final box
      point    Xmin, Xmax;            ///< box boundaries/extreme positions
      unsigned L;                     ///< level of box
      union {
	int    K;                     ///< auxiliary integer, could be index
	real   S;                     ///< auxiliary scalar, could be mass
      };
      real     R;                     ///< not used in tree, could be radius
      point    X;                     ///< not used in tree, could be centre
      /// pter to end of Dots
      Dot*Dend() const { return D0+N; }
      /// pter to last of Dots
      Dot*Dlast() const { return D0+N-1; }
      /// is *this a final box?
      bool is_final() const { return C==0; }
      /// right child
      Box*const&right() const { return C; }
      /// left child
      Box*      left () const { return C+1; }
    };
    /// BinaryTree requires a (pointer to) Initializer (argument to the
    /// constructor): it is used to initialize and re-initialize Dots.
    class Initializer {
    public:
      /// initializes ALL data of a Dot;
      /// used in tree construction
      /// \param[in] D Dot to be initialized
      virtual void InitDot(Dot*D) const = 0;
      /// re-initializes Dot data;
      /// will be called on all Dots of the (old) tree in rebuilding.
      /// \param[in] D Dot to be re-initialized
      virtual void ReInitDot(Dot*D) const = 0;
    };
    /// associated mutual tree walker
    typedef MutualBinaryTreeWalker<BinaryTree> MutualWalker;
    /// associated interactor
    typedef typename MutualWalker::Interactor Interactor;
    //@}
    /// \name data for class BinaryTree<Dim>
    //@{
  protected:
    const Initializer*INIT;           ///< initializing
  private:
    const unsigned    MAXD;           ///< maximum tree depth
    unsigned          NCRT;           ///< N_crit
    unsigned          NDOT;           ///< # dots
    unsigned          NALL;           ///< # boxes allocated
    unsigned          NBOX;           ///< # boxes used
    unsigned          NFIN;           ///< # final boxes
    unsigned          DPTH;           ///< tree depth
    Dot*              DOTS;           ///< array of dots
    Box*              ROOT;           ///< root box
    Box*              BOXN;           ///< end of boxes / free box
    //@}
#ifdef DEBUG
    /// write box No to stderr; used only for extensive debugging
    void WriteBox(const Box*B) const
    {
      std::cerr<<"Box #" << NoBox(B)
	       << (B->is_final()? " F" : " T");
    }
    /// check whether dot is in box; used only for extensive debugging
    bool CheckDot(const Box*B, const Dot*D) const
    {
      bool err = false;
      for(int d=0; d!=Dim; ++d) {
	if(D->X[d] > B->Xmax[d]) {
	  err = true;
	  std::cerr << "Dot#"<<int(D-DOTS)<<" i="<<D->I<<" not in ";
	  WriteBox(B);
	  std::cerr << " X[" << d << "]=" <<D->X[d]
		    << " > " << B->Xmax[d] << "= Xmax["<<d<<"]\n";
	} else if(D->X[d] < B->Xmin[d]) {
	  err = true;
	  std::cerr << "Dot#"<<int(D-DOTS)<<" i="<<D->I<<" not in ";
	  WriteBox(B);
	  std::cerr << " X[" << d << "]=" <<D->X[d]
		    << " < " << B->Xmin[d] << "= Xmin["<<d<<"]\n";
	}
      }
      return err;
    }
    /// return number of dots not in the box; used only for extensive debugging
    int CheckDots(const Box*B) const
    {
      int err = 0;
      for(const Dot*D = B->D0; D!=B->Dend(); ++D)
	if(CheckDot(B,D)) ++err;
      return err;
    }
    /// print list of x[i]; used only for extensive debugging
    void PrintList(const Box*B, int i, real s, size_t l)
    {
      const Dot* U=B->D0+l;
      const Dot* D=B->D0;
      std::cerr<<" i="<<i<<" s="<< std::setprecision(12) << s <<" dots left:\n";
      for(; D!=U; ++D)
	std::cerr << "   i="<<D->I<<" x=" << D->X[i] <<'\n';
      std::cerr<<" dots right:\n";
      U = B->Dend();
      for(; D!=U; ++D)
	std::cerr << "   i="<<D->I<<" x=" << D->X[i] <<'\n';
    }
    /// check result of SplitList(); used only for extensive debugging
    bool CheckListSplit(const Box*B, int i, real s, size_t l)
    {
      if(l == 0) {
	std::cerr<<"CheckListSplit(B#"<< NoBox(B)
		 <<"): l=0 (N="<<B->N<<")\n";
	PrintList(B,i,s,l);
	return true;
      }
      if(l >= B->N) {
	std::cerr<<"CheckListSplit(B#"<< NoBox(B)
		 <<"): l="<<l<<">=N="<<B->N<<'\n';
	PrintList(B,i,s,l);
	return true;
      }
      bool err = false;
      const Dot* const L=B->D0+l;
      for(const Dot*D = B->D0; D!=L; ++D)
	if(D->X[i] > s) {
	  std::cerr<<"CheckListSplit(B#"<< NoBox(B)
		   <<"): D#"<<size_t(D-DOTS)
		   <<": i="<<D->I<<" < l but right of s\n";
	  err = true;
	}
      for(const Dot*D = L; D!=B->Dend(); ++D)
	if(D->X[i] < s) {
	  std::cerr<<"CheckListSplit(B#"<< NoBox(B)
		   <<"): D#"<<size_t(D-DOTS)
		   <<": i="<<D->I<<" >=l but left of s\n";
	  err = true;
	}
      return err;
    }
#endif
    /// swap two dots; used in SplitList only.
    /// using memcpy for maximum efficiency
    static void swap(Dot*A, Dot*B)
    {
      Dot T;
      memcpy(&T, A, sizeof(Dot));
      memcpy( A, B, sizeof(Dot));
      memcpy( B,&T, sizeof(Dot));
    }
    /// split list of dots, return number of left dots.
    ///
    /// the list of dots of box B is re-shuffled (by swapping dots) such that
    /// on return the first l dots have x[i] <= s and the remaining dots have
    /// x[i] >=s. Dots with x[i]==s will not be swapped over.
    ///
    /// \return number of dots in left sub-box
    /// \param[in] B box the list of which is to be split
    /// \param[in] i dimension in which to split
    /// \param[in] s split point in that dimension
    static unsigned SplitList(const Box*B, int i, real s)
    {
      Dot*l,*u,*N=B->Dend();
      for(l=B->D0; l!=N && l->X[i]<s; ++l);
      if(l==N) return B->N;
      for(u=l+1;u!=N && u->X[i]>s; ++u);
      while(u!=N) {
	swap(l,u);
	for(++l; l!=N && l->X[i]<s; ++l);
	for(++u; u!=N && u->X[i]>s; ++u);
      }
      return  l - B->D0;
    };
    /// give a guess for the number of boxes needed; add safety margin
    size_t Nbox_alloc(unsigned nbox=0) const
    {
      if(NDOT <= NCRT) return 1;
      if(NCRT == 1)    return NDOT+NDOT-1;
      if(nbox == 0) {
	double fac = 1+0.37*(NCRT-1)/double(NCRT+7);
	nbox = unsigned(double(4*NDOT/(NCRT+1))/fac);
      }
      return 1+nbox+size_t(sqrt(double(nbox)));
    }
    /// makes a box final, updates DPTH & NFIN; called from SplitBox()
    void SetFinal(Box*B)
    {
      B->C = 0;
      if(B->L > DPTH) {
	DPTH = B->L;
	if(DPTH > MAXD)
	  WDutils_THROW("BinaryTree<%d,%s>: depth exceeds %d\n",
			Dim,nameof(real),MAXD);
      }
      ++NFIN;
    }
    /// initializes dots and sets values for root cell.
    void SetRoot(int in)
    {
      ROOT->D0   = DOTS;
      ROOT->N    = NDOT;
      ROOT->P    = 0;
      ROOT->L    = 0;
      ROOT->Xmin =
      ROOT->Xmax = DOTS->X;
      switch(in) {
      case 0: // dots are already initialized
	for(const Dot*Di=DOTS; Di!=DOTS+NDOT; ++Di)
	  Di->X.up_min_max(ROOT->Xmin,ROOT->Xmax);
	break;
      case 1: // dots need to be initialized
	for(Dot*Di=DOTS; Di!=DOTS+NDOT; ++Di) {
	  INIT->InitDot(Di);
	  Di->X.up_min_max(ROOT->Xmin,ROOT->Xmax);
	}
	break;
      case 2: // dots need to be re-initialized
	for(Dot*Di=DOTS; Di!=DOTS+NDOT; ++Di) {
	  INIT->ReInitDot(Di);
	  Di->X.up_min_max(ROOT->Xmin,ROOT->Xmax);
	}
	break;
      }
#ifdef DEBUG
      std::cerr<<" Root has: Xmin="<<ROOT->Xmin <<" Xmax="<<ROOT->Xmax<<'\n';
#endif
    }
    /// build the tree and set DPTH, NFIN.
    /// \return true if successfull (otherwise not enough boxes were allocated)
    bool stack_build() WDutils_THROWING
    {
      const Box*  BOXU = ROOT+NALL;
      Stack<Box*> STCK(MAXD+MAXD);
#ifdef DEBUG
      if(debug(1)) CheckDots(ROOT);
#endif
      STCK.push(ROOT);
      DPTH = 0;
      NFIN = 0;
      // while the stack is not empty, process:
      while(!STCK.is_empty()) {
	//  0    pop box off the stack
	Box *B = STCK.pop();
#ifdef TESTING
	std::cerr<<" splitting box#" << NoBox(B)
		 <<": D0#" << int(B->D0-DOTS) << " N=" << B->N;
#endif
	//  1    split list of dots
	int l=0,r=0,i=0;
	real s;
	do {
	  // 1.1 find split dimension and set s to be middle
	  for(int d=1; d!=Dim; ++d)
	    if(B->Xmax[d]-B->Xmin[d] > B->Xmax[i]-B->Xmin[i]) i=d;
	  s = 0.5*(B->Xmax[i]+B->Xmin[i]);
#ifdef TESTING
	  std::cerr<<" in dim "<<i<<" at s="<<s
		   <<" (Xmin="<<B->Xmin[i]<<" Xmax="<<B->Xmax[i]<<")\n";
#endif
	  // 1.2 split list
	  l = SplitList(B,i,s);
	  r = B->N-l;
	  // 1.3 if list of dots was not split, shrink boundaries
	  if     (l==0) B->Xmin[i]=s;
	  else if(r==0) B->Xmax[i]=s;
	} while(l==0 || r==0);
#ifdef DEBUG
	if(debug(1)) CheckListSplit(B,i,s,l);
#endif
	// 2  allocate child boxes and initialize their data
	B->C   = BOXN;
	BOXN  += 2;
	if(BOXN > BOXU) return false;
	Box*L  = B->C,       *R      = L+1;
	L->P   = B;           R->P   = B;
	L->D0  = B->D0;       R->D0  = B->D0 + l;
	L->N   = l;           R->N   = r;
	L->L   = B->L + 1;    R->L   = B->L + 1;
	L->Xmin= B->Xmin;     R->Xmin= B->Xmin;
	L->Xmax= B->Xmax;     R->Xmax= B->Xmax;
	L->Xmax[i]=s;         R->Xmin[i]=s;
#ifdef TESTING
	std::cerr<<" generating box#" << NoBox(L)
		 <<": n=" << L->N<<
		 <<" d0=" << (L->D0 - DOTS)<<'\n'
		 <<" generating box#" << NoBox(R)
		 << ": n=" << R->N<<
		 <<" d0=" << (R->D0 - DOTS)<<'\n';
#endif
#ifdef DEBUG
	if(debug(1)) {
	  CheckDots(L);
	  CheckDots(R);
	}
#endif
	// 3  process child boxes
	if(r<l) {
	  if(L->N > NCRT) STCK.push(L); else SetFinal(L);
	  if(R->N > NCRT) STCK.push(R); else SetFinal(R);
	} else {
	  if(R->N > NCRT) STCK.push(R); else SetFinal(R);
	  if(L->N > NCRT) STCK.push(L); else SetFinal(L);
	}
      } // while(!STCK.is_empty())
      return true;
    }
    //
    void build() WDutils_THROWING
    {
      while(!stack_build()) {
	unsigned N = min(NALL+NALL,NDOT-NCRT);
	warning("BinaryTree<%d,%s>::build(): allocated %d boxes too few "
		"couldn't build tree ... will re-allocate %d and start again",
		Dim,nameof(real),NALL,N);
	WDutils_DEL_A(ROOT);
	NALL = N;
	ROOT = WDutils_NEW(Box,NALL);
	BOXN = ROOT+1;
	DPTH = 0;
	NFIN = 0;
	SetRoot(0);
      }
    }
  public:
    /// constructor: build the binary tree
    /// \param[in] nd # dots
    /// \param[in] it Initializer to initialize dots
    /// \param[in] nc N_crit
    /// \param[in] md maximum tree depth
    BinaryTree(unsigned nd, const Initializer*it, unsigned nc=8,
	       unsigned md=200) :
      INIT  ( it ),
      MAXD  ( md ),
      NCRT  ( nc ),
      NDOT  ( nd ),
      NALL  ( Nbox_alloc() ),
      NFIN  ( 0 ),
      DPTH  ( 0 ),
      DOTS  ( WDutils_NEW16(DOT,NDOT) ),
      ROOT  ( WDutils_NEW(Box,NALL) ),
      BOXN  ( ROOT+1 )
    {
      SetRoot(1);
      if(NDOT > NCRT) 
	build();
      else
	SetFinal(ROOT);
      NBOX = BOXN - ROOT;
    }
    /// destructor
    ~BinaryTree();
    /// build the tree again, re-initializing the dots.
    /// \param[in] nc (optional) if non-zero, reset N_crit to this value
    void rebuild(unsigned nc=0)
    {
      unsigned NEWA;
      if(nc && nc!=NCRT) {
	NCRT = nc;
	NEWA = Nbox_alloc();
      } else
	NEWA = Nbox_alloc(NBOX);
      if(NEWA > NALL || NEWA < (NALL*8)/10) {
	WDutils_DEL_A(ROOT);
	NALL = NEWA;
	ROOT = WDutils_NEW(Box,NALL);
      }
      BOXN = ROOT+1;
      DPTH = 0;
      NFIN = 0;
      SetRoot(2);
      if(NDOT > NCRT) 
	build();
      else
	SetFinal(ROOT);
      NBOX = BOXN - ROOT;
    }
    /// \name functionality of a build tree
    //@{
    /// root box
    Box*const&Root() const { return ROOT; }
    /// first box
    Box*const&BeginBoxes() const { return ROOT; }
    /// end box
    Box*const&EndBoxes() const { return BOXN; }
    /// first box in reverse order: last box
    Box*      RBeginBoxes() const { return BOXN-1; }
    /// end box in reverse order: before first box
    Box*      REndBoxes() const { return ROOT-1; }
    /// running number of a given box
    unsigned NoBox(const Box*B) const { return B-ROOT; }
    /// first dot
    Dot*const&BeginDots() const { return DOTS; }
    /// end of dots
    Dot*EndDots() const { return DOTS+NDOT; }
    /// tree depth of root cell
    unsigned const&Depth() const { return DPTH; }
    /// N_crit
    unsigned const&Ncrit() const { return NCRT; }
    /// number of dots
    unsigned const&Ndots() const { return NDOT; }
    /// number of boxes used
    unsigned Nboxes() const { return NBOX; }
    /// number of boxes allocated
    unsigned Nalloc() const { return NALL; }
    /// number of final boxes
    unsigned Nfinal() const { return NFIN; }
    /// dump dot data to output
    void dump_dots(std::ostream&out) const
    {
      out << " dot    i      x\n";
      for(const Dot*D=DOTS; D!=DOTS+NDOT; ++D)
	out << 'D' << std::setfill('0') << std::setw(5) << int(D-DOTS) << ' '
	    << std::setfill('0') << std::setw(8) << D->I << ' '
	    << D->X <<'\n';
      out.flush();
    }
    /// dump box data to output
    void dump_boxes(std::ostream&out) const
    {
      out << " box    up     left   right  dot       N\n";
      for(Box*B=BeginBoxes(); B!=EndBoxes(); ++B) {
	out << (B->is_final()?'F':'T')
	    << std::setfill('0') << std::setw(5) << NoBox(B) <<' ';
	if(B->P)
	  out << 'T' <<std::setfill('0') << std::setw(5) << NoBox(B->P) <<' ';
	else
	  out<<" nil   ";
	if(B->C)
	  out << (B->C->is_final()?'F':'T')
	      << std::setfill('0') << std::setw(5) << NoBox(B->C) <<' '
	      << ((B->C+1)->is_final()?'F':'T')
	      << std::setfill('0') << std::setw(5) << NoBox(B->C+1) <<' ';
	else
	  out<<" nil    nil   ";
	out << 'D' << std::setfill('0')<<std::setw(5) << int(B->D0-DOTS) <<' '
	    << std::setfill(' ') << std::setw(5) << B->N << '\n';
      }
      out.flush();
    }
    /// for all boxes: set Box::Xmin and Box::Xmax to extreme Dot positions
    void SetBoxLimits()
    {
      for(Box*B=RBeginBoxes(); B!=REndBoxes(); --B) {
	if(B->is_final()) {
	  B->Xmin=B->Xmax=B->D0->X;
	  for(Dot*Di=B->D0+1; Di!=B->Dend(); ++Di)
	    Di->X.up_min_max(B->Xmin,B->Xmax);
	} else {
	  B->Xmin = B->left()->Xmin;
	  B->Xmax = B->left()->Xmax;
	  B->Xmin.up_min(B->right()->Xmin);
	  B->Xmax.up_max(B->right()->Xmax);
	}
      }
    }
    /// perform a mutual tree walk to interact as specified by interactor.
    /// see documentation for class MutualBinaryTreeWalker<> for details.
    void mutual_walk(Interactor*I) const;
    //@}
  };
  // ///////////////////////////////////////////////////////////////////////////
  template<int Dim, typename _X>
  struct traits< BinaryTree<Dim,_X> > {
    static const char *name () {
      return message("BinaryTree<%d,%s>",Dim,traits<_X>::name());
    }
  };
  // ///////////////////////////////////////////////////////////////////////////
  template<int Dim, typename _X>
  struct traits< typename BinaryTree<Dim,_X>::Dot > {
    static const char *name () {
      return message("BinaryTree<%d,%s>::Dot",Dim,traits<_X>::name());
    }
  };
  // ///////////////////////////////////////////////////////////////////////////
  template<int Dim, typename _X>
  struct traits< typename BinaryTree<Dim,_X>::Box > {
    static const char *name () {
      return message("BinaryTree<%d,%s>::Box",Dim,traits<_X>::name());
    }
  };
  // ///////////////////////////////////////////////////////////////////////////
  template<int Dim, typename _X>
  BinaryTree<Dim,_X>::~BinaryTree()
  {
    if(DOTS) { WDutils_DEL16(DOTS); DOTS=0; }
    if(ROOT) { WDutils_DEL_A(ROOT); ROOT=0; }
  }
  // ///////////////////////////////////////////////////////////////////////////
  template<typename A, typename B>
  struct traits< std::pair<A*,B*> > {
    static const char *name () {
      return message("std::pair<%s*,%s*>",nameof(A),nameof(B));
    }
  };
  // ///////////////////////////////////////////////////////////////////////////
  //
  //  class MutualBinaryTreeWalker
  //
  /// Implementing a mutual walk of a binary tree such as BinaryTree<>
  ///
  /// We implement an "Early-testing" mutual tree walk, which means that we try
  /// to perform any interaction as soon as it is generated and only put it on a
  /// stack if it needs splitting. Consequently, any interaction taken from the
  /// stack is splitted without further ado.\n
  /// This is faster than "late testing".
  ///
  // ///////////////////////////////////////////////////////////////////////////
  template<typename BinTree>
  class MutualBinaryTreeWalker {
  public:
    typedef typename BinTree::Dot Dot;
    typedef typename BinTree::Box Box;
    /// specifies how to walk the tree and what kind of interaction to do.
    ///
    /// The members interact() specify the interactions between tree nodes as
    /// well as box self-interactions;\n
    /// the member split_left() specifies which box of a mutual box-box
    /// interaction to split;\n
    /// the member left_first() specifies which sub-interaction of a splitted
    /// box-box is done first;\n 
    /// the member self_before_mutual() dictates if a box's self-interactions
    /// are to be done before its mutual interactions or vice versa.
    class Interactor {
    public:
      /// mutual dot-dot interaction, pure virtual.
      /// \param[in] L  left  interacting dot
      /// \param[in] R  right interacting dot
      virtual void interact(Dot*L, Dot*R) const = 0;
      /// mutual box-dot interaction, pure virtual.
      /// \param[in] B interacting Box
      /// \param[in] D interacting Dot
      /// \return was interaction successful? (otherwise we split box)
      virtual bool interact(Box*B, Dot*D) const = 0;
      /// mutual box-box interaction, pure virtual.
      /// \param[in] L  left  interacting Box
      /// \param[in] R right interacting Box
      /// \return was interaction successful? (otherwise we split bigger box)
      virtual bool interact(Box*L, Box*R) const = 0;
      /// box self-interaction
      /// \param[in] B self-interacting Box
      /// \return was interaction successful? (otherwise we split B)
      virtual bool interact(Box*B) const = 0;
      /// which box of a box-box interaction to split?;
      /// default: split box with more dots.
      /// \param[in] L  left  interacting box
      /// \param[in] R right interacting box
      /// \return split left (otherwise right)?
      virtual bool split_left(const Box*L, const Box*R) const {
	return L->N > R->N;
      }
      /// which of two box-box interactions to perform first?
      /// default: split left child box first
      /// \param[in] B unsplitted box of a splitted box-box interaction
      /// \param[in] L left  child of splitted box
      /// \param[in] R right child of splitted box
      /// \return consider B-L first (or B-R)?
      virtual bool left_first(const Box*B, const Box*L, const Box*R) const {
	return 1;
      }
      /// prioritize self-interactions over mutual interactions?
      /// default: do mutual interactions first
      /// If true, then unresolved mutual interactions between the children of 
      /// box B are done AFTER their (unresolved) self-interactions.
      /// \param[in] B box the self-interaction of which has to be splitted.
      virtual bool self_before_mutual(Box*B) const {
	return false;
      }
    };
  private:
    //
    typedef std::pair<Box*,Dot*>  BoxDot;
    typedef std::pair<Box*,Box*>  BoxBox;
    //
    const Interactor*IA;         ///< pter to interactor
    Stack<BoxDot>    BD;         ///< stack of box-dot interactions
    Stack<BoxBox>    BB;         ///< stack of box-box interactions
    //
    void perform(Dot*A, Dot*B) { IA->interact(A,B); }
    void perform(Box*A, Dot*B) { if(!IA->interact(A,B)) BD.push(BoxDot(A,B)); }
    void perform(Box*A, Box*B) { if(!IA->interact(A,B)) BB.push(BoxBox(A,B)); }
    void perform(Box*A)        { if(!IA->interact(A))   BB.push(BoxBox(A,0)); }
    /// clear the stack of box-dot interactions
    void clear_BD_stack()
    {
      while(!BD.is_empty()) {
	BoxDot I = BD.pop();
	if(I.first->is_final()) {
	  for(Dot*Di=I.first->D0; Di!=I.first->Dend(); ++Di)
	    IA->interact(Di,I.second);
	} else {
	  perform(I.first->left (),I.second);
	  perform(I.first->right(),I.second);
	}
      }
    }
    /// split a mutual box-box interaction
    /// \param[in] A box to be split
    /// \param[in] B box to be kept
    void split(Box*A, Box*B)
    {
      if(A->is_final()) {
	for(Dot*D=A->D0; D!=A->Dend(); ++D)
	  perform(B,D);
      } else {
	if(IA->left_first(B,A->left(),A->right())) {
	  perform(A->left (),B);
	  perform(A->right(),B);
	} else {
	  perform(A->right(),B);
	  perform(A->left (),B);
	}
      }
    }
    /// split a box self-interaction
    /// \param[in] A box to be split
    void split(Box*A)
    {
      if(A->is_final()) {
	for(Dot*Di=A->D0; Di!=A->Dlast(); ++Di)
	  for(Dot*Dj=Di+1; Dj!=A->Dend(); ++Dj)
	    perform(Di,Dj);
      } else {
	// order correct: what is stacked first will be done last!
	if(IA->self_before_mutual(A)) {
	  perform(A->left (),A->right());
	  perform(A->left ());
	  perform(A->right());
	} else {
	  perform(A->left ());
	  perform(A->right());
	  perform(A->left (),A->right());
	}
      }
    }
    /// clear the stack of box-box interactions, keep box-dot stack clear too
    void clear_BB_stack()
    {
      while(!BB.is_empty()) {
	BoxBox I = BB.pop();
	if     (0 == I.second)                    split(I.first);
	else if(IA->split_left(I.first,I.second)) split(I.first,I.second);
	else                                      split(I.second,I.first);
	clear_BD_stack();
      }
    }
  public:
    /// construction for interaction within one binary tree
    /// \param[in] i pter to interactor
    /// \param[in] d depth of tree
    MutualBinaryTreeWalker(const Interactor*i, size_t d) :
      IA(i), BD(d), BB(d+d) {}
    /// construction for interaction between two binary trees
    /// \param[in] i pter to interactor
    /// \param[in] d1 depth of tree of sources
    /// \param[in] d2 depth of tree of sinks
    MutualBinaryTreeWalker(const Interactor*i, size_t d1, size_t d2) :
      IA(i), BD(max(d1,d2)), BB(d1+d2) {}
    /// perform a mutual tree walk.
    /// If two distinct boxes are given, the mutual interaction between these is
    /// performed, including splitting them until no unresolved interaction
    /// remains.\n
    /// If only one box is given (or if both are the same), the mutual
    /// self-interaction of this box is performed, including splitting it until
    /// no unresolved interaction remains.
    /// \param[in] A interaction box
    /// \param[in] B (optional) second interacting box
    void walk(Box*A, Box*B=0)
    {
      if(B && B!=A)
	perform(A,B);
      else
	perform(A);
      clear_BB_stack();
    }
  };// class MutualBinaryTreeWalker
  // ///////////////////////////////////////////////////////////////////////////
  template<typename BinTree>
  struct traits< MutualBinaryTreeWalker<BinTree> > {
    static const char *name () {
      return message("MutualBinaryTreeWalker<%s>",traits<BinTree>::name());
    }
  };
  // ///////////////////////////////////////////////////////////////////////////
  template<int Dim, typename _X> inline
  void BinaryTree<Bin,_X>::mutual_walk(Interactor*I) const
  {
    Walker W(I,DPTH);
    W.walk(ROOT);
  }
};// namespace WDutils
////////////////////////////////////////////////////////////////////////////////
#endif // WDutils_included_bintree_h
