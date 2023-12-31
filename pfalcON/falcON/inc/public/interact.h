// -*- C++ -*-
////////////////////////////////////////////////////////////////////////////////
///
/// \file   inc/public/interact.h
///
/// \author Walter Dehnen
/// \date   2000-2006
///
/// \brief  contains class template falcON::MutualInteractor, which implements
///	    the mutual interaction algorithm used in falcON.
///
////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2000-2006  Walter Dehnen
//               2013       Benoit Lange, Pierre Fortin
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
//          22/11/2013  BL recursive implementation of DTT
//          22/11/2013  BL added support of OpenMP for parallel DTT
//          22/11/2013  BL added support of TBB for parallel DTT
// v p0.1   22/11/2013  BL added ISPC kernels
////////////////////////////////////////////////////////////////////////////////


extern int tct;

#ifndef falcON_included_interact_h
#define falcON_included_interact_h

#ifndef falcON_included_basic_h
#  include <public/basic.h>
#endif

#ifdef recursive
#   ifdef pfalcON
#       ifdef pfalcON_useTBB
#include "tbb/tbb_stddef.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"
#include "tbb/atomic.h"
using namespace tbb;
#       else
///OpenMP
#include <omp.h>
#       endif

#   endif
#endif
namespace falcON {
    
    // ///////////////////////////////////////////////////////////////////////////
    //
    // auxiliary stuff
    //
    // ///////////////////////////////////////////////////////////////////////////
    //----------------------------------------------------------------------------
    // struct iaction<>
    //   holds the left and right partner of an interaction
    //----------------------------------------------------------------------------
    template<typename A, typename B>
    struct iaction {
        A fst; // first object
        B snd; // second object
        iaction() :
        fst(0), snd(0) {
        } // constructor
        void set(A a, B b) { // set pair
            fst = a;
            snd = b;
        }
    };
}
//------------------------------------------------------------------------------
namespace WDutils {
    template<typename A, typename B> struct traits<falcON::iaction<A, B> > {
        static const char *name() {
            char __a[1024];
            strcpy(__a, traits < A > ::name());
            return message("iaction< %s, %s >", __a, traits < B > ::name());
            //       return
            // 	message("iaction< %s, %s >", traits<A>::name(), traits<B>::name());
        }
    };
}
//------------------------------------------------------------------------------
namespace falcON {
    //----------------------------------------------------------------------------
    // struct iastack<>
    //   a stack of iaction<>s organised in a linked list
    //----------------------------------------------------------------------------
    template<typename A, typename B>
    class iastack {
    private:
        typedef iaction<A, B> iact; // type of stack objects
        iact *IA, *pi; // first & current element
#ifdef DEBUG
        iact *IEND;
#endif
        //--------------------------------------------------------------------------
    public:
        explicit iastack(unsigned const&M) // constructor
        :
        IA(falcON_NEW(iact, M)), //   allocate memory
        pi(IA - 1) //   set pter to activ
#ifdef DEBUG
        , IEND ( IA + M )
#endif
        {
        }
        //--------------------------------------------------------------------------
        ~iastack() {
            falcON_DEL_A(IA);
        } // destructor: deallocate
        bool is_empty() const {
            return pi < IA;
        } // if stack empty?
        iact pop() {
            return *(pi--);
        } // give last:   pop
        void push(A a, B b) { // add element: push
            ++pi;
#ifdef DEBUG
            if(pi >= IEND) error("push()ing beyond end of iastack");
#endif
            pi->set(a, b);
        }
    };
    //----------------------------------------------------------------------------
    // struct saction<>
    //   holds one partner of an interaction
    //----------------------------------------------------------------------------
    template<typename A>
    struct saction {
        A obj; // object
        saction() :
        obj(0) {
        } // constructor
        void set(A a) { // set object
            obj = a;
        }
    };
} // namespace falcON {
//------------------------------------------------------------------------------
namespace WDutils {
    template<typename A> struct traits<falcON::saction<A> > {
        static const char *name() {
            return message("saction< %s >", traits < A > ::name());
        }
    };
}
//------------------------------------------------------------------------------
namespace falcON {
    //----------------------------------------------------------------------------
    // struct sastack<>
    //   a stack of saction<>s organised in a linked list
    //----------------------------------------------------------------------------
    template<typename A>
    class sastack {
    private:
        typedef saction<A> sact; // type of stack objects
        sact *SA, *pi; // first & current element
#ifdef DEBUG
        sact *SEND;
#endif
        //--------------------------------------------------------------------------
    public:
        explicit sastack(unsigned const&M) // constructor
        :
        SA(falcON_NEW(sact, M)), //   allocate memory
        pi(SA - 1) //   set pter to activ
#ifdef DEBUG
        , SEND ( SA + M )
#endif
        {
        }
        //--------------------------------------------------------------------------
        ~sastack() {
            falcON_DEL_A(SA);
        } // destructor: deallocate
        bool is_empty() const {
            return pi < SA;
        } // if stack empty?
        sact pop() {
            return *(pi--);
        } // give last:   pop
        void push(A a) { // add element: push
            ++pi;
#ifdef DEBUG
            if(pi >= SEND) error("push()ing beyond end of sastack");
#endif
            pi->set(a);
        }
    };
    // ///////////////////////////////////////////////////////////////////////////
    //
    // class falcON::MutualInteractor<>
    //
    /// encodes the mutual interaction algorithm as used in gravity.cc,
    /// partner.cc, and sph.cc; the order of nodes (first,second) is preserved
    /// when splitting a node.
    ///
    /// The mutual interaction algorithm is encoded as class template. The
    /// template argument must provide the methods performing the actual
    /// interactions. The class template \b MUST have the following member
    /// methods.
    /// \code
    /// class interactor {
    ///   public:
    ///   typedef leaf_iter;
    ///   typedef cell_iter;
    ///   bool take(cell_iter const&) const;
    ///   bool take(leaf_iter const&) const;
    ///   bool split_first(cell_iter const&, cell_iter const&) const;
    ///   bool interact(cell_iter const&) const;
    ///   bool interact(cell_iter const&, cell_iter const&) const;
    ///   bool interact(cell_iter const&, leaf_iter const&) const;
    ///   bool interact(leaf_iter const&, cell_iter const&) const;
    ///   void interact(leaf_iter const&, leaf_iter const&) const;
    /// };
    /// \endcode
    /// The typedefs \c leaf_iter and \c cell_iter refer to appropriate iterators
    /// (or pointers) used for leafs and cells, respectively. The methods \c take(
    /// \e node \c ) return true if the tree \e node shall be considered for
    /// interaction (for example, a SPH interaction will not be considered for a
    /// non-SPH leaf or cell). \c split_first() returns true if the first of a
    /// pair of interacting cells is to be split and false if the second is to be
    /// split. The methods \c interact() try to perform an individual mutual or
    /// self interaction and return \c true if that trial was successful.
    /// Otherwise, the interaction will be split and \c interact() called again
    /// on its sub-interactions.
    ///
    /// \version  June 2005  only put interactions that could not be done on the
    ///           stacks
    ///
    // ///////////////////////////////////////////////////////////////////////////
    template<typename INTERACTOR> class MutualInteractor
    {
        
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        // types
        //--------------------------------------------------------------------------
        typedef typename INTERACTOR::cell_iter cell_iter; // iterator over cells
        typedef typename INTERACTOR::leaf_iter leaf_iter; // iterator over leafs
#ifdef recursive
    public:
        const OctTree *TREE; // the tree to be used
        cell_iter rooot;
#ifdef pfalcON_useTBB
        
        mutable task_group g;
        
#endif
#endif
        
        //--------------------------------------------------------------------------
        typedef saction<cell_iter> cx_iact; // cell-self iaction
        typedef iaction<cell_iter, cell_iter> cc_iact; // cell-cell iaction
        typedef iaction<cell_iter, leaf_iter> cl_iact; // cell-leaf iaction
        typedef iaction<leaf_iter, cell_iter> lc_iact; // leaf-cell iaction
        //--------------------------------------------------------------------------
        typedef sastack<cell_iter> cx_stack; // stack: cell-self iaction
        typedef iastack<cell_iter, cell_iter> cc_stack; // stack: cell-cell iaction
        typedef iastack<cell_iter, leaf_iter> cl_stack; // stack: cell-leaf iaction
        typedef iastack<leaf_iter, cell_iter> lc_stack; // stack: leaf-cell iaction
        //--------------------------------------------------------------------------
        // data
        //--------------------------------------------------------------------------
        const INTERACTOR *IA; // interactor
#ifdef recursive
#else
        mutable cx_stack CX;                             // stack: cell-self iaction
        mutable cc_stack CC;                             // stack: cell-cell iaction
        mutable cl_stack CL;                             // stack: cell-leaf iaction
        mutable lc_stack LC;                             // stack: leaf-cell iaction
#endif
        mutable int ic;                                  // counter: cell iactions

        
        //--------------------------------------------------------------------------
        // private methods
        //--------------------------------------------------------------------------
#define LoopCKids(C,A)							\
LoopCellKids(typename cell_iter,C,A) if(INTERACTOR::take(A))
#define LoopLKids(C,A)							\
LoopLeafKids(typename cell_iter,C,A) if(INTERACTOR::take(A))
#define LoopCPairs(C,A,B)						\
LoopCellSecd(typename cell_iter,C,A,B) if(INTERACTOR::take(B))
#define LoopSPairs(C,A,B)						\
LoopLKids(C,A)							\
LoopLeafSecd(typename cell_iter,C,A+1,B) if(INTERACTOR::take(B))
#ifdef recursive
        void split_cell_leaf2( cell_iter const&c1, leaf_iter const&l2) const
        {
            cl_iact cl; //   pop new C-L iaction
            cl.set(c1, l2);
            
            LoopLKids	(cl.fst,l1) perform(l1,cl.snd); //   perform sub L-L
            LoopCKids(cl.fst,ci) perform(ci,cl.snd);//   perform sub C-L
        }
        
        void split_leaf_cell2( leaf_iter const&l1, cell_iter const&c2) const
        {
            lc_iact lc; //   pop new L-C iaction
            lc.set(l1,c2);
            LoopLKids(lc.snd,l2) perform(lc.fst,l2);//   perform sub L-L
            LoopCKids(lc.snd,ci) perform(lc.fst,ci);//   perform sub L-C
        }
        
        void split_cell_cell2( cell_iter const&c1, cell_iter const&c2) const
        {
            cc_iact cc;
            cc.set(c1,c2);
            split_cell_cell(cc);
        }
        void split_cell_self2( cell_iter const&cc) const
        {
            cx_iact cx;
            cx.set(cc);
            split_cell_self(cx);
        }
        void split_cell_leaf2_std(cell_iter const&c1, leaf_iter const&l2) const {
            cl_iact cl; //   pop new C-L iaction
            cl.set(c1,l2);
            LoopLKids	(cl.fst,l1) perform_std(l1,cl.snd); //   perform sub L-L
            LoopCKids(cl.fst,ci) perform_std(ci,cl.snd);//   perform sub C-L
        }
        
        void split_leaf_cell2_std(leaf_iter const&l1, cell_iter const&c2) const
        {
            lc_iact lc; //   pop new L-C iaction
            lc.set(l1,c2);
            LoopLKids(lc.snd,l2) perform_std(lc.fst,l2);//   perform sub L-L
            LoopCKids(lc.snd,ci) perform_std(lc.fst,ci);//   perform sub L-C
        }
        
        void split_cell_cell2_std(cell_iter const&c1, cell_iter const&c2) const
        {
            cc_iact cc;
            cc.set(c1,c2);
            split_cell_cell_std(cc);
        }
        
        void split_cell_self2_std(cell_iter const&cc) const
        {
            cx_iact cx;
            cx.set(cc);
            split_cell_self_std(cx);
        }
        
#endif
        ////FIn pfalcON_STD
#ifdef recursive
        void perform(int id1) const
        {


                        cell_iter const&cc = rooot.CellNo(id1) ;
            if(IA->interact(cc))
            {
#   if pfalcON
#   else
	      ++ic;
#   endif
                
            }
            else
            {
                
                //                int id1 = cc.index();
                if(number(cc) > tct)
                {
                    split_cell_self2(cc);
                }
                else
                {
                    split_cell_self2_std(cc);
                }
            }
        }
#endif
        /// try to perform a cell-cell interaction; otherwise put it on the CC stack
#ifdef recursive
        void perform(int id1 , int id2) const
        {

            cell_iter const&c1 = rooot.CellNo(id1) ;
            cell_iter const&c2 = rooot.CellNo(id2) ;

            if(IA->interact(c1,c2))
            {
#   if pfalcON
#   else
	      ++ic;
#   endif
                
            }
            else
            {
                if(number(c1) + number(c2) > tct)
                {
                    split_cell_cell2(c1,c2);
                }
                else
                {
                    split_cell_cell2_std(c1,c2);
                    
                }
            }
        }
#endif

        //--------------------------------------------------------------------------
        /// try to perform a cell self interaction; otherwise put it on the CX stack
        void perform(cell_iter const&cc) const
        {
#ifdef recursive
            if(IA->interact(cc))
            {
#   if pfalcON
#   else
	      ++ic;
#   endif
                
            }
            else
            {
                
                //                int id1 = cc.index();
                if(number(cc) > tct)
                {
                    split_cell_self2(cc);
                }
                else
                {
                    split_cell_self2_std(cc);
                }
            }
#else
            if(IA->interact(cc)) ++ic;
            else CX.push(cc);
#endif
        }
        /// try to perform a cell-cell interaction; otherwise put it on the CC stack
        void perform(cell_iter const&c1, cell_iter const&c2) const
        {
#ifdef recursive
            if(IA->interact(c1,c2))
            {
#   if pfalcON
#   else
	      ++ic;
#   endif
                
            }
            else
            {
                //                int id1 = c1.index();
                //              int id2 = c2.index();
                if(number(c1) + number(c2) > tct)
                {
                    split_cell_cell2(c1,c2);
                }
                else
                {
                    split_cell_cell2_std(c1,c2);
                    
                }
            }
#else
            if(IA->interact(c1,c2)) ++ic;
            else CC.push(c1,c2);
#endif
        }
        /// try to perform a cell-leaf interaction; otherwise put it on the CL stack
        void perform(cell_iter const&c1, leaf_iter const&l2) const
        {
#ifdef recursive
            if(!IA->interact(c1,l2))
            {
                if(number(c1) > tct)
                {
                    split_cell_leaf2(c1,l2);
                }
                else
                {
                    split_cell_leaf2_std(c1,l2);
                }
            }
#else
            if(!IA->interact(c1,l2)) CL.push(c1,l2);
#endif
        }
        /// try to perform a leaf-cell interaction; otherwise put it on the LC stack
        void perform(leaf_iter const&l1, cell_iter const&c2) const
        {
#ifdef recursive
            if(!IA->interact(l1,c2))
            {
                if(number(c2) > tct)
                {
                    split_leaf_cell2(l1,c2);
                }
                else
                {
                    split_leaf_cell2_std(l1,c2);
                }
            }
#else
            if(!IA->interact(l1,c2)) LC.push(l1,c2);
#endif
        }
        /// perform a leaf-leaf interaction
        void perform(leaf_iter const&l1, leaf_iter const&l2) const {
            IA->interact(l1,l2);
        }
        void perform_std(cell_iter const&cc) const
        {
            if(IA->interact(cc))
            {
#ifdef pfalcON
#else
	      ++ic;
#endif
                
            }
            else
            {
                split_cell_self2_std(cc);
            }
        }
        /// try to perform a cell-leaf interaction; otherwise put it on the CL stack
        void perform_std(cell_iter const&c1, leaf_iter const&l2) const
        {
            if(!IA->interact(c1,l2))
            {
                split_cell_leaf2_std(c1,l2);
            }
        }
        /// try to perform a cell-cell interaction; otherwise put it on the CC stack
        void perform_std(cell_iter const&c1, cell_iter const&c2) const
        {
            if(IA->interact(c1,c2))
            {
#ifdef pfalcON
#else
              ++ic;
#endif
                
            }
            else
            {
                
                split_cell_cell2_std(c1,c2);
            }
        }
        /// try to perform a leaf-cell interaction; otherwise put it on the LC stack
        void perform_std(leaf_iter const&l1, cell_iter const&c2) const
        {
            if(!IA->interact(l1,c2))
            {
                split_leaf_cell2_std(l1,c2);
            }
        }
        /// perform a leaf-leaf interaction
        void perform_std(leaf_iter const&l1, leaf_iter const&l2) const {
            IA->interact(l1,l2);
        }
        ////Fin pfalcON_STD
    private:
        //--------------------------------------------------------------------------
        /// clear the LC stack of leaf-cell interactions
#ifdef recursive
#else
        
        void clear_leaf_cell_stack() const {
            while(!LC.is_empty()) {                        // WHILE(LC non-empty)
                lc_iact lc = LC.pop();                       //   pop new L-C iaction
                LoopLKids(lc.snd,l2) perform(lc.fst,l2);     //   perform sub L-L
                LoopCKids(lc.snd,c2) perform(lc.fst,c2);     //   perform sub L-C
            }                                              // END WHILE
        }
        //--------------------------------------------------------------------------
        /// clear the CL stack of cell-leaf interactions
        void clear_cell_leaf_stack() const {
            while(!CL.is_empty()) {                        // WHILE(CL non-empty)
                cl_iact cl = CL.pop();                       //   pop new C-L iaction
                LoopLKids(cl.fst,l1) perform(l1,cl.snd);     //   perform sub L-L
                LoopCKids(cl.fst,c1) perform(c1,cl.snd);     //   perform sub C-L
            }                                              // END WHILE
        }
#endif
        //--------------------------------------------------------------------------
        /// split a CC interaction, keep CL & LC stacks clear
        void split_cell_cell(cc_iact&cc) const
        {
#ifdef recursive
#ifdef pfalcON_useTBB
            if(IA->split_first(cc.fst,cc.snd))
            { // IF(split 1st)
                LoopLKids(cc.fst,l1)
                {
                    perform(l1,cc.snd);//   perform sub L-C
                }
                //clear_leaf_cell_stack();//   clear the LC stack
                LoopCKids(cc.fst,c1)
                {
                    //                    g.run( [=] () mutable{ perform(c1,cc.snd);});
                    g.run([=]{this->perform(c1,cc.snd);});//   perform sub C-C
                    //perform(c1,cc.snd);
                }
            }
            else { // ELSE(split 2nd)
                LoopLKids(cc.snd,l2)
                {
                    perform(cc.fst,l2);//   perform sub C-L
                }
                //clear_cell_leaf_stack();//   clear the CL stack
                LoopCKids(cc.snd,c2)
                {
                    //                  g.run( [=] () mutable{perform( cc.fst,c2);});
                    g.run([=]{this->perform( cc.fst,c2);});//   perform sub C-C
                    //perform( cc.fst,c2);
                }
            } // ENDIF
#else
            if(IA->split_first(cc.fst,cc.snd)) { // IF(split 1st)
                LoopLKids(cc.fst,l1) perform(l1,cc.snd);//   perform sub L-C
                //clear_leaf_cell_stack();//   clear the LC stack
                LoopCKids(cc.fst,c1) perform(c1,cc.snd);//   perform sub C-C
            }
            else { // ELSE(split 2nd)
                LoopLKids(cc.snd,l2) perform(cc.fst,l2);//   perform sub C-L
                //clear_cell_leaf_stack();//   clear the CL stack
                LoopCKids(cc.snd,c2)
                {
                    int id1 = cc.fst.index();
                    int id2 = c2.index();
#pragma omp task
                    perform( id1,id2);//   perform sub C-C
                }
            } // ENDIF
#endif
#else
            if(IA->split_first(cc.fst,cc.snd)) { // IF(split 1st)
                LoopLKids(cc.fst,l1) perform(l1,cc.snd);//   perform sub L-C
                //clear_leaf_cell_stack();//   clear the LC stack
                LoopCKids(cc.fst,c1) perform(c1,cc.snd);//   perform sub C-C
            }
            else { // ELSE(split 2nd)
                LoopLKids(cc.snd,l2) perform(cc.fst,l2);//   perform sub C-L
                //clear_cell_leaf_stack();//   clear the CL stack
                LoopCKids(cc.snd,c2)
                {
                    perform( cc.fst,c2);//   perform sub C-C
                }
            } // ENDIF
#endif
        }
        
        //--------------------------------------------------------------------------
        /// split a cell-self interaction, keep CL & LC stacks clear
        void split_cell_self(cx_iact&cx) const
        {
#ifdef recursive
#ifdef pfalcON
#ifdef pfalcON_useTBB
            LoopSPairs(cx.obj,l1,l2)
            {
                perform(l1,l2); // perform sub L-L
            }
            LoopCKids(cx.obj,c1)
            { // LOOP(cell kids)
                LoopLKids (cx.obj,l2)
                {
                    perform(c1,l2);//   perform sub C-L
                }
                LoopCPairs(cx.obj,c1+1,c2)
                {
                    g.run([=]{this->perform(c1,c2);});//   perform sub C-C
                    //                    perform(c1,c2);

                }
            } // END LOOP
            LoopCKids(cx.obj,c1)
            { // LOOP(cell kids)
                // g.run( [=] () mutable {perform(c1);});
                g.run([=]{this->perform(c1);});
            }
#else
            LoopSPairs(cx.obj,l1,l2)
            {
                perform(l1,l2); // perform sub L-L
            }
            LoopCKids(cx.obj,c1)
            { // LOOP(cell kids)
                LoopLKids (cx.obj,l2)
                {
                    perform(c1,l2);//   perform sub C-L
                }
                LoopCPairs(cx.obj,c1+1,c2)
                {
                    int id1 = c1.index();
                    int id2 = c2.index();
#pragma omp task
                    perform(id1,id2);//   perform sub C-C
                }
            } // END LOOP
            LoopCKids(cx.obj,c1)
            { // LOOP(cell kids)
                int id1 = c1.index();
#pragma omp task
                perform(id1);
            }
#endif
#else
            LoopSPairs(cx.obj,l1,l2)
            {
                perform(l1,l2); // perform sub L-L
            }
            LoopCKids(cx.obj,c1)
            { // LOOP(cell kids)
                LoopLKids (cx.obj,l2)
                {
                    perform(c1,l2);//   perform sub C-L
                }
                LoopCPairs(cx.obj,c1+1,c2)
                {
                    int id1 = c1.index();
                    int id2 = c2.index();
                    perform(id1,id2);//   perform sub C-C
                }
            } // END LOOP
            LoopCKids(cx.obj,c1)
            { // LOOP(cell kids)
                int id1 = c1.index();
                perform(id1);
            }

#endif
#else

            LoopSPairs(cx.obj,l1,l2)
            {
                perform(l1,l2); // perform sub L-L
            }
            LoopCKids(cx.obj,c1) { // LOOP(cell kids)
                perform(c1);
                LoopLKids (cx.obj,l2) perform(c1,l2);//   perform sub C-L
                LoopCPairs(cx.obj,c1+1,c2) perform(c1,c2);//   perform sub C-C
            } // END LOOP
            clear_cell_leaf_stack();                       // clear the CL stack
            
#endif
        }
#ifdef recursive
        /// split a CC interaction, keep CL & LC stacks clear
        void split_cell_cell_std(cc_iact&cc) const {
            if(IA->split_first(cc.fst,cc.snd))
            { // IF(split 1st)
                LoopLKids(cc.fst,l1) perform_std(l1,cc.snd);//   perform sub L-C
                //clear_leaf_cell_stack();//   clear the LC stack
                LoopCKids(cc.fst,c1) perform_std(c1,cc.snd);//   perform sub C-C
            } else { // ELSE(split 2nd)
                LoopLKids(cc.snd,l2) perform_std(cc.fst,l2);//   perform sub C-L
                //clear_cell_leaf_stack();//   clear the CL stack
                LoopCKids(cc.snd,c2)
                {
                    perform_std( cc.fst,c2);//   perform sub C-C
                }
            } // ENDIF
        }
        //--------------------------------------------------------------------------
        /// split a cell-self interaction, keep CL & LC stacks clear
        void split_cell_self_std(cx_iact&cx) const
        {
            LoopSPairs(cx.obj,l1,l2) perform_std(l1,l2); // perform sub L-L
            LoopCKids(cx.obj,c1)
            { // LOOP(cell kids)
                LoopLKids (cx.obj,l2) perform_std(c1,l2);//   perform sub C-L
                LoopCPairs(cx.obj,c1+1,c2) perform_std(c1,c2);//   perform sub C-C
            } // END LOOP
            LoopCKids(cx.obj,c1) { // LOOP(cell kids)
                perform_std(c1);
            }
        }
        ////FIn pfalcON_STD
#undef LoopCKids
#undef LoopLKids
#undef LoopCPairs
#undef LoopSPairs
        
#else
        
        //--------------------------------------------------------------------------
#undef LoopCKids
#undef LoopLKids
#undef LoopCPairs
#undef LoopSPairs
        //--------------------------------------------------------------------------
        //--------------------------------------------------------------------------
        /// clear CC stack, keep CL & LC stacks clear
        void clear_cell_cell_stack() const {
            while(!CC.is_empty()) {
                cc_iact cc = CC.pop();
                split_cell_cell(cc);
            }
        }
        //--------------------------------------------------------------------------
        /// clear CX stack, keep CC, CL, & LC stacks clear
        void clear_cell_self_stack() const {
            while(!CX.is_empty()) {                        // WHILE(CX non-empty)
                cx_iact cx = CX.pop();                       //   pop new self iaction
                split_cell_self(cx);                         //   split it
                clear_cell_cell_stack();                     //   clear the CC stack
            }                                              // END WHILE
        }
        //--------------------------------------------------------------------------
        /// perform up to \e M cell-cell interactions,
        /// perform all occuring leaf interactions
        void work_cell_cell_stack(int const&M) const {
            while(!CC.is_empty() && ic < M) {
                cc_iact cc = CC.pop();
                split_cell_cell(cc);
            }
        }
        //--------------------------------------------------------------------------
        /// perform up to \e M cell-cell interactions,
        /// perform all occuring leaf interactions
        void work_cell_self_stack(int const&M) const {
            while(!CX.is_empty() && ic < M) {
                cx_iact cx = CX.pop();
                split_cell_self(cx);
                work_cell_cell_stack(M);
            }
        }
#endif
        
        
        //--------------------------------------------------------------------------
        // construction
        //--------------------------------------------------------------------------
        static unsigned n_x(unsigned const&d) {return Nsub*d+1;}
        static unsigned n_c(unsigned const&d) {return Nsub*(Nsub-1)/2+Nsub*d;}
        //--------------------------------------------------------------------------
    public:
#ifdef recursive
        /// construction for interaction within one tree
        ///
        /// \param ia (input) ^ to interactor, which must satisfy layout documented
        /// \param d1 (input) depth of tree
        MutualInteractor(INTERACTOR*const&ia,
                         unsigned const&d1) :
        IA ( ia )
        {
        }
        //--------------------------------------------------------------------------
        /// construction for interaction between two trees (not used in falcON)
        ///
        /// \param ia (input) ^ to interactor, which must satisfy layout documented
        /// \param d1 (input) depth of 1st tree
        /// \param d2 (input) depth of 2nd tree
        MutualInteractor(INTERACTOR*const&ia,
                         unsigned const&d1,
                         unsigned const&d2) :
        IA ( ia )
        {
        }
#else
        /// construction for interaction within one tree
        ///
        /// \param ia (input) ^ to interactor, which must satisfy layout documented
        /// \param d1 (input) depth of tree
        MutualInteractor(INTERACTOR*const&ia,
                         unsigned   const&d1) :
        IA ( ia ),                                     //   initialize interactor
        CX ( n_x(d1) ),                                //   initialize cell-self
        CC ( n_c(2*d1) ),                              //   initialize cell-cell
        CL ( n_c(2*d1) ),                              //   initialize cell-leaf
        LC ( n_c(2*d1) ) {}                            //   initialize leaf-cell
        //--------------------------------------------------------------------------
        /// construction for interaction between two trees (not used in falcON)
        ///
        /// \param ia (input) ^ to interactor, which must satisfy layout documented
        /// \param d1 (input) depth of 1st tree
        /// \param d2 (input) depth of 2nd tree
        MutualInteractor(INTERACTOR*const&ia,
                         unsigned   const&d1,
                         unsigned   const&d2) :
        IA ( ia ),                                     //   initialize interactor
        CX ( n_x(d1) ),                                //   initialize cell-self
        CC ( n_c(d2? d1+d2 : 2*d1) ),                  //   initialize cell-cell
        CL ( n_c(d2? d1+d2 : 2*d1) ),                  //   initialize cell-leaf
        LC ( n_c(d2? d1+d2 : 2*d1) ) {}                //   initialize leaf-cell
#endif
        
        //--------------------------------------------------------------------------
        /// \name routines for interactions in ordinary (sequential) code
        //@{
        /// perform a leaf-leaf interaction
        void leaf_leaf(leaf_iter const&a, leaf_iter const&b) const
        {
            IA->interact(a,b);
        }
        /// resolve a cell self interaction with all its sub-interactions
#ifdef recursive
        
        
        void cell_self(cell_iter const&a) const
        {
            perform(a);
        }
        void pfalcON_clear_cell_self_stack() const
        {
        }
        /// resolve a cell-cell interaction with all its sub-interactions
        void cell_cell(cell_iter const&a, cell_iter const&b) const falcON_THROWING
        {
            if(a==b) falcON_THROW("MutualInteractor::cell_cell(): self-interaction");
            perform(a,b);
        }
        /// resolve a cell-leaf interaction with all its sub-interactions
        void cell_leaf(cell_iter const&a, leaf_iter const&b) const
        {
            perform(a,b);
            //clear_cell_leaf_stack();
        }
        /// resolve a leaf-cell interaction with all its sub-interactions
        void leaf_cell(leaf_iter const&a, cell_iter const&b) const
        {
            perform(a,b);
        }
#else
        void cell_self(cell_iter const&a) const
        {
            perform(a);
            clear_cell_self_stack();
        }
        /// resolve a cell-cell interaction with all its sub-interactions
        void cell_cell(cell_iter const&a, cell_iter const&b) const falcON_THROWING
        {
            if(a==b) falcON_THROW("MutualInteractor::cell_cell(): self-interaction");
            perform(a,b);
            clear_cell_cell_stack();
        }
        /// resolve a cell-leaf interaction with all its sub-interactions
        void cell_leaf(cell_iter const&a, leaf_iter const&b) const
        {
            perform(a,b);
            clear_cell_leaf_stack();
        }
        /// resolve a leaf-cell interaction with all its sub-interactions
        void leaf_cell(leaf_iter const&a, cell_iter const&b) const
        {
            perform(a,b);
            clear_leaf_cell_stack();
        }
#endif
        
        
        //@}
        //--------------------------------------------------------------------------
        /// \name routines for interactions in MPI parallel code
        //
        // they allow to interrupt the clearing of the CX and/or CC stack, e.g., for
        // testing for the sending or receipt of an MPI message.
        // use init_..()   to add a new root-root interaction onto a stack
        // use work_..()   to perform at most a pre-defined number of cell iactions
        //                 (leaf-cell and leaf-leaf iaction are not counted)
        //                 returns true if still some work is left.
        // use finish_..() to clear the stack
        //@{
        /// initialize resolving a basic cell self interaction
        ///
        /// typically, this is used to start the local root-self interaction
        void init_cell_self(cell_iter const&a) const
        {
            perform(a);
        }
        /// partially resolve cell self interaction:
        /// perform up to \e M cell interactions
        ///
        /// works on resolving the cell self interaction previously initialized
        /// with init_cell_self().
        ///
        /// \return true if still work to be done
        /// \param  M (input) max number of cell interaction to be performed
#ifdef recursive
        /// finishes resolving the cell self interaction previously initialized
        /// with init_cell_self() and possibly worked on with work_cell_self()
        void finish_cell_self() const
        {
            
        }
        //==========================================================================
        /// initialize resolving a basic cell-cell interaction
        ///
        /// typically to be used for a (local) root - (remove) root interaction
        void init_cell_cell(cell_iter const&a, cell_iter const&b) const
        falcON_THROWING
        {
            if(a==b)
                falcON_THROW("MutualInteractor::init_cell_cell(): self-interaction");
            perform(a,b);
        }
#else
        
        bool work_cell_self(int const&M) const
        {
            ic = 0;
            work_cell_self_stack(M);
            return !(CX.is_empty() && CC.is_empty());
        }
        /// finish resolving a cell self interaction
        ///
        /// finishes resolving the cell self interaction previously initialized
        /// with init_cell_self() and possibly worked on with work_cell_self()
        void finish_cell_self() const
        {
            clear_cell_cell_stack();
            clear_cell_self_stack();
        }
        
        //==========================================================================
        /// initialize resolving a basic cell-cell interaction
        ///
        /// typically to be used for a (local) root - (remove) root interaction
        void init_cell_cell(cell_iter const&a, cell_iter const&b) const
        falcON_THROWING
        {
            if(a==b)
                falcON_THROW("MutualInteractor::init_cell_cell(): self-interaction");
            perform(a,b);
        }
        /// partially resolve cell-cell interaction:
        /// perform up to \e M cell interactions
        ///
        /// works on resolving the cell-cell interaction previously initialized
        /// with init_cell_cell().
        ///
        /// \return true if still work to be done
        /// \param  M (input) max number of cell interaction to be performed
        bool work_cell_cell(int const&M) const
        {
            ic = 0;
            work_cell_cell_stack(M);
            return !CC.is_empty();
        }
        /// finish resolving a cell-cell interaction
        ///
        /// finishes resolving the cell-cell interaction previously initialized
        /// with init_cell_cell() and possibly worked on with work_cell_cell()
        void finish_cell_cell() const
        {
            clear_cell_cell_stack();
        }
#endif
        
        //@}
        //--------------------------------------------------------------------------
    }; // class MutualInteractor<> {
    // ///////////////////////////////////////////////////////////////////////////
    //
    // class falcON::BasicIactor<ESTIMATOR>
    //
    // a base class for a possible template argument of MutualInteractor<>
    //
    // the template argument merely serves to provide the types cell_iterator and
    // leaf_iterator.
    //
    // ///////////////////////////////////////////////////////////////////////////
    template<typename ESTIMATOR> class BasicIactor {
        //--------------------------------------------------------------------------
        // types of class BasicIactor
        //--------------------------------------------------------------------------
    public:
        typedef ESTIMATOR estm_type; // type of estimator
        typedef typename estm_type::cell_iterator cell_iter; // iterator over cell
        typedef typename estm_type::leaf_iterator leaf_iter; // iterator over leaf
        //--------------------------------------------------------------------------
        // data of class BasicIactor
        //--------------------------------------------------------------------------
    private:
        const unsigned NCB, NCC, NCL; // params control direct
        //--------------------------------------------------------------------------
        // abstract methods, MUST be provided by derived class
        //--------------------------------------------------------------------------
    protected:
        virtual bool many(cell_iter const&, leaf_iter const&) const = 0;
        virtual bool many(cell_iter const&, cell_iter const&) const = 0;
        virtual bool many(cell_iter const&) const = 0;
        virtual void single(leaf_iter const&, leaf_iter const&) const = 0;
        virtual bool discard(cell_iter const&, leaf_iter const&) const = 0;
        virtual bool discard(cell_iter const&, cell_iter const&) const = 0;
    public:
        virtual bool split_first(cell_iter const&, cell_iter const&) const = 0;
        //--------------------------------------------------------------------------
        // other methods
        //--------------------------------------------------------------------------
    protected:
        explicit BasicIactor(const unsigned dir[4] = Default::direct) :
        NCB(dir[1]), NCC(dir[2]), NCL(dir[3]) {
        }
        virtual ~BasicIactor() {
        }
        //--------------------------------------------------------------------------
    public:
        bool interact(cell_iter const&A, leaf_iter const&B) const
        {
            
            if (!is_active(A) && !is_active(B))
                return true; // no actives -> no job
            if (discard(A, B))
                return true; // try to discard
            if (number(A) < NCB)
                return many(A, B); // perform many single
            return false; // must split
        }
        //--------------------------------------------------------------------------
        bool interact(leaf_iter const&B, cell_iter const&A) const
        {
            if (!is_active(A) && !is_active(B))
                return true; // no actives -> no job
            
            if (discard(A, B))
                return true; // try to discard
            
            if (number(A) < NCB)
                return many(A, B); // perform many single
            
            return false; // must split
        }
        //--------------------------------------------------------------------------
        bool interact(cell_iter const&A, cell_iter const&B) const
        {
            
            if (!is_active(A) && !is_active(B))
                return true; // no actives -> no job
            if (discard(A, B))
                return true; // try to discard
            if (number(A) < NCC && number(B) < NCC)
                return many(A, B); // perform many single
            return false; // must split
        }
        //--------------------------------------------------------------------------
        bool interact(cell_iter const&A) const
        {
            if (!is_active(A))
                return true; // no actives -> no job
            if (number(A) < NCL)
                return many(A); // perform many single
            return false; // must split
        }
        //--------------------------------------------------------------------------
        void interact(leaf_iter const&A, leaf_iter const&B) const
        {
            single(A, B);
        }
    };
    // ///////////////////////////////////////////////////////////////////////////
}// namespace falcON {
// /////////////////////////////////////////////////////////////////////////////
#endif // falcON_included_interact_h

