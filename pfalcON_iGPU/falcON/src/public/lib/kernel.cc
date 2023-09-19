// -*- C++ -*-
// /////////////////////////////////////////////////////////////////////////////
//
/// \file    src/public/kernel.cc
//
/// \brief   implements inc/public/kernel.h
/// \author  Walter Dehnen
/// \date    2000-2010,2012
//
// /////////////////////////////////////////////////////////////////////////////
//
// Copyright (C) 2008-2010  Walter Dehnen
//               2013       Benoit Lange, Pierre Fortin
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc., 675
// Mass Ave, Cambridge, MA 02139, USA.
//
////////////////////////////////////////////////////////////////////////////////
//          22/11/2013  BL recursive implementation of DTT
//          22/11/2013  BL added support of OpenMP for parallel DTT
//          22/11/2013  BL added support of TBB for parallel DTT
// v p0.1   22/11/2013  BL added ISPC kernels
////////////////////////////////////////////////////////////////////////////////

#include <public/types.h>
#include <public/kernel.h>
#include <public/tensor_set.h>
#include <utils/WDMath.h>

#ifdef pfalcON
uint32_t tct;
#ifdef pfalcON_useTBB
#include "tbb/tbb_stddef.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"
#include "tbb/atomic.h"
using namespace tbb;
#else
///OpenMP
#include <omp.h>
#endif
#endif 
#if iGPU
#include <public/cl_manip.h>
extern cl_manip * gpu;
extern falcON::OctTree *globalT;
#endif

#ifdef ispcpfalcON
#include <public/P2P.h>
#include <public/P2P2.h>
using namespace ispc;
unsigned nbSIMD;
bool mesureOK;
extern falcON::GravEstimator::Leaf::acpn_data *globalACPN;

#if __AVX__
unsigned thresholdCC = 8;
unsigned thresholdCC2 = 2;

unsigned thresholdCCx2 = 32;
unsigned thresholdCC2x2 = 13;

unsigned thresholdCS = 6;
unsigned thresholdCSx2 = 72;

#elif __SSE__
unsigned thresholdCC = 7;
unsigned thresholdCC2 = 3;

unsigned thresholdCCx2 = 10;
unsigned thresholdCC2x2 = 3;

unsigned thresholdCS = std::numeric_limits<unsigned>::max();
unsigned thresholdCSx2 = 8;

#elif  __MIC__
unsigned thresholdCC = 4;
unsigned thresholdCC2 = 2;

unsigned thresholdCCx2 = std::numeric_limits<unsigned>::max();
unsigned thresholdCC2x2 = std::numeric_limits<unsigned>::max();

unsigned thresholdCS = 3;
unsigned thresholdCSx2 = std::numeric_limits<unsigned>::max();

#else
    #error "This code should not compile without SSE, AVX or MIC instruction."
#endif

#endif // #ifdef ispcpfalcON

#ifdef traceKernels
double tempsPP;
double tempsM2L;
int nbPP;
int nbM2L;
int** typePP;
#endif
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Macros facilitating the Cell-Leaf and Cell-Cell interactions               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifdef falcON_SSE_CODE
#  define __ARG_D D,J
#else
#  define __ARG_D D
#endif

#define CellLeaf(A,B,D,J,R) {						       \
grav::Cset F;                                    /* to hold F^(n)        */  \
if(is_active(A)) {                               /* IF A is active       */  \
set_dPhi(F,R,__ARG_D);                         /*   F^(n) = d^nPhi/dR^n*/  \
add_C_B2C(A->Coeffs(),F);                      /*   C_A   = ...        */  \
if(is_active(B)) {                             /*   IF B is active, too*/  \
F.flip_sign_odd();                           /*     flip sign:F^(odd)*/  \
add_C_C2B(B->Coeffs(),F,A->poles());         /*     C_B   = ...      */  \
}                                              /*   ENDIF              */  \
} else if(is_active(B)) {                        /* ELIF B is active     */  \
R.negate();                                    /*   flip sign: R       */  \
set_dPhi(F,R,__ARG_D);                         /*   F^(n) = d^nPhi/dR^n*/  \
add_C_C2B(B->Coeffs(),F,A->poles());           /*   C_B   = ...        */  \
}                                                /* ENDIF                */  \
}

#define CellLeafAll(A,B,D,J,R) {					       \
grav::Cset F;                                    /* to hold F^(n)        */  \
set_dPhi(F,R,__ARG_D);                           /* F^(n) = d^nPhi/dR^n  */  \
add_C_B2C(A->Coeffs(),F);                        /* C_A   = ...          */  \
F.flip_sign_odd();                               /* F^(n) = d^nPhi/dR^n  */  \
add_C_C2B(B->Coeffs(),F,A->poles());             /* C_B   = ...          */  \
}

#define CellCell(A,B,D,J,R) {						       \
grav::Cset F;                                    /* to hold F^(n)        */  \
if(is_active(A)) {                               /* IF A is active       */  \
set_dPhi(F,R,__ARG_D);                         /*   F^(n) = d^nPhi/dR^n*/  \
add_C_C2C(A->Coeffs(),F,B->poles());           /*   C_A   = ...        */  \
if(is_active(B)) {                             /*   IF B is active, too*/  \
F.flip_sign_odd();                           /*     flip sign:F^(odd)*/  \
add_C_C2C(B->Coeffs(),F,A->poles());         /*     C_B   = ...      */  \
}                                              /*   ENDIF              */  \
} else if(is_active(B)) {                        /* ELIF B is active     */  \
R.negate();                                    /*   flip sign: R       */  \
set_dPhi(F,R,__ARG_D);                         /*   F^(n) = d^nPhi/dR^n*/  \
add_C_C2C(B->Coeffs(),F,A->poles());           /*   C_B   = ...        */  \
}                                                /* ENDIF                */  \
}

#define CellCellAll(A,B,D,J,R) {					       \
grav::Cset F;                                    /* to hold F^(n)        */  \
set_dPhi(F,R,__ARG_D);                           /* F^(n) = d^nPhi/dR^n  */  \
add_C_C2C(A->Coeffs(),F,B->poles());             /* C_A   = ...          */  \
F.flip_sign_odd();                               /* F^(n) = d^nPhi/dR^n  */  \
add_C_C2C(B->Coeffs(),F,A->poles());             /* C_B   = ...          */  \
}

////////////////////////////////////////////////////////////////////////////////

using namespace falcON;
using namespace falcON::grav;
////////////////////////////////////////////////////////////////////////////////
//
// class falcON::TaylorSeries
//
////////////////////////////////////////////////////////////////////////////////
inline void TaylorSeries::
shift_and_add(const grav::cell*const&c) {          // I: cell & its coeffs
     if(hasCoeffs(c)) {                               // IF(cell has had iaction)
        vect dX = cofm(c) - X;                         //   vector to shift by
        if(dX != zero && C != zero) {                  //   IF(dX != 0 AND C != 0)
            shift_by(C,dX);                              //     shift expansion
        }                                              //   ENDIF
        X = cofm(c);                                   //   set X to new position
        C.add_times(Coeffs(c), one/mass(c));           //   add cell's coeffs in
    }                                                // ENDIF
}
//------------------------------------------------------------------------------
inline void TaylorSeries::
extract_grav(leaf_iter const&L) const {            // I: leaf to get grav to
    eval_expn(L->Coeffs(),C,cofm(L)-X);              // evaluate expansion
}
////////////////////////////////////////////////////////////////////////////////
//
// class falcON::GravKernBase
//
////////////////////////////////////////////////////////////////////////////////
void GravKernBase::eval_grav(cell_iter    const&C,
                             TaylorSeries const&T) const
{
    TaylorSeries G(T);                               // G = copy of T
    G.shift_and_add(C);                              // shift G; G+=T_C
    take_coeffs(C);                                  // free memory: C's coeffs
    {
        
        for(cell_iter::leaf_child l  = C.begin_leafs();l != C.end_leaf_kids(); ++l)
        {
            
            {
                if(is_active(l))
                {   // LOOP C's active leaf kids
                    l->normalize_grav();                           //   pot,acc/=mass
                    if(!is_empty(G))
                        G.extract_grav(l);            //   add pot,acc due to G
                }
            }
        }
        
        LoopCellKids(cell_iter,C,c) if(is_active(c))     // LOOP C's active cell kids
        {
            
            eval_grav(c,G);                                //   recursive call
        }
    }
}
//------------------------------------------------------------------------------
void GravKernBase::eval_grav_all(cell_iter    const&C,
                                 TaylorSeries const&T) const
{
#ifdef pfalcON
#ifdef pfalcON_useTBB
    task_group g;
#endif
#endif
    TaylorSeries G(T);                               // G = copy of T
    G.shift_and_add(C);                              // shift G; G+=T_C
    // Memory release not done here in pfalcON (implies a performance bottelneck):
    //    take_coeffs(C);                                  // free memory: C's coeffs
    LoopLeafKids(cell_iter,C,l) {                    // LOOP C's leaf kids
        l->normalize_grav();                           //   pot,acc/=mass
        if(!is_empty(G)) G.extract_grav(l);            //   add pot,acc due to G
    }                                                // END LOOP
    //    LoopCellKids(cell_iter,C,c)                      // LOOP C's cell kids
    for(cell_iter::cell_child                        /* type of child cell     */\
        c  = C.begin_cell_kids();              /* from first child       */\
        c != C.end_cell_kids();                /* until beyond last      */\
        ++c)
#ifdef pfalcON
#ifdef pfalcON_useTBB
      {
        if(number(c) > tct){
	  g.run([=]{this->eval_grav_all(c,G); });                           //   recursive call
	  //                eval_grav_all(c,G);                            //   recursive call
        }
        else{
	  eval_grav_all(c,G);                            //   recursive call
        }
      }
    g.wait();
#else
    {
      if(number(c) > tct){
#pragma omp task //default(shared) shared(C)
	eval_grav_all(c,G);                            //   recursive call
      }
      else {
	eval_grav_all(c,G);                            //   recursive call
      } 
    }
#endif
#else
    {
      eval_grav_all(c,G);                            //   recursive call
    }
}
#endif

}
//------------------------------------------------------------------------------
real GravKernBase::Psi(kern_type k, real Xq, real Eq)
{
    switch(k) {
        case p1: {
            real   x = one/(Xq+Eq), d0=sqrt(x), d1=d0*x, hq=half*Eq;
            return d0 + hq*d1;
        }
        case p2: {
            real   x = one/(Xq+Eq), d0=sqrt(x), d1=d0*x, d2=3*d1*x, hq=half*Eq;
            return d0 + hq*(d1+hq*d2);
        }
        case p3: {
            real   x = one/(Xq+Eq), d0=sqrt(x), d1=d0*x, d2=3*d1*x, d3=5*d2*x,
            hq=half*Eq;
            return d0 + hq*(d1+half*hq*(d2+hq*d3));
        }
        default:
            return WDutils::invsqrt(Xq+Eq);
    }
}
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// 1. Single body-body interaction (these are extremely rare)                 //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
#define P0(MUM)					\
x  = one/(Rq+EQ);				\
D0 = MUM*sqrt(x);				\
R *= D0*x;
//------------------------------------------------------------------------------
#define P1(MUM)					\
x  = one/(Rq+EQ);				\
D0 = MUM*sqrt(x);				\
register real D1 = D0*x;			\
register real hq = half*EQ;			\
D0+= hq*D1;					\
D1+= hq*3*D1*x;				\
R *= D1;
//------------------------------------------------------------------------------
#define P2(MUM)						\
x  = one/(Rq+EQ);					\
D0 = MUM*sqrt(x);					\
register real D1 = D0*x, D2= 3*D1*x, D3= 5*D2*x;	\
register real hq = half*EQ;				\
D0+= hq*(D1+hq*D2);					\
D1+= hq*(D2+hq*D3);					\
R *= D1;
//------------------------------------------------------------------------------
#define P3(MUM)							\
x  = one/(Rq+EQ);						\
D0 = MUM*sqrt(x);						\
register real D1 = D0*x, D2= 3*D1*x, D3= 5*D2*x, D4= 7*D3*x;	\
register real hq = half*EQ;					\
register real qq = half*hq;					\
D0+= ((hq*D3+D2)*qq+D1)*hq;					\
D1+= ((hq*D4+D3)*qq+D2)*hq;					\
R *= D1;
//------------------------------------------------------------------------------
#define P0_I(MUM)				\
EQ = square(eph(A)+eph(B));			\
P0(MUM)
//------------------------------------------------------------------------------
#define P1_I(MUM)				\
EQ = square(eph(A)+eph(B));			\
P1(MUM)
//------------------------------------------------------------------------------
#define P2_I(MUM)				\
EQ = square(eph(A)+eph(B));			\
P2(MUM)
//------------------------------------------------------------------------------
#define P3_I(MUM)				\
EQ = square(eph(A)+eph(B));			\
P3(MUM)
//==============================================================================
void GravKern::single(leaf_iter const &A, leaf_iter const&B) const
{
    vect R  = cofm(A)-cofm(B);
    real Rq = norm(R),x,D0;
    if(INDI_SOFT)
        switch(KERN) {
            case p1: { P1_I(mass(A)*mass(B)) } break;
            case p2: { P2_I(mass(A)*mass(B)) } break;
            case p3: { P3_I(mass(A)*mass(B)) } break;
            default: { P0_I(mass(A)*mass(B)) } break;
        }
    else
        switch(KERN) {
            case p1: { P1(mass(A)*mass(B)) } break;
            case p2: { P2(mass(A)*mass(B)) } break;
            case p3: { P3(mass(A)*mass(B)) } break;
            default: { P0(mass(A)*mass(B)) } break;
        }
    if(is_active(A)) { A->pot()-=D0; A->acc()-=R; }
    if(is_active(B)) { B->pot()-=D0; B->acc()+=R; }
}
//------------------------------------------------------------------------------
void GravKernAll::single(leaf_iter const &A, leaf_iter const&B) const
{
    vect R  = cofm(A)-cofm(B);
    real Rq = norm(R),x,D0;
    if(INDI_SOFT)
        switch(KERN) {
            case p1: { P1_I(mass(A)*mass(B)) } break;
            case p2: { P2_I(mass(A)*mass(B)) } break;
            case p3: { P3_I(mass(A)*mass(B)) } break;
            default: { P0_I(mass(A)*mass(B)) } break;
        }
    else
        switch(KERN) {
            case p1: { P1(mass(A)*mass(B)) } break;
            case p2: { P2(mass(A)*mass(B)) } break;
            case p3: { P3(mass(A)*mass(B)) } break;
            default: { P0(mass(A)*mass(B)) } break;
        }
    A->pot()-=D0; A->acc()-=R;
    B->pot()-=D0; B->acc()+=R;
}
//------------------------------------------------------------------------------
#undef P0
#undef P1
#undef P2
#undef P3
#undef P0_I
#undef P1_I
#undef P2_I
#undef P3_I
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// 2. Cell-Node interactions with SSE instructions                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
#ifdef falcON_SSE_CODE
#  include <proper/kernel_SSE.h>
#else
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// 2. Cell-Node interactions without SSE instructions                         //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//==============================================================================
//
// macros for computing the gravity
//
// DSINGL       called after loading a leaf-leaf interaction
//
//==============================================================================
//
// for direct summation code, we assume:
// - real  D0          contains Mi*Mj   on input
// - real  D1          contains R^2+e^2 on input
// - real  EP[3]       are set to e^2, e^2/2 and e^2/4 for global softening
// - real  EQ          is set to e^2                   for individual softening
//
//==============================================================================
//
// NOTE that we changed the definition of the D_n by a sign:
//
// D_n = (-1/r d/dr)^n g(r) at r=|R|
//
//==============================================================================
# define DSINGL_P0_G				\
register real					\
XX  = one/D1;					\
D0 *= sqrt(XX);				\
D1  = XX * D0;
# define DSINGL_P0_I				\
DSINGL_P0_G
//------------------------------------------------------------------------------
# define DSINGL_P1_G				\
register real					\
XX  = one/D1;					\
D0 *= sqrt(XX);				\
D1  = XX * D0;				\
XX *= 3  * D1;          /* XX == T2 */	\
D0 += HQ * D1;				\
D1 += HQ * XX;
# define DSINGL_P1_I				\
HQ  = half*EQ;				\
DSINGL_P1_G
//------------------------------------------------------------------------------
# define DSINGL_P2_G				\
register real					\
XX  = one/D1;					\
D0 *= sqrt(XX);				\
D1  = XX * D0;				\
register real					\
D2  = 3 * XX * D1;				\
XX *= 5 * D2;           /* XX == T3 */	\
D0 += HQ*(D1+HQ*D2);				\
D1 += HQ*(D2+HQ*XX);
# define DSINGL_P2_I				\
HQ = half*EQ;					\
DSINGL_P2_G
//------------------------------------------------------------------------------
# define DSINGL_P3_G				\
register real					\
XX  = one/D1;					\
D0 *= sqrt(XX);				\
D1  =     XX * D0;				\
register real					\
D2  = 3 * XX * D1;				\
register real					\
D3  = 5 * XX * D2;				\
XX *= 7 * D3;           /* XX == T4 */	\
D0 += HQ*(D1+QQ*(D2+HQ*D3));			\
D1 += HQ*(D2+QQ*(D3+HQ*XX));
# define DSINGL_P3_I				\
HQ = half*EQ;					\
QQ = half*HQ;					\
DSINGL_P3_G
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// 2.1 direct summation of many-body interactions                             //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//
// macros for organizing the gravity computation via direct summation.
//
// we have to take care for active and non-active
//
//==============================================================================
//
// These macros assume:
//
// - vect    X0     position of left leaf
// - real    M0     mass of left leaf
// - vect    F0     force for left leaf (if used)
// - real    P0     potential for left leaf (if used)
// - vect    dR     to be filled with  X0-X_j
// - real    D0     to be filled with  M0*M_j
// - real    D1     to be filled with  norm(dR[J])+eps^2  on loading
//
//==============================================================================
#define LOAD_G					\
dR = X0 - cofm(B);				\
D1 = norm(dR) + EQ;				\
D0 = M0 * mass(B);
//------------------------------------------------------------------------------
#define LOAD_I					\
dR = X0 - cofm(B);				\
EQ = square(E0+eph(B));			\
D1 = norm(dR) + EQ;				\
D0 = M0 * mass(B);
//------------------------------------------------------------------------------
#define PUT_LEFT				\
dR *= D1;					\
P0 -= D0;					\
F0 -= dR;
//------------------------------------------------------------------------------
#define PUT_RGHT				\
dR       *= D1;				\
B->pot() -= D0;				\
B->acc() += dR;
//------------------------------------------------------------------------------
#define PUT_BOTH				\
dR       *= D1;				\
P0       -= D0;				\
F0       -= dR;				\
B->pot() -= D0;				\
B->acc() += dR;
//------------------------------------------------------------------------------
#define PUT_SOME				\
dR *= D1;					\
P0 -= D0;					\
F0 -= dR;					\
if(is_active(B)) {				\
B->pot() -= D0;				\
B->acc() += dR;				\
}
//------------------------------------------------------------------------------
#define GRAV_ALL(LOAD,DSINGL,PUT)		\
for(register leaf_iter B=B0; B!=BN; ++B) {	\
LOAD						\
DSINGL					\
PUT						\
}
//------------------------------------------------------------------------------
#define GRAV_FEW(LOAD,DSINGL)			\
for(register leaf_iter B=B0; B!=BN; ++B)	\
if(is_active(B)) {				\
LOAD					\
DSINGL					\
PUT_RGHT					\
}
//------------------------------------------------------------------------------
#define START_G					\
const    real      M0=mass(A);		\
const    vect      X0=cofm(A);		\
vect      dR;			\
register real      D0,D1;
//------------------------------------------------------------------------------
#define START_I					\
const    real      E0=eph(A);			\
const    real      M0=mass(A);		\
const    vect      X0=cofm(A);		\
vect      dR;			\
register real      D0,D1;
//==============================================================================
// now defining auxiliary inline functions for the computation of  N
// interactions. There are the following 10 cases:
// - each for the cases YA, YS, YN, NA, NS
// - each for global and individual softening
//==============================================================================
namespace {
    using namespace falcON; using namespace falcON::grav;
    //////////////////////////////////////////////////////////////////////////////
#define DIRECT(START,LOAD,DSINGL)				\
static void many_YA(ARGS) {					\
START; register real P0(zero); vect F0(zero);		\
GRAV_ALL(LOAD,DSINGL,PUT_BOTH)				\
A->pot()+=P0;  A->acc()+=F0;				\
}								\
static void many_YS(ARGS) {					\
START; register real P0(zero); vect F0(zero);		\
GRAV_ALL(LOAD,DSINGL,PUT_SOME)				\
A->pot()+=P0; A->acc()+=F0;				\
}								\
static void many_YN(ARGS) {					\
START; register real P0(zero); vect F0(zero);		\
GRAV_ALL(LOAD,DSINGL,PUT_LEFT)				\
A->pot()+=P0; A->acc()+=F0;				\
}								\
static void many_NA(ARGS) {					\
START;							\
GRAV_ALL(LOAD,DSINGL,PUT_RGHT)				\
}								\
static void many_NS(ARGS) {					\
START;							\
GRAV_FEW(LOAD,DSINGL)					\
}
    //////////////////////////////////////////////////////////////////////////////
    template<kern_type, bool> struct __direct;
    //----------------------------------------------------------------------------
#define ARGS					\
leaf_iter const&A,				\
leaf_iter const&B0,				\
leaf_iter const&BN,				\
real&EQ, real&, real&
    
    template<> struct __direct<p0,0> {
        DIRECT(START_G,LOAD_G,DSINGL_P0_G);
    };
    template<> struct __direct<p0,1> {
        DIRECT(START_I,LOAD_I,DSINGL_P0_I);
    };
    //----------------------------------------------------------------------------
#undef  ARGS
#define ARGS					\
leaf_iter const&A,				\
leaf_iter const&B0,				\
leaf_iter const&BN,				\
real&EQ, real&HQ, real&
    
    template<> struct __direct<p1,0> {
        DIRECT(START_G,LOAD_G,DSINGL_P1_G);
    };
    template<> struct __direct<p1,1> {
        DIRECT(START_I,LOAD_I,DSINGL_P1_I);
    };
    //----------------------------------------------------------------------------
    template<> struct __direct<p2,0> {
        DIRECT(START_G,LOAD_G,DSINGL_P2_G);
    };
    template<> struct __direct<p2,1> {
        DIRECT(START_I,LOAD_I,DSINGL_P2_I);
    };
    //----------------------------------------------------------------------------
#undef  ARGS
#define ARGS					\
leaf_iter const&A,				\
leaf_iter const&B0,				\
leaf_iter const&BN,				\
real&EQ, real&HQ, real&QQ
    
    template<> struct __direct<p3,0> {
        DIRECT(START_G,LOAD_G,DSINGL_P3_G);
    };
    template<> struct __direct<p3,1> {
        DIRECT(START_I,LOAD_I,DSINGL_P3_I);
    };
#undef LOAD_G
#undef LOAD_I
#undef START_G
#undef START_I
#undef DIRECT
    //////////////////////////////////////////////////////////////////////////////
    template<bool I> struct Direct {
        static void many_YA(kern_type KERN, ARGS) {
            switch(KERN) {
                case p1: __direct<p1,I>::many_YA(A,B0,BN,EQ,HQ,QQ); break;
                case p2: __direct<p2,I>::many_YA(A,B0,BN,EQ,HQ,QQ); break;
                case p3: __direct<p3,I>::many_YA(A,B0,BN,EQ,HQ,QQ); break;
                default: __direct<p0,I>::many_YA(A,B0,BN,EQ,HQ,QQ); break;
            } }
        static void many_YS(kern_type KERN, ARGS) {
            switch(KERN) {
                case p1: __direct<p1,I>::many_YS(A,B0,BN,EQ,HQ,QQ); break;
                case p2: __direct<p2,I>::many_YS(A,B0,BN,EQ,HQ,QQ); break;
                case p3: __direct<p3,I>::many_YS(A,B0,BN,EQ,HQ,QQ); break;
                default: __direct<p0,I>::many_YS(A,B0,BN,EQ,HQ,QQ); break;
            } }
        static void many_YN(kern_type KERN, ARGS) {
            switch(KERN) {
                case p1: __direct<p1,I>::many_YN(A,B0,BN,EQ,HQ,QQ); break;
                case p2: __direct<p2,I>::many_YN(A,B0,BN,EQ,HQ,QQ); break;
                case p3: __direct<p3,I>::many_YN(A,B0,BN,EQ,HQ,QQ); break;
                default: __direct<p0,I>::many_YN(A,B0,BN,EQ,HQ,QQ); break;
            } }
        static void many_NA(kern_type KERN, ARGS) {
            switch(KERN) {
                case p1: __direct<p1,I>::many_NA(A,B0,BN,EQ,HQ,QQ); break;
                case p2: __direct<p2,I>::many_NA(A,B0,BN,EQ,HQ,QQ); break;
                case p3: __direct<p3,I>::many_NA(A,B0,BN,EQ,HQ,QQ); break;
                default: __direct<p0,I>::many_NA(A,B0,BN,EQ,HQ,QQ); break;
            } }
        static void many_NS(kern_type KERN, ARGS) {
            switch(KERN) {
                case p1: __direct<p1,I>::many_NS(A,B0,BN,EQ,HQ,QQ); break;
                case p2: __direct<p2,I>::many_NS(A,B0,BN,EQ,HQ,QQ); break;
                case p3: __direct<p3,I>::many_NS(A,B0,BN,EQ,HQ,QQ); break;
                default: __direct<p0,I>::many_NS(A,B0,BN,EQ,HQ,QQ); break;
            } }
    };
    //////////////////////////////////////////////////////////////////////////////
}                                                  // END: unnamed namespace
#undef ARGS
//=============================================================================
// we now can define the cell-leaf and cell-self interaction via direct sums
//==============================================================================
#define ARGS KERN,B,A.begin_leafs(),A.end_leaf_desc(),EQ,HQ,QQ
void GravKern::direct(cell_iter const&A, leaf_iter const&B) const
{
#ifdef traceKernels
    nbPP++;
    double temp = my_gettimeofday();
    
#endif
#if defined(pfalcON) && (! defined(iGPU))
	_LockpfalcON(A);
	_LockpfalcON(B);
    
#endif
    
    if(INDI_SOFT)
        if(is_active(B)) {
            if     (al_active(A)) Direct<1>::many_YA(ARGS);
            else if(is_active(A)) Direct<1>::many_YS(ARGS);
            else                  Direct<1>::many_YN(ARGS);
        } else {
            if     (al_active(A)) Direct<1>::many_NA(ARGS);
            else if(is_active(A)) Direct<1>::many_NS(ARGS);
        }
        else
        {
            if(is_active(B)) {
                if     (al_active(A)) Direct<0>::many_YA(ARGS);
                else if(is_active(A)) Direct<0>::many_YS(ARGS);
                else                  Direct<0>::many_YN(ARGS);
            } else {
                if     (al_active(A)) Direct<0>::many_NA(ARGS);
                else if(is_active(A)) Direct<0>::many_NS(ARGS);
            }
        }
#if defined(pfalcON) && (! defined(iGPU))
	_unLockpfalcON(A);
	_unLockpfalcON(B);
    
#endif
#ifdef traceKernels
    tempsPP += my_gettimeofday() - temp;
#endif
}

#ifdef ispcpfalcON
//=============================================================================
// sfalcON Kernels
//=============================================================================

//=============================================================================
// ISPC Cell-Cell kernels
//=============================================================================

void GravKernAll::cellCellIpfalcONScal(leaf_iter const&A0, unsigned NA,
                                       leaf_iter const&B0, unsigned NB) const
{
    leaf_iter A = A0;
    leaf_iter B = B0;
    for(unsigned i = 0 ; i < NA; i++)
    {
        A = A0 + i;
        real M0 = mass(A);
        real X0i = (A)->POS[0];
        real X1i = (A)->POS[1];
        real X2i = (A)->POS[2] ;
        
        real dRx = X0i;
        dRx -= (B)->POS[0];
        real dRy = X1i;
        dRy -= (B)->POS[1];
        real dRz = X2i;
        dRz -= (B)->POS[2];
        real D0,D1;
        real P0(zero);
        real F0x = 0;
        real F0y = 0;
        real F0z = 0;
        for(unsigned j = 0; j< NB -1; j++)
        {
            B = B0 + j;
            D1 = dRx*dRx + dRy*dRy + dRz*dRz + EQ;
            
            D0 = M0 * mass(B);
            real XX = one/D1;
            D0 *= sqrt(XX);
            D1 = XX * D0;
            dRx *= D1;
            
            dRy *= D1;
            dRz *= D1;
            P0 -= D0;
            F0x -= dRx;
            F0y -= dRy;
            F0z -= dRz;
            
            B->pot() -= D0;
            B->acc()[0] += dRx;
            B->acc()[1] += dRy;
            B->acc()[2] += dRz;
            dRx = X0i;
            dRx -= (B+1)->POS[0];
            dRy = X1i;
            dRy -= (B+1)->POS[1];
            dRz = X2i;
            dRz -= (B+1)->POS[2];
        }
        B = B0 + NB -1;
        D1 = dRx*dRx + dRy*dRy + dRz*dRz + EQ;
        D0 = M0 * mass(B);
        real XX = one/D1;
        D0 *= sqrt(XX);
        D1 = XX * D0;
        dRx *= D1;
        dRy *= D1;
        dRz *= D1;
        P0 -= D0;
        F0x -= dRx;
        F0y -= dRy;
        F0z -= dRz;
        B->pot() -= D0;
        B->acc()[0] += dRx;
        B->acc()[1] += dRy;
        B->acc()[2] += dRz;
        
        A->pot()+=P0;
        A->acc()[0] += F0x;
        A->acc()[1] += F0y;
        A->acc()[2] += F0z;
        
        
    }
}

void GravKernAll::cellCellIpfalcONx1(leaf_iter const&A0, unsigned NA,
                                     leaf_iter const&B0, unsigned NB) const
{
    ispc::cellCellX2((ispc::pfalcONstruct *)(A0),
                     (ispc::pfalcONstruct *)(B0) ,
                     ((int) NA),
                     ((int) NB),
                     EQ,
                     (ispc::pfalcONstructPROP *)globalACPN);
    
    
}
void GravKernAll::cellCellIpfalcONx2(leaf_iter const&A0, unsigned NA,
                                     leaf_iter const&B0, unsigned NB) const
{
    ispc::cellCellX22((ispc::pfalcONstruct *)(A0),
                      (ispc::pfalcONstruct *)(B0) ,
                      ((int) NA),
                      ((int) NB),
                      EQ,
                      (ispc::pfalcONstructPROP *)globalACPN);
    
    
}

void GravKernAll::cellCellIpfalcON(leaf_iter const&A0, unsigned NA,
                                   leaf_iter const&B0, unsigned NB) const
{

    unsigned reste1 = (NA & ~(nbSIMD - 1));
    unsigned reste2 = (NA & (nbSIMD - 1));
    unsigned NAB = reste1 +NB;

    if(reste1 >= nbSIMD*2 && NAB >= thresholdCCx2 && NB >= thresholdCC2x2)
    {
        cellCellIpfalcONx2(A0,reste1,B0,NB);
    }
    else if(NAB >= thresholdCC && NB >= thresholdCC2)
    {
        cellCellIpfalcONx1(A0,reste1,B0,NB);
    }
    
    else
    {
        cellCellIpfalcONScal(A0,reste1,B0,NB);
    }
    
    if(reste2!=0)
    {
        NAB = reste2 +NB;
        if(reste2 == 1)
        {
            leaf_iter A = (A0)+(NA-reste2);
            cellLeafIpfalcON(B0,A,NB);
        }
        else if(NAB >= thresholdCC && NB >= thresholdCC2)
        {
            cellCellIpfalcONx1(A0+(NA-reste2),reste2,B0,NB);
        }
        
        else
        {
            cellCellIpfalcONScal(A0+(NA-reste2),reste2,B0,NB);
        }
    }

    
}

//=============================================================================
// end of ISPC Cell-Cell kernels
//==============================================================================


//=============================================================================
// ISPC Cell-Self kernels
//=============================================================================
void GravKernAll::cellSelfIpfalcONScal(leaf_iter &A0, unsigned N1) const
{
    leaf_iter A = A0;
    leaf_iter B = A0+1;
    for(unsigned i = 0 ; i < N1 - 1; i++)
    {
        A = A0 + i;
        real M0 = mass(A);
        real X0i = (A)->POS[0];
        real X1i = (A)->POS[1];
        real X2i = (A)->POS[2] ;
        
        real dRx = X0i;
        dRx -= (B)->POS[0];
        real dRy = X1i;
        dRy -= (B)->POS[1];
        real dRz = X2i;
        dRz -= (B)->POS[2];
        real D0,D1;
        real P0(zero);
        real F0x = 0;
        real F0y = 0;
        real F0z = 0;
        for(unsigned j = i+1; j< N1 -1; j++)
        {
            B = A0 + j;
            D1 = dRx*dRx + dRy*dRy + dRz*dRz + EQ;
            D0 = M0 * mass(B);
            real XX = one/D1;
            D0 *= sqrt(XX);
            D1 = XX * D0;
            dRx *= D1;
            dRy *= D1;
            dRz *= D1;
            P0 -= D0;
            
            F0x -= dRx;
            F0y -= dRy;
            F0z -= dRz;
            B->pot() -= D0;
            B->acc()[0] += dRx;
            B->acc()[1] += dRy;
            B->acc()[2] += dRz;
            
            dRx = X0i;
            dRx -= (B+1)->POS[0];
            dRy = X1i;
            dRy -= (B+1)->POS[1];
            dRz = X2i;
            dRz -= (B+1)->POS[2];
        }
        B = A0 + N1 -1;
        D1 = dRx*dRx + dRy*dRy + dRz*dRz + EQ;
        D0 = M0 * mass(B);
        real XX = one/D1;
        D0 *= sqrt(XX);
        D1 = XX * D0;
        dRx *= D1;
        dRy *= D1;
        dRz *= D1;
        P0 -= D0;
        F0x -= dRx;
        F0y -= dRy;
        F0z -= dRz;
        B->pot() -= D0;
        B->acc()[0] += dRx;
        B->acc()[1] += dRy;
        B->acc()[2] += dRz;
        
        A->pot()+=P0;
        A->acc()[0] += F0x;
        A->acc()[1] += F0y;
        A->acc()[2] += F0z;
    }
}

void GravKernAll::cellSelfIpfalcONx1(leaf_iter &A0, unsigned N1) const
{
    {
        ispc::cellSelfX2((ispc::pfalcONstruct *)A0,N1, EQ,
                         (ispc
                          ::pfalcONstructPROP *)globalACPN);
        
    }
}
void GravKernAll::cellSelfIpfalcONx2(leaf_iter &A0, unsigned N1) const
{
    {
        ispc::cellSelfX22((ispc::pfalcONstruct *)A0,N1, EQ,
                          (ispc
                           ::pfalcONstructPROP *)globalACPN);
        
    }
}
void GravKernAll::cellSelfIpfalcON(leaf_iter &A0, unsigned N1) const
{

    if(N1 ==1 )
    {
        
    }
    else
    {
        unsigned reste1 = (N1 & ~(nbSIMD - 1));
        unsigned reste2 = (N1 & (nbSIMD - 1));
        
        if(reste1 >= thresholdCSx2)
        {
            cellSelfIpfalcONx2(A0, reste1);
        }
        else if(reste1 >= thresholdCS)
        {
            cellSelfIpfalcONx1(A0, reste1);
        }
        else
        {
            if(reste1 > 1)
                cellSelfIpfalcONScal(A0,reste1);
        }
        if(reste2 > 1 )
        {
            leaf_iter A = A0+(N1-reste2);
            if(reste2 >= thresholdCS)
            {
                
                cellSelfIpfalcONx1(A,reste2);
            }
            else
            {
                cellSelfIpfalcONScal(A,reste2);
            }
            cellCellIpfalcON(A0,reste1,A,reste2);
        }
        
    }
    
}

//=============================================================================
// end of ISPC Cell-Self kernels
//=============================================================================

//=============================================================================
// ISPC Cell-Leaf kernels
//=============================================================================

void GravKernAll::cellLeafIpfalcONScal(leaf_iter const&A, leaf_iter const&B, unsigned N1) const
{
    leaf_iter       A0;
    for(unsigned i = 0 ; i < N1; i++)
    {
        A0 = A + i;
        real M0 = mass(A0);
        real X0i = (A0)->POS[0];
        real X1i = (A0)->POS[1];
        real X2i = (A0)->POS[2] ;
        
        real dRx = X0i;
        dRx -= (B)->POS[0];
        
        real dRy = X1i;
        dRy -= (B)->POS[1];
        
        real dRz = X2i;
        dRz -= (B)->POS[2];
        
        real D0,D1;
        real P0(zero);
        real F0x = 0;
        real F0y = 0;
        real F0z = 0;
        D1 = dRx*dRx + dRy*dRy + dRz*dRz + EQ;
        D0 = M0 * mass(B);
        real XX = one/D1;
        D0 *= sqrt(XX);
        D1 = XX * D0;
        dRx *= D1;
        dRy *= D1;
        dRz *= D1;
        P0 -= D0;
        F0x -= dRx;
        F0y -= dRy;
        F0z -= dRz;
        B->pot() -= D0;
        B->acc()[0] += dRx;
        B->acc()[1] += dRy;
        B->acc()[2] += dRz;
        
        (A0)->pot()+=P0;
        (A0)->acc()[0] += F0x;
        (A0)->acc()[1] += F0y;
        (A0)->acc()[2] += F0z;
        
    }
}
void GravKernAll::cellLeafIpfalcONx1(leaf_iter const&A, leaf_iter const&B, unsigned N1) const
{
    ispc::cellLeaf((ispc::pfalcONstruct *)A,(ispc::pfalcONstruct *)B,N1, EQ,
                   (ispc::pfalcONstructPROP *)globalACPN);
    
}
void GravKernAll::cellLeafIpfalcONx2(leaf_iter const&A, leaf_iter const&B, unsigned N1) const
{
    ispc::cellLeaf2((ispc::pfalcONstruct *)A,(ispc::pfalcONstruct *)B,N1, EQ,
                    (ispc::pfalcONstructPROP *)globalACPN);
    
}
void GravKernAll::cellLeafIpfalcON(leaf_iter const&A, leaf_iter const&B, unsigned N1) const
{
    cellLeafIpfalcONScal(A,B,N1);
    
}

//=============================================================================
//end of ISPC Cell-Leaf kernels
//=============================================================================

//=============================================================================
// Benchmark for P2P kernels
//=============================================================================
void GravKernAll::saveData(mesureErreur* tab, grav::cell_iter const&CA, int l) const
{
    leaf_iter A = CA.begin_leafs();
    leaf_iter       A0;
    for(int i = 0 ; i < l; i++)
    {
        A0 = A + i;
        
        tab[i].aw = (A0)->pot();
        tab[i].ax = (A0)->acc()[0];
        tab[i].ay = (A0)->acc()[1];
        tab[i].az = (A0)->acc()[2];
    }
}

void GravKernAll::loadData(mesureErreur* tab, grav::cell_iter const&CA, int l) const
{
    leaf_iter A = CA.begin_leafs();
    leaf_iter       A0;
    for(int i = 0 ; i < l; i++)
    {
        A0 = A + i;
        
        (A0)->pot() = tab[i].aw;
        (A0)->acc()[0] = tab[i].ax;
        (A0)->acc()[1] = tab[i].ay;
        (A0)->acc()[2] = tab[i].az;
    }
}

bool GravKernAll::diffError(mesureErreur* a, mesureErreur* b, int l)
{
    bool test = false;
    for(int i = 0 ; i < l; i++)
    {
        if(
           (a[i].aw - b[i].aw)*1000000 > 0.01f ||
           (a[i].ax - b[i].ax)*1000000 > 0.01f ||
           (a[i].ay - b[i].ay)*1000000 > 0.01f ||
           (a[i].az - b[i].az)*1000000 > 0.01f )
        {
            printf("\t%d w :  %f; x : %f; y : %f; z : %f \n",
                   i,
                   (a[i].aw - b[i].aw)*1000000,
                   (a[i].ax - b[i].ax)*1000000,
                   (a[i].ay - b[i].ay)*1000000,
                   (a[i].az - b[i].az)*1000000 );
            test = true;
        }
        if(
           (a[i].aw - b[i].aw)*1000000 < -0.01f ||
           (a[i].ax - b[i].ax)*1000000 < - 0.01f ||
           (a[i].ay - b[i].ay)*1000000 < - 0.01f ||
           (a[i].az - b[i].az)*1000000 < - 0.01f )
        {
            printf("\t%d w :  %f; x : %f; y : %f; z : %f \n",
                   i,
                   (a[i].aw - b[i].aw)*1000000,
                   (a[i].ax - b[i].ax)*1000000,
                   (a[i].ay - b[i].ay)*1000000,
                   (a[i].az - b[i].az)*1000000 );
            test = true;
        }
    }
    return test;
}

void GravKernAll::benchmark(grav::cell_iter const&CA,
                            grav::cell_iter const&CB) const
{
  //    const unsigned  NA=number(CA), NB=number(CB);
  //    leaf_iter A0 = CA.begin_leafs(),B0 = CB.begin_leafs();
    leaf_iter A;
    if(!mesureOK)
    {
        mesureOK = true;
        
#ifdef thresholdSIMD
        for(int i = 2; i <= 4*nbSIMD; i++)
        {
            int l = i;
#elif thresholdSIMD256
            for(int i = 2; i <= 256; i++)
            {
                int l = i;
#else
                for(int i = 1; i <= 11; i++)
                {
                    int l = pow(2,i);
#endif
                    mesureErreur* ori = new mesureErreur[l];
                    mesureErreur* falcon = new mesureErreur[l];
                    mesureErreur* falconi = new mesureErreur[l];
                    mesureErreur* falconi1 = new mesureErreur[l];
                    mesureErreur* falconi2 = new mesureErreur[l];
                    mesureErreur* falconif = new mesureErreur[l];
                    
                    
                    saveData(ori,CA,l);
                    
                    int iter = 500;
                    double temp;
                    temp = my_gettimeofday();
                    
                    for(int k = 0; k <iter; k++)
                    {
                        A = CA.begin_leafs();
                        cellSelfpfalcON(A, l);
                        
                    }
                    
                    double tempsNOVECTcs = my_gettimeofday() - temp;
                    saveData(falcon, CA, l);
                    loadData(ori, CA, l);
                    temp = my_gettimeofday();
                    for(int k = 0; k <iter; k++)
                    {
                        A = CA.begin_leafs();
                        cellSelfIpfalcONScal(A, l);
                        
                    }
                    
                    double tempscs2 = my_gettimeofday() - temp;
                    saveData(falconi, CA, l);
                    
                    loadData(ori, CA, l);
                    temp = my_gettimeofday();
                    for(int k = 0; k <iter; k++)
                    {
                        A = CA.begin_leafs();
                        cellSelfIpfalcONx1(A, l);
                        
                    }
                    
                    double tempscs3 = my_gettimeofday() - temp;
                    saveData(falconi1, CA, l);
                    
                    loadData(ori, CA, l);
                    temp = my_gettimeofday();
                    for(int k = 0; k <iter; k++)
                    {
                        A = CA.begin_leafs();
                        cellSelfIpfalcONx2(A, l);
                        
                    }
                    
                    double tempscs4 = my_gettimeofday() - temp;
                    saveData(falconi2, CA, l);
                    
                    
                    loadData(ori, CA, l);
                    temp = my_gettimeofday();
                    for(int k = 0; k <iter; k++)
                    {
                        
                        A = CA.begin_leafs();
                        cellSelfIpfalcON(A, l);
                        
                    }
                    
                    
                    double tempscs5 = my_gettimeofday() - temp;
                    saveData(falconif, CA, l);
                    loadData(ori, CA, l);
                    printf("%d ; %lf  ; %lf ; %lf ; %lf; %lf\n", l , tempsNOVECTcs,tempscs2,tempscs3,tempscs4,tempscs5);
#ifdef evalError
                    if(diffError(falcon, falconi, l))
                        printf("\t CS Diff Err falcon, falconi\n");
                    
                    if(diffError(falcon, falconi1, l))
                        printf("\t CS Diff Err falcon, falconi1\n");
                    
                    if(diffError(falcon, falconi2, l))
                        printf("\t CS Diff Err falcon, falconi2\n");
                    
                    if(diffError(falcon, falconif, l))
                        printf("\t CS Diff Err falcon, falconif\n");
                    
#endif
                    delete [] ori;
                    delete [] falcon;
                    delete [] falconi;
                    delete [] falconi1;
                    delete [] falconi2;
                    delete [] falconif;
                }
#ifdef thresholdSIMD
                for(int i = 1; i <= 4*nbSIMD; i++)
                {
                    int l = i;
#elif thresholdSIMD256
                    for(int i = 1; i <= 256; i++)
                    {
                        int l = i;
#else
                        for(int i = 0; i <= 11; i++)
                        {
                            int l = pow(2,i);
#endif
                            int iter = 500;
                            double temp;
                            
                            
#ifdef thresholdSIMD
                            for(int j = 1; j <= i; j++)
                                
#else
                                int j = l;
#endif
                            {
                                mesureErreur* ori = new mesureErreur[l];
                                mesureErreur* falcon = new mesureErreur[l];
                                mesureErreur* falconi = new mesureErreur[l];
                                mesureErreur* falconi1 = new mesureErreur[l];
                                mesureErreur* falconi2 = new mesureErreur[l];
                                
                                mesureErreur* falconif = new mesureErreur[l];
                                saveData(ori,CA,l);
                                mesureErreur* _ori = new mesureErreur[l];
                                mesureErreur* _falcon = new mesureErreur[l];
                                mesureErreur* _falconi = new mesureErreur[l];
                                mesureErreur* _falconi1 = new mesureErreur[l];
                                mesureErreur* _falconi2 = new mesureErreur[l];
                                
                                mesureErreur* _falconif = new mesureErreur[l];
                                saveData(_ori,CB,j);
                                
                                if (j ==1 || l==1)
                                {
                                    int t = j;
                                    if(l > t)
                                        t = l;
                                    temp = my_gettimeofday();
                                    loadData(ori, CA, 1);
                                    loadData(_ori, CB, t);
                                    for(int k = 0; k <iter; k++)
                                    {
                                        
                                        cellLeafpfalcON(CA.begin_leafs(),CB.begin_leafs(),t);
                                        
                                    }
                                    double tempsNOVECTcc = my_gettimeofday() - temp;
                                    saveData(falcon, CA, 1);
                                    saveData(_falcon, CB, t);
                                    loadData(ori, CA, 1);
                                    loadData(_ori, CB, t);
                                    
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        cellLeafpfalcON(CA.begin_leafs(),CB.begin_leafs(),t);
                                    }
                                    double tempscc2 = my_gettimeofday() - temp;
                                    saveData(falconi, CA, 1);
                                    saveData(_falconi, CB, t);
                                    loadData(ori, CA, 1);
                                    loadData(_ori, CB, t);
                                    
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        
                                        cellLeafIpfalcONx1(CA.begin_leafs(),CB.begin_leafs(),t);
                                        
                                        
                                    }
                                    double tempscc3 = my_gettimeofday() - temp;
                                    saveData(falconi1, CA, 1);
                                    saveData(_falconi1, CB, t);
                                    loadData(ori, CA, 1);
                                    loadData(_ori, CB, t);
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        
                                        cellLeafIpfalcONx2(CA.begin_leafs(),CB.begin_leafs(),t);
                                        
                                        
                                    }
                                    double tempscc4 = my_gettimeofday() - temp;
                                    saveData(falconi2, CA, 1);
                                    saveData(_falconi2, CB, t);
                                    
                                    
                                    
                                    loadData(ori, CA, 1);
                                    loadData(_ori, CB, t);
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        
                                        cellLeafIpfalcON(CA.begin_leafs(),CB.begin_leafs(),t);
                                        
                                        
                                    }
                                    double tempscc5 = my_gettimeofday() - temp;
                                    saveData(falconif, CA, 1);
                                    saveData(_falconif, CB, t);
                                    loadData(ori, CA, 1);
                                    loadData(_ori, CB, t);
                                    printf("%d ; %d ; %lf ;  %lf  ; %lf ; %lf ; %lf\n", l ,j, tempsNOVECTcc ,tempscc2,tempscc3,tempscc4, tempscc5);
#ifdef evalError
                                    if(diffError(falcon, falconi, 1))
                                        printf("\t CL Diff Err falcon, falconi\n");
                                    
                                    if(diffError(falcon, falconi1, 1))
                                        printf("\t CL Diff Err falcon, falconi1\n");
                                    
                                    if(diffError(falcon, falconi2, 1))
                                        printf("\t CL Diff Err falcon, falconi2\n");
                                    
                                    if(diffError(falcon, falconif, 1))
                                        printf("\t CL Diff Err falcon, falconif\n");
                                    
                                    
                                    if(diffError(_falcon, _falconi, j))
                                        printf("\t CL Diff Err falcon, _falconi\n");
                                    
                                    if(diffError(_falcon, _falconi1, j))
                                        printf("\t CL Diff Err falcon, _falconi1\n");
                                    
                                    if(diffError(_falcon, _falconi2, j))
                                        printf("\t CL Diff Err falcon, _falconi2\n");
                                    
                                    if(diffError(_falcon, _falconif, j))
                                        printf("\t CL Diff Err falcon, _falconif\n");
                                    
#endif
                                }
                                else
                                {
                                    loadData(ori, CA, l);
                                    loadData(_ori, CB, j);
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        cellCellpfalcON( CA.begin_leafs(),l,CB.begin_leafs(),j);
                                        
                                    }
                                    double tempsNOVECTcc = my_gettimeofday() - temp;
                                    saveData(falcon, CA, l);
                                    saveData(_falcon, CB, j);
                                    loadData(ori, CA, l);
                                    loadData(_ori, CB, j);
                                    
                                    
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        
                                        cellCellIpfalcONScal(CA.begin_leafs(),l,CB.begin_leafs(),j);
                                    }
                                    double tempscc2 = my_gettimeofday() - temp;
                                    saveData(falconi, CA, l);
                                    saveData(_falconi, CB, j);
                                    
                                    loadData(ori, CA, l);
                                    loadData(_ori, CB, j);
                                    
                                    
                                    
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        
                                        cellCellIpfalcONx1(CA.begin_leafs(),l,CB.begin_leafs(),j);
                                        
                                        
                                    }
                                    double tempscc3 = my_gettimeofday() - temp;
                                    saveData(falconi1, CA, l);
                                    saveData(_falconi1, CB, j);
                                    
                                    loadData(ori, CA, l);
                                    loadData(_ori, CB, j);
                                    
                                    
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        
                                        cellCellIpfalcONx2(CA.begin_leafs(),l,CB.begin_leafs(),j);
                                        
                                        
                                    }
                                    double tempscc4 = my_gettimeofday() - temp;
                                    
                                    saveData(falconi2, CA, l);
                                    saveData(_falconi2, CB, j);
                                    
                                    loadData(ori, CA, l);
                                    loadData(_ori, CB, j);
                                    
                                    
                                    
                                    
                                    temp = my_gettimeofday();
                                    for(int k = 0; k <iter; k++)
                                    {
                                        
                                        cellCellIpfalcON(CA.begin_leafs(),l,CB.begin_leafs(),j);
                                        
                                        
                                    }
                                    double tempscc5 = my_gettimeofday() - temp;
                                    
                                    saveData(falconif, CA, l);
                                    saveData(_falconif, CB, j);
                                    
                                    loadData(ori, CA, l);
                                    loadData(_ori, CB, j);
                                    
                                    printf("%d ; %d ; %lf ;  %lf  ; %lf ; %lf ; %lf \n", l ,j, tempsNOVECTcc ,tempscc2,tempscc3,tempscc4,tempscc5);
#ifdef evalError
                                    if(diffError(falcon, falconi, l))
                                        printf("\t CC Diff Err falcon, falconi\n");
                                    
                                    if(diffError(falcon, falconi1, l))
                                        printf("\t CC Diff Err falcon, falconi1\n");
                                    
                                    if(diffError(falcon, falconi2, l))
                                        printf("\t CC Diff Err falcon, falconi2\n");
                                    
                                    if(diffError(falcon, falconif, l))
                                        printf("\t CC Diff Err falcon, falconif\n");
                                    
                                    
                                    if(diffError(_falcon, _falconi, j))
                                        printf("\t CL Diff Err falcon, _falconi\n");
                                    
                                    if(diffError(_falcon, _falconi1, j))
                                        printf("\t CL Diff Err falcon, _falconi1\n");
                                    
                                    if(diffError(_falcon, _falconi2, j))
                                        printf("\t CL Diff Err falcon, _falconi2\n");
                                    
                                    if(diffError(_falcon, _falconif, j))
                                        printf("\t CL Diff Err falcon, _falconif\n");
                                    
#endif
                                    delete [] ori;
                                    delete [] falcon;
                                    delete [] falconi;
                                    delete [] falconi1;
                                    delete [] falconi2;
                                    delete [] falconif;
                                    delete [] _ori;
                                    delete [] _falcon;
                                    delete [] _falconi;
                                    delete [] _falconi1;
                                    delete [] _falconi2;
                                    delete [] _falconif;
                                    
                                }
                                
                            }
                        }
                    }
                }
//=============================================================================
// end of Benchmark for P2P kernels
//=============================================================================
#endif
                void GravKernAll::cellSelfpfalcON(leaf_iter &A, unsigned N1) const
                {
		  for(unsigned Nk=N1; Nk; --Nk,++A) Direct<0>::many_YA( KERN,A,A+1,A+1+Nk,EQ,HQ,QQ);
                }
                void GravKernAll::cellCellpfalcON(leaf_iter const&A0, unsigned NA,
                                                  leaf_iter const&B0, unsigned NB) const
                {
		  many_AA( A0,NA,B0, NB);
                }
                void GravKernAll::cellLeafpfalcON(leaf_iter const&A, leaf_iter const&B, unsigned N1) const
                {
                    Direct<0>::many_YA(KERN,B,A,A+N1,EQ,HQ,QQ);
                }
                
                
                //---------- End pfalcON -----------
                void GravKernAll::direct(grav::cell_iter const&CA,
                                         grav::cell_iter const&CB) const
                {
#ifdef traceKernels
                    if(number(CA) > number(CB))
                        typePP[number(CA)][number(CB)]++;
                    else
                        typePP[number(CB)][number(CA)]++;
                    nbPP++;
                    double temp = my_gettimeofday();
#endif
#define ARGS_A CA.begin_leafs(),NA,CB.begin_leafs(),NB
#define ARGS_B CB.begin_leafs(),NB,CA.begin_leafs(),NA
                    
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(CA);
                    _LockpfalcON(CB);
#endif
                    
                    
#ifdef mesure
                    benchmark(CA,CB);
#endif
                    
                    const unsigned  NA=number(CA), NB=number(CB);
#ifdef ispcpfalcON
                    if(NA > NB)
                    {
                        cellCellIpfalcON(ARGS_A);
                    }
                    else
                    {
                        cellCellIpfalcON(ARGS_B);
                    }

#else

                    if(NA%4 > NB%4)
                    {
                        cellCellpfalcON( ARGS_A);
                    }
                    else
                    {
                        cellCellpfalcON(ARGS_B);
                    }
#endif
                    
#if defined(pfalcON) && (! defined(iGPU))
                    _unLockpfalcON(CA);
                    _unLockpfalcON(CB);
#endif
#ifdef traceKernels
                    tempsPP += my_gettimeofday() - temp;
#endif
                }
#undef ARGS_A
#undef ARGS_B
                
                void GravKernAll::direct(cell_iter const&A, leaf_iter const&B) const
                {
#ifdef traceKernels
                    typePP[number(A)][1]++;
                    nbPP++;
                    double temp = my_gettimeofday();
#endif
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(A);
                    _LockpfalcON(B);
#endif
                    
                    if(INDI_SOFT)
                    {
                        Direct<1>::many_YA(ARGS);
                    }
                    else
                    {
                        const unsigned  N1 = number(A)-1;
                        
#ifdef ispcpfalcON
                        {
                        cellLeafIpfalcON(A.begin_leafs(),B,N1);
                        }
#else
                        cellLeafpfalcON(A.begin_leafs(),B, N1);
                        //Direct<0>::many_YA(ARGS);
#endif
                    }
                    
#if defined(pfalcON) && (! defined(iGPU))
                    _unLockpfalcON(A);
                    _unLockpfalcON(B);
#endif
#ifdef traceKernels
                    tempsPP += my_gettimeofday() - temp;
#endif
                }
#undef ARGS
                //------------------------------------------------------------------------------
#define ARGS KERN,A,A+1,A+1+Nk,EQ,HQ,QQ
                void GravKern::direct(cell_iter const&C) const
                {
#ifdef traceKernels
                    nbPP++;
                    double temp = my_gettimeofday();
                    
#endif
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(C);
#endif
                    
                    
                    const unsigned  N1 = number(C)-1;
                    leaf_iter       A  = C.begin_leafs();
                    if(INDI_SOFT)
                        if(al_active(C))
                            for(unsigned Nk=N1; Nk; --Nk,++A) Direct<1>::many_YA(ARGS);
                        else
                            for(unsigned Nk=N1; Nk; --Nk,++A)
                                if(is_active(A))                Direct<1>::many_YS(ARGS);
                                else                            Direct<1>::many_NS(ARGS);
                                else
                                {
                                    if(al_active(C))
                                        for(unsigned Nk=N1; Nk; --Nk,++A) Direct<0>::many_YA(ARGS);
                                    else
                                        for(unsigned Nk=N1; Nk; --Nk,++A)
                                            if(is_active(A))                Direct<0>::many_YS(ARGS);
                                            else                            Direct<0>::many_NS(ARGS);
                                }
#if defined(pfalcON) && (! defined(iGPU))
                    _unLockpfalcON(C);
#endif
#ifdef traceKernels
                    tempsPP += my_gettimeofday() - temp;
#endif
                }
                //------------------------------------------------------------------------------
                
                void GravKernAll::direct(cell_iter const&C) const
                {
#ifdef traceKernels
                    nbPP++;
                    double temp = my_gettimeofday();
                    typePP[number(C)][0]++;
#endif
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(C);
#endif
                    
                    const unsigned  N1 = number(C)-1;
                    leaf_iter       A  = C.begin_leafs();
                    
                    if(INDI_SOFT)
                    {
                        for(unsigned Nk=N1; Nk; --Nk,++A) Direct<1>::many_YA(ARGS);
                    }
                    else
                    {
#ifdef ispcpfalcON
                        cellSelfIpfalcON(  A, N1);
#else
                        cellSelfpfalcON( A, N1);
#endif
                        
                    }
#if defined(pfalcON) && (! defined(iGPU))
                    _unLockpfalcON(C);
                    
#endif
#ifdef traceKernels
                    tempsPP += my_gettimeofday() - temp;
#endif
                }
#undef ARGS
                //==============================================================================
                // we now define non-inline functions for the computation of many direct
                // interactions between NA and NB leafs
                // there are 8 cases, depending on whether all, some, or none of either A or
                // B are active (case none,none is trivial).
                //
                // these functions are called by GravKern::direct(cell,cell), which is inline
                // in kernel.h, or by GravKern::flush_scc() below.
                //==============================================================================
                void GravKernBase::many_AA(leaf_iter const&A0, unsigned NA,
                                           leaf_iter const&B0, unsigned NB) const
                {
                    const leaf_iter AN=A0+NA, BN=B0+NB;
                    if(INDI_SOFT)
                    {
                        for(leaf_iter A=A0; A!=AN; ++A)
                        {
                            Direct<1>::many_YA(KERN,A,B0,BN,EQ,HQ,QQ);
                        }
                    }
                    else
                    {
                        for(leaf_iter A=A0; A!=AN; ++A)
                        {
                            Direct<0>::many_YA(KERN,A,B0,BN,EQ,HQ,QQ);
                        }
                    }
                }
                //------------------------------------------------------------------------------
                void GravKernBase::many_AS(leaf_iter const&A0, unsigned NA,
                                           leaf_iter const&B0, unsigned NB) const
                {
                    const    leaf_iter AN=A0+NA, BN=B0+NB;
                    if(INDI_SOFT)
                        for(leaf_iter A=A0; A!=AN; ++A) Direct<1>::many_YS(KERN,A,B0,BN,EQ,HQ,QQ);
                    else
                        for(leaf_iter A=A0; A!=AN; ++A) Direct<0>::many_YS(KERN,A,B0,BN,EQ,HQ,QQ);
                }
                //------------------------------------------------------------------------------
                void GravKernBase::many_AN(leaf_iter const&A0, unsigned NA,
                                           leaf_iter const&B0, unsigned NB) const
                {
                    const leaf_iter AN=A0+NA, BN=B0+NB;
                    if(INDI_SOFT)
                        for(leaf_iter A=A0; A!=AN; ++A) Direct<1>::many_YN(KERN,A,B0,BN,EQ,HQ,QQ);
                    else
                        for(leaf_iter A=A0; A!=AN; ++A) Direct<0>::many_YN(KERN,A,B0,BN,EQ,HQ,QQ);
                }
                //------------------------------------------------------------------------------
                void GravKernBase::many_SA(leaf_iter const&A0, unsigned NA,
                                           leaf_iter const&B0, unsigned NB) const
                {
                    const leaf_iter AN=A0+NA, BN=B0+NB;
                    if(INDI_SOFT) {
                        for(leaf_iter A=A0; A!=AN; ++A)
                            if(is_active(A)) Direct<1>::many_YA(KERN,A,B0,BN,EQ,HQ,QQ);
                            else             Direct<1>::many_NA(KERN,A,B0,BN,EQ,HQ,QQ);
                    } else
                        for(leaf_iter A=A0; A!=AN; ++A)
                            if(is_active(A)) Direct<0>::many_YA(KERN,A,B0,BN,EQ,HQ,QQ);
                            else             Direct<0>::many_NA(KERN,A,B0,BN,EQ,HQ,QQ);
                }
                //------------------------------------------------------------------------------
                void GravKernBase::many_SS(leaf_iter const&A0, unsigned NA,
                                           leaf_iter const&B0, unsigned NB) const
                {
                    const leaf_iter AN=A0+NA, BN=B0+NB;
                    if(INDI_SOFT) {
                        for(leaf_iter A=A0; A!=AN; ++A)
                            if(is_active(A)) Direct<1>::many_YS(KERN,A,B0,BN,EQ,HQ,QQ);
                            else             Direct<1>::many_NS(KERN,A,B0,BN,EQ,HQ,QQ);
                    } else
                        for(leaf_iter A=A0; A!=AN; ++A)
                            if(is_active(A)) Direct<0>::many_YS(KERN,A,B0,BN,EQ,HQ,QQ);
                            else             Direct<0>::many_NS(KERN,A,B0,BN,EQ,HQ,QQ);
                }
                //------------------------------------------------------------------------------
                void GravKernBase::many_SN(leaf_iter const&A0, unsigned NA,
                                           leaf_iter const&B0, unsigned NB) const
                {
                    const leaf_iter AN=A0+NA, BN=B0+NB;
                    if(INDI_SOFT) {
                        for(leaf_iter A=A0; A!=AN; ++A)
                            if(is_active(A)) Direct<1>::many_YN(KERN,A,B0,BN,EQ,HQ,QQ);
                    } else
                        for(leaf_iter A=A0; A!=AN; ++A)
                            if(is_active(A)) Direct<0>::many_YN(KERN,A,B0,BN,EQ,HQ,QQ);
                }
                //------------------------------------------------------------------------------
                void GravKernBase::many_NA(leaf_iter const&A0, unsigned NA,
                                           leaf_iter const&B0, unsigned NB) const
                {
                    const leaf_iter AN=A0+NA, BN=B0+NB;
                    if(INDI_SOFT)
                        for(leaf_iter A=A0; A!=AN; ++A) Direct<1>::many_NA(KERN,A,B0,BN,EQ,HQ,QQ);
                    else
                        for(leaf_iter A=A0; A!=AN; ++A) Direct<0>::many_NA(KERN,A,B0,BN,EQ,HQ,QQ);
                }
                //------------------------------------------------------------------------------
                void GravKernBase::many_NS(leaf_iter const&A0, unsigned NA,
                                           leaf_iter const&B0, unsigned NB) const
                {
                    const leaf_iter AN=A0+NA, BN=B0+NB;
                    if(INDI_SOFT)
                        for(leaf_iter A=A0; A!=AN; ++A) Direct<1>::many_NS(KERN,A,B0,BN,EQ,HQ,QQ);
                    else
                        for(leaf_iter A=A0; A!=AN; ++A) Direct<0>::many_NS(KERN,A,B0,BN,EQ,HQ,QQ);
                }
                ////////////////////////////////////////////////////////////////////////////////
                //                                                                            //
                // 2.2 approximate gravity                                                    //
                //                                                                            //
                ////////////////////////////////////////////////////////////////////////////////
                namespace {
                    using namespace falcON; using namespace falcON::grav;
                    //////////////////////////////////////////////////////////////////////////////
#define LOAD_G					\
real D[ND];					\
real XX=one/(Rq+EQ);				\
D[0] = mass(A)*mass(B);
                    //////////////////////////////////////////////////////////////////////////////
#define LOAD_I					\
real D[ND];					\
EQ   = square(eph(A)+eph(B));			\
real XX=one/(Rq+EQ);				\
D[0] = mass(A)*mass(B);			\
__setE<P>::s(EQ,HQ,QQ);
                    //////////////////////////////////////////////////////////////////////////////
#define ARGS_B					\
cell_iter const&A,				\
leaf_iter const&B,				\
vect           &R,				\
real const     &Rq,				\
real&EQ, real&HQ, real&QQ
                    //////////////////////////////////////////////////////////////////////////////
#define ARGS_C					\
cell_iter const&A,				\
cell_iter const&B,				\
vect           &R,				\
real const     &Rq,				\
real&EQ, real&HQ, real&QQ
                    //////////////////////////////////////////////////////////////////////////////
                    //                                                                          //
                    // class __setE<kern_type>                                                  //
                    // class __block<kern_type, order>                                          //
                    //                                                                          //
                    //////////////////////////////////////////////////////////////////////////////
                    template<kern_type>          struct __setE;
                    template<kern_type,int>      struct __block;
#define sv   static void
                    //////////////////////////////////////////////////////////////////////////////
                    // kern_type = p0
                    //////////////////////////////////////////////////////////////////////////////
                    template<> struct __setE<p0> {
                        sv s(real,real&,real&) {
                        } };
                    //----------------------------------------------------------------------------
                    template<> struct __block<p0,1> : public __setE<p0> {
                        enum { ND=2 };
                        sv b(real&X, real D[ND], real, real, real) {
                            D[0] *= sqrt(X);
                            D[1]  = X * D[0];
                        } };
                    template<int K> struct __block<p0,K> : public __setE<p0> {
                        enum { ND=K+1, F=K+K-1 };
                        sv b(real&X, real D[ND], real EQ, real HQ, real QQ) {
                            __block<p0,K-1>::b(X,D,EQ,HQ,QQ);
                            D[K] = int(F) * X * D[K-1];
                        } };
                    //////////////////////////////////////////////////////////////////////////////
                    // kern_type = p1
                    //////////////////////////////////////////////////////////////////////////////
                    template<> struct __setE<p1> {
                        sv s(real EQ, real&HQ, real&) {
                            HQ = half * EQ;
                        } };
                    //----------------------------------------------------------------------------
                    template<> struct __block<p1,1> : public __setE<p1> {
                        enum { ND=3 };
                        sv b(real&X, real D[ND], real, real HQ, real) {
                            D[0] *= sqrt(X);
                            D[1]  =     X * D[0];
                            D[2]  = 3 * X * D[1];
                            D[0] += HQ*D[1];
                            D[1] += HQ*D[2];
                        } };
                    template<int K> struct __block<p1,K> : public __setE<p1> {
                        enum { ND=K+2, F=K+K+1 };
                        sv b(real&X, real D[ND], real EQ, real HQ, real QQ) {
                            __block<p1,K-1>::b(X,D,EQ,HQ,QQ);
                            D[K+1] = int(F) * X * D[K];
                            D[K]  += HQ  * D[K+1];
                        } };
                    //////////////////////////////////////////////////////////////////////////////
                    // kern_type = p2
                    //////////////////////////////////////////////////////////////////////////////
                    template<> struct __setE<p2> {
                        sv s(real EQ, real&HQ, real&) {
                            HQ = half * EQ;
                        } };
                    //----------------------------------------------------------------------------
                    template<> struct __block<p2,1> : public __setE<p2> {
                        enum { ND=4 };
                        sv b(real&X, real D[ND], real, real HQ, real) {
                            D[0] *= sqrt(X);
                            D[1]  =     X * D[0];
                            D[2]  = 3 * X * D[1];
                            D[3]  = 5 * X * D[2];
                            D[0] += HQ*(D[1]+HQ*D[2]);
                            D[1] += HQ*(D[2]+HQ*D[3]);
                        } };
                    template<int K> struct __block<p2,K> : public __setE<p2> {
                        enum { ND=K+3, F=K+K+3 };
                        sv b(real&X, real D[ND], real EQ, real HQ, real QQ) {
                            __block<p2,K-1>::b(X,D,EQ,HQ,QQ);
                            D[K+2] = int(F) * X * D[K+1];
                            D[K]  += HQ*(D[K+1]+HQ*D[K+2]);
                        } };
                    //////////////////////////////////////////////////////////////////////////////
                    // kern_type = p3
                    //////////////////////////////////////////////////////////////////////////////
                    template<> struct __setE<p3> {
                        sv s(real EQ, real&HQ, real&QQ) {
                            HQ = half * EQ;
                            QQ = half * QQ;
                        } };
                    //----------------------------------------------------------------------------
                    template<> struct __block<p3,1> : public __setE<p3> {
                        enum { ND=5 };
                        sv b(real&X, real D[ND], real, real HQ, real QQ) {
                            D[0] *= sqrt(X);
                            D[1]  =     X * D[0];
                            D[2]  = 3 * X * D[1];
                            D[3]  = 5 * X * D[2];
                            D[4]  = 7 * X * D[3];
                            D[0] += HQ*(D[1]+QQ*(D[2]+HQ*D[3]));
                            D[1] += HQ*(D[2]+QQ*(D[3]+HQ*D[4]));
                        } };
                    template<int K> struct __block<p3,K> : public __setE<p3> {
                        enum { ND=K+4, F=K+K+5 };
                        sv b(real&X, real D[ND], real EQ, real HQ, real QQ) {
                            __block<p3,K-1>::b(X,D,EQ,HQ,QQ);
                            D[K+3] = int(F) * X * D[K+2];
                            D[K]  += HQ*(D[K+1]+QQ*(D[K+2]+HQ*D[K+3]));
                        } };
                    //////////////////////////////////////////////////////////////////////////////
                    //                                                                          //
                    // class kernel<kern_type, order, all, indi_soft>                           //
                    //                                                                          //
                    //////////////////////////////////////////////////////////////////////////////
                    template<kern_type,int,bool,bool=0> struct kernel;
                    //////////////////////////////////////////////////////////////////////////////
#ifdef iGPU_M2L
		    template<kern_type P, int K> struct kernel<P,K,0,0> : private __block<P,K> {
                        enum { ND = __block<P,K>::ND };
                        sv a(cell_iter const&A,leaf_iter const&B,vect &R, real const &Rq, real&EQ,\
			     __attribute__((unused)) real&HQ, 
			     __attribute__((unused)) real&QQ) {
			  real D[ND];					
			  real XX=one/(Rq+EQ);				
			  D[0] = mass(A)*mass(B);
			  D[0] *= sqrt(XX);
                          D[1]  = XX * D[0];
			  D[2] = 3 * XX * D[1];
			  D[3] = 5 * XX * D[2];
			  // kernel::b(XX,D,EQ,HQ,QQ); CellLeaf(A,B,D,0,R);
			  // Code extracted from: 
			  // #define CellLeaf(A,B,D,J,R) {		
			  grav::Cset F;                                    /* to hold F^(n)        */ 
			  if(is_active(A)) {                               /* IF A is active       */ 
			  set_dPhi(F,R,D);                         /*   F^(n) = d^nPhi/dR^n*/ 
			  add_C_B2C(A->Coeffs(),F);                      /*   C_A   = ...        */ 
			  if(is_active(B)) {                             /*   IF B is active, too*/ 
			  F.flip_sign_odd();                           /*     flip sign:F^(odd)*/ 
			  add_C_C2B(B->Coeffs(),F,A->poles());         /*     C_B   = ...      */ 
			  }                                              /*   ENDIF              */ 
			  } else if(is_active(B)) {                        /* ELIF B is active     */ 
			    R.negate();                                    /*   flip sign: R       */ 
			    set_dPhi(F,R,D);                         /*   F^(n) = d^nPhi/dR^n*/ 
			    add_C_C2B(B->Coeffs(),F,A->poles());           /*   C_B   = ...        */ 
			  }                                                /* ENDIF                */ 
			  //  } // end of CellLeaf(A,B,D,J,R) 

			}
                        sv a(ARGS_C) { LOAD_G kernel::b(XX,D,EQ,HQ,QQ); CellCell(A,B,D,0,R); }
                    };

#else
                    template<kern_type P, int K> struct kernel<P,K,0,0> : private __block<P,K> {
                        enum { ND = __block<P,K>::ND };
                        sv a(ARGS_B) { LOAD_G kernel::b(XX,D,EQ,HQ,QQ); CellLeaf(A,B,D,0,R); }
                        sv a(ARGS_C) { LOAD_G kernel::b(XX,D,EQ,HQ,QQ); CellCell(A,B,D,0,R); }
                    };
#endif
                    template<kern_type P, int K> struct kernel<P,K,0,1> : private __block<P,K> {
                        enum { ND = __block<P,K>::ND };
                        sv a(ARGS_B) { LOAD_I kernel::b(XX,D,EQ,HQ,QQ); CellLeaf(A,B,D,0,R); }
                        sv a(ARGS_C) { LOAD_I kernel::b(XX,D,EQ,HQ,QQ); CellCell(A,B,D,0,R); }
                    };
#ifdef iGPU_M2L

                    template<kern_type P, int K> struct kernel<P,K,1,0> : private __block<P,K> {
                        enum { ND = __block<P,K>::ND };
                        sv a(ARGS_B) { 
			  LOAD_G kernel::b(XX,D,EQ,HQ,QQ); 
			  CellLeafAll(A,B,D,0,R); 
			}
                        sv a(cell_iter const&A, cell_iter const&B, vect &R, real const &Rq,real&EQ, 
			     __attribute__((unused)) real&HQ, 
			     __attribute__((unused)) real&QQ) { 
			  real D[ND];					
			  real XX=one/(Rq+EQ);				
			  D[0] = mass(A)*mass(B);
			  D[0] *= sqrt(XX);
                          D[1]  = XX * D[0];
			  D[2] = 3 * XX * D[1];
			  D[3] = 5 * XX * D[2];
			  grav::Cset F;                               /* to hold F^(n)        */  
			  set_dPhi(F,R, D);                           /* F^(n) = d^nPhi/dR^n  */ 
			  add_C_C2C(A->Coeffs(),F,B->poles());        /* C_A   = ...          */ 
			  F.flip_sign_odd();                          /* F^(n) = d^nPhi/dR^n  */ 
			  add_C_C2C(B->Coeffs(),F,A->poles());        /* C_B   = ...          */ 
			  //	  LOAD_G kernel::b(XX,D,EQ,HQ,QQ); CellCellAll(A,B,D,0,R); }
			}
		    };
#else
		    template<kern_type P, int K> struct kernel<P,K,1,0> : private __block<P,K> {
                        enum { ND = __block<P,K>::ND };
                        sv a(ARGS_B) { LOAD_G kernel::b(XX,D,EQ,HQ,QQ); CellLeafAll(A,B,D,0,R); }
                        sv a(ARGS_C) { LOAD_G kernel::b(XX,D,EQ,HQ,QQ); CellCellAll(A,B,D,0,R); }
                    };
#endif

#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 1)
                    // gcc 4.1.2 gives crashing code, if this is inlined.
                    template<kern_type P, int K> struct kernel<P,K,1,1> : private __block<P,K> {
                        enum { ND = __block<P,K>::ND };
                        sv a(ARGS_B);
                        sv a(ARGS_C);
                    };
                    template<kern_type P, int K> void kernel<P,K,1,1>::a(ARGS_B)
                    { LOAD_I kernel::b(XX,D,EQ,HQ,QQ); CellLeafAll(A,B,D,0,R); }
                    template<kern_type P, int K> void kernel<P,K,1,1>::a(ARGS_C)
                    { LOAD_I kernel::b(XX,D,EQ,HQ,QQ); CellCellAll(A,B,D,0,R); }
#else
                    template<kern_type P, int K> struct kernel<P,K,1,1> : private __block<P,K> {
                        enum { ND = __block<P,K>::ND };
                        sv a(ARGS_B) { LOAD_I kernel::b(XX,D,EQ,HQ,QQ); CellLeafAll(A,B,D,0,R); }
                        sv a(ARGS_C) { LOAD_I kernel::b(XX,D,EQ,HQ,QQ); CellCellAll(A,B,D,0,R); }
                    };
#endif
#undef sv
                    //////////////////////////////////////////////////////////////////////////////
                }                                                  // END: unnamed namespace
                ////////////////////////////////////////////////////////////////////////////////
#define ARGS A,B,R,Rq,EQ,HQ,QQ
                void GravKern::approx(cell_iter const&A, leaf_iter const&B,
                                      vect           &R, real      Rq) const
                {
#ifdef traceKernels
                    nbM2L++;
                    double temp = my_gettimeofday();
#endif
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(A);
                    _LockpfalcON(B);
#endif
                    
                    if(is_active(A)) give_coeffs(A);
                    if(INDI_SOFT)
                        switch(KERN) {
                            case p1: kernel<p1,ORDER,0,1>::a(ARGS); break;
                            case p2: kernel<p2,ORDER,0,1>::a(ARGS); break;
                            case p3: kernel<p3,ORDER,0,1>::a(ARGS); break;
                            default: kernel<p0,ORDER,0,1>::a(ARGS); break;
                        }
                    else
                        switch(KERN) {
                            case p1: kernel<p1,ORDER,0,0>::a(ARGS); break;
                            case p2: kernel<p2,ORDER,0,0>::a(ARGS); break;
                            case p3: kernel<p3,ORDER,0,0>::a(ARGS); break;
                            default: kernel<p0,ORDER,0,0>::a(ARGS); break;
                        }
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(A);
                    _LockpfalcON(B);
#endif
#ifdef traceKernels
                    tempsM2L += my_gettimeofday() - temp;
#endif
                }
                //------------------------------------------------------------------------------
                void GravKernAll::approx(cell_iter const&A, leaf_iter const&B,
                                         vect           &R, real      Rq) const
                {
#ifdef traceKernels
                    nbM2L++;
                    double temp = my_gettimeofday();
#endif
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(A);
                    _LockpfalcON(B);
#endif
                    
                    give_coeffs(A);
                    if(INDI_SOFT)
                        switch(KERN) {
                            case p1: kernel<p1,ORDER,1,1>::a(ARGS); break;
                            case p2: kernel<p2,ORDER,1,1>::a(ARGS); break;
                            case p3: kernel<p3,ORDER,1,1>::a(ARGS); break;
                            default: kernel<p0,ORDER,1,1>::a(ARGS); break;
                        }
                    else
                        switch(KERN) {
                            case p1: kernel<p1,ORDER,1,0>::a(ARGS); break;
                            case p2: kernel<p2,ORDER,1,0>::a(ARGS); break;
                            case p3: kernel<p3,ORDER,1,0>::a(ARGS); break;
                            default: kernel<p0,ORDER,1,0>::a(ARGS); break;
                        }
#if defined(pfalcON) && (! defined(iGPU))
                    _unLockpfalcON(A);
                    _unLockpfalcON(B);
                    
#endif
#ifdef traceKernels
                    tempsM2L += my_gettimeofday() - temp;
#endif
                }
                //------------------------------------------------------------------------------
                void GravKern::approx(cell_iter const&A, cell_iter const&B,
                                      vect           &R, real      Rq) const
                {
#ifdef traceKernels
                    nbM2L++;
                    double temp = my_gettimeofday();
#endif
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(A);
                    _LockpfalcON(B);
#endif
                    
                    if(is_active(A)) give_coeffs(A);
                    if(is_active(B)) give_coeffs(B);
                    if(INDI_SOFT)
                        switch(KERN) {
                            case p1: kernel<p1,ORDER,0,1>::a(ARGS); break;
                            case p2: kernel<p2,ORDER,0,1>::a(ARGS); break;
                            case p3: kernel<p3,ORDER,0,1>::a(ARGS); break;
                            default: kernel<p0,ORDER,0,1>::a(ARGS); break;
                        }
                    else
                        switch(KERN) {
                            case p1: kernel<p1,ORDER,0,0>::a(ARGS); break;
                            case p2: kernel<p2,ORDER,0,0>::a(ARGS); break;
                            case p3: kernel<p3,ORDER,0,0>::a(ARGS); break;
                            default: kernel<p0,ORDER,0,0>::a(ARGS); break;
                        }
#if defined(pfalcON) && (! defined(iGPU))
                    _unLockpfalcON(A);
                    _unLockpfalcON(B);
#endif
#ifdef traceKernels
                    tempsM2L += my_gettimeofday() - temp;
#endif
                }
                //------------------------------------------------------------------------------
                void GravKernAll::approx(cell_iter const&A, cell_iter const&B,
                                         vect           &R, real      Rq) const
                {
#ifdef traceKernels
                    nbM2L++;
                    double temp = my_gettimeofday();
#endif
#if defined(pfalcON) && (! defined(iGPU))
                    _LockpfalcON(A);
                    _LockpfalcON(B);
#endif
                    
                    give_coeffs(A);
                    give_coeffs(B);
                    if(INDI_SOFT)
                        switch(KERN) {
                            case p1: kernel<p1,ORDER,1,1>::a(ARGS); break;
                            case p2: kernel<p2,ORDER,1,1>::a(ARGS); break;
                            case p3: kernel<p3,ORDER,1,1>::a(ARGS); break;
                            default: kernel<p0,ORDER,1,1>::a(ARGS); break;
                        }
                    else
                        switch(KERN) {
                            case p1: kernel<p1,ORDER,1,0>::a(ARGS); break;
                            case p2: kernel<p2,ORDER,1,0>::a(ARGS); break;
                            case p3: kernel<p3,ORDER,1,0>::a(ARGS); break;
                            default: kernel<p0,ORDER,1,0>::a(ARGS); break;
                        }
#if defined(pfalcON) && (! defined(iGPU))
                    _unLockpfalcON(A);
                    _unLockpfalcON(B);
#endif
#ifdef traceKernels
                    tempsM2L += my_gettimeofday() - temp;
#endif
                }
                ////////////////////////////////////////////////////////////////////////////////
#endif // falcON_SSE_CODE
                
                
