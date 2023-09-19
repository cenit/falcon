// -*- C++ -*-                                                                 |
//-----------------------------------------------------------------------------+
//                                                                             |
/// /file inc/public/kernel.h                                                  |
//                                                                             |
// Copyright (C) 2000-2010  Walter Dehnen                                      |
//               2013       Benoit Lange, Pierre Fortin                        |
//                                                                             |
// This program is free software; you can redistribute it and/or modify        |
// it under the terms of the GNU General Public License as published by        |
// the Free Software Foundation; either version 2 of the License, or (at       |
// your option) any later version.                                             |
//                                                                             |
// This program is distributed in the hope that it will be useful, but         |
// WITHOUT ANY WARRANTY; without even the implied warranty of                  |
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU           |
// General Public License for more details.                                    |
//                                                                             |
// You should have received a copy of the GNU General Public License           |
// along with this program; if not, write to the Free Software                 |
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                   |
//                                                                             |
//-----------------------------------------------------------------------------+
//                                                                             |
// design note: experiments have shown that, strange enough, a code with       |
//              GravKern and GravKernAll being templates with template         |
//              parameter being the kern_type, is somewhat slower (with gcc    |
//              3.2.2: 4% for SSE code). This affects only the approximate     |
//              part of gravity.                                               |
//                                                                             |
//-----------------------------------------------------------------------------+
////////////////////////////////////////////////////////////////////////////////
//          22/11/2013  BL recursive implementation of DTT
//          22/11/2013  BL added support of OpenMP for parallel DTT
//          22/11/2013  BL added support of TBB for parallel DTT
// v p0.1   22/11/2013  BL added ISPC kernels
////////////////////////////////////////////////////////////////////////////////


#ifndef falcON_included_kernel_h
#define falcON_included_kernel_h

#ifndef falcON_included_gravity_h
#  include <public/gravity.h>
#endif

#if defined(falcON_SSE_CODE) && !defined(falcON_included_simd_h)
#  include <proper/simd.h>
#endif
#ifdef pfalcON
#ifndef pfalcON_useTBB
//OpenMP
#include <omp.h>
#endif
#endif
#ifdef ispcpfalcON
#include <public/P2P.h>
using namespace ispc;
extern unsigned nbSIMD;

#endif
extern double tempsPP;
extern double tempsM2L;
extern int nbPP;
extern int nbM2L;
////////////////////////////////////////////////////////////////////////////////
namespace falcON {
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // class falcON::TaylorSeries                                               //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////
    class TaylorSeries {
        vect       X;                                  // expansion center
        grav::Cset C;                                  // expansion coefficients
    public:
        explicit
        TaylorSeries(vect         const&x) : X(x), C(zero) {}
        TaylorSeries(TaylorSeries const&T) : X(T.X) { C = T.C; }
        //--------------------------------------------------------------------------
        friend bool is_empty(TaylorSeries const&T) {
            return T.C == zero; }
        //--------------------------------------------------------------------------
        inline void shift_and_add(const grav::cell*const&);
        inline void extract_grav (grav::leaf*const&) const;
        //--------------------------------------------------------------------------
    };
    //////////////////////////////////////////////////////////////////////////////
    //
    // class falcON::GravKernBase
    //
    //////////////////////////////////////////////////////////////////////////////
    class GravKernBase {
        //--------------------------------------------------------------------------
        // data
        //--------------------------------------------------------------------------
    public:
        struct mesureErreur
        {
            float aw,ax,ay,az;
        };

        double my_gettimeofday() const
        {
            struct timeval tmp_time;
            gettimeofday(&tmp_time, NULL);
            return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
        }
        double my_gettimeofday()
        {
            struct timeval tmp_time;
            gettimeofday(&tmp_time, NULL);
            return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
        }
        
    protected:
        const kern_type       KERN;                    // softening kernel
        const bool            INDI_SOFT;               // use individual eps?
        mutable real          EPS, EQ;                 // eps, eps^2
#ifdef  falcON_SSE_CODE
        mutable fvec4         fHQ, fQQ;                // eps^2/2, eps^2/4
#else
        mutable real          HQ, QQ;                  // eps^2/2, eps^2/4
#endif
        //--------------------------------------------------------------------------
    private:
#ifdef pfalcON
#else
        falcON::pool         *COEFF_POOL;              // pool for TaylorCoeffs
#endif
        mutable int           NC, MAXNC;               // # TaylorCoeffs ever used
        //--------------------------------------------------------------------------
        // blocking of 4 cell-leaf interactions
        //--------------------------------------------------------------------------
#ifdef  falcON_SSE_CODE
    public:
               struct acl_block {
            int              NR;
            grav::cell_pter  A[4];
            grav::leaf_pter  B[4];
            vect             dX[4];
            fvec4            D0,D1;
            void load_g(grav::cell*a, grav::leaf*b, vect&dR, real Rq)
            {
                A [NR] = a;
                B [NR] = b;
                dX[NR] = dR;
                D0[NR] = mass(a)*mass(b);
                D1[NR] = Rq;
                ++NR;
            }
            fvec4            EQ;
            void load_i(grav::cell*a, grav::leaf*b, vect&dR, real Rq)
            {
                A [NR] = a;
                B [NR] = b;
                dX[NR] = dR;
                EQ[NR] = square(eph(a)+eph(b));
                D0[NR] = mass(a)*mass(b);
                D1[NR] = Rq+EQ[NR];
                ++NR;
            }
            acl_block() : NR(0) {}
            bool is_full () const { return NR==4; }
            bool is_empty() const { return NR==0; }
            void reset   ()       { NR=0; }
        };
        mutable acl_block ACL;
        //--------------------------------------------------------------------------
        // blocking of 4 cell-cell interactions
        //--------------------------------------------------------------------------
        struct acc_block {
            int              NR;
            grav::cell_pter  A[4], B[4];
            vect             dX[4];
            fvec4            D0,D1;
            void load_g(grav::cell*a, grav::cell*b, vect&dR, real Rq)
            {
                A [NR] = a;
                B [NR] = b;
                dX[NR] = dR;
                D0[NR] = mass(a)*mass(b);
                D1[NR] = Rq;
                ++NR;
            }
            fvec4            EQ;
            void load_i(grav::cell*a, grav::cell*b, vect&dR, real Rq)
            {
                A [NR] = a;
                B [NR] = b;
                dX[NR] = dR;
                EQ[NR] = square(eph(a)+eph(b));
                D0[NR] = mass(a)*mass(b);
                D1[NR] = Rq+EQ[NR];
                ++NR;
            }
            acc_block() : NR(0) {}
            bool is_full () const { return NR==4; }
            bool is_empty() const { return NR==0; }
            void reset   ()       { NR=0; }
        };
        mutable acc_block ACC;
#endif // falcON_SSE_CODE
        //--------------------------------------------------------------------------
        // protected methods
        //--------------------------------------------------------------------------
    protected:
        GravKernBase(
                     kern_type const&k,                // I: type of kernel
                     real      const&e,                // I: softening length
                     bool      const&s,                // I: type of softening
                     unsigned  const&np) :             // I: initial pool size
        KERN       ( k ),                            // set softening kernel
        INDI_SOFT  ( s ),                            // set softening type
        EPS        ( e ),                            // set softening length
        EQ         ( e*e ),
#ifdef falcON_SSE_CODE
        fHQ        ( half * EQ ),
        fQQ        ( quarter * EQ ),
#else
        HQ         ( half * EQ ),
        QQ         ( quarter * EQ ),
#endif
        NC         ( 0 ),
        MAXNC      ( 0 )
        {
#ifdef pfalcON
#else
            COEFF_POOL =new falcON::pool(max(4u,np), grav::NCOEF*sizeof(real)) ;
#endif
        }
        //--------------------------------------------------------------------------
        ~GravKernBase()
        {
#ifdef pfalcON
#else
            
            falcON_DEL_O(COEFF_POOL);
#endif
        }
        //--------------------------------------------------------------------------
        void give_coeffs(grav::cell_pter const&C) const
        {
#ifdef pfalcON
#else
            if(COEFF_POOL && !hasCoeffs(C))
            {
                register grav::Cset*X = static_cast<grav::Cset*>(COEFF_POOL->alloc());
                X->set_zero();
                C->setCoeffs(X);
                ++NC;
            }
#endif
        }
        //--------------------------------------------------------------------------
        void take_coeffs(grav::cell_pter const&C) const
        {
            if( hasCoeffs(C))
            {
#ifdef pfalcON
	      grav::Cset*X = (grav::Cset*) C->returnCoeffs();
	      delete X;
	      C->resetCoeffs();
#else
                COEFF_POOL->free(C->returnCoeffs());
                C->resetCoeffs();
                update_max(MAXNC,NC--);
#endif
            }
        }
        //--------------------------------------------------------------------------
#define ARGS__ grav::leaf_iter const&, unsigned, 	\
grav::leaf_iter const&, unsigned
        void many_AA(ARGS__) const;
        void many_AS(ARGS__) const;
        void many_AN(ARGS__) const;
        void many_SA(ARGS__) const;
        void many_SS(ARGS__) const;
        void many_SN(ARGS__) const;
        void many_NA(ARGS__) const;
        void many_NS(ARGS__) const;
#undef ARGS__
        //--------------------------------------------------------------------------
#ifdef pfalcON
        //pfalcON
    public:
#endif
        void eval_grav    (grav::cell_iter const&, TaylorSeries const&) const;
        void eval_grav_all(grav::cell_iter const&, TaylorSeries const&) const;
        //--------------------------------------------------------------------------
        // public methods
        //--------------------------------------------------------------------------
        //pfalcON
    public:
#ifdef pfalcON
        void _LockpfalcON(grav::cell_iter const&A) const
        {
#ifdef pfalcON_useTBB
#ifdef useLockInt
            while( A->lock.compare_and_swap(1,0));
#else
            int oldVal,newVal  ;
            do
            {
                oldVal =  A->val & 0x7FFFFFFF;
                newVal =  A->val | 0x80000000;
            }
            while( A->val.compare_and_swap(newVal,oldVal));
#endif
#else
//OpenMP
#ifdef useLockInt
            int temp = 1;
            do 
            {
#pragma omp atomic capture seq_cst
                {
                    temp = A->lock;
                    A->lock |=1;
                }
            } while (temp); 
#else
            int temp = 1;
	    do 
            {
#pragma omp atomic capture seq_cst
                {
                    temp = A->val;
                    A->val |= 0x80000000;
                }
            } while (temp & 0x80000000); 
            
#endif
#endif
        }
        
        static void _unLockpfalcON(grav::cell_iter const&A) 
        {
            
#ifdef pfalcON_useTBB
            
            
#ifdef useLockInt
            A->lock = 0;
#else
            A->val = A->val & 0x7FFFFFFF;
#endif
            
#else
//OpenMP
#ifdef useLockInt
#pragma omp atomic write seq_cst
            A->lock = 0;
            
#else
#pragma omp atomic update seq_cst
            A->val &= 0x7FFFFFFF;
#endif
#endif
        }
        void _LockpfalcON(grav::leaf_iter const&B)const
        {
#ifdef pfalcON_useTBB
#ifdef useLockInt
            while( B->lock.compare_and_swap(1,0));
#else
            int oldVal,newVal  ;
            do
            {
                oldVal =  B->FLAGS.val & 0x7FFFFFFF;
                newVal =  B->FLAGS.val | 0x80000000;
            }
            while( B->FLAGS.val.compare_and_swap(newVal,oldVal));
#endif
#else
//OpenMP
#ifdef useLockInt
            int temp = 1;
            do 
            {
#pragma omp atomic capture seq_cst
                {
                    temp = B->lock;
                    B->lock |=1;
                }
            } while (temp); 
            
#else
            int temp = 1;
	    do 
            {
#pragma omp atomic capture seq_cst
                {
                    temp = B->FLAGS.val;
                    B->FLAGS.val |= 0x80000000;
                }
            } while (temp & 0x80000000); 

#endif
#endif
        }
        static void _unLockpfalcON(grav::leaf_iter const&B) 
        {
#ifdef pfalcON_useTBB
            
#ifdef useLockInt
            B->lock = 0;
#else
            B->FLAGS.val = B->FLAGS.val & 0x7FFFFFFF;
#endif
#else
//OpenMP
#ifdef useLockInt
#pragma omp atomic write seq_cst
            B->lock = 0;
            
#else
#pragma omp atomic update seq_cst
            B->FLAGS.val &= 0x7FFFFFFF;
#endif
#endif
        }
        
#endif
        
        
        int      const&coeffs_used() const { return MAXNC; }
        unsigned       chunks_used() const
        {
#ifdef pfalcON
            return 0u;
#else
            return COEFF_POOL?  COEFF_POOL->N_chunks() : 0u;
#endif
        }
        /// given X^2 and Eps^2, compute negative gravitational potential
        static real Psi(kern_type k, real Xq, real Eq);
        //--------------------------------------------------------------------------
        const real&current_eps  ()     const { return EPS; }
        const real&current_epsq ()     const { return EQ; }
        //--------------------------------------------------------------------------
        void reset_eps(real e)
        {
            EPS = e;
            EQ  = e*e;
#ifdef falcON_SSE_CODE
            fHQ = half * EQ;
            fQQ = quarter * EQ;
#else
            HQ  = half * EQ;
            QQ  = quarter * EQ;
#endif
        }
        //--------------------------------------------------------------------------
    };
    //////////////////////////////////////////////////////////////////////////////
    //
    // class falcON::GravKern
    //
    // This class implements the direct summation and approximate computation
    // of gravity between tree nodes.
    //
    //////////////////////////////////////////////////////////////////////////////
    class GravKern : public GravKernBase
    {
#ifdef falcON_SSE_CODE
        //--------------------------------------------------------------------------
        // blocking of 4 cell-node interactions
        //--------------------------------------------------------------------------
        void flush_acl() const;
        void flush_acc() const;
#endif
        //--------------------------------------------------------------------------
        // main purpose methods
        //--------------------------------------------------------------------------
    protected:
        GravKern(kern_type const&k,                    // I: type of kernel
                 real      const&e,                    // I: softening length
                 bool      const&s,                    // I: type of softening
                 unsigned  const&np) :                 // I: initial pool size
        GravKernBase(k,e,s,np) {}
        //--------------------------------------------------------------------------
        // single leaf-leaf interaction
        //--------------------------------------------------------------------------
        void single(grav::leaf_iter const&, grav::leaf_iter const&) const;
        //--------------------------------------------------------------------------
        // cell-leaf interaction via direct summation
        //--------------------------------------------------------------------------
        void direct(grav::cell_iter const&, grav::leaf_iter const&) const;
        //--------------------------------------------------------------------------
        // cell-cell interaction via direct summation
        //--------------------------------------------------------------------------
        void direct(grav::cell_iter const&, grav::cell_iter const&) const;
        //--------------------------------------------------------------------------
        // cell-self interaction via direct summation
        //--------------------------------------------------------------------------
        void direct(grav::cell_iter const&) const;
        //--------------------------------------------------------------------------
        // cell-leaf interaction via approximation
        //--------------------------------------------------------------------------
        void approx(grav::cell_iter const&, grav::leaf_iter const&,
                    vect&, real) const;
        //--------------------------------------------------------------------------
        // cell-cell interaction via approximation
        //--------------------------------------------------------------------------
        void approx(grav::cell_iter const&, grav::cell_iter const&,
                    vect&, real) const;
        //--------------------------------------------------------------------------
        void flush_buffers() const {
#ifdef falcON_SSE_CODE
            if(!ACL.is_empty()) flush_acl();
            if(!ACC.is_empty()) flush_acc();
#endif
        }
        //--------------------------------------------------------------------------
        // destruction: flush buffers
        //--------------------------------------------------------------------------
        ~GravKern() { flush_buffers(); }
        //--------------------------------------------------------------------------
    };
    //////////////////////////////////////////////////////////////////////////////
    //
    // class falcON::GravKernAll
    //
    // Like GravKern, except that all cells and leafs are assumed active.
    //
    //////////////////////////////////////////////////////////////////////////////
    class GravKernAll : public GravKernBase
    {
#ifdef falcON_SSE_CODE
        //--------------------------------------------------------------------------
        // blocking of 4 cell-node interactions
        //--------------------------------------------------------------------------
        void flush_acl() const;
        void flush_acc() const;
#endif
        
        
        
        //--------------------------------------------------------------------------
        // main purpose methods
        //--------------------------------------------------------------------------
    public:
        GravKernAll(kern_type const&k,                 // I: type of kernel
                    real      const&e,                 // I: softening length
                    bool      const&s,                 // I: type of softening
                    unsigned  const&np) :              // I: initial pool size
        GravKernBase(k,e,s,np) {}
        //--------------------------------------------------------------------------
        // single leaf-leaf interaction
        //--------------------------------------------------------------------------
        void single(grav::leaf_iter const&, grav::leaf_iter const&) const;
        //--------------------------------------------------------------------------
        // cell-leaf interaction via direct summation
        //--------------------------------------------------------------------------
        void direct(grav::cell_iter const&, grav::leaf_iter const&) const;
        //--------------------------------------------------------------------------
        // cell-cell interaction via direct summation
        //--------------------------------------------------------------------------
        void direct(grav::cell_iter const&, grav::cell_iter const&) const;
        //--------------------------------------------------------------------------
        // cell-self interaction via direct summation
        //--------------------------------------------------------------------------
        void direct(grav::cell_iter const&) const;
        //--------------------------------------------------------------------------
        // cell-leaf interaction via approximation
        //--------------------------------------------------------------------------
        void approx(grav::cell_iter const&, grav::leaf_iter const&,
                    vect&, real) const;
        //--------------------------------------------------------------------------
        // cell-cell interaction via approximation
        //--------------------------------------------------------------------------
        void approx(grav::cell_iter const&, grav::cell_iter const&,
                    vect&, real) const;
    public:
#ifdef ispcpfalcON
        //--------------------------------------------------------------------------
        // ISPC Kernels gang width 1
        //--------------------------------------------------------------------------
        void cellCellIpfalcONx1(grav::leaf_iter const&A0, unsigned NA,
                                grav::leaf_iter const&B0, unsigned NB) const;

                void cellSelfIpfalcONx1(grav::leaf_iter &A0, unsigned NA) const;
        
                void cellLeafIpfalcONx1(grav::leaf_iter const&A, grav::leaf_iter const&B, unsigned N1) const;
        //--------------------------------------------------------------------------
        // end of ISPC Kernels gang width 1
        //--------------------------------------------------------------------------

        //--------------------------------------------------------------------------
        // ISPC Kernels gang width 2
        //--------------------------------------------------------------------------
        void cellCellIpfalcONx2(grav::leaf_iter const&A0, unsigned NA,
                                grav::leaf_iter const&B0, unsigned NB) const;

                void cellSelfIpfalcONx2(grav::leaf_iter &A0, unsigned NA) const;
        
                void cellLeafIpfalcONx2(grav::leaf_iter const&A, grav::leaf_iter const&B, unsigned N1) const;
        //--------------------------------------------------------------------------
        // end of ISPC Kernels gang width 2
        //--------------------------------------------------------------------------

        //--------------------------------------------------------------------------
        // scalar kernel
        //--------------------------------------------------------------------------
        void cellCellIpfalcONScal(grav::leaf_iter const&A0, unsigned NA,
                                  grav::leaf_iter const&B0, unsigned NB) const;
        
                void cellSelfIpfalcONScal(grav::leaf_iter &A0, unsigned NA) const;

                void cellLeafIpfalcONScal(grav::leaf_iter const&A, grav::leaf_iter const&B, unsigned N1) const;
        //--------------------------------------------------------------------------
        // end of scalar kernel
        //--------------------------------------------------------------------------

        
        //--------------------------------------------------------------------------
        // Hybrid kernel
        //--------------------------------------------------------------------------
        void cellCellIpfalcON(grav::leaf_iter const&A0, unsigned NA,
                              grav::leaf_iter const&B0, unsigned NB) const;
        
        void cellSelfIpfalcON(grav::leaf_iter &A0, unsigned NA) const;

        void cellLeafIpfalcON(grav::leaf_iter const&A, grav::leaf_iter const&B, unsigned N1) const;
        //--------------------------------------------------------------------------
        // end of Hybrid kernel
        //--------------------------------------------------------------------------
        
        //--------------------------------------------------------------------------
        // Benchmark tools
        //--------------------------------------------------------------------------
        void benchmark(grav::cell_iter const&CA,
                       grav::cell_iter const&CB) const;
        void saveData(mesureErreur*, grav::cell_iter const&CA, int l) const;
        void loadData(mesureErreur*, grav::cell_iter const&CA, int l) const;
        bool diffError(mesureErreur*, mesureErreur*, int l);
        
        //--------------------------------------------------------------------------
        // end of Benchmark tools
        //--------------------------------------------------------------------------
        
        

#endif
        void cellSelfpfalcON(grav::leaf_iter &A0, unsigned NA) const;
        void cellCellpfalcON(grav::leaf_iter const&A0, unsigned NA,
                             grav::leaf_iter const&B0, unsigned NB) const;
        void cellLeafpfalcON(grav::leaf_iter const&A, grav::leaf_iter const&B, unsigned N1) const;
        
        //--------------------------------------------------------------------------
        void flush_buffers() const {
#ifdef falcON_SSE_CODE
            if(!ACL.is_empty()) flush_acl();
            if(!ACC.is_empty()) flush_acc();
#endif
        }
        //--------------------------------------------------------------------------
        // destruction: flush buffers
        //--------------------------------------------------------------------------
        ~GravKernAll() { flush_buffers(); }
        //--------------------------------------------------------------------------
    };
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // inline functions                                                         //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////
    inline void GravKern::direct(grav::cell_iter const&CA,
                                 grav::cell_iter const&CB) const
    {
#ifdef traceKernels
        nbPP++;
        double temp = my_gettimeofday();
#endif
#define ARGS_A CA.begin_leafs(),NA,CB.begin_leafs(),NB
#define ARGS_B CB.begin_leafs(),NB,CA.begin_leafs(),NA
        
#ifdef pfalcON
        _LockpfalcON(CA);
        _LockpfalcON(CB);
#endif
        const unsigned NA=number(CA), NB=number(CB);
        if(NA%4 > NB%4) {
            if       (al_active(CA))
                if     (al_active(CB)) many_AA(ARGS_A);    // active: all  A, all  B
                else if(is_active(CB)) many_AS(ARGS_A);    // active: all  A, some B
                else                   many_AN(ARGS_A);    // active: all  A, no   B
                else if  (is_active(CA))
                    if     (al_active(CB)) many_SA(ARGS_A);    // active: some A, all  B
                    else if(is_active(CB)) many_SS(ARGS_A);    // active: some A, some B
                    else                   many_SN(ARGS_A);    // active: some A, no   B
                    else
                        if     (al_active(CB)) many_NA(ARGS_A);    // active: no   A, all  B
                        else if(is_active(CB)) many_NS(ARGS_A);    // active: no   A, some B
        } else {
            if       (al_active(CB))
                if     (al_active(CA)) many_AA(ARGS_B);    // active: all  B, all  A
                else if(is_active(CA)) many_AS(ARGS_B);    // active: all  B, some A
                else                   many_AN(ARGS_B);    // active: all  B, no   A
                else if(is_active(CB))
                    if     (al_active(CA)) many_SA(ARGS_B);    // active: some B, all  A
                    else if(is_active(CA)) many_SS(ARGS_B);    // active: some B, some A
                    else                   many_SN(ARGS_B);    // active: some B, no   A
                    else
                        if     (al_active(CA)) many_NA(ARGS_B);    // active: no   B, all  A
                        else if(is_active(CA)) many_NS(ARGS_B);    // active: no   B, some A
        }
#ifdef pfalcON
        _unLockpfalcON(CA);
        _unLockpfalcON(CB);
#endif
#ifdef traceKernels
        tempsPP += my_gettimeofday() - temp;
#endif
    }
    //----------------------------------------------------------------------------
#undef ARGS_A
#undef ARGS_B
    //////////////////////////////////////////////////////////////////////////////
#ifdef falcON_SSE_CODE
    //----------------------------------------------------------------------------
    // blocking of approximate interactions: just fill the blocks
    //----------------------------------------------------------------------------
    inline void GravKern::approx(grav::cell_iter const&A,
                                 grav::leaf_iter const&B,
                                 vect                 &dR,
                                 real            const&Rq) const
    {
#ifdef traceKernels
        nbM2L++;
        double temp = my_gettimeofday();
#endif
#ifdef pfalcON
        _LockpfalcON(A);
        _LockpfalcON(B);
#endif
        if(INDI_SOFT) ACL.load_i(A,B,dR,Rq);
        else          ACL.load_g(A,B,dR,Rq+EQ);
        if(ACL.is_full()) flush_acl();
#ifdef pfalcON
        _unLockpfalcON(A);
        _unLockpfalcON(B);
#endif
#ifdef traceKernels
        tempsM2L += my_gettimeofday() - temp;
#endif
    }
    //----------------------------------------------------------------------------
    inline void GravKern::approx(grav::cell_iter const&A,
                                 grav::cell_iter const&B,
                                 vect                 &dR,
                                 real            const&Rq) const
    {
#ifdef traceKernels
        nbM2L++;
        double temp = my_gettimeofday();
#endif
#ifdef pfalcON
        _LockpfalcON(A);
        _LockpfalcON(B);
        
#endif
        if(INDI_SOFT) ACC.load_i(A,B,dR,Rq);
        else          ACC.load_g(A,B,dR,Rq+EQ);
        if(ACC.is_full()) flush_acc();
#ifdef pfalcON
        _unLockpfalcON(A);
        _unLockpfalcON(B);
#endif
#ifdef traceKernels
        tempsM2L += my_gettimeofday() - temp;
#endif
    }
    //----------------------------------------------------------------------------
    inline void GravKernAll::approx(grav::cell_iter const&A,
                                    grav::leaf_iter const&B,
                                    vect                 &dR,
                                    real            const&Rq) const
    {
#ifdef traceKernels
        nbM2L++;
        double temp = my_gettimeofday();
#endif
#ifdef pfalcON
        _LockpfalcON(A);
        _LockpfalcON(B);
#endif
        if(INDI_SOFT) ACL.load_i(A,B,dR,Rq);
        else          ACL.load_g(A,B,dR,Rq+EQ);
        if(ACL.is_full()) flush_acl();
#ifdef pfalcON
        _unLockpfalcON(A);
        _unLockpfalcON(B);
#endif
#ifdef traceKernels
        tempsM2L += my_gettimeofday() - temp;
#endif
    }
    //----------------------------------------------------------------------------
    inline void GravKernAll::approx(grav::cell_iter const&A,
                                    grav::cell_iter const&B,
                                    vect                 &dR,
                                    real            const&Rq) const
    {
#ifdef traceKernels
        nbM2L++;
        double temp = my_gettimeofday();
#endif
#ifdef pfalcON
        _LockpfalcON(A);
        _LockpfalcON(B);
#endif
        if(INDI_SOFT) ACC.load_i(A,B,dR,Rq);
        else          ACC.load_g(A,B,dR,Rq+EQ);
        if(ACC.is_full()) flush_acc();
#ifdef pfalcON
        _unLockpfalcON(A);
        _unLockpfalcON(B);
#endif
#ifdef traceKernels
        tempsM2L += my_gettimeofday() - temp;
#endif
    }
#endif
    //////////////////////////////////////////////////////////////////////////////
}
////////////////////////////////////////////////////////////////////////////////
#endif                                             // falcON_included_kernel.h
