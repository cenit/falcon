// -*- C++ -*-
////////////////////////////////////////////////////////////////////////////////
///
/// \file    src/public/lib/gravity.cc
///
/// \author  Walter Dehnen
///
/// \date    2000-2010
///
/// \brief   implements inc/public/gravity.h
///
////////////////////////////////////////////////////////////////////////////////
//
// Copyright (C) 2000-2010  Walter Dehnen
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
// v p0.1   22/11/2013  BL added ISPC kernel
////////////////////////////////////////////////////////////////////////////////

#include <public/gravity.h>
#include <body.h>
#include <public/interact.h>
#include <public/kernel.h>
#include <numerics.h>
#ifdef pfalcON
#ifdef pfalcON_useTBB
#include "tbb/tbb_stddef.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"
#include "tbb/atomic.h"
#include <thread>         // std::this_thread::sleep_for
#include <chrono> 
#include <unistd.h>
using namespace tbb;
#else
//OpenMP
#include <omp.h>
#endif

falcON::grav::Cset*  COEF_ACPN;
#endif 
#ifdef iGPU
#include <public/cl_manip.h>
#include <vector>
#include <fstream>
#include <string.h> // for memset() 
extern cl_manip *gpu;
extern falcON::OctTree *globalT;
#define LEAF_FLAG (1 << 30)
float EQ; 
#endif

#ifdef profile
//int* n_par_cell;
int n_par_cell[2048];
unsigned  nb_P2P_t = 0;
unsigned  nb_P2M_t = 0;
#endif

#ifdef recursive 
double my_gettimeofday()
{
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}
#endif 

#ifdef ispcpfalcON
falcON::GravEstimator::Leaf::acpn_data *globalACPN;

#endif


#ifdef iGPU

#define KERNEL_DISPLAY_LEVEL 1 
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
double time_P2P = 0;
double time_P2PLeafTgt = 0;
double time_M2L = 0;
unsigned nb_P2P = 0;
unsigned nb_P2PLeafTgt = 0;
unsigned nb_M2L = 0;
#endif 

#define DTT_EVAL_DISPLAY_LEVEL 1 
double t_start_DTT = 0; 

#define START_KERNEL_DISPLAY_LEVEL 3 

#define GFLOPS_DISPLAY_LEVEL 1
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
// For both pair and own computations:  
#define FLOPS_PER_INTERACTION 24 
unsigned long long interaction_nb = 0; 
#define PAIR_INTERACTION_COUNT(N_A, N_B) ((N_A) * (N_B))
// We do not count here the interaction of a particle 
// with itself (even if it is still performed in the Ocl P2P 
// kernel): 
#define OWN_INTERACTION_COUNT(N) ((N) * ((N)-1)) 
#endif 


using namespace falcON; 
void run_kernel_M2L(int thread_num){
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]);
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
  double t1 = my_gettimeofday();   
#ifdef pfalcON
#pragma omp atomic
#endif
  nb_M2L++;   
  DISPLAY(START_KERNEL_DISPLAY_LEVEL, std::cout << "Starting run_kernel_M2L() at: " << t1 - t_start_DTT << std::endl;);
#endif 

  gpu->unmap_buffer(&(p_t->M2L_interBuf_clBuf), p_t->M2L_interBuf, CQ_M2L);  
  gpu->unmap_buffer(&(p_t->M2L_nomutual_indexing_clBuf), p_t->M2L_nomutual_indexing, CQ_M2L);  
  gpu->unmap_buffer(&(p_t->M2L_nomutual_indexing_start_clBuf), p_t->M2L_nomutual_indexing_start, CQ_M2L);  
  
  gpu->run_M2L(p_t->M2L_nomutual_indexing_start_ind, EQ, thread_num);
  
  gpu->map_buffer(&(p_t->M2L_interBuf_clBuf), gpu->buf_size*sizeof(int), (void **) &(p_t->M2L_interBuf), CQ_M2L, CL_FALSE /* non-blocking */); 
  gpu->map_buffer(&(p_t->M2L_nomutual_indexing_clBuf), gpu->M2L_globalMaxBlockNb * sizeof(int), (void **) &(p_t->M2L_nomutual_indexing), CQ_M2L, CL_FALSE /* non-blocking */); 
  gpu->map_buffer(&(p_t->M2L_nomutual_indexing_start_clBuf), 2*gpu->NbCells*sizeof(int), (void **) &(p_t->M2L_nomutual_indexing_start), CQ_M2L, CL_TRUE /* blocking */); 
 
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif 
  time_M2L += my_gettimeofday() - t1; 	
#endif 
}


void run_kernel_P2P(int thread_num){
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]); 
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
  double t1 = my_gettimeofday();   
#ifdef pfalcON
#pragma omp atomic
#endif 
  nb_P2P++;   
  DISPLAY(START_KERNEL_DISPLAY_LEVEL, std::cout << "Starting run_kernel_P2P() at: " << t1 - t_start_DTT << std::endl;);
#endif 

  gpu->unmap_buffer(&(p_t->P2P_interBuf_clBuf), p_t->P2P_interBuf, CQ_P2P);  
  gpu->unmap_buffer(&(p_t->P2P_nomutual_indexing_clBuf), p_t->P2P_nomutual_indexing, CQ_P2P);  
  gpu->unmap_buffer(&(p_t->P2P_nomutual_indexing_start_clBuf), p_t->P2P_nomutual_indexing_start, CQ_P2P);  
  
  gpu->run_P2P(p_t->P2P_nomutual_indexing_start_ind, EQ, thread_num);

  gpu->map_buffer(&(p_t->P2P_interBuf_clBuf), gpu->buf_size*sizeof(int), (void **) &(p_t->P2P_interBuf), CQ_P2P,  CL_FALSE /* non-blocking */); 
  gpu->map_buffer(&(p_t->P2P_nomutual_indexing_clBuf), gpu->P2P_globalMaxBlockNb * sizeof(int), (void **) &(p_t->P2P_nomutual_indexing), CQ_P2P, CL_FALSE /* non-blocking */); 
  gpu->map_buffer(&(p_t->P2P_nomutual_indexing_start_clBuf), 2*gpu->NbCells*sizeof(int), (void **) &(p_t->P2P_nomutual_indexing_start), CQ_P2P,  CL_TRUE /* blocking */); 
  
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif 
  time_P2P += my_gettimeofday() - t1; 	
#endif 
}

void run_kernel_P2PLeafTgt(int thread_num){
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]); 
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
  double t1 = my_gettimeofday();   
#ifdef pfalcON
#pragma omp atomic
#endif 
  nb_P2PLeafTgt++;   
  DISPLAY(START_KERNEL_DISPLAY_LEVEL, std::cout << "Starting run_kernel_P2PLeafTgt() at: " << t1 - t_start_DTT << std::endl;); 
#endif 
  
  gpu->unmap_buffer(&(p_t->P2PLeafTgt_interBuf_clBuf), p_t->P2PLeafTgt_interBuf, CQ_P2PLEAFTGT);  
  gpu->unmap_buffer(&(p_t->P2PLeafTgt_nomutual_indexing_clBuf), p_t->P2PLeafTgt_nomutual_indexing, CQ_P2PLEAFTGT);  
  gpu->unmap_buffer(&(p_t->P2PLeafTgt_nomutual_indexing_start_clBuf), p_t->P2PLeafTgt_nomutual_indexing_start, CQ_P2PLEAFTGT);  

  gpu->run_P2PLeafTgt(p_t->P2PLeafTgt_nomutual_indexing_start_ind, EQ, thread_num);

  gpu->map_buffer(&(p_t->P2PLeafTgt_interBuf_clBuf), gpu->buf_size*sizeof(int), (void **) &(p_t->P2PLeafTgt_interBuf), CQ_P2PLEAFTGT, CL_FALSE /* non-blocking */); 
  gpu->map_buffer(&(p_t->P2PLeafTgt_nomutual_indexing_clBuf), gpu->P2PLeafTgt_globalMaxBlockNb * sizeof(int), (void **) &(p_t->P2PLeafTgt_nomutual_indexing), CQ_P2PLEAFTGT, CL_FALSE /* non-blocking */); 
  gpu->map_buffer(&(p_t->P2PLeafTgt_nomutual_indexing_start_clBuf), 2*gpu->NbLeafs*sizeof(int), (void **) &(p_t->P2PLeafTgt_nomutual_indexing_start), CQ_P2PLEAFTGT, CL_TRUE /* blocking */); 
  
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif 
  time_P2PLeafTgt += my_gettimeofday() - t1; 	
#endif 
}


#define CLEAR_KERNEL_DISPLAY_LEVEL 2
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
double time_clear_M2L_buffers=0; 
double time_clear_P2P_buffers=0; 
double time_clear_P2PLeafTgt_buffers=0; 
#endif 

void clear_M2L_buffers(int thread_num){
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]); 
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
  double t1 = my_gettimeofday(); 
#endif
  p_t->M2L_globalNextBlockNb = 1;
  // Content of M2L_nomutual_indexing_start[] not initialized 
  p_t->M2L_nomutual_indexing_start_ind = 0;
  // Using memset(0) (wasting first block in M2L_interBuf):
  memset(p_t->M2L_currentCellBlockNb, 0, gpu->NbCells*sizeof(int));
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif 
  time_clear_M2L_buffers += (my_gettimeofday() - t1);  
#endif
}

void clear_P2P_buffers(int thread_num){
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]); 
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
  double t1 = my_gettimeofday(); 
#endif
  p_t->P2P_globalNextBlockNb = 1;
  // Content of P2P_nomutual_indexing_start[] not initialized 
  p_t->P2P_nomutual_indexing_start_ind = 0; 
  // Using memset(0) (wasting first block in P2P_interBuf):
  memset(p_t->P2P_currentBlockNb, 0,  gpu->NbCells*sizeof(int));
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif 
  time_clear_P2P_buffers += (my_gettimeofday() - t1); 
#endif
}

void clear_P2PLeafTgt_buffers(int thread_num){
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]); 
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
  double t1 = my_gettimeofday(); 
#endif
  p_t->P2PLeafTgt_globalNextBlockNb = 1;
  // Content of P2PLeafTgt_nomutual_indexing_start[] not initialized 
  p_t->P2PLeafTgt_nomutual_indexing_start_ind = 0; 
  // Using memset(0) (wasting first block in P2PLeafTgt_interBuf):
  memset(p_t->P2PLeafTgt_currentBlockNb, 0, gpu->NbLeafs*sizeof(int));
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif 
  time_clear_P2PLeafTgt_buffers += (my_gettimeofday() - t1);  
#endif
}


#define M2L_NOMUTUAL_INDEXING_START__CELLA_IND(i) (2*i) 
#define M2L_NOMUTUAL_INDEXING_CELL_INIT() {				\
  currentCellBlockNb_LA = p_t->M2L_currentCellBlockNb[LA] = (p_t->M2L_globalNextBlockNb)++; \
  int A_ind = M2L_NOMUTUAL_INDEXING_START__CELLA_IND((p_t->M2L_nomutual_indexing_start_ind)++); \
  p_t->M2L_nomutual_indexing_start[A_ind]   = LA;				\
  p_t->M2L_nomutual_indexing_start[A_ind+1] = currentCellBlockNb_LA;		\
  p_t->M2L_nomutual_indexing[currentCellBlockNb_LA] = 0;			\
}

// LA : index of target cell for which we will add 'LB' index: 
void add_CellIndex2nomutual_M2L(int LA, int LB){
  int currentCellBlockNb_LA;
#ifdef pfalcON
  int thread_num = omp_get_thread_num();
#else
  int thread_num = 0;
#endif
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]);
  
  if ((currentCellBlockNb_LA = p_t->M2L_currentCellBlockNb[LA]) == 0){
    if (p_t->M2L_globalNextBlockNb >= gpu->M2L_globalMaxBlockNb){
      run_kernel_M2L(thread_num);	  
      clear_M2L_buffers(thread_num); 
    }
    M2L_NOMUTUAL_INDEXING_CELL_INIT();
  }

  if (p_t->M2L_nomutual_indexing[currentCellBlockNb_LA] < M2L_INTERBUF_BS){
    /* p_t->M2L_nomutual_indexing[currentCellBlockNb_LA] is used here to store the sub-index 
     * (between 0 and M2L_INTERBUF_BS-1) for the next LB writing: */
    p_t->M2L_interBuf[currentCellBlockNb_LA*M2L_INTERBUF_BS
		      + ((p_t->M2L_nomutual_indexing[currentCellBlockNb_LA])++)] = LB;
  }
  else {
    /* p_t->M2L_nomutual_indexing[currentCellBlockNb_LA] == M2L_INTERBUF_BS:
     * we now use p_t->M2L_nomutual_indexing[currentCellBlockNb_LA] to store the next block number 
     * (Rq: we add M2L_INTERBUF_BS to distinguish from the last block) */
    if (p_t->M2L_globalNextBlockNb < gpu->M2L_globalMaxBlockNb){
      int newBlockNb = (p_t->M2L_globalNextBlockNb)++; 
      p_t->M2L_nomutual_indexing[currentCellBlockNb_LA] = newBlockNb + M2L_INTERBUF_BS; 
      p_t->M2L_currentCellBlockNb[LA] = newBlockNb;
      p_t->M2L_nomutual_indexing[newBlockNb] = 0;     
      p_t->M2L_interBuf[newBlockNb*M2L_INTERBUF_BS + ((p_t->M2L_nomutual_indexing[newBlockNb])++)] = LB; 
    }
    else { 
      run_kernel_M2L(thread_num);	  
      clear_M2L_buffers(thread_num);       
      M2L_NOMUTUAL_INDEXING_CELL_INIT();
      p_t->M2L_interBuf[currentCellBlockNb_LA*M2L_INTERBUF_BS
			+ ((p_t->M2L_nomutual_indexing[currentCellBlockNb_LA])++)] = LB;
    }
  }
}

#define P2P_NOMUTUAL_INDEXING_START__CELLA_IND(i) (2*i) 
#define P2P_NOMUTUAL_INDEXING_CELL_INIT() { \
    currentBlockNb_LA = p_t->P2P_currentBlockNb[shifted_LA] = (p_t->P2P_globalNextBlockNb)++; \
    int A_ind = P2P_NOMUTUAL_INDEXING_START__CELLA_IND((p_t->P2P_nomutual_indexing_start_ind)++); \
    p_t->P2P_nomutual_indexing_start[A_ind]   = LA; \
    p_t->P2P_nomutual_indexing_start[A_ind+1] = currentBlockNb_LA; \
    p_t->P2P_nomutual_indexing[currentBlockNb_LA] = 0; \
}
// LA : index of target cell for which we will add 'LB' index: 
void add_index2nomutual_P2P(int LA, int LB){
  int currentBlockNb_LA; 
  int shifted_LA = LA; // LA is NOT a leaf  
#ifdef pfalcON
  int thread_num = omp_get_thread_num();
#else
  int thread_num = 0;
#endif
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]);

  if ((currentBlockNb_LA = p_t->P2P_currentBlockNb[shifted_LA]) == 0){
    if (p_t->P2P_globalNextBlockNb >= gpu->P2P_globalMaxBlockNb){
      run_kernel_P2P(thread_num);	  
      clear_P2P_buffers(thread_num);       
    }
    P2P_NOMUTUAL_INDEXING_CELL_INIT();
  }

  if (p_t->P2P_nomutual_indexing[currentBlockNb_LA] < P2P_INTERBUF_BS){
    /* p_t->P2P_nomutual_indexing[currentBlockNb_LA] is used here to store the sub-index 
     * (between 0 and P2P_INTERBUF_BS-1) for the next LB writing: */
    p_t->P2P_interBuf[currentBlockNb_LA*P2P_INTERBUF_BS
		      + ((p_t->P2P_nomutual_indexing[currentBlockNb_LA])++)] = LB;
  }
  else {
    /* p_t->P2P_nomutual_indexing[currentBlockNb_LA] == P2P_INTERBUF_BS:
     * we now use p_t->P2P_nomutual_indexing[currentBlockNb_LA] to store the next block number 
     * (Rq: we add M2L_INTERBUF_BS to distinguish from the last block) */
    if (p_t->P2P_globalNextBlockNb < gpu->P2P_globalMaxBlockNb){
      //    std::cout << "In add_index2nomutual_P2P(): new block for cell #" << LA << " (p_t->P2P_globalNextBlockNb="<< p_t->P2P_globalNextBlockNb << ")" << std::endl; 
      int newBlockNb = (p_t->P2P_globalNextBlockNb)++; 
      p_t->P2P_nomutual_indexing[currentBlockNb_LA] = newBlockNb + P2P_INTERBUF_BS; 
      p_t->P2P_currentBlockNb[shifted_LA] = newBlockNb;
      p_t->P2P_nomutual_indexing[newBlockNb] = 0; 
      p_t->P2P_interBuf[newBlockNb*P2P_INTERBUF_BS + ((p_t->P2P_nomutual_indexing[newBlockNb])++)] = LB; 
    }
    else {
      run_kernel_P2P(thread_num);	  
      clear_P2P_buffers(thread_num);       
      P2P_NOMUTUAL_INDEXING_CELL_INIT();
      p_t->P2P_interBuf[currentBlockNb_LA*P2P_INTERBUF_BS
			+ ((p_t->P2P_nomutual_indexing[currentBlockNb_LA])++)] = LB;
    } 
  }
}




#define P2PLEAFTGT_NOMUTUAL_INDEXING_START__CELLA_IND(i) (2*i) 
#define P2PLEAFTGT_NOMUTUAL_INDEXING_CELL_INIT() { \
    currentBlockNb_LA = p_t->P2PLeafTgt_currentBlockNb[shifted_LA] = (p_t->P2PLeafTgt_globalNextBlockNb)++; \
    int A_ind = P2PLEAFTGT_NOMUTUAL_INDEXING_START__CELLA_IND((p_t->P2PLeafTgt_nomutual_indexing_start_ind)++); \
    p_t->P2PLeafTgt_nomutual_indexing_start[A_ind]   = LA;		\
    p_t->P2PLeafTgt_nomutual_indexing_start[A_ind+1] = currentBlockNb_LA; \
    p_t->P2PLeafTgt_nomutual_indexing[currentBlockNb_LA] = 0;		\
}
// LA : index of target cell for which we will add 'LB' index: 
// C++ template can be use to merge codes for add_index2nomutual_P2PLeafTgt() and add_index2nomutual_P2P() XXX
void add_index2nomutual_P2PLeafTgt(int LA, int LB){
  int currentBlockNb_LA; 
  // LA denotes a leaf 
  int shifted_LA = (LA & (~LEAF_FLAG));
#ifdef pfalcON
  int thread_num = omp_get_thread_num();
#else
  int thread_num = 0;
#endif
  InterBufs_t *p_t = &(gpu->InterBufs[thread_num]);
  
  if ((currentBlockNb_LA = p_t->P2PLeafTgt_currentBlockNb[shifted_LA]) == 0){
    if (p_t->P2PLeafTgt_globalNextBlockNb >= gpu->P2PLeafTgt_globalMaxBlockNb){
      run_kernel_P2PLeafTgt(thread_num);	  
      clear_P2PLeafTgt_buffers(thread_num);       
    }
    P2PLEAFTGT_NOMUTUAL_INDEXING_CELL_INIT();
  }

  if (p_t->P2PLeafTgt_nomutual_indexing[currentBlockNb_LA] < P2P_INTERBUF_BS){
    /* p_t->P2PLeafTgt_nomutual_indexing[currentBlockNb_LA] is used here to store the sub-index 
     * (between 0 and P2P_INTERBUF_BS-1) for the next LB writing: */
    p_t->P2PLeafTgt_interBuf[currentBlockNb_LA*P2P_INTERBUF_BS
			     + ((p_t->P2PLeafTgt_nomutual_indexing[currentBlockNb_LA])++)] = LB;
  }
  else {
    /* p_t->P2PLeafTgt_nomutual_indexing[currentBlockNb_LA] == P2P_INTERBUF_BS:
     * we now use p_t->P2PLeafTgt_nomutual_indexing[currentBlockNb_LA] to store the next block number 
     * (Rq: we add M2L_INTERBUF_BS to distinguish from the last block) */
    if (p_t->P2PLeafTgt_globalNextBlockNb < gpu->P2PLeafTgt_globalMaxBlockNb){
      //    std::cout << "In add_index2nomutual_P2PLeafTgt(): new block for cell #" << LA << " (p_t->P2PLeafTgt_globalNextBlockNb="<< p_t->P2PLeafTgt_globalNextBlockNb << ")" << std::endl; 
      int newBlockNb = (p_t->P2PLeafTgt_globalNextBlockNb)++; 
      p_t->P2PLeafTgt_nomutual_indexing[currentBlockNb_LA] = newBlockNb + P2P_INTERBUF_BS; 
      p_t->P2PLeafTgt_currentBlockNb[shifted_LA] = newBlockNb;
      p_t->P2PLeafTgt_nomutual_indexing[newBlockNb] = 0; 
      p_t->P2PLeafTgt_interBuf[newBlockNb*P2P_INTERBUF_BS + (p_t->P2PLeafTgt_nomutual_indexing[newBlockNb]++)] = LB; 
    }
    else {
      run_kernel_P2PLeafTgt(thread_num);	  
      clear_P2PLeafTgt_buffers(thread_num);       
      P2PLEAFTGT_NOMUTUAL_INDEXING_CELL_INIT();
      p_t->P2PLeafTgt_interBuf[currentBlockNb_LA*P2P_INTERBUF_BS
			       + ((p_t->P2PLeafTgt_nomutual_indexing[currentBlockNb_LA])++)] = LB;
    } 
  }
}

  

#endif












using namespace falcON;
////////////////////////////////////////////////////////////////////////////////
namespace falcON {
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // auxiliary stuff for class GravMAC                                        //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // class falcON::InvertZ                                                    //
    //                                                                          //
    // methods for inverting                                                    //
    //                                                                          //
    //         theta^(P+2)/(1-theta)^2 * y^a = 1                                //
    //                                                                          //
    // for  1/theta(y)                                                          //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////
    class InvertZ {
    private:
        static const unsigned N = 1000, N1=N-1;        // size of tables
        const unsigned P;                              // expansion order
        const     real A,hA,sA;                        // parameters
        real          *Z,*Y;                           // tables
        //--------------------------------------------------------------------------
        real z(real y) const {                         // z(y) = 1/theta - 1
            if(y < Y[ 0]) return std::pow(y,hA);
            if(y > Y[N1]) return std::pow(y,sA);
            return Polev(y,Y,Z,N);
        }
    public:
        //--------------------------------------------------------------------------
        InvertZ(real     a,                            // I: power a
                unsigned p) :                          // I: order P
        P   ( p ),
        A   ( a ),
        hA  ( half * A ),
        sA  ( A/(P+2.) ),
        Z   ( falcON_NEW(real,N) ),
        Y   ( falcON_NEW(real,N) )
        {
            double _z,iA=1./A,
            zmin = 1.e-4,
            zmax = 1.e4,
            lmin = log(zmin),
            dlz  = (log(zmax)-lmin)/double(N1);
            for(unsigned i=0; i!=N; ++i) {
                _z   = std::exp(lmin+i*dlz);
                Z[i] = _z;
                Y[i] = pow(_z*_z*pow(1+_z,P),iA);
            }
        }
        //--------------------------------------------------------------------------
        ~InvertZ() {
            falcON_DEL_A(Z);
            falcON_DEL_A(Y);
        }
        //--------------------------------------------------------------------------
        real invtheta(real y) const {
            return one + z(y);
        }
    };
    //////////////////////////////////////////////////////////////////////////////
} // namespace falcON {
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// class falcON::GravMAC                                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
GravMAC::GravMAC(MAC_type mc,
                 real     t0,
                 unsigned p) :
MAC  ( mc ),
P    ( p ),
TH0  ( min(one, falcON::abs(t0)) ),
iTH0 ( one/TH0)
{
    switch(MAC) {
        case const_theta:
            IZ = 0;
            break;
        case theta_of_M:
            // th^(p+2)    M  (d-2)/d   th0^(p+2)
            // --------  (---)        = ---------
            // (1-th)^2   M0            (1-th0)^2
            IZ = new InvertZ(third,P);
            break;
        case theta_of_M_ov_r:
            // th^(p+2)    Q  (d-2)/(d-1)   th0^(p+2)               M
            // --------  (---)            = ---------  with  Q := -----
            // (1-th)^2   Q0                (1-th0)^2             r_max
            IZ = new InvertZ(half,P);
            break;
        case theta_of_M_ov_rq:
            // th^(p+2)    S     th0^(p+2)                M
            // --------  (---) = ---------  with  S := -------
            // (1-th)^2   S0     (1-th0)^2             r_max^2
            IZ = new InvertZ(one,P);
            break;
    }
}
//------------------------------------------------------------------------------
void GravMAC::reset(MAC_type mc,
                    real     t0,
                    unsigned p) {
    TH0  = min(one,abs(t0));
    iTH0 = one/TH0;
    if(MAC != mc || P != p) {
        if(IZ) falcON_DEL_O(IZ);
        MAC  = mc;
        P    = p;
        switch(MAC) {
            case const_theta:
                IZ = 0;
                break;
            case theta_of_M:
                IZ = new InvertZ(third,P);
                break;
            case theta_of_M_ov_r:
                IZ = new InvertZ(half,P);
                break;
            case theta_of_M_ov_rq:
                IZ = new InvertZ(one,P);
                break;
        }
    }
}
//------------------------------------------------------------------------------
void GravMAC::reset_theta(real t0)
{
    TH0  = min(one,abs(t0));
    iTH0 = one/TH0;
}
//------------------------------------------------------------------------------
void GravMAC::set_rcrit(const GravEstimator*G) const {
    switch(MAC) {
        case const_theta:
            LoopCellsDown(grav::cell_iter,G->my_tree(),Ci)
            Ci->set_rcrit(iTH0);
            break;
        case theta_of_M: {
            real
            M0 = mass(G->root()),
            iF = pow(square(1-TH0)/pow(TH0,P+2u), 3u) / M0;
            LoopCellsDown(grav::cell_iter,G->my_tree(),Ci)
            Ci->set_rcrit(IZ->invtheta(mass(Ci)*iF));
        } break;
        case theta_of_M_ov_r: {
            int  i  = 0;
            real Q0 = mass(G->root()) / rmax(G->root());
            real *Q = falcON_NEW(real,G->my_tree()->N_cells());
            LoopCellsDown(grav::cell_iter,G->my_tree(),Ci) {
                Q[i] = mass(Ci)/rmax(Ci);
                if(Q[i] > Q0) Q0 = Q[i];
                ++i;
            }
            real iF = square(square(1-TH0)/pow(TH0,P+2u)) / Q0;
            i = 0;
            LoopCellsDown(grav::cell_iter,G->my_tree(),Ci)
            Ci->set_rcrit(IZ->invtheta(iF*Q[i++]));
            falcON_DEL_A(Q);
        } break;
        case theta_of_M_ov_rq: {
            int  i  = 0;
            real S0 = mass(G->root()) / square(rmax(G->root()));
            real *S = falcON_NEW(real,G->my_tree()->N_cells());
            LoopCellsDown(grav::cell_iter,G->my_tree(),Ci) {
                S[i] = mass(Ci)/square(rmax(Ci));
                if(S[i] > S0) S0 = S[i];
                ++i;
            }
            real iF = square(1-TH0)/pow(TH0,P+2u) / S0;
            i = 0;
            LoopCellsDown(grav::cell_iter,G->my_tree(),Ci)
            Ci->set_rcrit(IZ->invtheta(iF*S[i++]));
            falcON_DEL_A(S);
        } break;
    }
}
//------------------------------------------------------------------------------
GravMAC::~GravMAC()
{
    if(IZ) falcON_DEL_O(IZ);
}
////////////////////////////////////////////////////////////////////////////////
namespace {
    using namespace grav;
    //////////////////////////////////////////////////////////////////////////////
    //
    // auxiliary stuff for class GravEstimator
    //
    //////////////////////////////////////////////////////////////////////////////
    inline real bmax(vect const&com, cell_iter const&C)
    // This routines returns the distance from the cell's cofm (com)
    // to its most distant corner.
    {
        return sqrt( square(radius(C)+abs(com[0]-center(C)[0])) +
                    square(radius(C)+abs(com[1]-center(C)[1])) +
                    square(radius(C)+abs(com[2]-center(C)[2])) );
    }
    //////////////////////////////////////////////////////////////////////////////
    //
    // class GravIactBase
    //
    //////////////////////////////////////////////////////////////////////////////
    class GravIactBase {
        //--------------------------------------------------------------------------
        // types required in interact.h
        //--------------------------------------------------------------------------
    public:
        typedef grav::leaf_iter leaf_iter;
        typedef grav::cell_iter cell_iter;
        //--------------------------------------------------------------------------
        // static methods
        //--------------------------------------------------------------------------
        static bool take(grav::leaf_iter const&) { return true; }
        static bool take(grav::cell_iter const&) { return true; }
        //--------------------------------------------------------------------------
        // data
        //--------------------------------------------------------------------------
    private:
        unsigned              N_PRE[3], N_POST[3];     // direct sums control
    protected:
        GravStats* const      STAT;                    // statistics
        real                  RFAQ;
        //--------------------------------------------------------------------------
        // protected methods
        //--------------------------------------------------------------------------
        bool do_direct_pre (cell_iter const&A, leaf_iter const&) const
        {
            return number(A) < N_PRE[0];
        }
        //--------------------------------------------------------------------------
        bool do_direct_post(cell_iter const&A, leaf_iter const&) const
        {
	  return is_twig(A) || number(A) < N_POST[0];
	}
        //--------------------------------------------------------------------------
        bool do_direct_post(cell_iter const&A, cell_iter const&B) const
        {
            return (is_twig(A) && is_twig(B)) ||
            (number(A) < N_POST[1] && number(B) < N_POST[1]);
        }
        //--------------------------------------------------------------------------
        bool do_direct(cell_iter const&A) const
        {
            return is_twig(A) || number(A) < N_PRE[2];
        }
        //--------------------------------------------------------------------------
        static bool well_separated(cell_iter const&A, cell_iter const&B, real Rq)
        {
            return Rq > square(rcrit(A)+rcrit(B));
        }
        //--------------------------------------------------------------------------
        bool well_separated(cell_iter const&A, leaf_iter const&, real Rq) const
        {
            return RFAQ * Rq > rcrit2(A);
        }
        //--------------------------------------------------------------------------
    protected:
        GravIactBase(GravStats* t, unsigned const nd[4]= Default::direct)
        : STAT ( t ), RFAQ ( one )
        {
#ifdef iGPU_P2P
	  // NB: in order to always perform P2P with cells having <= Ncrit
	  // bodies, we rely on the minimum of each direct[] value w.r.t. Ncrit: 
	  size_t ncrit = gpu->get_ncrit();     
	  N_PRE [0] = (nd[0] <= ncrit ? nd[0] : ncrit); // C-B direct before try
	  N_PRE [1] = 0u;                               // C-C direct before try
	  N_PRE [2] = (nd[3] <= ncrit ? nd[3] : ncrit); // C-S direct before try
	  N_POST[0] = (nd[1] <= ncrit ? nd[1] : ncrit); // C-B direct after fail
	  N_POST[1] = (nd[2] <= ncrit ? nd[2] : ncrit); // C-C direct after fail
	  N_POST[2] = 0u;                               // C-S direct after fail
	  std::cout << "P2P settings: N_PRE[0]=" << N_PRE[0] 
		    << " N_PRE[2]=" << N_PRE[2]
		    << " N_POST[0]=" << N_POST[0]
		    << " N_POST[1]=" << N_POST[1] << std::endl; 
#else
	  N_PRE [0] = nd[0];                           // C-B direct before try
	  N_PRE [1] = 0u;                              // C-C direct before try
	  N_PRE [2] = nd[3];                           // C-S direct before try
	  N_POST[0] = nd[1];                           // C-B direct after fail
	  N_POST[1] = nd[2];                           // C-C direct after fail
	  N_POST[2] = 0u;                              // C-S direct after fail
#endif 	  
	}
        //--------------------------------------------------------------------------
    public:
        bool split_first(cell_iter const&A, cell_iter const&B) const
        {
            return is_twig(B) || (!is_twig(A) && rmax(A) > rmax(B));
        }
    };
    //////////////////////////////////////////////////////////////////////////////
    //
    // class falcON::GravIact
    //
    // This class is at the heart of the algorithm. It serves as INTERACTOR in
    // the template class MutualInteract<>, defined in interact.h, which
    // encodes the interaction phase in a most general way.
    // class falcON::GravIact has member functions for leaf-leaf,
    // leaf-cell, cell-leaf, cell-cell, and cell-self interactions (methods
    // interact()), as well as a function for the evaluation phase, method
    // evaluate_grav().
    //
    // NOTE. We organize the cells' Taylor coefficient: at a cell's first
    // interaction, memory for its coefficients is taken from a pre-allocated
    // pool and returned to the pool when the cell eventually passes through
    // evaluation phase.
    //
    //////////////////////////////////////////////////////////////////////////////
    class GravIact :
    public GravIactBase,
    public GravKern
    {
        GravIact           (GravIact const&);          // not implemented
        GravIact& operator=(GravIact const&);          // not implemented
        //--------------------------------------------------------------------------
        // public methods
        //--------------------------------------------------------------------------
    public:
        GravIact(kern_type k,                           // I: type of kernel
                 GravStats*t,                           // I: statistics
                 real      e,                           // I: softening length
                 unsigned  np,                          // I: initial pool size
                 bool      s    = Default::soften,      //[I: use individual eps?]
                 unsigned  const nd[4]= Default::direct)//[I: direct sum control]
        : GravIactBase  ( t,nd ),
        GravKern      ( k,e,s,np ) {}
        //--------------------------------------------------------------------------
        // interaction phase
        //--------------------------------------------------------------------------
        void direct_summation(cell_iter const&A) const {
            direct(A);
        }
        //--------------------------------------------------------------------------
        void direct_summation(cell_iter const&A, leaf_iter const&B) const {
            direct(A,B);
        }
        //--------------------------------------------------------------------------
        bool interact(cell_iter const&A) const
        {

            if(!is_active(A)) return true;               // no interaction -> DONE
            if(do_direct(A)) {                           // IF(suitable)
                direct(A);                                 //   perform BB iactions
#if !defined(pfalcON) && !defined(recursive)
                STAT->record_direct_CX(A);                 //   record stats
#ifdef WRITE_IACT_INFO
                std::cerr<<": direct\n";
#endif
#endif
                return true;                               //   DONE
            }                                            // ELSE
            return false;                                //   must be splitted
        }
        //--------------------------------------------------------------------------
        bool interact(cell_iter const&A, cell_iter const&B) const {

            if(!(is_active(A)||is_active(B)))return true;// no interaction -> DONE
            vect dX = cofm(A)-cofm(B);                   // compute dX = X_A - X_B
            real Rq = norm(dX);                          // and dX^2
            if(well_separated (A,B,Rq))
            {                // IF(well separated)
	      approx(A,B,dX,Rq);                         //   interact
#if !defined(pfalcON) && !defined(recursive)
                STAT->record_approx_CC(A,B);               //   record stats
#ifdef WRITE_IACT_INFO
                std::cerr<<": approx\n";
#endif
#endif
                return true;                               //   DONE
            }                                            // ENDIF
            if(do_direct_post(A,B))
            {                    // IF(suitable)
                direct(A,B);                               //   perform BB iactions
#if !defined(pfalcON) && !defined(recursive)
                STAT->record_direct_CC(A,B);               //   record stats
#ifdef WRITE_IACT_INFO
                std::cerr<<": direct\n";
#endif
#endif
                return true;                               //   DONE
            }                                            // ELSE
            return false;                                //   must be splitted
        }
        //--------------------------------------------------------------------------
        bool interact(cell_iter const&A, leaf_iter const&B) const {

            if(!(is_active(A)||is_active(B)))return true;// no interaction -> DONE
            if(do_direct_pre(A,B)) {                     // IF(suitable)
                direct(A,B);                               //   perform BB iactions
#if !defined(pfalcON) && !defined(recursive)
                STAT->record_direct_CB(A,B);               //   record statistics
#ifdef WRITE_IACT_INFO
                std::cerr<<": direct\n";
#endif
#endif
                return true;                               //   DONE
            }                                            // ENDIF
            vect dX = cofm(A)-cofm(B);                   // compute R = x_A-x_B
            real Rq = norm(dX);                          // compute R^2
            if(well_separated(A,B,Rq)) {                 // IF(well separated)
	      approx(A,B,dX,Rq);                         //   interact
#if !defined(pfalcON) && !defined(recursive)
                STAT->record_approx_CB(A,B);               //   record statistics
#ifdef WRITE_IACT_INFO
                std::cerr<<": approx\n";
#endif
#endif
                return true;                               //   DONE
            }                                            // ENDIF
            if(do_direct_post(A,B)) {                    // IF(suitable)
                direct(A,B);                               //   perform BB iactions
#if !defined(pfalcON) && !defined(recursive)
                STAT->record_direct_CB(A,B);               //   record statistics
#ifdef WRITE_IACT_INFO
                std::cerr<<": direct\n";
#endif
#endif
                return true;                               //   DONE
            }                                            // ELSE
            return false;                                //   cell must be splitted
        }
        //--------------------------------------------------------------------------
        bool interact(leaf_iter const&A, cell_iter const&B) const {

            return interact(B,A);
        }
        //--------------------------------------------------------------------------
        void interact(leaf_iter const&A, leaf_iter const&B) const {

            if(!(is_active(A) || is_active(B))) return;  // no interaction -> DONE
            single(A,B);                                 // perform interaction
#if !defined(pfalcON) && !defined(recursive)
            STAT->record_BB(A,B);                        // record statistics
#endif
        }
        //--------------------------------------------------------------------------
        void evaluate(cell_iter const&C) const {       // evaluation phase
            flush_buffers();                             // finish interactions
            eval_grav(C,TaylorSeries(cofm(C)));          // start recursion
        }
        //--------------------------------------------------------------------------
        void set_sink(real e, real f)
        {
            reset_eps(e);
            RFAQ = f*f;
        }
        //--------------------------------------------------------------------------
        void unset_sink(real e)
        {
            reset_eps(e);
            RFAQ = one;
        }
    };
    //////////////////////////////////////////////////////////////////////////////
    //
    // class falcON::GravIactAll
    //
    // Like GravIact, except that all cells and leafs are assumed active.
    //
    //////////////////////////////////////////////////////////////////////////////
    class GravIactAll :
    public GravIactBase,
    public GravKernAll
    {
        GravIactAll           (GravIactAll const&);    // not implemented
        GravIactAll& operator=(GravIactAll const&);    // not implemented
        //--------------------------------------------------------------------------
        // public methods
        //--------------------------------------------------------------------------
    public:
        GravIactAll(kern_type    k,                     // I: type of kernel
                    GravStats*   t,                     // I: statistics
                    real         e,                     // I: softening length
                    unsigned     np,                    // I: initial pool size
                    bool         s    =Default::soften, //[I: use individual eps?]
                    unsigned const nd[4]=Default::direct  //[I: direct sum control]
        ) :
        GravIactBase ( t,nd ),
        GravKernAll  ( k,e,s,np ) {
	}

        //-------------------------------------------------------------------------
        // data
        //-------------------------------------------------------------------------
        
        mutable unsigned int i = 0;
       

        //--------------------------------------------------------------------------
        // interaction phase
        //--------------------------------------------------------------------------
        void direct_summation(cell_iter const&A) const {
#ifdef iGPU_P2P
	  int LA = (int) globalTree->NoCell(A);
	  add_index2nomutual_P2P(LA, LA);
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif   
	  interaction_nb += OWN_INTERACTION_COUNT(number(A));
#endif
#ifdef profile
	  n_par_cell[NA - 1] += 1;
	  nb_P2P_t++;
#endif	  
#else         
   direct(A);	    
#endif
	}
        //--------------------------------------------------------------------------
        void direct_summation(cell_iter const&A, leaf_iter const&B) const {
#ifdef iGPU_P2P
	  int LA = (int) globalTree->NoCell(A);
	  int LB = ((int) globalTree->NoLeaf(B) | LEAF_FLAG);
	  add_index2nomutual_P2P(LA, LB);
	  add_index2nomutual_P2PLeafTgt(LB, LA);
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif   
	  interaction_nb += PAIR_INTERACTION_COUNT(number(A), 1);
#endif
#ifdef profile
	  n_par_cell[NA - 1] += 1;
	  n_par_cell[NB - 1] += 1;
	  nb_P2P_t++;
#endif
#else         
	    direct(A,B);

#endif
        }
        //--------------------------------------------------------------------------
        bool interact(cell_iter const&A) const
        {
            if(do_direct(A)) {                           // IF(suitable)
#ifdef iGPU_P2P
	      int LA = (int) globalTree->NoCell(A);
	      add_index2nomutual_P2P(LA, LA);
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif   
	      interaction_nb += OWN_INTERACTION_COUNT(number(A));
#endif
#ifdef profile
	      n_par_cell[NA - 1] += 1;
	      n_par_cell[NB - 1] += 1;
	      nb_P2P_t++;
#endif
#else               
	      direct(A);                                 //   perform BB iactions
#endif
#if !defined(pfalcON) && !defined(recursive)
                STAT->record_direct_CX(A);                 //   record stats
#ifdef WRITE_IACT_INFO
                std::cerr<<": direct\n";
#endif
#endif
                return true;                               //   DONE
            }                                            // ELSE
            return false;                                //   must be splitted
        }
        //--------------------------------------------------------------------------
        bool interact(cell_iter const&A, cell_iter const&B) const {
            vect dX = cofm(A)-cofm(B);                   // compute dX = X_A - X_B
            real Rq = norm(dX);                          // and dX^2
            if(well_separated (A,B,Rq))
            {                // IF(well separated)
#ifdef iGPU_M2L
	      int LA = (int) globalTree->NoCell(A);
	      int LB = (int) globalTree->NoCell(B);
	      add_CellIndex2nomutual_M2L(LA, LB);
	      add_CellIndex2nomutual_M2L(LB, LA);
#else              
	      approx(A,B,dX,Rq);                         //   interact
#endif
#if !defined(pfalcON) && !defined(recursive)
	      STAT->record_approx_CC(A,B);               //   record stats
#ifdef WRITE_IACT_INFO
	      std::cerr<<": approx\n";
#endif
#endif
	      return true;                               //   DONE
            }                                            // ENDIF
            if(do_direct_post(A,B))
            {                    // IF(suitable)
#ifdef iGPU_P2P
	      int LA = (int) globalTree->NoCell(A);
	      int LB = (int) globalTree->NoCell(B);
	      add_index2nomutual_P2P(LA, LB);
	      add_index2nomutual_P2P(LB, LA);
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif   
	      interaction_nb += 2*PAIR_INTERACTION_COUNT(number(A), number(B));
#endif
#ifdef profile
	      n_par_cell[NA - 1] += 1;
	      n_par_cell[NB - 1] += 1;
	      nb_P2P_t++;
#endif
#else
	      direct(A,B);                               //   perform BB iactions
#endif
#ifndef pfalcON
	      STAT->record_direct_CC(A,B);               //   record stats
#ifdef WRITE_IACT_INFO
	      std::cerr<<": direct\n";
#endif
#endif
	      return true;                               //   DONE
            }                                            // ELSE
            return false;                                //   SPLIT <
        }
        //--------------------------------------------------------------------------
        bool interact(cell_iter const&A, leaf_iter const&B) const {

	  if(do_direct_pre(A,B)) {                     // IF(suitable)
#ifdef iGPU_P2P
	    int LA = (int) globalTree->NoCell(A);
	    int LB = ((int) globalTree->NoLeaf(B) | LEAF_FLAG);
	    add_index2nomutual_P2P(LA, LB);
	    add_index2nomutual_P2PLeafTgt(LB, LA);
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif   
	  interaction_nb += PAIR_INTERACTION_COUNT(number(A), 1);
#endif
#ifdef profile
	    n_par_cell[NA - 1] += 1;
	    n_par_cell[NB - 1] += 1;
	    nb_P2P_t++;
#endif
#else
	    direct(A,B);                               //   perform BB iactions
#endif
#if !defined(pfalcON) && !defined(recursive)
	    STAT->record_direct_CB(A,B);               //   record statistics
#ifdef WRITE_IACT_INFO
	    std::cerr<<": direct\n";
#endif
#endif
	    return true;                               //   DONE
	  }                                            // ENDIF

#ifndef iGPU_P2P
            vect dX = cofm(A)-cofm(B);                   // compute R = x_A-x_B
            real Rq = norm(dX);                          // compute R^2
            if(well_separated(A,B,Rq)) {                 // IF(well separated) 
	      approx(A,B,dX,Rq);                         //   interact
#if !defined(pfalcON) && !defined(recursive)
	      STAT->record_approx_CB(A,B);               //   record statistics
#ifdef WRITE_IACT_INFO
	      std::cerr<<": approx\n";
#endif
#endif
	      return true;                               //   DONE
            }                                            // ENDIF
#endif // ! iGPU_P2P

            if(do_direct_post(A,B)) {                    // IF(suitable)
#ifdef iGPU_P2P
	      int LA = (int) globalTree->NoCell(A);
	      int LB = ((int) globalTree->NoLeaf(B) | LEAF_FLAG);
	      add_index2nomutual_P2P(LA, LB);
	      add_index2nomutual_P2PLeafTgt(LB, LA);
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif   
	  interaction_nb += PAIR_INTERACTION_COUNT(number(A), 1);
#endif
#ifdef profile   
	      n_par_cell[NA - 1] += 1;
	      n_par_cell[NB - 1] += 1;
	      nb_P2P_t++;
#endif
#else               
	      direct(A,B);                               //   perform BB iactions
#endif
#if !defined(pfalcON) && !defined(recursive)
	      STAT->record_direct_CB(A,B);               //   record statistics
#ifdef WRITE_IACT_INFO
	      std::cerr<<": direct\n";
#endif
#endif
	      return true;                               //   DONE
            }                                            // ELSE
            return false;                                //   cell must be splitted
        }
        //--------------------------------------------------------------------------
        bool interact(leaf_iter const&A, cell_iter const&B) const {
            return interact(B,A);
        }
        //--------------------------------------------------------------------------
        void interact(leaf_iter const&A, leaf_iter const&B) const {
#ifdef iGPU_P2P
	  int LA = ((int) globalTree->NoLeaf(A) | LEAF_FLAG);
	  int LB = ((int) globalTree->NoLeaf(B) | LEAF_FLAG);
	  add_index2nomutual_P2PLeafTgt(LA, LB);
	  add_index2nomutual_P2PLeafTgt(LB, LA);
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
#ifdef pfalcON
#pragma omp atomic
#endif   
	  interaction_nb += PAIR_INTERACTION_COUNT(1, 1);
#endif
#ifdef profile
	  n_par_cell[NA - 1] += 1;
	  n_par_cell[NB - 1] += 1;
	  nb_P2P_t++;
#endif
#else
	  single(A,B);                                 // perform interaction
#endif
#if !defined(pfalcON) && !defined(recursive)
	  STAT->record_BB(A,B);                        // record statistics
#endif
        }
        //--------------------------------------------------------------------------
        void evaluate(cell_iter const&C) const {       // evaluation phase
            flush_buffers();                             // finish interactions
            eval_grav_all(C,TaylorSeries(cofm(C)));      // start recursion
            
        }
        //--------------------------------------------------------------------------
        void set_sink(real e, real f)
        {
            reset_eps(e);
            RFAQ = f*f;
        }
        //--------------------------------------------------------------------------
        void unset_sink(real e)
        {
            reset_eps(e);
            RFAQ = one;
        }
    };
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // UpdateLeafs()                                                            //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////
    unsigned UpdateLeafs(const OctTree*tree, bool i_soft)
    {
        unsigned n=0;
        if(i_soft) {
            CheckMissingBodyData(tree->my_bodies(),
                                 fieldset::m|fieldset::e|fieldset::f);
            if(debug(1))
                LoopLeafs(grav::leaf,tree,Li) {
                    Li->copy_from_bodies_mass(tree->my_bodies());
                    Li->copy_from_bodies_eph (tree->my_bodies());
                    Li->copy_from_bodies_flag(tree->my_bodies());
                    if(is_active(Li)) ++n;
                    if(mass(Li) <= zero)
                        falcON_THROW("GravEstimator: mass of body #%d=%f "
                                     "but falcON requires positive masses\n",
                                     tree->my_bodies()->bodyindex(mybody(Li)),
                                     mass(Li));
                }
            else
                LoopLeafs(grav::leaf,tree,Li) {
                    Li->copy_from_bodies_mass(tree->my_bodies());
                    Li->copy_from_bodies_eph (tree->my_bodies());
                    Li->copy_from_bodies_flag(tree->my_bodies());
                    if(is_active(Li)) ++n;
                }
        } else {
            CheckMissingBodyData(tree->my_bodies(),fieldset::m|fieldset::f);
            if(debug(1))
                LoopLeafs(grav::leaf,tree,Li) {
                    Li->copy_from_bodies_mass(tree->my_bodies());
                    Li->copy_from_bodies_flag(tree->my_bodies());
                    if(is_active(Li)) ++n;
                    if(mass(Li) <= zero)
                        falcON_THROW("GravEstimator: mass of body #%d=%f "
                                     "but falcON requires positive masses\n",
                                     tree->my_bodies()->bodyindex(mybody(Li)),
                                     mass(Li));
                }
            else
                LoopLeafs(grav::leaf,tree,Li) {
                    Li->copy_from_bodies_mass(tree->my_bodies());
                    Li->copy_from_bodies_flag(tree->my_bodies());
                    if(is_active(Li)) ++n;
                }
        }
        return n;
    }
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // is_act<bool>()                                                           //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////
    template<bool> struct __IsAct;
    template<> struct __IsAct<0> {
        template<typename T> static bool is(T const&t) { return is_active(t); }
    };
    template<> struct __IsAct<1> {
        template<typename T> static bool is(T const&) { return 1; }
    };
    
    template<bool ALL> inline
    bool is_act(const grav::leaf*L) { return __IsAct<ALL>::is(L); }
    template<bool ALL> inline
    bool is_act(flags F) { return __IsAct<ALL>::is(F); }
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // UpdateBodiesGrav<ALL_ACT>()                                              //
    //                                                                          //
    // for active OR all leafs:                                                 //
    // - copy pot & acc to their associated bodies                              //
    // - optionally copy eps                                                    //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////
    template<bool ALL_ACT>
    void UpdateBodiesGrav(const OctTree*T,
                          real          G
#ifdef falcON_ADAP
                          , bool          U
#endif
    )
    {
#ifdef falcON_ADAP
        if(U) {
            CheckMissingBodyData(T->my_bodies(),
                                 fieldset::e|fieldset::a|fieldset::p);
            if(G!=one) {
                LoopLeafs(grav::leaf,T,Li) if(is_act<ALL_ACT>(Li)) {
                    Li->copy_to_bodies_eps (T->my_bodies());
                    Li->copy_to_bodies_grav(T->my_bodies(),G);
                }
            } else {
                LoopLeafs(grav::leaf,T,Li) if(is_act<ALL_ACT>(Li)) {
                    Li->copy_to_bodies_eps (T->my_bodies());
                    Li->copy_to_bodies_grav(T->my_bodies());
                }
            }
        } else {
#endif
            CheckMissingBodyData(T->my_bodies(),fieldset::a|fieldset::p);
            if(G!=one) {
                LoopLeafs(grav::leaf,T,Li) if(is_act<ALL_ACT>(Li))
                    Li->copy_to_bodies_grav(T->my_bodies(),G);
            } else {
                LoopLeafs(grav::leaf,T,Li) if(is_act<ALL_ACT>(Li))
                    Li->copy_to_bodies_grav(T->my_bodies());
            }
#ifdef falcON_ADAP
        }
#endif
    }
    //////////////////////////////////////////////////////////////////////////////
    //                                                                          //
    // ResetBodiesGrav<ALL_ACT>()                                               //
    //                                                                          //
    // - reset pot & acc of active OR all bodies to zero                        //
    //                                                                          //
    //////////////////////////////////////////////////////////////////////////////
    template<bool ALL_ACT> void ResetBodiesGrav(const bodies*B)
    {
        CheckMissingBodyData(B,fieldset::a|fieldset::p);
        LoopAllBodies(B,b)
        if(is_act<ALL_ACT>(flag(b))) {
            b.pot() = zero;
            b.acc() = zero;
        }
    }
}                                                  // END: unnamed namespace
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// class falcON::GravEstimator                                                //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
void GravEstimator::update_leafs()
{
    if(TREE==0)
        falcON_Error("GravEstimator: no tree");        // IF no tree, FATAL ERROR
    if(! TREE->is_used_for_grav() )                  // IF tree not used by grav
        reset();                                       //   reset allocation & flags
    if( TREE->my_bodies()->srce_data_changed() )     // IF body source are changed
        LEAFS_UPTODATE = 0;                            //   leafs are out of date
    if(! LEAFS_UPTODATE ) {                          // IF leafs are out of date
        NLA_needed = UpdateLeafs(TREE, INDI_SOFT);     //   update leafs
        LEAFS_UPTODATE = 1;                            //   leafs are up to date now
        CELLS_UPTODATE = 0;                            //   but cells are not
        TREE->my_bodies()->mark_srce_data_read();      //   mark bodies: srce read
    }                                                // ENDIF
}
//------------------------------------------------------------------------------
#ifdef falcON_ADAP
# include <proper/gravity_ind.cc>                 // GravEstimator::adjust_eph()
#endif
//------------------------------------------------------------------------------
GravEstimator::~GravEstimator() {
#ifdef ispcpfalcON
    if(LEAF_ACPN) falcON_DEL_A(LEAF_ACPN);//WDutils_DEL_aligned(32,LEAF_ACPN);//
#else
#ifndef iGPU
    if(CELL_SRCE) falcON_DEL_A(CELL_SRCE);
    if(LEAF_ACPN) falcON_DEL_A(LEAF_ACPN);
#else 
    // CELL_SRCE and LEAF_ACPN will be unmapped and the corresponding
    // OpenCL buffers will be released in cl_manip::~cl_manip() 
#endif
#endif
}
//------------------------------------------------------------------------------
unsigned GravEstimator::pass_up(const GravMAC*MAC,
                                bool          REUSE)
{
    // passes up: flag, mass, cofm, rmax[, eph], multipoles; sets rcrit
    report REPORT("GravEstimator::pass_up_for_approx()");
    int n=0;                                         // counter: active cells
    if(INDI_SOFT) {                                  // IF(individual eps_i)
        // 1    with eps_i: pass flag, mass, N*eps/2, cofm, rmax, multipoles
#define MASS_WEIGHTED_SOFTENING 1
        LoopCellsUp(grav::cell_iter,TREE,Ci) {         //   LOOP cells upwards
            Ci->reset_active_flag();                     //     reset activity flag
            Ci->reset_sink_flag();                       //     reset sink flag
            real eh (zero);                              //     reset eps/2
            real mon(zero);                              //     reset monopole
            vect com(zero);                              //     reset dipole
            LoopCellKids(cell_iter,Ci,c) {               //     LOOP sub-cells c
                eh  += eph (c) *
#ifdef MASS_WEIGHTED_SOFTENING
                mass(c)                                //       sum up M * eps/2
#else
                number(c)
#endif
                ;
                mon += mass(c);                            //       sum up monopole
                com += mass(c) * cofm(c);                  //       sum up dipole
                Ci->add_active_flag(c);                    //       add in activity flag
                Ci->add_sink_flag(c);                      //       add in sink flag
            }                                            //     END LOOP
            LoopLeafKids(cell_iter,Ci,l) {               //     LOOP sub-leafs s
#ifdef MASS_WEIGHTED_SOFTENING
                eh  += mass(l) * eph(l);                   //       sum up M * eps/2
#else
                eh  += eph (l);                            //       sum up eps/2
#endif
                mon += mass(l);                            //       sum up monopole
                com += mass(l) * cofm(l);                  //       sum up dipole
                Ci->add_active_flag(l);                    //       add in activity flag
                Ci->add_sink_flag(l);                      //       add in sink flag
            }                                            //     END LOOP
            if(is_active(Ci)) n++;                       //     count active cells
            Ci->mass() = mon;                            //     set mass
            mon        = (mon==zero)? zero:one/mon;      //     1/mass
            com       *= mon;                            //     cofm = dipole/mass
#ifdef MASS_WEIGHTED_SOFTENING
            eh        /= mass(Ci);                       //     mean eps/2
#else
            eh        /= number(Ci);                     //     mean eps/2
#endif
            Ci->eph()  = eh;                             //     set eps/2
            Mset P(zero);                                //     reset multipoles
            real dmax(zero);                             //     reset d_max
            LoopLeafKids(cell_iter,Ci,l) {               //     LOOP sub-leafs s
                vect Xi = cofm(l); Xi-= com;               //       distance vector
                update_max(dmax,norm(Xi));                 //       update d_max^2
                P.add_body(Xi,mass(l));                    //       add multipoles
            }                                            //     END LOOP
            if(has_leaf_kids(Ci)) dmax = sqrt(dmax);     //     d_max due to sub-leafs
            LoopCellKids(cell_iter,Ci,c) {               //     LOOP sub-cells c
                vect Xi = cofm(c); Xi-= com;               //       distance vector
                real Xq = norm(Xi);                        //       distance^2
                real x  = dmax - rmax(c);                  //       auxiliary
                if(zero>x || Xq>square(x))                 //       IF(d>d_max)
                    dmax = sqrt(Xq) + rmax(c);               //         set d_max = d
                P.add_cell(Xi,mass(c),poles(c));           //       add multipoles
            }                                            //     END LOOP
            Ci->rmax() = REUSE? dmax :                   //     r_max=d_max
            min(dmax,bmax(com,Ci));         //     r_max=min(d_max,b_max)
            Ci->cofm() = com;                            //     set dipole = mass*cofm
            Ci->poles()= P;                              //     assign multipoles
        }                                              //   END LOOP
    } else {                                         // ELSE (no individual eps)
        // 2    without eps_i: pass flag, mass, cofm, rmax, multipoles
        LoopCellsUp(grav::cell_iter,TREE,Ci) {         //   LOOP cells upwards
            Ci->reset_active_flag();                     //     reset activity flag
            Ci->reset_sink_flag();                       //     reset sink flag
#if defined(pfalcON) && (! defined(iGPU))
	    falcON::GravKernBase::_unLockpfalcON(Ci);    //     clear lock 
#endif
            real mon(zero);                              //     reset monopole
            vect com(zero);                              //     reset dipole
            LoopCellKids(cell_iter,Ci,c) {               //     LOOP sub-cells c
                mon += mass(c);                            //       sum up monopole
                com += mass(c) * cofm(c);                  //       sum up dipole
                Ci->add_active_flag(c);                    //       add in activity flag
                Ci->add_sink_flag(c);                      //       add in sink flag
            }                                            //     END LOOP
            LoopLeafKids(cell_iter,Ci,l) {               //     LOOP sub-leafs s
                mon += mass(l);                            //       sum up monopole
                com += mass(l) * cofm(l);                  //       sum up dipole
                Ci->add_active_flag(l);                    //       add in activity flag
                Ci->add_sink_flag(l);                      //       add in sink flag
            }                                            //     END LOOP
            if(is_active(Ci)) n++;                       //     count active cells
            Ci->mass() = mon;                            //     set mass
            mon        = (mon==zero)? zero:one/mon;      //     1/mass
            com       *= mon;                            //     cofm = dipole/mass
            Mset P(zero);                                //     reset multipoles
            real dmax(zero);                             //     reset d_max
            LoopLeafKids(cell_iter,Ci,l) {               //     LOOP sub-leafs s
                vect Xi = cofm(l); Xi-= com;               //       distance vector
                update_max(dmax,norm(Xi));                 //       update d_max^2
                P.add_body(Xi,mass(l));                    //       add multipoles
            }                                            //     END LOOP
            if(has_leaf_kids(Ci)) dmax = sqrt(dmax);     //     d_max due to sub-leafs
            LoopCellKids(cell_iter,Ci,c) {               //     LOOP sub-cells c
                vect Xi = cofm(c); Xi-= com;               //       distance vector
                real Xq = norm(Xi);                        //       distance^2
                real x  = dmax - rmax(c);                  //       auxiliary
                if(zero>x || Xq>square(x))                 //       IF(d>d_max)
                    dmax = sqrt(Xq) + rmax(c);               //         set d_max = d
                P.add_cell(Xi,mass(c),poles(c));           //       add multipoles
            }                                            //     END LOOP
            Ci->rmax() = REUSE? dmax :                   //     r_max=d_max
            min(dmax,bmax(com,Ci));         //     r_max=min(d_max,b_max)
            Ci->cofm() = com;                            //     set dipole = mass*cofm
            Ci->poles()= P;                              //     assign multipoles
        }                                              //   END LOOP
    }                                                // ENDIF
    // 3  normalize multipoles
    for(unsigned i=0; i!=TREE->N_cells(); ++i)       // LOOP cell sources
        CELL_SRCE[i].normalize_poles();                //   normalize multipoles
    // 4  set rcrit
    if(MAC) MAC->set_rcrit(this);                    // set r_crit for all cells
#ifdef pfalcON
    // 5  give coeffs (local expansion) to each cell in pfalcON
    // (waste of memory, but avoid (possible) bottleneck with memory 
    // allocations in the parallel dual tree traversal) 
    unsigned i = 0;
    LoopCellsDown(grav::cell_iter,TREE,Ci) {
      if (! Ci->hasCoeffs()){
	falcON::grav::Cset*X = COEF_ACPN + i;
	X->set_zero();
	Ci->setCoeffs(X);
#ifdef iGPU
	Ci->ID2 = i;
#endif 
	i++;
      }
    }
#endif
    return n;                                        // return # active cells
}
 
#ifdef iGPU
 void GravEstimator::map_PotAcc(){
   gpu->map_buffer(gpu->cl_bufs + CLBUF_POTACC, gpu->NbLeafs*sizeof(Leaf::acpn_data), (void **) &(LEAF_ACPN), CQ_P2P);  
   gpu->cl_bufs_map_ptr[CLBUF_POTACC] = LEAF_ACPN; 
 }
 void GravEstimator::map_MPoles(){
   gpu->map_buffer(gpu->cl_bufs + CLBUF_MPOLES, gpu->NbCells*sizeof(Cell::srce_data), (void **) &(CELL_SRCE), CQ_M2L);  
   gpu->cl_bufs_map_ptr[CLBUF_MPOLES] = CELL_SRCE; 
 }
 void GravEstimator::map_Local_Coefs(){
   gpu->map_buffer(gpu->cl_bufs + CLBUF_LOCAL_COEFS, gpu->NbCells*sizeof(falcON::grav::Cset), (void **) &(COEF_ACPN), CQ_M2L);  
   gpu->cl_bufs_map_ptr[CLBUF_LOCAL_COEFS] = COEF_ACPN; 
 }
 
 void GravEstimator::unmap_PotAcc(){
   gpu->unmap_buffer(gpu->cl_bufs + CLBUF_POTACC, LEAF_ACPN, CQ_P2P);  
   gpu->cl_bufs_map_ptr[CLBUF_POTACC] = NULL; 
 }
 void GravEstimator::unmap_MPoles(){
  gpu->unmap_buffer(gpu->cl_bufs + CLBUF_MPOLES, CELL_SRCE, CQ_M2L);  
  gpu->cl_bufs_map_ptr[CLBUF_MPOLES] = NULL; 
 }
 void GravEstimator::unmap_Local_Coefs(){
  gpu->unmap_buffer(gpu->cl_bufs + CLBUF_LOCAL_COEFS, COEF_ACPN, CQ_M2L);  
  gpu->cl_bufs_map_ptr[CLBUF_LOCAL_COEFS] = NULL; 
 }
#endif

//------------------------------------------------------------------------------
bool GravEstimator::prepare(const GravMAC*MAC,
                            bool          al)
{
    SET_I
    if(al) NLA_needed = TREE->N_leafs();             // all leafs are active
    if(NLA_needed==0) {
        falcON_Warning("in GravEstimator::prepare(): no body active");
        return 1;
    }
    //  - allocate memory for leaf acc/pot/num properties for active leafs
    if(NLA!=NLA_needed)
    {                            // IF #active leafs changed
        NLA = NLA_needed;                              //   # new allocation

#ifdef ispcpfalcON
        if(LEAF_ACPN) falcON_DEL_A(LEAF_ACPN);         //   delete old allocation
        LEAF_ACPN=falcON_NEW(Leaf::acpn_data,NLA);     //   allocate memory
        globalACPN = LEAF_ACPN;
#else
#ifdef iGPU
	gpu->NbLeafs = NLA; 
	if(LEAF_ACPN) falcON_DEL_A(LEAF_ACPN);

	gpu->create_buffer(gpu->cl_bufs + CLBUF_POTACC, gpu->NbLeafs*sizeof(Leaf::acpn_data), (void **) &(LEAF_ACPN), CQ_P2P);
#else
        if(LEAF_ACPN) falcON_DEL_A(LEAF_ACPN);         //   delete old allocation
        LEAF_ACPN=falcON_NEW(Leaf::acpn_data,NLA);     //   allocate memory
#endif
#endif
    }                                                // ENDIF
    const bool all = al || NLA==TREE->N_leafs();     // are all active?
    Leaf::acpn_data*si=LEAF_ACPN;                    // pter to leafs' acpn data
    if(all)
    {// IF all leafs
#if defined (pfalcON) || defined (iGPU)
        int i=0;
#endif
        LoopLeafs(Leaf,TREE,Li)
        {                      //   LOOP leafs
            si->reset();                                 //     reset acpn data
            Li->set_acpn(si++);                          //     set leaf: acpn
#if defined(pfalcON) && (! defined(iGPU))
	    falcON::GravKernBase::_unLockpfalcON(Li);    //     clear lock 
#endif
#ifdef ispcpfalcON
            Li->idACPN = i;
            i++;
#endif

#ifdef iGPU
	    Li->idACPN = i;
            i++;
#endif
        } //   END LOOP
        
    }
    else                                             // ELSE (only active)
        LoopLeafs(Leaf,TREE,Li)                        //   LOOP leafs
        if(is_active(Li)) {                          //     IF(leaf is active)
            si->reset();                               //       reset acpn data
            Li->set_acpn(si++);                        //       set leaf: acpn
        } else                                       //     ELSE
            Li->set_acpn(0);                           //       leaf has no acpn
    // update cell source properties
    if(!CELLS_UPTODATE ||                            // IF sources out-of-date
       NCT != TREE->N_cells()) {                     //    OR changed in number
        //    - allocate memory for cell source properties & reset their coeff pter
        if(NCT     < TREE->N_cells() ||                //   IF #cells too large
           NCT+NCT > TREE->N_cells()   ) {             //   OR #cells too small
#ifndef iGPU
	  if(CELL_SRCE) falcON_DEL_A(CELL_SRCE);       //     delete old allocation
#else 
	  if(CELL_SRCE) {
	    std::cout << "In GravEstimator::prepare(), OpenCL buffers must be reallocated!" << std::endl; 
	    exit(EXIT_FAILURE); 
	  }
#endif
            NCT = TREE->N_cells();                       //     # new allocation
#ifdef iGPU
#if defined(pfalcON) && (! defined(pfalcON_useTBB))
	    if (omp_in_parallel()){ 
	      std::cout << "GravEstimator::prepare() called within OpenMP parallel region ..." << std::endl; 
	      exit(EXIT_FAILURE); 
	    }
#endif 
	    gpu->NbCells = NCT;
	    std::cout << "NbCells=" << gpu->NbCells << std::endl; 

	    // OpenCL buffer for multipole expansions: 
	    gpu->create_buffer(gpu->cl_bufs + CLBUF_MPOLES, gpu->NbCells*sizeof(Cell::srce_data), (void **) &(CELL_SRCE), CQ_M2L);
	    // OpenCL buffer for local expansions: 
	    gpu->create_buffer(gpu->cl_bufs + CLBUF_LOCAL_COEFS, gpu->NbCells*sizeof(falcON::grav::Cset), (void **) &(COEF_ACPN), CQ_M2L);
	    
	    // InterBufs buffers:
	    if (P2P_INTERBUF_BS % gpu->P2P_wg_size != 0){ 
	      std::cerr << "Error: P2P_INTERBUF_BS must be a multiple of P2P_wg_size." << std::endl; exit(EXIT_FAILURE); 
	    }
	    if (M2L_INTERBUF_BS % gpu->M2L_wg_size != 0){ 
	      std::cerr << "Error: M2L_INTERBUF_BS must be a multiple of M2L_wg_size." << std::endl; exit(EXIT_FAILURE); 
	    }
	    gpu->M2L_globalMaxBlockNb        = gpu->buf_size/M2L_INTERBUF_BS; // greatest multiple of M2L_INTERBUF_BS <= buf_size 
	    gpu->P2P_globalMaxBlockNb        = gpu->buf_size/P2P_INTERBUF_BS; // greatest multiple of P2P_INTERBUF_BS <= buf_size  
	    gpu->P2PLeafTgt_globalMaxBlockNb = gpu->buf_size/P2P_INTERBUF_BS; // greatest multiple of P2P_INTERBUF_BS <= buf_size  

	    gpu->InterBufs = new InterBufs_t[gpu->num_threads]; 
#ifdef pfalcON
#pragma omp parallel for schedule(static,1)
#endif
	    for (int t=0; t<gpu->num_threads; t++){
	      InterBufs_t *p_t = &(gpu->InterBufs[t]);

	      /// M2L InterBufs: 
	      gpu->create_buffer(&(p_t->M2L_interBuf_clBuf), gpu->buf_size*sizeof(int), (void **) &(p_t->M2L_interBuf), CQ_M2L);
	      gpu->create_buffer(&(p_t->M2L_nomutual_indexing_clBuf), gpu->M2L_globalMaxBlockNb*sizeof(int), (void **) &(p_t->M2L_nomutual_indexing), CQ_M2L);
	      gpu->create_buffer(&(p_t->M2L_nomutual_indexing_start_clBuf), 2*gpu->NbCells*sizeof(int), (void **) &(p_t->M2L_nomutual_indexing_start), CQ_M2L);
	      p_t->M2L_currentCellBlockNb = new int[gpu->NbCells];
	      clear_M2L_buffers(t);

	      /// P2P InterBufs: 
	      gpu->create_buffer(&(p_t->P2P_interBuf_clBuf), gpu->buf_size*sizeof(int), (void **) &(p_t->P2P_interBuf), CQ_P2P);
	      gpu->create_buffer(&(p_t->P2P_nomutual_indexing_clBuf), gpu->P2P_globalMaxBlockNb*sizeof(int), (void **) &(p_t->P2P_nomutual_indexing), CQ_P2P);
	      gpu->create_buffer(&(p_t->P2P_nomutual_indexing_start_clBuf), 2*gpu->NbCells*sizeof(int), (void **) &(p_t->P2P_nomutual_indexing_start), CQ_P2P);
	      p_t->P2P_currentBlockNb = new int[gpu->NbCells];
	      clear_P2P_buffers(t);

	      /// P2PLeafTgt InterBufs: 
	      gpu->create_buffer(&(p_t->P2PLeafTgt_interBuf_clBuf), gpu->buf_size*sizeof(int), (void **) &(p_t->P2PLeafTgt_interBuf), CQ_P2PLEAFTGT);
	      gpu->create_buffer(&(p_t->P2PLeafTgt_nomutual_indexing_clBuf), gpu->P2PLeafTgt_globalMaxBlockNb*sizeof(int), (void **) &(p_t->P2PLeafTgt_nomutual_indexing), CQ_P2PLEAFTGT);
	      gpu->create_buffer(&(p_t->P2PLeafTgt_nomutual_indexing_start_clBuf), 2*gpu->NbLeafs*sizeof(int), (void **) &(p_t->P2PLeafTgt_nomutual_indexing_start), CQ_P2PLEAFTGT);
	      p_t->P2PLeafTgt_currentBlockNb = new int[gpu->NbLeafs];
	      clear_P2PLeafTgt_buffers(t);

	    } // for t 

#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
	    time_clear_M2L_buffers=0; // reset 
	    time_clear_P2P_buffers=0; // reset 
	    time_clear_P2PLeafTgt_buffers=0; // reset 
#endif
	    if (gpu->use_Intel_iGPU){ 
	      // "Warm-up" kernel run for Intel iGPU (required for correct timing measurements 
	      // of following kernel executions):  
	      // (Rq: done here so that the OpenCL buffers are already map. Indeed for the "warm-up" kernels to be effective, 
	      //  it seems that the Intel SDK identifies the kernels based on the first args (which must hence be the same ...).)
	      gpu->warmup_run();
	    }
#else    
	    CELL_SRCE=falcON_NEW(Cell::srce_data,NCT);   //     allocate memory
#endif
        }                                              //   ENDIF
#ifdef iGPU
	unsigned i = 0;
#endif
        Cell::srce_data*ci=CELL_SRCE;                  //   pter to cell's source
        LoopCellsDown(grav::cell_iter,TREE,Ci) {       //   LOOP cells
#ifdef iGPU
	  Ci->ID1 = i;
	  i++;  
#endif

	  Ci->set_srce(ci++);                          //     give memory to cell
	  Ci->resetCoeffs();                           //     reset cell: Coeffs
        }                                              //   END LOOP
#if defined(pfalcON) && (! defined(iGPU))
	// Buffer for local expansions:
	COEF_ACPN = new falcON::grav::Cset[NCT]; 
#endif 
        //    - pass source properties up the tree, count active cells
        NCA = pass_up(MAC,TREE->is_re_used());         //   pass source data up tree
        if(debug(11)) {
            std::ofstream dump;
            dump.open("/tmp/leafs");
            TREE->dump_leafs<Leaf>(dump);
            dump.open("/tmp/cells");
            TREE->dump_cells<Cell>(dump);
            if(debug(11)) DebugInfo("GravEstimator::prepare(): "
                                    "leafs dumped to file \"/tmp/leafs\" "
                                    "and cells to file \"/tmp/cells\"\n");
        }
        CELLS_UPTODATE = 1;                            //   update up-to-date flag
    } else {                                         // ELSE
        Cell::srce_data*ci=CELL_SRCE;                  //   pter to cell's source
        LoopCellsDown(grav::cell_iter,TREE,Ci)         //   LOOP cells
        Ci->set_srce(ci++);                          //     give memory to cell
    }                                                // ENDIF
#ifdef iGPU
    unmap_PotAcc();
    unmap_MPoles();
    unmap_Local_Coefs();
    // remapping CELLS & LEAFS as read-only for incoming approx(): 
    globalTree->unmap_cells();
    globalTree->map_cells(CL_MAP_READ);
    globalTree->unmap_leafs();
    globalTree->map_leafs(CL_MAP_READ);
#endif
    return all;                                      // return all
}
//------------------------------------------------------------------------------
void GravEstimator::exact(bool       al
#ifdef falcON_ADAP
                          ,real       Nsoft,
                          unsigned   Nref,
                          real       emin,
                          real       efac
#endif
)
{
    if(GRAV==zero) {
        falcON_Warning("GravEstimator::exact(): G=0\n");
        if(al) ResetBodiesGrav<1>(TREE->my_bodies());
        else   ResetBodiesGrav<0>(TREE->my_bodies());
        return;
    }
   
    update_leafs();
#ifdef falcON_ADAP
    adjust_eph(al,Nsoft,emin,EPS,Nref,efac);
#endif
    const bool all = prepare(0,al);
    if(N_active_cells()==0)
        return falcON_Warning("GravEstimator::exact(): nobody active");
    STATS->reset(
#ifdef WRITE_IACTION_INFO
                 TREE
#endif
                 );
    if(TREE->my_bodies()->N_bodies(bodytype::sink) && EPSSINK != EPS)
        falcON_Warning("GravEstimator::exact(): will ignore eps_sink\n");
    if(all) {
        GravIactAll K(KERNEL,STATS,EPS,0,INDI_SOFT);
        K.direct_summation(root());
        LoopLeafs(Leaf,TREE,Li) Li->normalize_grav();
    } else {
        GravIact K(KERNEL,STATS,EPS,0,INDI_SOFT);
        K.direct_summation(root());
        LoopLeafs(Leaf,TREE,Li) if(is_active(Li)) Li->normalize_grav();
    }
#ifdef falcON_ADAP
    if(all) UpdateBodiesGrav<1>(TREE,GRAV,INDI_SOFT && Nsoft);
    else    UpdateBodiesGrav<0>(TREE,GRAV,INDI_SOFT && Nsoft);
#else
    if(all) UpdateBodiesGrav<1>(TREE,GRAV);
    else    UpdateBodiesGrav<0>(TREE,GRAV);
#endif
    TREE->mark_grav_usage();
}
//------------------------------------------------------------------------------
void GravEstimator::approx(const GravMAC*GMAC,
                           bool          al
#ifdef falcON_ADAP
                           ,
                           real          Nsoft,
                           unsigned      Nref,
                           real          emin,
                           real          efac
#endif
)
{
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
  double t_Inter = 0.0, t_Eval = 0.0, t_Prepare = 0.0, t_MemRelease = 0.0;
  double t_start;
#endif 
#ifdef profile
  size_t ncrit = gpu->get_ncrit();
  printf("ncrit %d\n",ncrit);
  for(int i = 0; i< 1024; i++){
    n_par_cell[i] = 0;
  }
#endif
  
    if(GRAV==zero) {
      falcON_Warning("[GravEstimator::approx()]: G=0\n");
      if(al) ResetBodiesGrav<1>(TREE->my_bodies());
      else   ResetBodiesGrav<0>(TREE->my_bodies());
      return;
    }
    SET_I
    report REPORT("GravEstimator::approx()");
    // update leafs' source fields (mass, flags, epsh) & count active
    update_leafs();
    SET_T(" time: GravEstimator::update_leafs():  ");
#ifdef falcON_ADAP
    // adjust epsh of leafs
    adjust_eph(al,Nsoft,emin,EPS,Nref,efac);
    SET_T(" time: GravEstimator::adjust_eph():    ");
#endif
    // prepare tree: allocate memory (leafs & cells), pass up source, count active
    t_start = my_gettimeofday(); 
    const bool all = prepare(GMAC,al);
    t_Prepare += (my_gettimeofday() - t_start); 
    if(!all && N_active_cells()==0)
        return falcON_Warning("[GravEstimator::approx()]: nobody active");
    SET_T(" time: GravEstimator::prepare():       ");
    report REPORT2("interaction & evaluation");
    STATS->reset(
#ifdef WRITE_IACTION_INFO
                 TREE
#endif
                 );
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
    t_start_DTT =  my_gettimeofday(); 
#endif

    Ncsize = 4+(all? TREE->N_cells() : N_active_cells())/16;    
    if(all) {                                        // IF all are active
#ifdef recursive
      GravIactAll GK(KERNEL,STATS,EPS,Ncsize,INDI_SOFT,DIR);
      //   init gravity kernel
      MutualInteractor<GravIactAll> MI(&GK,TREE->depth()-1);
      MI.rooot = root();
        
      //   init mutual interactor
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
      t_start = my_gettimeofday();
#endif 
#ifdef iGPU 
      EQ = GK.EQ; 
#endif 
#ifdef pfalcON
#ifdef pfalcON_useTBB
      MI.split_cell_self2(root());
      MI.g.wait();
#else // #ifdef pfalcON_useTBB
	//OpenMP
#pragma omp parallel
      {
#pragma omp single nowait
	{
	  MI.split_cell_self2(root());
	}
#pragma omp barrier
#ifdef iGPU
	int thread_num = omp_get_thread_num();
	InterBufs_t *p_t = &(gpu->InterBufs[thread_num]);
#ifdef iGPU_P2P
	if (p_t->P2P_nomutual_indexing_start_ind > 0) run_kernel_P2P(thread_num);	  
	if (p_t->P2PLeafTgt_nomutual_indexing_start_ind > 0) run_kernel_P2PLeafTgt(thread_num); 
#endif 
#ifdef iGPU_M2L
	if (p_t->M2L_nomutual_indexing_start_ind > 0) run_kernel_M2L(thread_num);	  
#endif 
#endif
      } // omp parallel 
      
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
      printf("P2P %lf \n", time_P2P);
      printf("P2PLeafTgt %lf \n", time_P2PLeafTgt);
      printf("M2L %lf \n" , time_M2L);
      printf(" nbM2L %d nb P2P %d nb P2PLeafTgt %d\n", nb_M2L, nb_P2P, nb_P2PLeafTgt);	 
#endif
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
      printf("time_clear_M2L_buffers %lf \n", time_clear_M2L_buffers);  
      printf("time_clear_P2P_buffers %lf \n", time_clear_P2P_buffers); 
      printf("time_clear_P2PLeafTgt_buffers %lf \n", time_clear_P2PLeafTgt_buffers); 
#endif
#if INFO_DISPLAY_LEVEL >= GFLOPS_DISPLAY_LEVEL
      printf("### P2P interaction count = %lld \t P2P Gflop/s = %g \n", 
	     interaction_nb, (((double) interaction_nb * FLOPS_PER_INTERACTION) / time_P2P) / 1e9); 
#endif 
#ifdef profile
      printf("nb_P2P_t %d",nb_P2P_t);
      printf("nb_P2M_t %d",nb_P2M_t);
      
      for(int i =0; i< 2048; i++){
	printf(" nb particle %d ncell %d\n", i+1,n_par_cell[i]);
      }
#endif
	
#endif // #ifdef pfalcON_useTBB
#else  // #ifdef pfalcON
	// recursive & sequential (no task) 
	MI.split_cell_self2_std(root());
#ifdef iGPU
	int thread_num = 0;
	InterBufs_t *p_t = &(gpu->InterBufs[thread_num]);
#ifdef iGPU_P2P
	if (p_t->P2P_nomutual_indexing_start_ind > 0) run_kernel_P2P(thread_num);	  
	if (p_t->P2PLeafTgt_nomutual_indexing_start_ind > 0) run_kernel_P2PLeafTgt(thread_num);
#endif
#ifdef iGPU_M2L 
	if (p_t->M2L_nomutual_indexing_start_ind > 0) run_kernel_M2L(thread_num);	  
#endif 
#if INFO_DISPLAY_LEVEL >= KERNEL_DISPLAY_LEVEL
	printf("P2P %lf \n", time_P2P);
	printf("P2PLeafTgt %lf \n", time_P2PLeafTgt);
	printf("M2L %lf \n" , time_M2L);
	printf(" nbM2L %d nb P2P %d nb P2PLeafTgt %d\n", nb_M2L, nb_P2P, nb_P2PLeafTgt);	 
#endif
#if INFO_DISPLAY_LEVEL >= CLEAR_KERNEL_DISPLAY_LEVEL
	printf("time_clear_M2L_buffers %lf \n", time_clear_M2L_buffers); 
	printf("time_clear_P2P_buffers %lf \n", time_clear_P2P_buffers);   
	printf("time_clear_P2PLeafTgt_buffers %lf \n", time_clear_P2PLeafTgt_buffers); 
#endif
#endif

	 
#ifdef profile
	 printf("nb_P2P_t %d",nb_P2P_t);
	 printf("nb_P2M_t %d",nb_P2M_t);
	 
	 for(int i =0; i< 1024; i++){	      
	   printf(" nb particle %d ncell %d\n", i+1,n_par_cell[i]);
	 }
#endif
#endif // #ifdef pfalcON

#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
        t_Inter += (my_gettimeofday() - t_start);
        t_start = my_gettimeofday();
#endif 

#ifdef pfalcON
#ifdef pfalcON_useTBB
        LoopCellKids(cell_iter,MI.rooot,c1)
        {   //   LOOP cell kids c1
            //     END LOOP
            GK.evaluate(c1);                             //     evaluation phase
        }                                              //   END LOOP
#else // ifdef pfalcON_useTBB
	//OpenMP
#pragma omp parallel// default(shared)
#pragma omp single nowait
	{
	  LoopCellKids(cell_iter,MI.rooot,c1)
	    { //   LOOP cell kids c1
	      //     END LOOP
	      if(number(c1) > tct){
#pragma omp task 
		GK.evaluate(c1);                             //     evaluation phase
	      }
	      else {
		GK.evaluate(c1);                             //     evaluation phase
	      }
	    }
	}
	// implicit barrier at the end of parallel region 
#endif // ifdef pfalcON_useTBB
	t_Eval += (my_gettimeofday() - t_start);
	t_start = my_gettimeofday();
	// Releasing memory for local expansions: 
	LoopCellsDown(grav::cell_iter,TREE,Ci) {       
	  GK.take_coeffs(Ci);  
	}
	delete [] COEF_ACPN; 
	t_MemRelease += (my_gettimeofday() - t_start);
	t_start = my_gettimeofday(); 
#else  // #ifdef pfalcON
        LoopCellKids(cell_iter,MI.rooot,c1)
	  {            //   LOOP cell kids c1
            //     END LOOP
            GK.evaluate(c1);                             //     evaluation phase
	  }                                              //   END LOOP
#endif //#ifdef pfalcON
	LoopLeafKids(cell_iter,root(),s1) {            //   LOOP leaf kids
	  s1->normalize_grav();                        //     evaluation phase
	}                                              //   END LOOP
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
        t_Eval += (my_gettimeofday() - t_start);
#endif 
	Ncoeffs = GK.coeffs_used();                    //   remember # coeffs used
	Nchunks = GK.chunks_used();                    //   remember # chunks used


#else // #ifdef recursive
        GravIactAll GK(KERNEL,STATS,EPS,Ncsize,INDI_SOFT,DIR);
        //   init gravity kernel
        MutualInteractor<GravIactAll> MI(&GK,TREE->depth()-1);
        //   init mutual interactor
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
        t_start = my_gettimeofday();
#endif
        LoopCellKids(cell_iter,root(),c1) {            //   LOOP cell kids c1
            MI.cell_self(c1);                            //     self-iaction c1
            LoopCellSecd(cell_iter,root(),c1+1,c2)       //     LOOP kids c2>c1
            MI.cell_cell(c1,c2);                       //       interaction c1,2
            LoopLeafKids(cell_iter,root(),s2) {          //     LOOP leaf kids s2
                if(is_sink(s2)) {                          //       IF s2 is sink
                    GK.set_sink(EPSSINK,FSINK);              //         switch to sink
                    // 	  GK.direct_summation(c1,s2);              //         interaction c1,s2
                    MI.cell_leaf(c1,s2);                     //         interaction c1,s2
                    GK.unset_sink(EPS);                      //         switch back
                } else                                     //       ELSE
                    MI.cell_leaf(c1,s2);                     //         interaction c1,s2
            }
            //     END LOOP
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
            t_Inter += (my_gettimeofday() - t_start);
            t_start = my_gettimeofday();
#endif
            GK.evaluate(c1);                             //     evaluation phase
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
            t_Eval += (my_gettimeofday() - t_start);
            t_start = my_gettimeofday();
#endif
        }                                              //   END LOOP
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
        t_start = my_gettimeofday();
#endif
        LoopLeafKids(cell_iter,root(),s1) {            //   LOOP leaf kids s1
            if(is_sink(s1)) {                            //     IF s1 is sink
                GK.set_sink(EPSSINK,FSINK);                //       switch to sink
                LoopLeafSecd(cell_iter,root(),s1+1,s2)     //       LOOP kids s2>s1
                GK.interact(s1,s2);                      //         interaction s1,s2
                GK.unset_sink(EPS);                        //       switch back
            } else {                                     //     ELSE
                LoopLeafSecd(cell_iter,root(),s1+1,s2)     //       LOOP kids s2>s1
                if(is_sink(s2)) {                          //       IF s2 is sink
                    GK.set_sink(EPSSINK,FSINK);              //         switch to sink
                    GK.interact(s1,s2);                      //         interaction s1,s2
                    GK.unset_sink(EPS);                      //         switch back
                } else                                     //       ELSE
                    GK.interact(s1,s2);                      //         interaction s1,s2
            }                                            //     ENDIF
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL            
            t_Inter += (my_gettimeofday() - t_start);
            t_start = my_gettimeofday();
#endif
            s1->normalize_grav();                        //     evaluation phase
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
            t_Eval += (my_gettimeofday() - t_start);
            t_start = my_gettimeofday();
#endif
        }                                              //   END LOOP
        Ncoeffs = GK.coeffs_used();                    //   remember # coeffs used
        Nchunks = GK.chunks_used();                    //   remember # chunks used
#endif // #ifdef recursive
    } else {                                         // ELSE: not all are active
        GravIact GK(KERNEL,STATS,EPS,Ncsize,INDI_SOFT,DIR);
        //   init gravity kernel
        MutualInteractor<GravIact> MI(&GK,TREE->depth()-1);
        //   init mutual interactor
        LoopCellKids(cell_iter,root(),c1) {            //   LOOP cell kids c1
            if(is_active(c1)) {                          //     IF active c1:
                MI.cell_self(c1);                          //       self-iaction c1
                LoopCellSecd(cell_iter,root(),c1+1,c2)     //       LOOP kids c2>c1
                MI.cell_cell(c1,c2);                     //         interaction c1,2
                LoopLeafKids(cell_iter,root(),s2) {        //       LOOP leaf kids s
                    if(is_sink(s2)) {                        //        IF s2 is sink
                        GK.set_sink(EPSSINK,FSINK);            //         switch to sink
                        GK.direct_summation(c1,s2);            //         interact by direct
                        GK.unset_sink(EPS);                    //         switch back
                    } else                                   //        ELSE
                        MI.cell_leaf(c1,s2);                   //         interaction c1,s2
                }                                          //       END LOOP
                GK.evaluate(c1);                           //       evaluation phase
            } else {                                     //     ELSE: inactive c1
                LoopCellSecd(cell_iter,root(),c1+1,c2)     //       LOOP kids c2>c1
                if(is_active(c2)) MI.cell_cell(c1,c2);   //         interaction c1,2
                LoopLeafKids(cell_iter,root(),s2)          //       LOOP leaf kids s
                if(is_active(s2)) {
                    if(is_sink(s2)) {                      //        IF s2 is sink
                        GK.set_sink(EPSSINK,FSINK);          //         switch to sink
                        GK.direct_summation(c1,s2);          //         interact by direct
                        GK.unset_sink(EPS);                  //         switch back
                    } else                                 //        ELSE
                        MI.cell_leaf(c1,s2);                 //         interaction c1,s2
                }                                        //       END LOOP
            }                                            //     ENDIF
        }                                              //   END LOOP
        LoopLeafKids(cell_iter,root(),s1) {            //   LOOP leaf kids s1
            if(is_active(s1)) {                          //     IF active s1:
                LoopLeafSecd(cell_iter,root(),s1+1,s2) {   //       LOOP kids s2>s1
                    if(is_sink(s1) || is_sink(s2)) {         //        IF either is sink
                        GK.set_sink(EPSSINK,FSINK);            //         switch to sink
                        GK.interact(s1,s2);                    //         interaction s1,2
                        GK.unset_sink(EPS);                    //         switch back
                    } else                                   //        ELSE
                        GK.interact(s1,s2);                    //         interaction s1,2
                }                                          //       END LOOP
                s1->normalize_grav();                      //       evaluation phase
            } else {                                     //     ELSE: inactive s1
                LoopLeafSecd(cell_iter,root(),s1+1,s2)     //       LOOP kids s2>s1
                if(is_active(s2)) {                      //         active s2 only
                    if(is_sink(s1) || is_sink(s2)) {       //         IF either is sink
                        GK.set_sink(EPSSINK,FSINK);          //          switch to sink
                        GK.interact(s1,s2);                  //          interaction s1,s2
                        GK.unset_sink(EPS);                  //          switch back
                    } else                                 //         ELSE
                        GK.interact(s1,s2);                  //          interaction s1,s2
                }                                        //       END LOOP
            }                                            //     ENDIF
        }                                              //   END LOOP
        Ncoeffs = GK.coeffs_used();                    //   remember # coeffs used
        Nchunks = GK.chunks_used();                    //   remember # chunks used
    }                                                // ENDIF

#ifdef iGPU
    map_PotAcc();
    map_MPoles();
    map_Local_Coefs(); 
    // remapping CELLS & LEAFS as read-write: 
    globalTree->unmap_cells();
    globalTree->map_cells();
    globalTree->unmap_leafs();
    globalTree->map_leafs();
#endif

    SET_T(" time: interaction & evaluation:        ");
#if INFO_DISPLAY_LEVEL >= DTT_EVAL_DISPLAY_LEVEL
#ifdef iGPU
    std::cerr << " time: pfalcON (elapsed time (s)): "
              << " Prepare = " << t_Prepare 
	      << " & Inter = " << t_Inter
	      << " & Eval = "  << t_Eval
	      << " & MemRelease = " <<  t_MemRelease
	      << " Inter+Eval = " << t_Inter + t_Eval
	      << " Global = " << my_gettimeofday() - t_start_DTT 
	      <<std::endl;
#else
    std::cerr << " time: pfalcON (elapsed time (s)): "
              << " Prepare = " << t_Prepare 
	      << " & Inter = " << t_Inter
	      << " & Eval = "  << t_Eval
	      << " & MemRelease = " <<  t_MemRelease
	      << " Global = " << my_gettimeofday() - t_start_DTT 
	      << std::endl;
              
#endif    
#endif 

#ifdef falcON_ADAP
    if(all) UpdateBodiesGrav<1>(TREE,GRAV,INDI_SOFT && Nsoft);
    else    UpdateBodiesGrav<0>(TREE,GRAV,INDI_SOFT && Nsoft);
#else
    if(all) UpdateBodiesGrav<1>(TREE,GRAV);
    else    UpdateBodiesGrav<0>(TREE,GRAV);
#endif
    TREE->mark_grav_usage();
    SET_T(" time: updating bodies gravity:         ");
}
//------------------------------------------------------------------------------
namespace {
    using namespace falcON;
    //============================================================================
    unsigned NX;
    real pdim(real const&x) { return cube(x); }
    //----------------------------------------------------------------------------
    struct number_density {
        static real dens(grav::cell_iter const&C)
        { return number(C)/(Nsub*pdim(radius(C))); }
    };
    //----------------------------------------------------------------------------
    struct surface_density {
        static real dens(grav::cell_iter const&C)
        { return mass(C)/(4*square(radius(C))); }
    };
    //----------------------------------------------------------------------------
    struct mass_density {
        static real dens(grav::cell_iter const&C)
        { return mass(C)/(Nsub*pdim(radius(C))); }
    };
    //============================================================================
    template<typename, bool> class guess;
    //----------------------------------------------------------------------------
    template<typename density> class guess<density,1> {
    public:
        static void do_it(cell_iter const&C, real d) {
            if(number(C)>NX || d==zero) d = density::dens(C);
            LoopLeafKids(grav::cell_iter,C,l) l->rho() = d;
            LoopCellKids(grav::cell_iter,C,c) do_it(c,d);
        }
    };
    //----------------------------------------------------------------------------
    template<typename density> class guess<density,0> {
    public:
        static void do_it(cell_iter const&C, real d) {
            if(number(C)>NX || d==zero) d = density::dens(C);
            LoopLeafKids(grav::cell_iter,C,l)
            if(is_active(l)) l->rho() = d;
            LoopCellKids(grav::cell_iter,C,c)
            if     (al_active(c)) guess<density,1>::do_it(c,d);
            else if(is_active(c))                   do_it(c,d);
        }
    };
    //----------------------------------------------------------------------------
    void UpdateBodiesRho(const OctTree*T,
                         bool          all)
    {
        if(all)
            LoopLeafs(grav::leaf,T,Li)
            Li->copy_to_bodies_rho(T->my_bodies());
        else
            LoopLeafs(grav::leaf,T,Li) if(is_active(Li))
                Li->copy_to_bodies_rho(T->my_bodies());
    }
}                                                  // END: unnamed namespace
//------------------------------------------------------------------------------
void GravEstimator::estimate_nd(bool al, unsigned Nx) const
{
    NX = Nx;
    if(al) guess<number_density,1>::do_it(root(),zero);
    else   guess<number_density,0>::do_it(root(),zero);
    UpdateBodiesRho(TREE,al);
}
//------------------------------------------------------------------------------
void GravEstimator::estimate_sd(bool al, unsigned Nx)
{
    update_leafs();
    prepare(0,al);
    NX = Nx;
    if(al) guess<surface_density,1>::do_it(root(),zero);
    else   guess<surface_density,0>::do_it(root(),zero);
    UpdateBodiesRho(TREE,al);
    TREE->mark_grav_usage();
}
//------------------------------------------------------------------------------
void GravEstimator::estimate_md(bool al, unsigned Nx)
{
    update_leafs();
    prepare(0,al);
    NX = Nx;
    if(al) guess<mass_density,1>::do_it(root(),zero);
    else   guess<mass_density,0>::do_it(root(),zero);
    UpdateBodiesRho(TREE,al);
    TREE->mark_grav_usage();
}
//------------------------------------------------------------------------------
void GravEstimator::dump_cells(std::ostream&o) const
{
    if(CELL_SRCE) TREE->dump_cells<Cell>(o);
    else          TREE->dump_cells<OctTree::Cell>(o);
}
//------------------------------------------------------------------------------
void GravEstimator::dump_leafs(std::ostream&o) const
{
    TREE->dump_leafs<Leaf>(o);
}
////////////////////////////////////////////////////////////////////////////////
