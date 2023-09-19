
#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define MY_RSQRT(x) rsqrt(x)
//#define MY_RSQRT(x) native_rsqrt(x)
//#define MY_RSQRT(x) half_rsqrt(x)

// for scalar value (mass) and position vector: 
typedef float4 ScalPos_t; 
#define SCAL(a) ((a).x)
#define POSX(a) ((a).y)
#define POSY(a) ((a).z)
#define POSZ(a) ((a).w)

typedef struct 
{
  ScalPos_t scal_pos;  
  unsigned int id;
  int padding[7];
} pfalcONstruct;

// for potential and acceleration/forces:  
typedef float4 PotAcc_t; 
#define POT(a) ((a).x)
#define FX(a)  ((a).y)
#define FY(a)  ((a).z)
#define FZ(a)  ((a).w)
#define NB_FIELDS_IN_POTACC_T 4

// See corresponding host class Cell in tree.h 
typedef struct
{
  int lock;
  int shifting1[2]; // for NLEAFS and NCELLS
  unsigned NUMBER;  // # leaf descendants
  unsigned FCLEAF;  // index of fst leaf desc
  int shifting2[5]; // for FCCELL, PACELL and CENTRE
  float POS[3];
  float RAD;
  int padding[4];
  int ID1;
  int ID2;
} cellstruct;


typedef struct
{
  float MASS;
  float EPSH;
  float POLS[6];

} srce_data;

#define LOCAL_EXP_SIZE 20
typedef struct
{
  float coeff[LOCAL_EXP_SIZE];
} coeffstruct;


#define LEAF_FLAG (1 << 30)

///// Read single value for all wi from global mem using local mem 
///// (from AMD_OpenCL_Programming_Optimization_Guide2.pdf (Aout 2015, sect. 2.1.1.2))
///// -> no perf gain 
/* #define BCAST_GMEM(wi_id, reg, g_mem, l_mem) {	\ */
/*   if (wi_id == 0) { l_mem = g_mem; }		        \ */
/*   barrier(CLK_LOCAL_MEM_FENCE);			\ */
/*   reg = l_mem;					\ */
/* } */
///// Standard version (directly read from global memory): 
#define BCAST_GMEM(wi_id, reg, g_mem) {	\
    reg = g_mem;			\
  }
///// With work_group_broadcast(): 
///// -> slower in practice (possibly due to the barrier) 
/* #define BCAST_GMEM(wi_id, reg, g_mem) {	    \ */
/*     if (wi_id == 0){ reg = g_mem; }		    \ */
/*     reg = work_group_broadcast(reg, 0 /\* root *\/);  \ */
/*   } */
  

__kernel void check_struct(__global unsigned long ptr_sizes[], __global int local_memory_size[]) {
  ptr_sizes[0] = (unsigned long) (&(((pfalcONstruct*)(0))->scal_pos));
  ptr_sizes[1] = (unsigned long) (&(((pfalcONstruct*)(0))->id));
  ptr_sizes[2] = (unsigned long) (&(((pfalcONstruct*)(0))->padding[5]));
  ptr_sizes[3] = sizeof(pfalcONstruct);
  local_memory_size[0] = (int) sizeof(ScalPos_t);
}





////////////////////////////////////////////////////////////////////////////////
/// P2P 

/// Compute both potential and forces with 23 flops (considering
/// 2 flops for rsqrt(): 1 flop for sqrt and 1 for div).
#define LOAD_AND_COMPUTE(J, contrib, tgt_scal_pos) {			\
    POSX(src_scal_pos_##J) = POSX(localBodySrc[J]);			\
    rx_##J = POSX(src_scal_pos_##J) - POSX(tgt_scal_pos);		\
    POSY(src_scal_pos_##J) = POSY(localBodySrc[J]);			\
    ry_##J = POSY(src_scal_pos_##J) - POSY(tgt_scal_pos);		\
    POSZ(src_scal_pos_##J) = POSZ(localBodySrc[J]);			\
    rz_##J = POSZ(src_scal_pos_##J) - POSZ(tgt_scal_pos);		\
    distSqr_##J = rx_##J * rx_##J + ry_##J * ry_##J + rz_##J * rz_##J;  \
    distSqr_##J += softeningSquared;					\
    SCAL(src_scal_pos_##J) =   SCAL(localBodySrc[J]);			\
    invDist_##J = MY_RSQRT((float)distSqr_##J);				\
    invDistCube_##J =  invDist_##J * invDist_##J * invDist_##J;	        \
    s_##J = SCAL(src_scal_pos_##J) * SCAL(tgt_scal_pos) * invDistCube_##J; \
    POT(contrib) -= s_##J  * distSqr_##J;				\
    FX(contrib)  += rx_##J * s_##J;					\
    FY(contrib)  += ry_##J * s_##J;					\
    FZ(contrib)  += rz_##J * s_##J;					\
}

// Compute both potential and forces with 23 flops (considering
/// 2 flops for rsqrt(): 1 flop for sqrt and 1 for div).
#define LOAD_OWN_AND_COMPUTE(J, contrib, tgt_scal_pos, tgt_ind) {	\
    POSX(src_scal_pos_##J) = POSX(localBodySrc[J]);			\
    rx_##J = POSX(src_scal_pos_##J) - POSX(tgt_scal_pos);		\
    POSY(src_scal_pos_##J) = POSY(localBodySrc[J]);			\
    ry_##J = POSY(src_scal_pos_##J) - POSY(tgt_scal_pos);		\
    POSZ(src_scal_pos_##J) = POSZ(localBodySrc[J]);			\
    rz_##J = POSZ(src_scal_pos_##J) - POSZ(tgt_scal_pos);		\
    distSqr_##J = rx_##J * rx_##J + ry_##J * ry_##J + rz_##J * rz_##J;  \
    distSqr_##J += softeningSquared;					\
    SCAL(src_scal_pos_##J) = (tgt_ind != J ? SCAL(localBodySrc[J]) : 0.0f); \
    invDist_##J = MY_RSQRT((float)distSqr_##J);				\
    invDistCube_##J =  invDist_##J * invDist_##J * invDist_##J;	        \
    s_##J = SCAL(src_scal_pos_##J) * SCAL(tgt_scal_pos) * invDistCube_##J; \
    POT(contrib) -= s_##J  * distSqr_##J;				\
    FX(contrib)  += rx_##J * s_##J;					\
    FY(contrib)  += ry_##J * s_##J;					\
    FZ(contrib)  += rz_##J * s_##J;					\
}






////////////////////////////////////////////////////////////////////////////////

#define LOAD_SRC_BODIES(src_ind, Nsrc) {			\
    if (wi_id < Nsrc){						\
      localBodySrc[wi_id] = positions[src_ind].scal_pos;	\
    } 								\
    barrier(CLK_LOCAL_MEM_FENCE); 				\
  }


#ifdef INTEL_iGPU
//#define UNROLL_FACTOR 1
//#define UNROLL_FACTOR 2
#define UNROLL_FACTOR 4
//#define UNROLL_FACTOR 8
#endif
#ifdef AMD_iGPU
#define UNROLL_FACTOR 4
#endif 


#if (UNROLL_FACTOR == 1)
#define LOOP(count_end,contrib,tgt_scal_pos) {		\
      for (; j1 < count_end; j1++){			\
	LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos);	\
      }							\
  }
#define LOOP_OWN(count_end,contrib,tgt_scal_pos,tgt_ind) {		\
      for (; j1 < count_end; j1++){					\
        LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind);		\
      }								        \
  } 
#endif // #if (UNROLL_FACTOR == 1)
#if (UNROLL_FACTOR == 2)
#define LOOP(count_end,contrib,tgt_scal_pos) {			\
    for (; j1 < count_end-1; j1++){				\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos);		\
    }								\
    if (j1 < count_end){					\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos);		\
    }								\
  }							
#define LOOP_OWN(count_end,contrib,tgt_scal_pos,tgt_ind) {		\
    for (; j1 < count_end-1; j1++){					\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind);		\
    }									\
    if (j1 < count_end){						\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind);		\
    }									\
  }								       
#endif // #if (UNROLL_FACTOR == 2)
#if (UNROLL_FACTOR == 4)
#define LOOP(count_end,contrib,tgt_scal_pos) {			\
    for (; j1 < count_end-3; j1++){				\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); 		\
    }								\
    for (; j1 < count_end; j1++){				\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos);		\
    }								\
  }							
#define LOOP_OWN(count_end,contrib,tgt_scal_pos,tgt_ind) {		\
    for (; j1 < count_end-3; j1++){					\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind);		\
    }									\
    for (; j1 < count_end; j1++){					\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind);		\
    }									\
  }								       
#endif // #if (UNROLL_FACTOR == 4)
#if (UNROLL_FACTOR == 8)
#define LOOP(count_end,contrib,tgt_scal_pos) {			\
    for (; j1 < count_end-7; j1++){				\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos); j1++;		\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos);		\
    }								\
    for (; j1 < count_end; j1++){				\
      LOAD_AND_COMPUTE(j1,contrib,tgt_scal_pos);		\
    }								\
  }							
#define LOOP_OWN(count_end,contrib,tgt_scal_pos,tgt_ind) {		\
    for (; j1 < count_end-7; j1++){					\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind); j1++;	\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind);		\
    }									\
    for (; j1 < count_end; j1++){					\
      LOAD_OWN_AND_COMPUTE(j1,contrib,tgt_scal_pos,tgt_ind);		\
    }									\
  }								       
#endif // #if (UNROLL_FACTOR == 8)



#if P2P_WG_SIZE_X < NCRIT // no multi-work-item 
#define CALC(Nsrc,Ntgt,contrib,index,index_N,tgt_scal_pos) { \
    if (index < Ntgt){						\
      int j1 = 0;	/* 'unsigned' faster than 'int' ... */	\
      ScalPos_t src_scal_pos_j1;				\
      float s_j1, rx_j1, ry_j1, rz_j1;				\
      float distSqr_j1, invDist_j1, invDistCube_j1;		\
      LOOP(Nsrc,contrib,tgt_scal_pos);				\
     }								\
  } 
#else // #if P2P_WG_SIZE_X < NCRIT 
#define CALC(Nsrc,Ntgt,contrib,index,index_N,tgt_scal_pos) { \
    if (index < Ntgt*multi_WI_dim){		      		\
      int j1 = 0;	/* 'unsigned' faster than 'int' ... */	\
      int count_end;						\
      ScalPos_t src_scal_pos_j1;				\
      float s_j1, rx_j1, ry_j1, rz_j1;				\
      float distSqr_j1, invDist_j1, invDistCube_j1;		\
      /* Balanced distribution the Nsrc bodies over the */	\
      /* 'multi_WI_dim' work-items: */				\
      int NsrcQ = Nsrc/multi_WI_dim;				\
      int NsrcR = Nsrc%multi_WI_dim;				\
      if (index_N < NsrcR){					\
	j1 = (NsrcQ+1)*index_N;					\
	count_end = j1+(NsrcQ+1);				\
      }								\
      else {							\
	j1 = (NsrcQ+1)*NsrcR + NsrcQ*(index_N-NsrcR);		\
	count_end = j1+NsrcQ;					\
      }								\
      LOOP(count_end,contrib,tgt_scal_pos); 			\
     }								\
  }
#endif // #else P2P_WG_SIZE_X < NCRIT



#if P2P_WG_SIZE_X < NCRIT // no multi-work-item
#define CALC_OWN(Nsrc,Ntgt,contrib,index,index_N,tgt_scal_pos,tgt_ind) { \
    if (index < Ntgt){						\
      int j1 = 0;	/* 'unsigned' faster than 'int' on AMD APU ... */	\
      ScalPos_t src_scal_pos_j1;				\
      float s_j1, rx_j1, ry_j1, rz_j1;				\
      float distSqr_j1, invDist_j1, invDistCube_j1;		\
      LOOP_OWN(Nsrc,contrib,tgt_scal_pos,tgt_ind);		\
    }								\
  }									
#else // #else // #if P2P_WG_SIZE_X < NCRIT 
#define CALC_OWN(Nsrc,Ntgt,contrib,index,index_N,tgt_scal_pos,tgt_ind) { \
    if (index < Ntgt*multi_WI_dim){					\
      int j1 = 0;	/* 'unsigned' faster than 'int' on AMD APU ... */	\
      int  count_end;						\
      ScalPos_t src_scal_pos_j1;				\
      float s_j1, rx_j1, ry_j1, rz_j1;				\
      float distSqr_j1, invDist_j1, invDistCube_j1;		\
      /* Balanced distribution the Nsrc bodies over the */	\
      /* 'multi_WI_dim' work-items: */				\
      int NsrcQ = Nsrc/multi_WI_dim;				\
      int NsrcR = Nsrc%multi_WI_dim;				\
      if (index_N < NsrcR){					\
	j1 = (NsrcQ+1)*index_N;					\
	count_end = j1+(NsrcQ+1);				\
      }								\
      else {							\
	j1 = (NsrcQ+1)*NsrcR + NsrcQ*(index_N-NsrcR);		\
	count_end = j1+NsrcQ;					\
      }								\
      LOOP_OWN(count_end,contrib,tgt_scal_pos,tgt_ind);		\
    }								\
  }									
#endif // #else // #if P2P_WG_SIZE_X < NCRIT 




// Could be improved, but not performance sensitive for large NCRIT values: 
#define MULTI_WI_REDUCTION_CONTRIB0(Ntgt) {				\
  if (multi_WI_dim > 1){						\
    barrier(CLK_LOCAL_MEM_FENCE);					\
    if(index0 < Ntgt*multi_WI_dim){					\
      localBodySrc[wi_id] = contrib0;					\
    }									\
    barrier(CLK_LOCAL_MEM_FENCE);					\
    while (multi_WI_dim >= 4){						\
      if ((multi_WI_dim & 1) && wi_id < Ntgt){				\
	localBodySrc[wi_id] += localBodySrc[wi_id+Ntgt*(multi_WI_dim & (~1))]; \
      }									\
      multi_WI_dim = multi_WI_dim>>1;					\
      if (index0 < Ntgt*multi_WI_dim) {					\
	localBodySrc[wi_id] += localBodySrc[wi_id+Ntgt*multi_WI_dim];	\
      }									\
      barrier(CLK_LOCAL_MEM_FENCE);					\
    } /* while */							\
    if (multi_WI_dim == 3) {						\
      if (wi_id < Ntgt){						\
	contrib0 = localBodySrc[wi_id] + localBodySrc[wi_id+Ntgt] + localBodySrc[wi_id+Ntgt*2]; \
      }									\
    }									\
    else { /* multi_WI_dim == 2 */					\
      if (wi_id < Ntgt) {						\
	contrib0 = localBodySrc[wi_id] + localBodySrc[wi_id+Ntgt];	\
      }									\
    }									\
  }									\
  }
  


__kernel 
__attribute__((reqd_work_group_size(P2P_WG_SIZE_X, 1, 1)))
__attribute__((vec_type_hint(float)))
void process_P2P(float softeningSquared,
		 __global pfalcONstruct const * restrict positions, 
		 __global int const * restrict interBuf,
		 __global  PotAcc_t * restrict ACPN, 
		 __local ScalPos_t * restrict localBodySrc,
		 __global int const * restrict P2P_nomutual_indexing,
		 __global cellstruct const * restrict Cell,
		 __global int const * restrict P2P_nomutual_indexing_start,
		 __local int * restrict B,
		 __local int * restrict NB 
#if P2P_WG_SIZE_X < NCRIT 
		 ,__local ScalPos_t * restrict localBodyTgt,
		 __local PotAcc_t  * restrict localContribs
#endif 
		 ){
  int A, NA;
  int cellA, cellB;
  int ind; // last+1 index for 'index'(= wi local id) OR next block index                                                                                                                                 
  int wg_id = get_group_id(1);
  int currentBlockNb;
  int src_ind; 
  PotAcc_t contrib0; 
  ScalPos_t tgt_scal_pos0;
  int index0;

#ifdef INTEL_iGPU /* "Warm-up" kernel: */
  if (softeningSquared < 0.0){ return; } 
#endif 

#if P2P_WG_SIZE_X < NCRIT 
  int wi_id = get_local_id(0);
#else
  // We have:  wi_id == index0 
#define wi_id index0 
  index0 =  get_local_id(0);     
#endif 
#define P2P_NOMUTUAL_INDEXING_START__CELLA_IND(i) (2*i) 
  BCAST_GMEM(wi_id, cellA, P2P_nomutual_indexing_start[P2P_NOMUTUAL_INDEXING_START__CELLA_IND(wg_id)]);  // target cell index 
  if (cellA & LEAF_FLAG){ /* target is a leaf */
    A = cellA & (~LEAF_FLAG);
    NA = 1; 
  }
  else { /* target is a cell */
    BCAST_GMEM(wi_id, A,  (int) Cell[cellA].FCLEAF);
    BCAST_GMEM(wi_id, NA, (int) Cell[cellA].NUMBER); 
  }
  
#if P2P_WG_SIZE_X >= NCRIT 
  // "multi-work-item" feature generalizing the "multi-threading" feature 
  // from NVIDIA SDK OpenCL 4.1: when P2P_WG_SIZE_X >= 2*NA, 
  // each target body is processed by 'multi-WI_dim' work-items.
  int multi_WI_dim_init = P2P_WG_SIZE_X / NA;
  //  int multi_WI_dim_init = 1; // deactivate multi-work-item
  int multi_WI_dim = multi_WI_dim_init;
#else // #if P2P_WG_SIZE_X >= NCRIT 
  // No multi-treading 
#endif // #else P2P_WG_SIZE_X >= NCRIT 


#if P2P_WG_SIZE_X < NCRIT 
  int NA_roundedUp = ((NA + P2P_WG_SIZE_X - 1) / P2P_WG_SIZE_X) * P2P_WG_SIZE_X;
  //  if (wi_id == 0) { printf("NA=%d P2P_WG_SIZE_X=%d NA_roundedUp=%d \n", NA, P2P_WG_SIZE_X, NA_roundedUp); } 
  // Using local memory to load all tgt bodies: 
  // (Rq: "A bare minimum SLM allocation size is 4k per workgroup" for Intel iGPUs, 
  // from https://software.intel.com/en-us/node/540442) 
  for (index0 = wi_id;
       index0 < NA_roundedUp; // stay in loop while at least one wi has a tgt particle 
       index0 += P2P_WG_SIZE_X){
    if (index0 < NA){
      localBodyTgt[index0] = positions[A + index0].scal_pos;
      localContribs[index0] = (PotAcc_t) (0.0f, 0.0f, 0.0f, 0.0f); 
    }
    // No local memory barrier required 
  } // for index0 

#else // #if P2P_WG_SIZE_X < NCRIT 
  int index_N0 = index0/NA;
  int tgt_ind0;
  // if (wi_id == 0){ printf("### A=%i NA=%i multi_WI_dim_init=%i\n", A, NA, multi_WI_dim_init);}
  if (multi_WI_dim > 1){ 
    // First load all tgt scal_pos in local mem (via localBodySrc[]):
    if (index0 < NA){
      localBodySrc[wi_id] = positions[A + index0].scal_pos;
    }
    barrier(CLK_LOCAL_MEM_FENCE); 
    if (index0 < multi_WI_dim_init*NA){ /* this wi will process a target body */
      tgt_ind0 = index0%NA; 
      tgt_scal_pos0 = localBodySrc[tgt_ind0];
    }
    barrier(CLK_LOCAL_MEM_FENCE); 
  }
  else { // no multi-work-item: 
    if (index0 < NA){
      tgt_ind0 = index0; 
      tgt_scal_pos0 = positions[A + index0].scal_pos;
    }
  }
  contrib0 = (PotAcc_t) (0.0f, 0.0f, 0.0f, 0.0f); 
#endif // #else // #if P2P_WG_SIZE_X < NCRIT 

  BCAST_GMEM(wi_id, currentBlockNb, P2P_nomutual_indexing_start[P2P_NOMUTUAL_INDEXING_START__CELLA_IND(wg_id)+1]); 
  BCAST_GMEM(wi_id, ind, P2P_nomutual_indexing[currentBlockNb]);
  
  bool do_own = false;      
#define NO_FCLEAF ((int) -1)
  while (ind != 0){ 

    // If last block (possibly incomplete), we have: 'ind' <= P2P_INTERBUF_BS
    // If not last block, we ensure: 'ind' > P2P_INTERBUF_BS 
    int stop = MIN(ind, P2P_INTERBUF_BS);  
    for (int shift_id = 0; shift_id < stop; shift_id += P2P_WG_SIZE_X){ 
      int block_id = shift_id + wi_id; 
      if (block_id < ind){ 
	cellB = interBuf[currentBlockNb * P2P_INTERBUF_BS + shift_id + wi_id]; /* src cell index */ 
	if (cellB & LEAF_FLAG){ /* src is a leaf */
	  B[wi_id]  = cellB & (~LEAF_FLAG);
	  NB[wi_id] = 1; 
	}
	else { /* src is a cell */	
	  B[wi_id]  = (int) Cell[cellB].FCLEAF;						
	  NB[wi_id] = (int) Cell[cellB].NUMBER;						
	}
      }
      else {
	B[wi_id]  = NO_FCLEAF;						
      }
      barrier(CLK_LOCAL_MEM_FENCE);      
      
      int nb_src_bodies = 0; 
      int idInBlock=0;
      int B_idInBlock = B[idInBlock]; 
      int NB_idInBlock = NB[idInBlock]; 
      bool new_load = true;
#define LOAD_NEXT() { if (++idInBlock < P2P_WG_SIZE_X){ B_idInBlock = B[idInBlock]; NB_idInBlock = NB[idInBlock]; }} 
      while (idInBlock < P2P_WG_SIZE_X) {
	if (B_idInBlock == NO_FCLEAF){ break; }
	if (B_idInBlock == A){ do_own = true; LOAD_NEXT(); continue; }
	int shift = wi_id - nb_src_bodies; 
	if (0 <= shift && shift < NB_idInBlock){
	  src_ind = B_idInBlock + shift;
	}
	int nb_required = P2P_WG_SIZE_X - nb_src_bodies;
	if (NB_idInBlock > nb_required){
	  B_idInBlock += nb_required; 
	  NB_idInBlock -= nb_required; 
	  nb_src_bodies += nb_required;
	}
	else {
	  nb_src_bodies += NB_idInBlock;
	  LOAD_NEXT(); 
	}
      
	if (nb_src_bodies == P2P_WG_SIZE_X){
	  LOAD_SRC_BODIES(src_ind, P2P_WG_SIZE_X);
#if P2P_WG_SIZE_X >= NCRIT 
	  // at least 1 source body for each of the 
	  // 'multi_WI_dim' wi processing each target body:  
	  multi_WI_dim = MIN(multi_WI_dim_init, P2P_WG_SIZE_X);  
	  CALC(P2P_WG_SIZE_X,NA,contrib0,index0,index_N0,tgt_scal_pos0);
#else
	  for (index0 = wi_id; index0 < NA_roundedUp; index0 += P2P_WG_SIZE_X){
	    if (index0 < NA){	  
	      tgt_scal_pos0 = localBodyTgt[index0];	      
	      contrib0 = (PotAcc_t) (0.0f, 0.0f, 0.0f, 0.0f); 
	    }
	    CALC(P2P_WG_SIZE_X,NA,contrib0,index0,index_N0,tgt_scal_pos0);
	    if (index0 < NA){	  
	      localContribs[index0] += contrib0;	      
	    }
	  } // for index0 
#endif 
	  // To ensure that previous computations are over
	  // and that we can reuse the local memory for the 
	  // next P2P (at the next loop iteration): 
	  barrier(CLK_LOCAL_MEM_FENCE);  
	  nb_src_bodies = 0; // reset nb_src_bodies  
	}
      } // while (idInBlock < P2P_WG_SIZE_X) 
    
      if (nb_src_bodies > 0){ 
	/* No distinction between "nb_src_bodies == P2P_WG_SIZE_X" and "nb_src_bodies < P2P_WG_SIZE_X". */ 
	LOAD_SRC_BODIES(src_ind, nb_src_bodies);
#if P2P_WG_SIZE_X >= NCRIT 
	// at least 1 source body for each of the 
	// 'multi_WI_dim' wi processing each target body:  
	multi_WI_dim = MIN(multi_WI_dim_init, nb_src_bodies);  
	CALC(nb_src_bodies,NA,contrib0,index0,index_N0,tgt_scal_pos0);
#else
	for (index0 = wi_id; index0 < NA_roundedUp; index0 += P2P_WG_SIZE_X){
	  if (index0 < NA){	  
	    tgt_scal_pos0 = localBodyTgt[index0];	      
	    contrib0 = (PotAcc_t) (0.0f, 0.0f, 0.0f, 0.0f); 
	  }
	  CALC(nb_src_bodies,NA,contrib0,index0,index_N0,tgt_scal_pos0);
	  if (index0 < NA){	  
	    localContribs[index0] += contrib0;	      
	  }
	} // for index0 
#endif 
	// To ensure that previous computations are over
	// and that we can reuse the local memory for the 
	// next P2P (at the next loop iteration): 
	barrier(CLK_LOCAL_MEM_FENCE);  
      } // if (nb_src_bodies > 0)

    } // for (shift_id) 

    if (ind <= P2P_INTERBUF_BS){ // last block 
      break; 
    }
    else { // not last block: prepare for next block 
      currentBlockNb = ind - P2P_INTERBUF_BS; 
      BCAST_GMEM(wi_id, ind, P2P_nomutual_indexing[currentBlockNb]);
    }
  } // while (ind != 0){ 
  
#if P2P_WG_SIZE_X >= NCRIT 
  // for CALC_OWN() and for MULTI_WI_REDUCTION(): 
  multi_WI_dim = multi_WI_dim_init;
#endif 
  
  if (do_own){
#if P2P_WG_SIZE_X >= NCRIT 
    if (index0 < NA){
      localBodySrc[index0] = tgt_scal_pos0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    CALC_OWN(NA,NA,contrib0,index0,index_N0,tgt_scal_pos0,tgt_ind0);
#else // #if P2P_WG_SIZE_X >= NCRIT 
    for (index0 = wi_id; index0 < NA_roundedUp; index0 += P2P_WG_SIZE_X){
      bool own_for_block0; 
      int k, k_stop = (NA / P2P_WG_SIZE_X) * P2P_WG_SIZE_X;
      
      if (index0 < NA){	  
	tgt_scal_pos0 = localBodyTgt[index0];	      
	contrib0 = (PotAcc_t) (0.0f, 0.0f, 0.0f, 0.0f); 
      }      
      
      // Complete blocks: 
      for (k=wi_id; k < k_stop; k += P2P_WG_SIZE_X){
	own_for_block0 = false;
	if (k == index0){
	  /* this block will require "own" computations: */
	  own_for_block0 = true; 
	}
	localBodySrc[wi_id] = localBodyTgt[k];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (own_for_block0){
	  CALC_OWN(P2P_WG_SIZE_X,NA,contrib0,index0,index_N0,tgt_scal_pos0,index0);
	}
	else {
	  CALC(P2P_WG_SIZE_X,NA,contrib0,index0,index_N0,tgt_scal_pos0);
	}
      } // for k 
      
      k_stop = NA - k_stop;
      if (k_stop > 0){ 
	// Last (possibly incomplete) block (we rely on the current value of k):
	own_for_block0 = false;
	if (k == index0){
	  /* this block will require "own" computations: */
	  own_for_block0 = true; 
	}
	if (k < NA){ 
	  localBodySrc[wi_id] = localBodyTgt[k];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (own_for_block0){
	  CALC_OWN(k_stop,NA,contrib0,index0,index_N0,tgt_scal_pos0,index0);
	}
	else {
	  CALC(k_stop,NA,contrib0,index0,index_N0,tgt_scal_pos0);
	}
      }

      if (index0 < NA){	  
	localContribs[index0] += contrib0;	      
      }
    } // for index0 
#endif // #else // #if P2P_WG_SIZE_X >= NCRIT 
  } // if (do_own)

#if P2P_WG_SIZE_X >= NCRIT 
  MULTI_WI_REDUCTION_CONTRIB0(NA);
#endif 
  
#if P2P_WG_SIZE_X >= NCRIT 
  if (index0 < NA) {
    int id = positions[A+ index0].id;
    ACPN[id] += contrib0;
  }
#else
  for (index0 = wi_id; index0 < NA_roundedUp; index0 += P2P_WG_SIZE_X){
    if (index0 < NA) {
      int id = positions[A+ index0].id;
      ACPN[id] += localContribs[index0];
    }
  } // for index0    
#endif 


#if P2P_WG_SIZE_X >= NCRIT 
#undef wi_id
#endif 
}//end kernel
































////////////////////////////////////////////////////////////////////////////////
/// M2L 

__kernel 
__attribute__((reqd_work_group_size(M2L_WG_SIZE_X, 1, 1)))
__attribute__((vec_type_hint(float)))
void process_M2L(__global cellstruct const * restrict Cell, 
		 __global srce_data const * restrict M, 
		 float EQ,
		 __global coeffstruct * restrict C, 
		 __global int const * restrict interBuf,
		 __global int const * restrict M2L_nomutual_indexing,
		 __global int const * restrict M2L_nomutual_indexing_start,
		 __local coeffstruct *C_local_mem) {
  int A,B;
  int id1a, id2a;
  int id1b, id2b;
  float D[4];
  float XX;
  float dXRq_x, dXRq_y, dXRq_z, dXRq;
  float dXRq_x2, dXRq_y2, dXRq_z2;
  float a[LOCAL_EXP_SIZE];
  float coeff[4];
  float M_POLS0;
  float M_POLS1;
  float M_POLS2;
  float M_POLS3;
  float M_POLS4;
  float M_POLS5;
  float CellAx, CellAy, CellAz; 
  float CellAm; 

#ifdef INTEL_iGPU /* "Warm-up" kernel: */
  if (EQ < 0.0){ return; } 
#endif 

  int ind; // last+1 index for wi_id OR next block index 
  coeffstruct CellACoeffs = {{0.0f}}; 
  int currentBlockNb;
  int wg_id = get_group_id(1); 
  int wi_id = get_local_id(0); 
#define M2L_NOMUTUAL_INDEXING_START__CELLA_IND(i) (2*i) 
  BCAST_GMEM(wi_id, A, M2L_nomutual_indexing_start[M2L_NOMUTUAL_INDEXING_START__CELLA_IND(wg_id)]);  // target cell index 
  BCAST_GMEM(wi_id, currentBlockNb, M2L_nomutual_indexing_start[M2L_NOMUTUAL_INDEXING_START__CELLA_IND(wg_id)+1]); 
  BCAST_GMEM(wi_id, ind, M2L_nomutual_indexing[currentBlockNb]);

  CellAx = Cell[A].POS[0];
  CellAy = Cell[A].POS[1];
  CellAz = Cell[A].POS[2];
  id1a = Cell[A].ID1;
  id2a = Cell[A].ID2;
  CellAm = M[id1a].MASS;


  while (ind != 0){ 
    // If last block (possibly incomplete), we have: 'ind' <= M2L_INTERBUF_BS
    // If not last block, we ensure: 'ind' > M2L_INTERBUF_BS 
    int stop = MIN(ind, M2L_INTERBUF_BS);  
    for (int block_id = wi_id; block_id < stop; block_id += M2L_WG_SIZE_X){ 
      B = interBuf[currentBlockNb * M2L_INTERBUF_BS + block_id]; 
    
      ///// M2L computation core: 
      id1b = Cell[B].ID1;
      id2b = Cell[B].ID2;
      
      dXRq_x = CellAx - Cell[B].POS[0];
      dXRq_y = CellAy - Cell[B].POS[1];
      dXRq_z = CellAz - Cell[B].POS[2];
      dXRq_x2 = dXRq_x*dXRq_x;
      dXRq_y2 = dXRq_y*dXRq_y;
      dXRq_z2 = dXRq_z*dXRq_z;
      dXRq = dXRq_x2 + dXRq_y2 + dXRq_z2;  
      XX   = 1.0f/(dXRq+EQ);  
      D[0] = CellAm*M[id1b].MASS;
      D[0] *= sqrt(XX);
      D[1] = XX * D[0];
      D[2] = 3 * XX * D[1];
      D[3] = 5 * XX * D[2];
      M_POLS5 = M[id1b].POLS[5];
      a[ 0] = D[0];
      float t =-D[1];
      a[ 1] = dXRq_x*t;
      a[ 2] = dXRq_y*t;
      a[ 3] = dXRq_z*t;
      t     = D[2]*dXRq_x;
      M_POLS0 = M[id1b].POLS[0];
      a[ 4] = t*dXRq_x - D[1];
      a[ 5] = t*dXRq_y;
      a[ 6] = t*dXRq_z;
      t     = D[2]*dXRq_y;
      M_POLS1 = M[id1b].POLS[1];
      a[ 7] = t*dXRq_y - D[1];
      a[ 8] = t*dXRq_z;
      t     = D[2]*dXRq_z;
      a[ 9] = t*dXRq_z - D[1];
      M_POLS2 = M[id1b].POLS[2];
      float D2_3 = 3*D[2];
      t     = D[3]*dXRq_x2;
      a[10] = (D2_3-t)*dXRq_x;
      float D2_t = D[2]-t;
      a[11] = D2_t*dXRq_y;
      a[12] = D2_t*dXRq_z;
      M_POLS3 = M[id1b].POLS[3];
      t     =  D[3]*dXRq_y2;
      D2_t  = D[2]-t;
      a[13] = D2_t*dXRq_x;
      a[16] = (D2_3-t)*dXRq_y;
      a[17] = D2_t*dXRq_z;
      t     =  D[3]*dXRq_z2;
      D2_t  = D[2]-t;
      M_POLS4 = M[id1b].POLS[4];
      a[15] = D2_t*dXRq_x;
      a[18] = D2_t*dXRq_y;
      a[19] = (D2_3-t)*dXRq_z;
      a[14] = -D[3]*dXRq_x*dXRq_y*dXRq_z;
    
      
      coeff[0]  = a[0] + a[4]*M_POLS0  + a[9]*M_POLS5  + a[7]*M_POLS3 ; 
      coeff[0] +=  2.0f*(a[5]*M_POLS1  + a[6]*M_POLS2  + a[8]*M_POLS4) ;
      coeff[1]  = a[1] + a[10]*M_POLS0 + a[15]*M_POLS5 + a[13]*M_POLS3 ; 
      coeff[1] +=  2.0f*(a[11]*M_POLS1 + a[12]*M_POLS2 + a[14]*M_POLS4); 
      coeff[2]  = a[2] + M_POLS0*a[11] + M_POLS3*a[16] + M_POLS5*a[18];
      coeff[2] += 2.0f*(M_POLS1*a[13]  + M_POLS2*a[14] + M_POLS4*a[17]);   
      coeff[3]  = a[3] + M_POLS0*a[12] + M_POLS3*a[17] + M_POLS5*a[19];
      coeff[3] +=  2.0f*(M_POLS1*a[14] + M_POLS2*a[15] + M_POLS4*a[18]) ;
      ///// End of M2L computation core 
    
      CellACoeffs.coeff[0]  += coeff[0];
      CellACoeffs.coeff[1]  += coeff[1];
      CellACoeffs.coeff[2]  += coeff[2];
      CellACoeffs.coeff[3]  += coeff[3];
      CellACoeffs.coeff[4]  += a[4];
      CellACoeffs.coeff[5]  += a[5];
      CellACoeffs.coeff[6]  += a[6];
      CellACoeffs.coeff[7]  += a[7];
      CellACoeffs.coeff[8]  += a[8];
      CellACoeffs.coeff[9]  += a[9];
      CellACoeffs.coeff[10] += a[10];
      CellACoeffs.coeff[11] += a[11];
      CellACoeffs.coeff[12] += a[12];
      CellACoeffs.coeff[13] += a[13];
      CellACoeffs.coeff[14] += a[14];
      CellACoeffs.coeff[15] += a[15];
      CellACoeffs.coeff[16] += a[16];
      CellACoeffs.coeff[17] += a[17];
      CellACoeffs.coeff[18] += a[18];
      CellACoeffs.coeff[19] += a[19];

    } // for (block_id) 

    if (ind <= M2L_INTERBUF_BS){ // last block 
      break; 
    }
    else { // not last block: prepare for next block 
      currentBlockNb = ind - M2L_INTERBUF_BS; 
      ind = M2L_nomutual_indexing[currentBlockNb]; 
    }
  } // while (ind != 0)
  

  /////////////////////////////////////////////////////////////////////  
  // Load data into local memory
  C_local_mem[wi_id] = CellACoeffs;
  barrier(CLK_LOCAL_MEM_FENCE);


#ifdef AMD_iGPU 
  /////////////////////////////////////////////////////////////////////
  /////// NB: we assume here M2L_WG_SIZE_X >= 64 ///////

  /***** Bank conflicts for AMP GCN APU/GPU  */
  /* (from AMD APP SDK OpenCL Optimization Guide (August 2015)): */
  /* All AMD Southern Islands, Sea Islands, and Volcanic Islands GPUs (collectively */ 
  /* referred to as GCN devices) contain a 64 kB LDS for each compute unit; */
  /* although only 32 kB can be allocated per work-group. The LDS contains 32- */
  /* banks, each bank is four bytes wide and 256 bytes deep; the bank address is */
  /* determined by bits 6:2 in the address. */
  
  ///// Custom reduction bis: with increased parallelism and ok with bank conflicts
  ///// (since each wi of each half wavefront (see below) accesses a distinct bank)
  int part_size  = M2L_WG_SIZE_X / 2;
  int part_id    = wi_id / part_size;
  int part_wi_id = wi_id % part_size;
  int shift      = part_id*part_size;
  ///// Remark: no bank conflict between distinct half wave-fronts for AMD GCN
  // APU/GPU (see AMD APP SDK OpenCL Optimization Guide (August 2015, Sect. 2.2):
  // "Bank conflicts are determined by what addresses are accessed on each half wavefront
  // boundary. Threads 0 through 31 are checked for conflicts as are threads 32
  // through 63 within a wavefront."
  if (part_wi_id < LOCAL_EXP_SIZE){
    /// Manual unroll (assuming M2L_WG_SIZE_X is a multiple of 4): 
    float contrib = 0.0f;
    for (int ii=0; ii<part_size; ii+=4){
      contrib += C_local_mem[shift + ii  ].coeff[part_wi_id];
      contrib += C_local_mem[shift + ii+1].coeff[part_wi_id];
      contrib += C_local_mem[shift + ii+2].coeff[part_wi_id];
      contrib += C_local_mem[shift + ii+3].coeff[part_wi_id];
    }
    C_local_mem[shift].coeff[part_wi_id] = contrib;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (wi_id < LOCAL_EXP_SIZE){
    C[id2a].coeff[wi_id] += C_local_mem[0].coeff[wi_id] + C_local_mem[part_size].coeff[wi_id];
  }
  

#endif // #ifdef AMD_iGPU 
#ifdef INTEL_iGPU 
  /////////////////////////////////////////////////////////////////////  
  /***** Bank conflicts for AMP GCN APU/GPU  */
  /* From https://software.intel.com/en-us/node/540426 : "[...] shared
   * local memory is organized as 16 banks at 4-byte granularity." */ 
  /////////////////////////////////// M2L_WG_SIZE_X == 4 : ///////////////////////////////////
#if M2L_WG_SIZE_X == 4
  for (int coef_id=wi_id; coef_id<LOCAL_EXP_SIZE; coef_id += 4){  
      float contrib = 0.0;
      ///// Manual unroll:
      contrib += C_local_mem[0].coeff[coef_id];
      contrib += C_local_mem[1].coeff[coef_id]; 
      contrib += C_local_mem[2].coeff[coef_id]; 
      contrib += C_local_mem[3].coeff[coef_id]; 
      /////
      C[id2a].coeff[coef_id] += contrib;
    } // for coef_id 
#endif // #if M2L_WG_SIZE_X == 4
  /////////////////////////////////// M2L_WG_SIZE_X == 8 : ///////////////////////////////////
#if M2L_WG_SIZE_X == 8
#define REMAINDER_SIZE 4
    for (int coef_id=wi_id; coef_id<LOCAL_EXP_SIZE-REMAINDER_SIZE; coef_id += 8){
      float contrib = 0.0; 
      ///// Manual unroll:
      for (int ii=0; ii<8; ii+=4){ 
	contrib += C_local_mem[ii  ].coeff[coef_id];
	contrib += C_local_mem[ii+1].coeff[coef_id]; 
	contrib += C_local_mem[ii+2].coeff[coef_id]; 
	contrib += C_local_mem[ii+3].coeff[coef_id]; 
      }
      /////
      C[id2a].coeff[coef_id] += contrib;
    } // for coef_id
    // Last REMAINDER_SIZE=4 coefficients (based on "custom reduction bis" for AMD iGPU) 
#define PART_SIZE 4 // = M2L_WG_SIZE_X / 2 
    int part_id    = wi_id / PART_SIZE;
    int part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2 
    int shift      = part_id*PART_SIZE;
    int coef_id    = (LOCAL_EXP_SIZE-REMAINDER_SIZE) + part_wi_id;
    float contrib   = 0.0; 
    ///// Manual unroll:
    contrib += C_local_mem[shift  ].coeff[coef_id];
    contrib += C_local_mem[shift+1].coeff[coef_id];
    contrib += C_local_mem[shift+2].coeff[coef_id];
    contrib += C_local_mem[shift+3].coeff[coef_id];
    /////
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }    
#endif // #if M2L_WG_SIZE_X == 8
  /////////////////////////////////// M2L_WG_SIZE_X == 16 : ///////////////////////////////////
#if M2L_WG_SIZE_X == 16
#define REMAINDER_SIZE 4
    float contrib = 0.0;
    ///// Manual unroll:
    for (int ii=0; ii<16; ii+=4){ 
      contrib += C_local_mem[ii  ].coeff[wi_id];
      contrib += C_local_mem[ii+1].coeff[wi_id];
      contrib += C_local_mem[ii+2].coeff[wi_id];
      contrib += C_local_mem[ii+3].coeff[wi_id];
    }
    /////
    C[id2a].coeff[wi_id] += contrib;
    // Last REMAINDER_SIZE=4 coefficients (based on "custom reduction bis" for AMD iGPU) 
#define PART_SIZE 4 // = M2L_WG_SIZE_X / 4 
    int part_id    = wi_id / PART_SIZE;
    int part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2  
    int shift      = part_id*PART_SIZE;
    int coef_id    = (LOCAL_EXP_SIZE-REMAINDER_SIZE) + part_wi_id;
    contrib  = 0.0;
    ///// Manual unroll:
    contrib += C_local_mem[shift  ].coeff[coef_id];
    contrib += C_local_mem[shift+1].coeff[coef_id];
    contrib += C_local_mem[shift+2].coeff[coef_id];
    contrib += C_local_mem[shift+3].coeff[coef_id];
    /////
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 2*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 2*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }    
#endif // #if M2L_WG_SIZE_X == 16
  /////////////////////////////////// M2L_WG_SIZE_X == 32 : ///////////////////////////////////
#if M2L_WG_SIZE_X == 32
    // NB : we treat the first 16 coefficients together to match the 16 banks of the shared local memory. 
#define REMAINDER_SIZE 4
    // First 16 coefficients (based on "custom reduction bis" for AMD iGPU) 
#define PART_SIZE 16 // = M2L_WG_SIZE_X / 2 
    int part_id    = wi_id / PART_SIZE;
    int part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2  
    int shift      = part_id*PART_SIZE;
    int coef_id    = part_wi_id;
    float contrib  = 0.0;
    ///// Manual unroll:
    for (int ii=0; ii<PART_SIZE; ii+=4){ 
      contrib += C_local_mem[shift+ii  ].coeff[coef_id];
      contrib += C_local_mem[shift+ii+1].coeff[coef_id];
      contrib += C_local_mem[shift+ii+2].coeff[coef_id];
      contrib += C_local_mem[shift+ii+3].coeff[coef_id];
    }
    ///// 
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }        
    // Last REMAINDER_SIZE=4 coefficients (based on "custom reduction bis" for AMD iGPU) 
#undef PART_SIZE 
#define PART_SIZE 4 // = M2L_WG_SIZE_X / 8 
    part_id    = wi_id / PART_SIZE;
    part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2  
    shift      = part_id*PART_SIZE;
    coef_id    = (LOCAL_EXP_SIZE-REMAINDER_SIZE) + part_wi_id;
    contrib  = 0.0;
    ///// Manual unroll:
    contrib += C_local_mem[shift  ].coeff[coef_id];
    contrib += C_local_mem[shift+1].coeff[coef_id];
    contrib += C_local_mem[shift+2].coeff[coef_id];  
    contrib += C_local_mem[shift+3].coeff[coef_id];
    /////
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 4*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 4*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 2*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 2*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }        
#endif // #if M2L_WG_SIZE_X == 32
  /////////////////////////////////// M2L_WG_SIZE_X == 64 : ///////////////////////////////////
#if M2L_WG_SIZE_X == 64
    // NB : we treat the first 16 coefficients together to match the 16 banks of the shared local memory. 
#define REMAINDER_SIZE 4
    // First 16 coefficients (based on "custom reduction bis" for AMD iGPU) 
#define PART_SIZE 16 // = M2L_WG_SIZE_X / 4 
    int part_id    = wi_id / PART_SIZE;
    int part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2  
    int shift      = part_id*PART_SIZE;
    int coef_id    = part_wi_id;
    float contrib  = 0.0; 
    ///// Manual unroll:
    for (int ii=0; ii<PART_SIZE; ii+=4){ 
      contrib += C_local_mem[shift+ii  ].coeff[coef_id];
      contrib += C_local_mem[shift+ii+1].coeff[coef_id];
      contrib += C_local_mem[shift+ii+2].coeff[coef_id];
      contrib += C_local_mem[shift+ii+3].coeff[coef_id];
    }
    /////
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 2*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 2*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }        
    // Last REMAINDER_SIZE=4 coefficients (based on "custom reduction bis" for AMD iGPU) 
#undef PART_SIZE 
#define PART_SIZE 4 // = M2L_WG_SIZE_X / 16 
    part_id    = wi_id / PART_SIZE;
    part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2 
    shift      = part_id*PART_SIZE;
    coef_id    = (LOCAL_EXP_SIZE-REMAINDER_SIZE) + part_wi_id;
    contrib  = 0.0; 
    ///// Manual unroll:
    contrib += C_local_mem[shift  ].coeff[coef_id];
    contrib += C_local_mem[shift+1].coeff[coef_id];
    contrib += C_local_mem[shift+2].coeff[coef_id];
    contrib += C_local_mem[shift+3].coeff[coef_id];
    ///// 
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 8*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 8*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 4*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 4*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 2*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 2*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }            
#endif // #if M2L_WG_SIZE_X == 64  
  /////////////////////////////////// M2L_WG_SIZE_X == 128 : ///////////////////////////////////
#if M2L_WG_SIZE_X == 128
    // NB : we treat the first 16 coefficients together to match the 16 banks of the shared local memory. 
#define REMAINDER_SIZE 4
    // First 16 coefficients (based on "custom reduction bis" for AMD iGPU) 
#define PART_SIZE 16 // = M2L_WG_SIZE_X / 8 
    int part_id    = wi_id / PART_SIZE;
    int part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2  
    int shift      = part_id*PART_SIZE;
    int coef_id    = part_wi_id;
    float contrib  = 0.0; 
    ///// Manual unroll:
    for (int ii=0; ii<PART_SIZE; ii+=4){ 
      contrib += C_local_mem[shift+ii  ].coeff[coef_id];
      contrib += C_local_mem[shift+ii+1].coeff[coef_id];
      contrib += C_local_mem[shift+ii+2].coeff[coef_id];
      contrib += C_local_mem[shift+ii+3].coeff[coef_id];
    }
    /////
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 4*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 4*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 2*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 2*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }        
    // Last REMAINDER_SIZE=4 coefficients (based on "custom reduction bis" for AMD iGPU) 
#undef PART_SIZE 
#define PART_SIZE 4 // = M2L_WG_SIZE_X / 32
    part_id    = wi_id / PART_SIZE;
    part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2 
    shift      = part_id*PART_SIZE;
    coef_id    = (LOCAL_EXP_SIZE-REMAINDER_SIZE) + part_wi_id;
    contrib  = 0.0; 
    ///// Manual unroll:
    contrib += C_local_mem[shift  ].coeff[coef_id];
    contrib += C_local_mem[shift+1].coeff[coef_id];
    contrib += C_local_mem[shift+2].coeff[coef_id];
    contrib += C_local_mem[shift+3].coeff[coef_id];
    ///// 
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 16*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 16*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 8*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 8*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 4*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 4*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 2*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 2*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }            
#endif // #if M2L_WG_SIZE_X == 128  
  /////////////////////////////////// M2L_WG_SIZE_X == 256 : ///////////////////////////////////
#if M2L_WG_SIZE_X == 256
    // NB : we treat the first 16 coefficients together to match the 16 banks of the shared local memory. 
#define REMAINDER_SIZE 4
    // First 16 coefficients (based on "custom reduction bis" for AMD iGPU) 
#define PART_SIZE 16 // = M2L_WG_SIZE_X / 16 
    int part_id    = wi_id / PART_SIZE;
    int part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2  
    int shift      = part_id*PART_SIZE;
    int coef_id    = part_wi_id;
    float contrib  = 0.0; 
    ///// Manual unroll:
    for (int ii=0; ii<PART_SIZE; ii+=4){ 
      contrib += C_local_mem[shift+ii  ].coeff[coef_id];
      contrib += C_local_mem[shift+ii+1].coeff[coef_id];
      contrib += C_local_mem[shift+ii+2].coeff[coef_id];
      contrib += C_local_mem[shift+ii+3].coeff[coef_id];
    }
    /////
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 8*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 8*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 4*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 4*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 2*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 2*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }        
    // Last REMAINDER_SIZE=4 coefficients (based on "custom reduction bis" for AMD iGPU) 
#undef PART_SIZE 
#define PART_SIZE 4 // = M2L_WG_SIZE_X / 64
    part_id    = wi_id / PART_SIZE;
    part_wi_id = wi_id &(PART_SIZE-1); // = "wi_id % PART_SIZE" since PART_SIZE is a power of 2 
    shift      = part_id*PART_SIZE;
    coef_id    = (LOCAL_EXP_SIZE-REMAINDER_SIZE) + part_wi_id;
    contrib  = 0.0; 
    ///// Manual unroll:
    contrib += C_local_mem[shift  ].coeff[coef_id];
    contrib += C_local_mem[shift+1].coeff[coef_id];
    contrib += C_local_mem[shift+2].coeff[coef_id];
    contrib += C_local_mem[shift+3].coeff[coef_id];
    ///// 
    C_local_mem[shift].coeff[coef_id] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 32*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 32*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 16*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 16*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 8*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 8*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 4*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 4*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < 2*PART_SIZE){
      C_local_mem[shift].coeff[coef_id] += C_local_mem[shift + 2*PART_SIZE].coeff[coef_id];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wi_id < PART_SIZE){
      C[id2a].coeff[coef_id] += C_local_mem[0].coeff[coef_id] + C_local_mem[PART_SIZE].coeff[coef_id];
    }            
#endif // #if M2L_WG_SIZE_X == 256
#endif // #ifdef INTEL_iGPU 

}
