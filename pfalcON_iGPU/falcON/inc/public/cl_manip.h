// -*- C++ -*-
////////////////////////////////////////////////////////////////////////////////
///
/// \file    inc/cl_manip.h
///
/// \author  Maxime Touche, Pierre Fortin 
///
/// \date    2014, 2016, 2017 
///
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
//#include <tree.h>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <string.h>
#include <iostream>

#define PROGRAM_FILE "Ocl_kernels.cl"

#define KERNEL_P2P "process_P2P"
#define KERNEL_M2L "process_M2L"
#define KERNEL_CHECK_STRUCT "check_struct"

#define CLBUF_LEAF        0
#define CLBUF_CELL        1
#define CLBUF_POTACC      2
#define CLBUF_LOCAL_COEFS 3
#define CLBUF_MPOLES      4
#define NB_CLBUF          5

#define CQ_M2L        0
#define CQ_P2P        1
#define CQ_P2PLEAFTGT 2
#define NB_CQ         3 

// Warning: these must be multiples of the corresponding work-group sizes: 
#define P2P_INTERBUF_BS 192
#define M2L_INTERBUF_BS 192





////////////////////////////////////////////////////////////////////////////////
///// Various data structures for CPU-GPU handling of M2L and P2P computations: 

   typedef struct 
   {
     unsigned int id;
     float SCAL;  
     float a[3];
     int padding[7];
   } clpfalcONstruct;

   typedef struct 
   {
     float POT[4]; 
   } pfalcONstructPROP;


   typedef struct 
   {
     cl_kernel kernel_P2P;
     cl_kernel kernel_M2L;
   } kernel_t;



////////////////////////////////////////////////////////////////////////////////
/// Global variables: 

typedef struct {
  int *M2L_nomutual_indexing; 
  cl_mem M2L_nomutual_indexing_clBuf; // corresponding OpenCL buffer  
  int *M2L_nomutual_indexing_start; 
  cl_mem M2L_nomutual_indexing_start_clBuf; // corresponding OpenCL buffer  
  int M2L_nomutual_indexing_start_ind; 
  int *M2L_interBuf; 
  cl_mem M2L_interBuf_clBuf; 
  int M2L_globalNextBlockNb;
  int *M2L_currentCellBlockNb; // for each cell 
  
  int *P2P_nomutual_indexing; 
  cl_mem P2P_nomutual_indexing_clBuf; // corresponding OpenCL buffer  
  int *P2P_nomutual_indexing_start; 
  cl_mem P2P_nomutual_indexing_start_clBuf; // corresponding OpenCL buffer  
  int P2P_nomutual_indexing_start_ind; 
  int *P2P_interBuf; 
  cl_mem P2P_interBuf_clBuf; 
  int P2P_globalNextBlockNb;
  int *P2P_currentBlockNb; // for each cell 
  
  int *P2PLeafTgt_nomutual_indexing; 
  cl_mem P2PLeafTgt_nomutual_indexing_clBuf; // corresponding OpenCL buffer  
  int *P2PLeafTgt_nomutual_indexing_start; 
  cl_mem P2PLeafTgt_nomutual_indexing_start_clBuf; // corresponding OpenCL buffer  
  int P2PLeafTgt_nomutual_indexing_start_ind; 
  int *P2PLeafTgt_interBuf; 
  cl_mem P2PLeafTgt_interBuf_clBuf; 
  int P2PLeafTgt_globalNextBlockNb;
  int *P2PLeafTgt_currentBlockNb; // for each leaf 
} InterBufs_t; 



  class cl_manip {
	
	cl_manip(); // not used

    
  public :  
    cl_platform_id platform;
    cl_device_id device;
    cl_program program;
    int num_threads; // Number of OpenMP threads
    //	cl_uint num_platforms;
    kernel_t *kernels; // kernels for each thread 
    cl_mem *cl_bufs;
    void **cl_bufs_map_ptr; // saved for unmapping 
    // InterBufs: zero-copy buffers for CPU-GPU data sharing: 
    InterBufs_t *InterBufs; 
    int M2L_globalMaxBlockNb; // greatest possible block number
    int P2P_globalMaxBlockNb; // greatest possible block number
    int P2PLeafTgt_globalMaxBlockNb; // greatest possible block number
    int ncrit;
    size_t buf_size;
    int P2P_local_memory_size;
    int M2L_wg_size;
    int P2P_wg_size;
    cl_context context;
    cl_command_queue *queue;
    unsigned NbCells;
    unsigned NbLeafs;
    bool use_Intel_iGPU; 
    
    /////////////////////////////////////////////////////////////
    // Constructor
    ////////////////////////////////////////////////////////////

    cl_manip(cl_device_type device_type, 
	     int num_threads_arg, 
	     int ncrit_arg, 
	     unsigned buf_size_arg, 
	     int M2L_wg_size_arg, 
	     int P2P_wg_size_arg,
	     int argc, const char **argv) : 
      num_threads(num_threads_arg), 
      ncrit(ncrit_arg), 
      buf_size(buf_size_arg), 
      M2L_wg_size(M2L_wg_size_arg), 
      P2P_wg_size(P2P_wg_size_arg),
      NbCells(0), 
      NbLeafs(0),
      use_Intel_iGPU(false)
    { 
      get_platform();
      get_device(device_type);
      create_context();
      create_queues();
      // Allocate cl_bufs: 
      cl_bufs = new cl_mem[NB_CLBUF]; 
      cl_bufs_map_ptr = new void*[NB_CLBUF]; 
      for (int i=0; i<NB_CLBUF; i++){ cl_bufs_map_ptr[i] = NULL; }
      // Allocate InterBufs:
      InterBufs = new InterBufs_t[num_threads]; 
      init_program(argc, argv);
    }

    ~cl_manip(){
      /* Destroy cl_bufs: */
      for (int i=0; i<NB_CLBUF; i++){
	if (cl_bufs_map_ptr[i] != NULL){ // buffer still mapped 
	  unmap_buffer(cl_bufs + i, cl_bufs_map_ptr[i], 
		       (i == CLBUF_LEAF || i == CLBUF_POTACC ? CQ_P2P : CQ_M2L));  
	}
	clReleaseMemObject(cl_bufs[i]);   
      }
      delete [] cl_bufs; 
      
      /* Destroy InterBufs buffers: */
      for (int t=0; t<num_threads; t++){
	InterBufs_t *p_t = InterBufs+t;
	
	clEnqueueUnmapMemObject(queue[CQ_M2L], p_t->M2L_interBuf_clBuf, (void *) p_t->M2L_interBuf, 0, NULL, NULL);
	clReleaseMemObject(p_t->M2L_interBuf_clBuf);
	clEnqueueUnmapMemObject(queue[CQ_M2L], p_t->M2L_nomutual_indexing_clBuf,(void *) p_t->M2L_nomutual_indexing, 0, NULL, NULL);      
	clReleaseMemObject(p_t->M2L_nomutual_indexing_clBuf);
	clEnqueueUnmapMemObject(queue[CQ_M2L], p_t->M2L_nomutual_indexing_start_clBuf,(void *) p_t->M2L_nomutual_indexing_start, 0, NULL, NULL);      
	clReleaseMemObject(p_t->M2L_nomutual_indexing_start_clBuf);
	delete [] p_t->M2L_currentCellBlockNb;
	
	clEnqueueUnmapMemObject(queue[CQ_P2P], p_t->P2P_interBuf_clBuf, (void *) p_t->P2P_interBuf, 0, NULL, NULL);
	clReleaseMemObject(p_t->P2P_interBuf_clBuf);
	clEnqueueUnmapMemObject(queue[CQ_P2P], p_t->P2P_nomutual_indexing_clBuf,(void *) p_t->P2P_nomutual_indexing, 0, NULL, NULL);      
	clReleaseMemObject(p_t->P2P_nomutual_indexing_clBuf);
	clEnqueueUnmapMemObject(queue[CQ_P2P], p_t->P2P_nomutual_indexing_start_clBuf,(void *) p_t->P2P_nomutual_indexing_start, 0, NULL, NULL);      
	clReleaseMemObject(p_t->P2P_nomutual_indexing_start_clBuf);
	delete [] p_t->P2P_currentBlockNb;
	
	clEnqueueUnmapMemObject(queue[CQ_P2PLEAFTGT], p_t->P2PLeafTgt_interBuf_clBuf, (void *) p_t->P2PLeafTgt_interBuf, 0, NULL, NULL);
	clReleaseMemObject(p_t->P2PLeafTgt_interBuf_clBuf);
	clEnqueueUnmapMemObject(queue[CQ_P2PLEAFTGT], p_t->P2PLeafTgt_nomutual_indexing_clBuf,(void *) p_t->P2PLeafTgt_nomutual_indexing, 0, NULL, NULL);      
	clReleaseMemObject(p_t->P2PLeafTgt_nomutual_indexing_clBuf);
	clEnqueueUnmapMemObject(queue[CQ_P2PLEAFTGT], p_t->P2PLeafTgt_nomutual_indexing_start_clBuf,(void *) p_t->P2PLeafTgt_nomutual_indexing_start, 0, NULL, NULL);      
	clReleaseMemObject(p_t->P2PLeafTgt_nomutual_indexing_start_clBuf);
	delete [] p_t->P2PLeafTgt_currentBlockNb;
      }
      delete [] InterBufs; 
      
      for(int i=0; i < NB_CQ; ++i){      
	clReleaseCommandQueue(queue[i]);
      }
      delete [] queue; 
      
      for (int i=0; i < num_threads; i++){
	clReleaseKernel(kernels[i].kernel_P2P);
	clReleaseKernel(kernels[i].kernel_M2L);
      }
      delete [] kernels;
      
      clReleaseContext(context);
      clReleaseProgram(program);
    }
    

    /////////////////////////////////////////////////////////////
    // Accessor
    ////////////////////////////////////////////////////////////
    int get_ncrit() const{
      return ncrit;
    }

    
    //////////////////////////////////////////////////////////////
    // Methods
    /////////////////////////////////////////////////////////////
    void get_platform();
		
    void get_device(cl_device_type device_type);
			
    void create_context();

    void create_queues();

    void init_program(__attribute__((unused)) int argc, const char **argv);

    void warmup_run();

    void map_buffer(cl_mem *p_clBuf, 
		    size_t size, 
		    void ** p_map_ptr, 
		    int cq_num, 
		    cl_bool blocking_map = CL_TRUE,
		    cl_map_flags map_flags = CL_MAP_READ|CL_MAP_WRITE){
      cl_int err;
      *p_map_ptr = clEnqueueMapBuffer(queue[cq_num], *p_clBuf, blocking_map, map_flags, 0, size, 0, NULL, NULL, &err);
      if(err <0) { std::cout << "Couldn\'t map buffer in map_buffer() with err = " <<err <<std::endl; exit(1);}
    }


    void unmap_buffer(cl_mem *p_clBuf, void * map_ptr, int cq_num){
      clEnqueueUnmapMemObject(queue[cq_num], *p_clBuf, map_ptr, 0, NULL, NULL);
    }


    void create_buffer(cl_mem *p_clBuf, 
		       size_t size, /* in bytes */
		       cl_mem_flags mem_flags, 
		       void ** p_map_ptr = NULL, 
		       int cq_num = -1, 
		       cl_bool blocking_map = CL_TRUE,
		       cl_map_flags map_flags = CL_MAP_READ|CL_MAP_WRITE){
      cl_int err;
      *p_clBuf = clCreateBuffer(context, mem_flags, size, NULL, &err);
      if(err <0) { std::cout << "Couldn\'t create buffer in create_buffer() with err = " <<err <<std::endl; exit(1);}
      if (p_map_ptr != NULL){
	map_buffer(p_clBuf, size, p_map_ptr, cq_num, blocking_map, map_flags);
      }
    }

    void create_buffer(cl_mem *p_clBuf, 
		       size_t size, /* in bytes */
		       void ** p_map_ptr = NULL, 
		       int cq_num = -1, 
		       cl_bool blocking_map = CL_TRUE,
		       cl_map_flags map_flags = CL_MAP_READ|CL_MAP_WRITE){
      create_buffer(p_clBuf, size, 
		    CL_MEM_ALLOC_HOST_PTR|CL_MEM_READ_WRITE, // simulating default value for 'mem_flags'
		    p_map_ptr, cq_num, blocking_map, map_flags); 
    }

    void run_P2P(unsigned nin, float EQ, int thread_num); 
    void run_P2PLeafTgt(unsigned nin, float EQ, int thread_num); 
    void run_M2L(unsigned nin, float EQ, int thread_num); 
    void check_pfalcONstruct_sizes(uint64_t *ptr_sizes,int nb_fields);

  };
   

