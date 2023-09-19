// -*- C++ -*-
////////////////////////////////////////////////////////////////////////////////
///
/// \file    inc/cl_manip.cc
///
/// \author  Maxime Touche, Pierre Fortin
///
/// \date    2014, 2016 
///
////////////////////////////////////////////////////////////////////////////////

#include <public/cl_manip.h>
#include <public/gravity.h>
#include <omp.h>
#include <sys/time.h>
using namespace falcON;


////////////////////////////////////////////////////////////////////////////////
///// Various data structures for CPU-GPU handling of M2L and P2P computations: 

void cl_manip::check_pfalcONstruct_sizes(uint64_t *ptr_sizes,int nb_fields)
{
  cl_mem pfalcONstruct_sizes;
  cl_mem P2P_local_memory_size_obj;
  cl_kernel kernel_check;
  cl_int err;
  size_t local_work_size[2];
  size_t global_work_size[2];

  pfalcONstruct_sizes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint64_t)*nb_fields, NULL, &err);
  if(err != CL_SUCCESS) {
    std::cout << "Couldn't create buffer 'pfalcONstruct_sizes': err=" << err << std::endl;
    exit(1);
  }

  P2P_local_memory_size_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
  if(err != CL_SUCCESS) {
    std::cout << "Couldn't create buffer 'P2P_local_memory_size_obj': err=" << err << std::endl;
    exit(1);
  }

  kernel_check = clCreateKernel(program, KERNEL_CHECK_STRUCT, &err);
  if(err != CL_SUCCESS) {
    std::cout << "Couldn't create kernel 'kernel_check': err=" << err << std::endl;
    exit(1);
  }

  err = 0;
  err  = clSetKernelArg(kernel_check, 0, sizeof(cl_mem), &pfalcONstruct_sizes);
  err  = clSetKernelArg(kernel_check, 1, sizeof(cl_mem), &P2P_local_memory_size_obj);
  
  if (err != CL_SUCCESS){
    std::cout<<"Error: Failed to set kernel arguments for kernel_check! \n err = "<<err <<"\n"<<std::endl;
    exit(1);      
  }
 
  local_work_size[0] = 1;
  local_work_size[1] = 1;
  global_work_size[0]= 1; 
  global_work_size[1]= 1;
  
  err = clEnqueueNDRangeKernel(queue[0], kernel_check, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  if(err != CL_SUCCESS){
    std::cout<<"Error: Failed to launch kernel in check_struct !\n err = "<< err <<std::endl;
    exit(1); 
   }
  err = clFinish(queue[0]);
  
 
  if(err != CL_SUCCESS){
    std::cout<<"Error: Failed to wait queue in checkstruct !\n err = "<< err <<std::endl;
    exit(1);
  }
  
  err=clEnqueueReadBuffer(queue[0], pfalcONstruct_sizes,CL_TRUE,0, sizeof(uint64_t)*nb_fields, (void*)ptr_sizes,0,NULL,NULL);
  if(err != CL_SUCCESS){
    std::cout<<"Error: Failed to read buffer in checkstruct !\n err = "<< err <<std::endl;
    exit(1);
  }
  err=clEnqueueReadBuffer(queue[0],P2P_local_memory_size_obj ,CL_TRUE,0, sizeof(int), (void*) &P2P_local_memory_size,0,NULL,NULL);
  if(err != CL_SUCCESS){
    std::cout<<"Error: Failed to read buffer in checkstruct !\n err = "<< err <<std::endl;
    exit(1);
  }
  
  clReleaseMemObject(pfalcONstruct_sizes);
  clReleaseMemObject(P2P_local_memory_size_obj);
  clReleaseKernel(kernel_check);

}



// Get the first OpenCl platform available

double time_test = 0;
double time_test1 = 0;


void cl_manip::get_platform(){
  cl_int err;
  //If multiple platforms are available, this takes only the first platform
  err = clGetPlatformIDs(1, &platform,NULL); //access first platform
  if (err != CL_SUCCESS) {
    std::cout << "Couldn\'t find any platforms" << std::endl;
    exit(1);
  }
}

// Get the fist device of type device_type ( by default a CPU)
void cl_manip::get_device(cl_device_type device_type = CL_DEVICE_TYPE_CPU){
  cl_int err;
  cl_uint num_devices;
  err = clGetDeviceIDs(platform, device_type ,0 ,NULL, &num_devices); //find number of devices
  if (err !=CL_SUCCESS) {
    std::cout << "Couldn\'t find any devices of type "<< device_type << std::endl;
    exit(1);
  } 
  clGetDeviceIDs(platform, device_type, 1, &device, NULL); 
}

// Create the OpenCl context
void cl_manip::create_context(){
  cl_int err;
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS){
    std::cout << "Couldn\'t create the context" << std::endl;
    exit(1);
  }
}

//Create command queues
void cl_manip::create_queues(){
  cl_int err;
  queue = new cl_command_queue[NB_CQ];
  for(int i = 0; i < NB_CQ; ++i){
    queue[i] = clCreateCommandQueue(context, device,
				    /* in-order command queue */
#ifdef OCL_EVENT_PROFILING
				    CL_QUEUE_PROFILING_ENABLE /* for events profiling */, 
#else 
				    0, 
#endif 
				    &err);
    if (err < 0){
      std::cout << "Couldn\'t create command queue #" << i << std::endl;
      exit(1);
    }
  }
}




// Initialize the OpenCL program:
void cl_manip::init_program(__attribute__((unused)) int argc, const char **argv){  
         
  FILE *fp;
  char *source_str;
  size_t source_size;
  cl_int err;
  char path_cl_file[1000]={0};
  const char *slash_last_pos = NULL; 

  //Look for directory of .cl files based on argv[0]:
  slash_last_pos = strrchr(argv[0], '/'); 
  if (slash_last_pos != NULL){
    strncpy(path_cl_file, argv[0], (size_t) (slash_last_pos - argv[0] +1));
    path_cl_file[(size_t) (slash_last_pos - argv[0] +2)] = '\0'; 
    strcat(path_cl_file, "../src/public/lib/"); 
  }
  /* else (no '/' in argv[0]: look into current working directory) */ 
  strcat(path_cl_file, PROGRAM_FILE);
  std::cout << "Loading: " << path_cl_file << " ... " << std::endl ; 

  //Open .cl file
  fp = fopen(path_cl_file, "r");
  if (fp == NULL)
    {
      std::cout << "pfalcON_iGPU error: cannot open .cl file ("
		<< path_cl_file << ")" << std::endl;
      exit(1);
    }
  fseek(fp, 0, SEEK_END);
  source_size = ftell(fp);
  rewind(fp);
  source_str =(char*)malloc(source_size +1);
  source_str[source_size] ='\0';
  fread(source_str, sizeof(char), source_size, fp);
  fclose(fp);
  
  //Create program 
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
				      (const size_t *)&source_size, &err);
  
  if(err != CL_SUCCESS) {
    std::cout << "Couldn\'t create program with err = "<< err <<"\n "<<std::endl;
    exit(1);
  }
  free(source_str);
  err = 0;
  //Build program
  //  char cl_compile_args[1024] = "-cl-std=CL2.0 -cl-unsafe-math-optimizations -cl-fast-relaxed-math -I. ";
  char cl_compile_args[1024] = "-cl-unsafe-math-optimizations -cl-fast-relaxed-math -I. ";

  {  // For reqd_work_group_size():
    char s[100];
    sprintf(s,  " -DM2L_WG_SIZE_X=%i ", M2L_wg_size); 
    strcat(cl_compile_args, s);
    sprintf(s,  " -DP2P_WG_SIZE_X=%i ", P2P_wg_size); 
    strcat(cl_compile_args, s);

    // For {M2L,P2P}_INTERBUF_BS: 
    sprintf(s,  " -DM2L_INTERBUF_BS=%i ", M2L_INTERBUF_BS); 
    strcat(cl_compile_args, s);
    sprintf(s,  " -DP2P_INTERBUF_BS=%i ", P2P_INTERBUF_BS); 
    strcat(cl_compile_args, s);
    
    // For P2P kernel: 
    sprintf(s,  " -DNCRIT=%i ", this->get_ncrit()); 
    strcat(cl_compile_args, s);
  }

  // Integrated GPU architecture: 
  { char *s = NULL;
    size_t size = 0; 
    if (clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &size) != CL_SUCCESS){
      std::cout<<"Error in clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &size)\n "<<std::endl;
      exit(1);
    }
    s = (char *)malloc(size);
    if (clGetDeviceInfo(device, CL_DEVICE_VENDOR, size, s, NULL) != CL_SUCCESS){
      std::cout<<"Error in clGetDeviceInfo(device, CL_DEVICE_VENDOR, size, s, NULL)\n "<<std::endl;
      exit(1);
    }

    if (strcasestr(s, "Advanced Micro Devices") != NULL){
      strcat(cl_compile_args, " -DAMD_iGPU ");
    }
    if (strcasestr(s, "Intel") != NULL){
      strcat(cl_compile_args, " -DINTEL_iGPU ");
      use_Intel_iGPU = true; 
      if (M2L_wg_size != 4   && 
	  M2L_wg_size != 8   && 
	  M2L_wg_size != 16  && 
	  M2L_wg_size != 32  && 
	  M2L_wg_size != 64  && 
	  M2L_wg_size != 128 && 
	  M2L_wg_size != 256){ // see M2L kernel code
	std::cout << "M2L_wg_size not supported in M2L kernel code compiled with -DINTEL_iGPU."<<std::endl;
	exit(1);
      }
    } 

    free(s);
  }

  

  std::cout <<  "Building OpenCL program with: " << cl_compile_args << std::endl; 
  err =  clBuildProgram(program, 0,NULL, cl_compile_args, NULL, NULL);  
  if (err != CL_SUCCESS){
      size_t logSize;
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
			     &logSize);

      char *programLog = new char[logSize]; 
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
			     logSize, programLog, NULL);
      std::cout <<  "Error in clBuildProgram(): "
	        << "(logSize=" << logSize << ") " 
		<< programLog << std::endl;

      delete[] programLog;
      exit (1);
    }
  
  if(err != CL_SUCCESS) {
    std::cout << "Couldn\'t build program with err = "<< err <<"\n"<<std::endl;
    exit(1);
  }

  //Create kernels
  kernels = new kernel_t[num_threads]; 
  for (int i=0; i<num_threads; ++i){
    kernels[i].kernel_P2P = clCreateKernel(program, KERNEL_P2P, &err);
    if (!kernels[i].kernel_P2P || err != CL_SUCCESS){
      std::cout<<"Error: Failed to create P2P kernel for thread #" << i << " \n err = "<< err<<"\n "<<std::endl;
      exit(1);
    }
    kernels[i].kernel_M2L = clCreateKernel(program, KERNEL_M2L, &err);
    if (!kernels[i].kernel_M2L || err != CL_SUCCESS){
      std::cout<<"Error: Failed to create M2L kernel for thread #" << i << " \n err = "<< err<<"\n "<<std::endl;
      exit(1);
    }    
  }
}




void cl_manip::warmup_run(){
#ifdef pfalcON
#pragma omp parallel for schedule(static,1)
#endif
  for (int t=0; t<num_threads; t++){
    run_P2P(1, -1.0 /* EQ => "warm-up" kernel */, t /* thread_num */);
    run_P2PLeafTgt(1, -1.0 /* EQ => "warm-up" kernel */, t /* thread_num */);
    run_M2L(1, -1.0 /* EQ => "warm-up" kernel */, t /* thread_num */); 
  } // for t 
}

  




   
void cl_manip::run_P2P(unsigned nin, float EQ, int thread_num){  
  cl_int err;
  size_t local_work_size[2];
  size_t global_work_size[2];
  int local_mem_size;
  cl_kernel kernel_cp = kernels[thread_num].kernel_P2P; 
  int size_wg = P2P_wg_size; 
  int arg_num = 0;
  InterBufs_t *p_t = InterBufs + thread_num; 
  
  //  std::cout<<"run_P2P(): Nb interactions = " << nin << std::endl;
  
  // Without multi-threading in P2P kernel: 
  //local_mem_size = P2P_local_memory_size*ncrit;
  // With multi-threading in P2P kernel: 
  local_mem_size = P2P_local_memory_size*size_wg;

  err = 0; 

  err  = clSetKernelArg(kernel_cp, arg_num++, sizeof(float), &EQ);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_LEAF);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->P2P_interBuf_clBuf)); 
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_POTACC);
  err |= clSetKernelArg(kernel_cp, arg_num++, local_mem_size, NULL); 
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->P2P_nomutual_indexing_clBuf));
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_CELL);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->P2P_nomutual_indexing_start_clBuf));

  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(int)*size_wg, NULL); 
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(unsigned)*size_wg, NULL); 
  if (size_wg < this->get_ncrit()){
    err |= clSetKernelArg(kernel_cp, arg_num++, P2P_local_memory_size*(this->get_ncrit()), NULL); 
    err |= clSetKernelArg(kernel_cp, arg_num++, P2P_local_memory_size*(this->get_ncrit()), NULL); 
  } 


#define KERNEL_RUN_DISPLAY_LEVEL 2 
#if INFO_DISPLAY_LEVEL >= KERNEL_RUN_DISPLAY_LEVEL
  if (0){ // (EQ >= 0.0){ // not "warm-up" kernel
    std::cout<<"run_P2P(): local mem (src bodies) size = " 
	     << local_mem_size + (sizeof(int)+sizeof(unsigned))*size_wg + 
      (int) (size_wg < this->get_ncrit() ? 2*P2P_local_memory_size*(this->get_ncrit()) : 0)
	     << " bytes" << std::endl;
  }
#endif  
  if (err != CL_SUCCESS){
    std::cout<<"Error: Failed to set kernel arguments! \n err = "<<err <<"\n"<<std::endl;
    exit(1);
  }


  local_work_size[0] = size_wg;
  local_work_size[1] = 1;
  global_work_size[0]= (size_t) size_wg;
  global_work_size[1]= (size_t) nin;
 
#if INFO_DISPLAY_LEVEL >= KERNEL_RUN_DISPLAY_LEVEL
  if (EQ >= 0.0){ // not "warm-up" kernel
    std::cout << "Running P2P kernel with local_work_size[0]=" << local_work_size[0] 
	      << ", global_work_size[0]="    << global_work_size[0] 
	      << ", global_work_size[1]=" << global_work_size[1] 
	      << ", P2P_globalNextBlockNb=" << p_t->P2P_globalNextBlockNb 
	      << " by thread #" << thread_num << std::endl ; 
  }
#endif 

  cl_event event;  
  err = clEnqueueNDRangeKernel(queue[CQ_P2P], kernel_cp, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
  if(err != CL_SUCCESS){ std::cout<<"Error: Failed to launch kernel in run_P2P !\n err = "<< err <<std::endl; exit(1); }
  err = clWaitForEvents(1, &event);
#ifdef OCL_EVENT_PROFILING
  cl_ulong time_start=0, time_end=0;
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
  if (EQ >= 0.0){ // not "warm-up" kernel
    printf("P2P-event: %lf\n", (double) (time_end - time_start) / 1e9); 
  }
#endif 
  clReleaseEvent(event);

  if(err != CL_SUCCESS){
    std::cout<<"Error: Failed to wait for kernel in run_P2P !\n err = "<< err <<std::endl;
    exit(1);
  }
}





// C++ template can be use to merge codes for run_P2P() and run_P2PLeafTgt() XXX
void cl_manip::run_P2PLeafTgt(unsigned nin, float EQ, int thread_num){  
  cl_int err;
  size_t local_work_size[2];
  size_t global_work_size[2];
  int local_mem_size;
  cl_kernel kernel_cp = kernels[thread_num].kernel_P2P; 
  int size_wg = P2P_wg_size; 
  int arg_num = 0;
  InterBufs_t *p_t = InterBufs + thread_num; 
  
  //  std::cout<<"run_P2P(): Nb interactions = " << nin << std::endl;

  // Without multi-threading in P2P kernel: 
  //local_mem_size = P2P_local_memory_size*ncrit;
  // With multi-threading in P2P kernel: 
  local_mem_size = P2P_local_memory_size*size_wg;

  err = 0; 
  err  = clSetKernelArg(kernel_cp, arg_num++, sizeof(float), &EQ);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_LEAF);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->P2PLeafTgt_interBuf_clBuf)); 
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_POTACC);
  err |= clSetKernelArg(kernel_cp, arg_num++, local_mem_size, NULL); 
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->P2PLeafTgt_nomutual_indexing_clBuf));
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_CELL);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->P2PLeafTgt_nomutual_indexing_start_clBuf));

  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(int)*size_wg, NULL); 
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(unsigned)*size_wg, NULL); 
  if (size_wg < this->get_ncrit()){
    err |= clSetKernelArg(kernel_cp, arg_num++, P2P_local_memory_size*(this->get_ncrit()), NULL); 
    err |= clSetKernelArg(kernel_cp, arg_num++, P2P_local_memory_size*(this->get_ncrit()), NULL); 
  } 


#if INFO_DISPLAY_LEVEL >= KERNEL_RUN_DISPLAY_LEVEL
  if (0){ // (EQ >= 0.0){ // not "warm-up" kernel
    std::cout<<"run_P2P(): local mem (src bodies) size = " 
	     << local_mem_size + (sizeof(int)+sizeof(unsigned))*size_wg + 
      (int) (size_wg < this->get_ncrit() ? 2*P2P_local_memory_size*(this->get_ncrit()) : 0)
	     << " bytes" << std::endl;
  }
#endif 
  if (err != CL_SUCCESS){
    std::cout<<"Error: Failed to set kernel arguments! \n err = "<<err <<"\n"<<std::endl;
    exit(1);
  }
 
  local_work_size[0] = size_wg;
  local_work_size[1] = 1;
  global_work_size[0]= (size_t) size_wg;
  global_work_size[1]= (size_t) nin;
 
#if INFO_DISPLAY_LEVEL >= KERNEL_RUN_DISPLAY_LEVEL
  if (EQ >= 0.0){ // not "warm-up" kernel
    std::cout << "Running P2P kernel (P2PLeafTgt) with local_work_size[0]=" << local_work_size[0] 
	      << ", global_work_size[0]="    << global_work_size[0] 
	      << ", global_work_size[1]=" << global_work_size[1]  
	      << ", P2PLeafTgt_globalNextBlockNb=" << p_t->P2PLeafTgt_globalNextBlockNb 
  	      << " by thread #" << thread_num << std::endl ; 
  }
#endif 

  cl_event event; 
  err = clEnqueueNDRangeKernel(queue[CQ_P2PLEAFTGT], kernel_cp, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
  if(err != CL_SUCCESS){ std::cout<<"Error: Failed to launch kernel in run_P2PLeafTgt !\n err = "<< err <<std::endl; exit(1); }
  err = clWaitForEvents(1, &event); 
#ifdef OCL_EVENT_PROFILING
  cl_ulong time_start=0, time_end=0;
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
  if (EQ >= 0.0){ // not "warm-up" kernel
    printf("P2PLeafTgt-event: %lf\n", (double) (time_end - time_start) / 1e9); 
  }
#endif 
  clReleaseEvent(event);

  if(err != CL_SUCCESS){
    std::cout<<"Error: Failed to wait for kernel in run_P2PLeafTgt !\n err = "<< err <<std::endl;
    exit(1);
  }
}







void cl_manip::run_M2L(unsigned nin, float EQ, int thread_num){ 
  cl_int err;
  size_t local_work_size[2];
  size_t global_work_size[2];
  unsigned ni = nin;
  cl_kernel kernel_cp = kernels[thread_num].kernel_M2L; 
  int arg_num = 0;
  InterBufs_t *p_t = InterBufs + thread_num; 

  err = 0;
  err  = clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_CELL);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_MPOLES); 
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(float), &EQ);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), cl_bufs + CLBUF_LOCAL_COEFS);
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->M2L_interBuf_clBuf)); 
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->M2L_nomutual_indexing_clBuf));
  err |= clSetKernelArg(kernel_cp, arg_num++, sizeof(cl_mem), &(p_t->M2L_nomutual_indexing_start_clBuf));
  int local_mem_size  = sizeof(grav::Cset) * M2L_wg_size;
#if INFO_DISPLAY_LEVEL >= KERNEL_RUN_DISPLAY_LEVEL
  if (0){ // (EQ >= 0.0){ // not "warm-up" kernel
    std::cout<<"run_M2L(): local mem size = " << local_mem_size << " bytes"<<std::endl;
  }
#endif   
  clSetKernelArg(kernel_cp, arg_num++, local_mem_size, NULL);
  if (err != CL_SUCCESS){
    std::cout<<"Error: Failed to set kernel arguments! in M2L \n"<<std::endl;
    exit(1);      
  }

  local_work_size[0] = M2L_wg_size;
  local_work_size[1] = 1;
  // 1 WG per CellA (target cell)  
  global_work_size[0]= (size_t) M2L_wg_size; 
  global_work_size[1]= (size_t) ni; 

#if INFO_DISPLAY_LEVEL >= KERNEL_RUN_DISPLAY_LEVEL
  if (EQ >= 0.0){ // not "warm-up" kernel
    std::cout << "Running M2L kernel with local_work_size[0]=" << local_work_size[0] 
	      << ", global_work_size[0]="    << global_work_size[0] 
	      << ", global_work_size[1]=" << global_work_size[1] 
	      << ", M2L_globalNextBlockNb=" << p_t->M2L_globalNextBlockNb 
	      << " by thread #" << thread_num << std::endl ; 
  }
#endif 
  
  cl_event event; 
  err = clEnqueueNDRangeKernel(queue[CQ_M2L], kernel_cp, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
  if(err != CL_SUCCESS){ std::cout<<"Error: Failed to launch kernel in run_M2L !\n err = "<< err <<"\n"<<std::endl; exit(1); }  
  err = clWaitForEvents(1, &event);  
#ifdef OCL_EVENT_PROFILING
  cl_ulong time_start, time_end;
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
  if (EQ >= 0.0){ // not "warm-up" kernel
    printf("M2L-event: %lf\n", (double) (time_end - time_start) / 1e9); 
  }
#endif 
  clReleaseEvent(event);
  
  if(err != CL_SUCCESS){
    std::cout<<"Error: Failed to wait for kernel in run_M2L !\n err = "<< err <<std::endl;
    exit(1);
  }
}

