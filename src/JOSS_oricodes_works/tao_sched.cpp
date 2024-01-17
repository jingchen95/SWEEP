/* assembly_sched.cxx -- integrated work stealing with assembly scheduling */
#include "tao.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <time.h>
#include <sstream>
#include <cstring>
#include <unistd.h> 
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <numeric>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include "xitao_workspace.h"
using namespace xitao;

struct read_format {
  uint64_t nr;
  struct {
    uint64_t value;
    uint64_t id;
  } values[];
};
// std::ofstream pmc;

#if defined(Haswell)
int num_sockets;
#endif

#if (defined DynaDVFS)
// int freq_dec;
// int freq_inc;
// 0 denotes highest frequency, 1 denotes lowest. 
// e.g. in 00, first 0 is denver, second 0 is a57. env= 0*2+0 = 0
int env;
// int dynamic_time_change;
int current_freq;
#endif

// #if defined(TX2)
// std::ifstream Cluster_A("/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq"); // edit Denver cluster frequency
// std::ifstream Cluster_B("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"); // edit A57 cluster frequency
// long cur_ddr_freq = 1866000000;

// long cur_freq[NUMSOCKETS] = {2035200, 2035200}; /*starting frequency is 2.04GHz for both clusters */
// #endif

int ptt_freq_index[NUMSOCKETS] = {0};
// int PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS] = {0};
int start_coreid[NUMSOCKETS] = {0, START_CLUSTER_B};
int end_coreid[NUMSOCKETS] = {START_CLUSTER_B, XITAO_MAXTHREADS};
int PTT_finish_state[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS] = {0}; // First: 2.04 or 1.11; Second: two clusters; third: the number of kernels (assume < XITAO_MAXTHREADS)
int global_training_state[XITAO_MAXTHREADS] = {0}; // array size: the number of kernels (assume < XITAO_MAXTHREADS)
bool global_training = false;

int status[XITAO_MAXTHREADS];
int status_working[XITAO_MAXTHREADS];
int Sched, num_kernels;
int maySteal_DtoA, maySteal_AtoD;
std::atomic<int> DtoA(0);

// define the topology
int gotao_sys_topo[5] = TOPOLOGY;

#ifdef NUMTASKS
int NUM_WIDTH_TASK[XITAO_MAXTHREADS] = {0};
#endif

#ifdef DVFS
float compute_bound_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS] = {0.0};
float memory_bound_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS] = {0.0};
float cache_intensive_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS] = {0.0};
#elif (defined DynaDVFS) // Currently only consider 4 combinations: max&max, max&min, min&max, min&min
float compute_bound_power[4][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
float memory_bound_power[4][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
float cache_intensive_power[4][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
#elif (defined ERASE)
float compute_bound_power[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
float memory_bound_power[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
float cache_intensive_power[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0};
#else
float runtime_power[10][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0}; // TX2 Power Profiles: 10 groups by memory-boundness level
float idle_power[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS] = {0.0};
#endif

struct timespec tim, tim2;
cpu_set_t affinity_setup;
int TABLEWIDTH;
int worker_loop(int);

#ifdef PowerProfiling
std::ofstream out("KernelTaskTime.txt");
#endif

#ifdef NUMTASKS_MIX
//std::vector<int> num_task(XITAO_MAXTHREADS * XITAO_MAXTHREADS, 0);
int num_task[XITAO_MAXTHREADS][XITAO_MAXTHREADS * XITAO_MAXTHREADS] = {0}; /*First parameter: assume an application has XITAO_MAXTHREADS different kernels at most*/
#endif

int PTT_flag[XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::chrono::time_point<std::chrono::system_clock> t3;
// std::mutex mtx;
std::condition_variable cv;
bool finish = false;

enum { NS_PER_SECOND = 1000000000 };
void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td){
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0){
      td->tv_nsec += NS_PER_SECOND;
      td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0){
      td->tv_nsec -= NS_PER_SECOND;
      td->tv_sec++;
    }
}

// std::vector<thread_info> thread_info_vector(XITAO_MAXTHREADS);
//! Allocates/deallocates the XiTAO's runtime resources. The size of the vector is equal to the number of available CPU cores. 
/*!
  \param affinity_control Set the usage per each cpu entry in the cpu_set_t
 */
int set_xitao_mask(cpu_set_t& user_affinity_setup) {
  if(!gotao_initialized) {
    resources_runtime_conrolled = true;                                    // make this true, to refrain from using XITAO_MAXTHREADS anywhere
    int cpu_count = CPU_COUNT(&user_affinity_setup);
    runtime_resource_mapper.resize(cpu_count);
    int j = 0;
    for(int i = 0; i < XITAO_MAXTHREADS; ++i) {
      if(CPU_ISSET(i, &user_affinity_setup)) {
        runtime_resource_mapper[j++] = i;
      }
    }
    if(cpu_count < gotao_nthreads) std::cout << "Warning: only " << cpu_count << " physical cores available, whereas " << gotao_nthreads << " are requested!" << std::endl;      
  } else {
    std::cout << "Warning: unable to set XiTAO affinity. Runtime is already initialized. This call will be ignored" << std::endl;      
  }  
}

void gotao_wait() {
//  gotao_master_waiting = true;
//  master_thread_waiting.notify_one();
//  std::unique_lock<std::mutex> lk(pending_tasks_mutex);
//  while(gotao_pending_tasks) pending_tasks_cond.wait(lk);
////  gotao_master_waiting = false;
//  master_thread_waiting.notify_all();
  while(PolyTask::pending_tasks > 0);
}
//! Initialize the XiTAO Runtime
/*!
  \param nthr is the number of XiTAO threads 
  \param thrb is the logical thread id offset from the physical core mapping
  \param nhwc is the number of hardware contexts
*/ 
int gotao_init_hw( int nthr, int thrb, int nhwc){
  gotao_initialized = true;
 	if(nthr>=0) gotao_nthreads = nthr;
  else {
    if(getenv("GOTAO_NTHREADS")) gotao_nthreads = atoi(getenv("GOTAO_NTHREADS"));
    else gotao_nthreads = XITAO_MAXTHREADS;
  }
  if(gotao_nthreads > XITAO_MAXTHREADS) {
    std::cout << "Fatal error: gotao_nthreads is greater than XITAO_MAXTHREADS of " << XITAO_MAXTHREADS << ". Make sure XITAO_MAXTHREADS environment variable is set properly" << std::endl;
    exit(0);
  }

  const char* layout_file = getenv("XITAO_LAYOUT_PATH");
  if(!resources_runtime_conrolled) {
    if(layout_file) {
      int line_count = 0;
      int cluster_count = 0;
      std::string line;      
      std::ifstream myfile(layout_file);
      int current_thread_id = -1; // exclude the first iteration
      if (myfile.is_open()) {
        bool init_affinity = false;
        while (std::getline(myfile,line)) {         
          size_t pos = 0;
          std::string token;
          if(current_thread_id >= XITAO_MAXTHREADS) {
            std::cout << "Fatal error: there are more partitions than XITAO_MAXTHREADS of: " << XITAO_MAXTHREADS  << " in file: " << layout_file << std::endl;    
            exit(0);    
          }
          int thread_count = 0;
          while ((pos = line.find(",")) != std::string::npos) {
            token = line.substr(0, pos);      
            int val = stoi(token);
            if(!init_affinity) static_resource_mapper[thread_count++] = val;  
            else { 
              if(current_thread_id + 1 >= gotao_nthreads) {
                  std::cout << "Fatal error: more configurations than there are input threads in:" << layout_file << std::endl;    
                  exit(0);
              }
              ptt_layout[current_thread_id].push_back(val);
              for(int i = 0; i < val; ++i) {     
                if(current_thread_id + i >= XITAO_MAXTHREADS) {
                  std::cout << "Fatal error: illegal partition choices for thread: " << current_thread_id <<" spanning id: " << current_thread_id + i << " while having XITAO_MAXTHREADS: " << XITAO_MAXTHREADS  << " in file: " << layout_file << std::endl;    
                  exit(0);           
                }
                inclusive_partitions[current_thread_id + i].push_back(std::make_pair(current_thread_id, val)); 
              }              
            }            
            line.erase(0, pos + 1);
          }          
          //if(line_count > 1) {
            token = line.substr(0, line.size());      
            int val = stoi(token);
            if(!init_affinity) static_resource_mapper[thread_count++] = val;
            else { 
              ptt_layout[current_thread_id].push_back(val);
              for(int i = 0; i < val; ++i) {                
                if(current_thread_id + i >= XITAO_MAXTHREADS) {
                  std::cout << "Fatal error: illegal partition choices for thread: " << current_thread_id <<" spanning id: " << current_thread_id + i << " while having XITAO_MAXTHREADS: " << XITAO_MAXTHREADS  << " in file: " << layout_file << std::endl;    
                  exit(0);           
                }
                inclusive_partitions[current_thread_id + i].push_back(std::make_pair(current_thread_id, val)); 
              }              
            }            
          //}
          if(!init_affinity) { 
            gotao_nthreads = thread_count; 
            init_affinity = true;
          }
          current_thread_id++;    
          line_count++;     
          //}
        }
        myfile.close();
      } else {
        std::cout << "Fatal error: could not open hardware layout path " << layout_file << std::endl;    
        exit(0);
      }
    } else {
        std::cout << "Warning: XITAO_LAYOUT_PATH is not set. Default values for affinity and symmetric resoruce partitions will be used" << std::endl;    
        for(int i = 0; i < XITAO_MAXTHREADS; ++i) 
          static_resource_mapper[i] = i; 
        std::vector<int> widths;             
        int count = gotao_nthreads;        
        std::vector<int> temp;        // hold the big divisors, so that the final list of widths is in sorted order 
        for(int i = 1; i < sqrt(gotao_nthreads); ++i){ 
          if(gotao_nthreads % i == 0) {
            widths.push_back(i);
            temp.push_back(gotao_nthreads / i); 
          } 
        }
        std::reverse(temp.begin(), temp.end());
        widths.insert(widths.end(), temp.begin(), temp.end());
        //std::reverse(widths.begin(), widths.end());        
        for(int i = 0; i < widths.size(); ++i) {
          for(int j = 0; j < gotao_nthreads; j+=widths[i]){
            ptt_layout[j].push_back(widths[i]);
          }
        }

        for(int i = 0; i < gotao_nthreads; ++i){
          for(auto&& width : ptt_layout[i]){
            for(int j = 0; j < width; ++j) {                
              inclusive_partitions[i + j].push_back(std::make_pair(i, width)); 
            }         
          }
        }
      } 
  } else {    
    if(gotao_nthreads != runtime_resource_mapper.size()) {
      std::cout << "Warning: requested " << runtime_resource_mapper.size() << " at runtime, whereas gotao_nthreads is set to " << gotao_nthreads <<". Runtime value will be used" << std::endl;
      gotao_nthreads = runtime_resource_mapper.size();
    }            
  }
#ifdef DEBUG
	std::cout << "XiTAO initialized with " << gotao_nthreads << " threads and configured with " << XITAO_MAXTHREADS << " max threads " << std::endl;
  // std::cout << "The platform has " << cluster_mapper.size() << " clusters.\n";
  // for(int i = 0; i < cluster_mapper.size(); i++){
  //   std::cout << "[DEBUG] Cluster " << i << " has " << cluster_mapper[i] << " cores.\n";
  // }
  for(int i = 0; i < static_resource_mapper.size(); ++i) {
    std::cout << "[DEBUG] Thread " << i << " is configured to be mapped to core id : " << static_resource_mapper[i] << std::endl;
    std::cout << "[DEBUG] PTT Layout Size of thread " << i << " : " << ptt_layout[i].size() << std::endl;
    std::cout << "[DEBUG] Inclusive partition size of thread " << i << " : " << inclusive_partitions[i].size() << std::endl;
    std::cout << "[DEBUG] leader thread " << i << " has partition widths of : ";
    for (int j = 0; j < ptt_layout[i].size(); ++j){
      std::cout << ptt_layout[i][j] << " ";
    }
    std::cout << std::endl;
    std::cout << "[DEBUG] thread " << i << " is contained in these [leader,width] pairs : ";
    for (int j = 0; j < inclusive_partitions[i].size(); ++j){
      std::cout << "["<<inclusive_partitions[i][j].first << "," << inclusive_partitions[i][j].second << "]";
    }
    std::cout << std::endl;
  }
#endif
  std::ifstream FreqReader, SupportedCPUFreq, SupportedDDRFreq; // Read the supported CPU Frequency Profile File
  for(int i = 0; i < NUMSOCKETS; ++i) {
    FreqReader.open("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[start_coreid[i]]) + "/cpufreq/scaling_cur_freq");
    if(FreqReader.fail()){
      std::cout << "Failed to open scaling_cur_freq(CPU) file!" << std::endl;
      std::cin.get();
      return 0;
    }
    std::string token;
    while(std::getline(FreqReader, token)) {
      std::istringstream line(token);
      while(line >> token) {
        long detected_cpu_freq = stol(token);
        cur_freq[i] = detected_cpu_freq;
      }
    }
    FreqReader.close();
  }

  FreqReader.open("/sys/kernel/debug/bpmp/debug/clk/emc/rate");
  if(FreqReader.fail()){
    std::cout << "Failed to open cur_freq(DDR) file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token;
  while(std::getline(FreqReader, token)) {
    std::istringstream line(token);
    while(line >> token) {
      long detected_ddr_freq = stol(token);
      cur_ddr_freq = detected_ddr_freq;
    }
  }
  FreqReader.close();

  // char buffer[256];  // Print out the current work path
  // char *val = getcwd(buffer, sizeof(buffer));
  // if (val) {
  //     std::cout << buffer << std::endl;
  // }
  SupportedCPUFreq.open("Supported_CPU_Freq.txt");
  SupportedDDRFreq.open("Supported_DDR_Freq.txt");
  if(SupportedCPUFreq.fail() || SupportedDDRFreq.fail()){
    std::cout << "Failed to open supported CPU / DDR frequency profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token1, token2;
  while(std::getline(SupportedDDRFreq, token1)) {
    std::istringstream line(token1);
    int ii = 0;
    while(line >> token1) {
      long ddrfreq = stol(token1);
      avail_ddr_freq[ii] = ddrfreq;
      ii++;
    }
  }
  while(std::getline(SupportedCPUFreq, token2)) {
    std::istringstream line(token2);
    int ii = 0; // Column index of power files
    int cluster_id = 0;
    while(line >> token2) {
      if(ii == 0){
        cluster_id = stoi(token2); // first column is DDR frequency index
      }else{
        long cpufreq = stol(token2);
        avail_freq[cluster_id][ii-1] = cpufreq;
      }
      ii++;
    }
  }
  SupportedCPUFreq.close();
  SupportedDDRFreq.close();

  for(int i = 0; i < NUMSOCKETS; ++i) {
    if(cur_freq[i] != avail_freq[i][0]){ // If the starting frequency is not the highest
      std::cout << "Detected frequency of cluster " << i << ": " << cur_freq[i] << ". Throttle to the highest now! \n";
      std::ofstream Throttle("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[start_coreid[i]]) + "/cpufreq/scaling_setspeed");
      if (!Throttle.is_open()){
        std::cerr << "[DEBUG] failed while opening the scaling_setspeed file! " << std::endl;
        return 0;
      }
      Throttle << std::to_string(avail_freq[i][0]) << std::endl;
      Throttle.close();
    }
  }

  if(cur_ddr_freq != avail_ddr_freq[0]){
    std::cout << "Detected frequency of DDR: " << cur_ddr_freq << ". Throttle to the highest now! \n";
    std::ofstream EMC("/sys/kernel/debug/bpmp/debug/clk/emc/rate"); // edit chip memory frequency - TX2 specific
    if (!EMC.is_open()){
      std::cerr << "[DEBUG] failed while opening the DDR setspeed file! " << std::endl;
      return 0;
    }
    EMC << std::to_string(avail_ddr_freq[0]) << std::endl;
    EMC.close();
  }
  cur_ddr_freq_index = 0; // Start by the highest DDR frequency
  cur_freq_index[NUMSOCKETS] = {0}; // Start by the highest CPU frequency

  if(nhwc>=0){
    gotao_ncontexts = nhwc;
  }
  else{
    if(getenv("GOTAO_HW_CONTEXTS")){
      gotao_ncontexts = atoi(getenv("GOTAO_HW_CONTEXTS"));
    }
    else{ 
      gotao_ncontexts = GOTAO_HW_CONTEXTS;
    }
  }

#if defined(Haswell)
  if(nhwc >= 0){
    num_sockets = nhwc;
  }
  else{
    if(getenv("NUMSOCKETS")){
      num_sockets = atoi(getenv("NUMSOCKETS"));
    }
    else{
      num_sockets = NUMSOCKETS;
    }
  } 
#endif

  if(thrb>=0){
    gotao_thread_base = thrb;
  }
  else{
    if(getenv("GOTAO_THREAD_BASE")){
      gotao_thread_base = atoi(getenv("GOTAO_THREAD_BASE"));
    }
    else{
      gotao_thread_base = GOTAO_THREAD_BASE;
    }
  }
/*
  starting_barrier = new BARRIER(gotao_nthreads + 1);
  tao_barrier = new cxx_barrier(2);
  for(int i = 0; i < gotao_nthreads; i++){
    t[i]  = new std::thread(worker_loop, i);   
  }
*/  
}

// Initialize gotao from environment vars or defaults
int gotao_init(int scheduler, int numkernels, int STEAL_DtoA, int STEAL_AtoD){
  starting_barrier = new BARRIER(gotao_nthreads);
  tao_barrier = new cxx_barrier(gotao_nthreads);
  for(int i = 0; i < gotao_nthreads; i++){
    t[i]  = new std::thread(worker_loop, i);
  }
  Sched = scheduler;
  num_kernels = numkernels;
  maySteal_DtoA = STEAL_DtoA;
  maySteal_AtoD = STEAL_AtoD;
#ifdef DynaDVFS
  current_freq = 2035200; // ERASE starting frequency is 2.04GHz for both clusters
  env = 0;
#endif
}

int gotao_start()
{
  /*
  if(Sched == 0){
  //Analyse DAG based on tasks in ready q and asign criticality values
  for(int j=0; j<gotao_nthreads; j++){
    //Iterate over all ready tasks for all threads
    for(std::list<PolyTask *>::iterator it = worker_ready_q[j].begin(); it != worker_ready_q[j].end(); ++it){
      //Call recursive function setting criticality
      (*it)->set_criticality();
    }
  }
  for(int j = 0; j < gotao_nthreads; j++){
    for(std::list<PolyTask *>::iterator it = worker_ready_q[j].begin(); it != worker_ready_q[j].end(); ++it){
      if ((*it)->criticality == critical_path){
        (*it)->marker = 1;
        (*it) -> set_marker(1);
        break;
      }
    }
  }
  }
  */
  starting_barrier->wait(gotao_nthreads+1);
}

int gotao_fini()
{
  resources_runtime_conrolled = false;
  gotao_can_exit = true;
  gotao_initialized = false;
  for(int i = 0; i < gotao_nthreads; i++){
    t[i]->join();
  }
}

void gotao_barrier()
{
  tao_barrier->wait();
}

int check_and_get_available_queue(int queue) {
  bool found = false;
  if(queue >= runtime_resource_mapper.size()) {
    return rand()%runtime_resource_mapper.size();
  } else {
    return queue;
  }  
}
// push work into polytask queue
// if no particular queue is specified then try to determine which is the local
// queue and insert it there. This has some overhead, so in general the
// programmer should specify some queue
int gotao_push(PolyTask *pt, int queue)
{
  if((queue == -1) && (pt->affinity_queue != -1)){
    queue = pt->affinity_queue;
  }
  else{
    if(queue == -1){
      queue = sched_getcpu();
    }
  }
  if(resources_runtime_conrolled) queue = check_and_get_available_queue(queue);
  LOCK_ACQUIRE(worker_lock[queue]);
  worker_ready_q[queue].push_front(pt);
  LOCK_RELEASE(worker_lock[queue]);
}

// Push work when not yet running. This version does not require locks
// Semantics are slightly different here
// 1. the tid refers to the logical core, before adjusting with gotao_thread_base
// 2. if the queue is not specified, then put everything into the first queue
int gotao_push_init(PolyTask *pt, int queue)
{
  if((queue == -1) && (pt->affinity_queue != -1)){
    queue = pt->affinity_queue;
  }
  else{
    if(queue == -1){
      queue = gotao_thread_base;
    }
  }
  if(resources_runtime_conrolled) queue = check_and_get_available_queue(queue);
  worker_ready_q[queue].push_front(pt);
}

// alternative version that pushes to the back
int gotao_push_back_init(PolyTask *pt, int queue)
{
  if((queue == -1) && (pt->affinity_queue != -1)){
    queue = pt->affinity_queue;
  }
  else{
    if(queue == -1){
      queue = gotao_thread_base;
    }
  }
  worker_ready_q[queue].push_back(pt);
}


long int r_rand(long int *s)
{
  *s = ((1140671485*(*s) + 12820163) % (1<<24));
  return *s;
}


void __xitao_lock()
{
  smpd_region_lock.lock();
  //LOCK_ACQUIRE(smpd_region_lock);
}
void __xitao_unlock()
{
  smpd_region_lock.unlock();
  //LOCK_RELEASE(smpd_region_lock);
}

int worker_loop(int nthread){
  // pmc.open("PMC.txt", std::ios_base::app);
  // std::ofstream timetask;
  // timetask.open("data_process.sh", std::ios_base::app);

  int phys_core;
  if(resources_runtime_conrolled) {
    if(nthread >= runtime_resource_mapper.size()) {
      LOCK_ACQUIRE(output_lck);
      std::cout << "Error: thread cannot be created due to resource limitation" << std::endl;
      LOCK_RELEASE(output_lck);
      exit(0);
    }
    phys_core = runtime_resource_mapper[nthread];
  } else {
    phys_core = static_resource_mapper[gotao_thread_base+(nthread%(XITAO_MAXTHREADS-gotao_thread_base))];   
  }
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] nthread: " << nthread << " mapped to physical core: "<< phys_core << std::endl;
  LOCK_RELEASE(output_lck);
#endif  
  unsigned int seed = time(NULL);
  cpu_set_t cpu_mask;
  CPU_ZERO(&cpu_mask);
  CPU_SET(phys_core, &cpu_mask);

  sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask); 
  // When resources are reclaimed, this will preempt the thread if it has no work in its local queue to do.
  
  PolyTask *st = nullptr;
  starting_barrier->wait(gotao_nthreads+1);  
  auto&&  partitions = inclusive_partitions[nthread];

  // Perf Event Counters
  struct perf_event_attr pea;
  int fd1;
  uint64_t id1, val1;
  char buf[4096];
  struct read_format* rf = (struct read_format*) buf;

  memset(&pea, 0, sizeof(struct perf_event_attr));
  pea.type = PERF_TYPE_HARDWARE;
  pea.size = sizeof(struct perf_event_attr);
  pea.config = PERF_COUNT_HW_CPU_CYCLES;
  pea.disabled = 1;
  pea.exclude_kernel = 0;
  pea.exclude_hv = 1;
  pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  fd1 = syscall(__NR_perf_event_open, &pea, 0, phys_core, -1, 0);
  ioctl(fd1, PERF_EVENT_IOC_ID, &id1);



  int idle_try = 0;
  int idle_times = 0;
  int SleepNum = 0;
  int AccumTime = 0;

  if(Sched == 1){
    for(int i=0; i<XITAO_MAXTHREADS; i++){ 
      status[i] = 1;
      status_working[i] = 0;
    }
  }
  bool stop = false;

  // Accumulation of tasks execution time
  // Goal: Get runtime idle time
#ifdef EXECTIME
  //std::chrono::time_point<std::chrono::system_clock> idle_start, idle_end;
  std::chrono::duration<double> elapsed_exe;
#endif

#ifdef OVERHEAD_PTT
  std::chrono::duration<double> elapsed_ptt;
#endif

#ifdef PTTaccuracy
  float MAE = 0.0f;
  std::ofstream PTT("PTT_Accuracy.txt");
#endif

#ifdef Energyaccuracy
  float EnergyPrediction = 0.0f;
#endif

  while(true){    
    int random_core = 0;
    AssemblyTask *assembly = nullptr;
    SimpleTask *simple = nullptr;

  // 0. If a task is already provided via forwarding then exeucute it (simple task)
  //    or insert it into the assembly queues (assembly task)
    if( st && !stop){
      if(st->type == TASK_SIMPLE){
        SimpleTask *simple = (SimpleTask *) st;
        simple->f(simple->args, nthread);
  #ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Distributing simple task " << simple->taskid << " with width " << simple->width << " to workers " << nthread << std::endl;
        LOCK_RELEASE(output_lck);
  #endif
  #ifdef OVERHEAD_PTT
        st = simple->commit_and_wakeup(nthread, elapsed_ptt);
  #else
        st = simple->commit_and_wakeup(nthread);
  #endif
        simple->cleanup();
        //delete simple;
      }
      else 
      if(st->type == TASK_ASSEMBLY){
        AssemblyTask *assembly = (AssemblyTask *) st;
// #if defined(TX2)
//         if(Sched == 3){
//           assembly->leader = nthread / assembly->width * assembly->width; // homogenous calculation of leader core
//         }
// #endif
#if defined(Haswell) || defined(CATS)
        assembly->leader = nthread / assembly->width * assembly->width;
#endif
        /* In some applications, pretty loose task dependencies (very high parallelism), so here we need to check if tasks are going to run with best config.
        Otherwise, tasks will be executed with the original setting config */
        if(global_training == true && assembly->tasktype < num_kernels){ 
          if(assembly->get_bestconfig_state() == false){    
#if defined Exhastive_Search
            assembly->find_best_config(nthread, assembly);
#endif
#if defined Optimized_Search
            assembly->optimized_search(nthread, assembly);
#endif
            assembly->set_bestconfig_state(true);
          }else{
            if(assembly->get_bestconfig == false){
              assembly->update_best_config(nthread, assembly);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[Jing-DEBUG] assembly->get_bestconfig_state() = " << assembly->get_bestconfig_state() <<". Task " << assembly->taskid << " is wrong here? " << std::endl;
              LOCK_RELEASE(output_lck);
#endif
            }
          }
        }
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Distributing " << assembly->kernel_name << " task " << assembly->taskid << " with width " << assembly->width << " to workers [" << assembly->leader << "," << assembly->leader + assembly->width << ")" << std::endl;
        LOCK_RELEASE(output_lck);
#endif
        /* After getting the best config, and before distributing to AQs: 
        (1) Coarse-grained task: check if it is needed to tune the frequency; 
        (2) Fine-grained task: check the WQs of the cluster include N consecutive same tasks, that the total execution time of these N tasks > threshold, then search for the best frequency and then tune the frequency */
        if(global_training == true && assembly->get_bestconfig_state() == true){
#ifdef AcrossCLustersTest
          int best_cluster = assembly->leader < 2? 0:1; // MatrixMul test across clusters, A57 steal task from Denver, so best cluster becomes 1
#else
          int best_cluster = assembly->get_best_cluster();
#endif
          int best_width = assembly->get_best_numcores();
          if(assembly->granularity_fine == true && assembly->get_enable_cpu_freq_change() == false && assembly->get_enable_ddr_freq_change() == false){ /* (2) Fine-grained tasks && not allowed to do frequency change currently */
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Assembly Task " << assembly->taskid << " is a fine-grained task. " << std::endl;
          LOCK_RELEASE(output_lck);
#endif        
          int consecutive_fine_grained = 1; /* assembly already is one fine-grained task, so initilize to 1 */
          for(int i = 0; i < 8; i++){ /* Assume the maximum of each work queue size is 8 */
            int task_of_this_round = 0; // count the task of the same type in each round (startcore to endcore)
            for(int j = start_coreid[best_cluster]; j < end_coreid[best_cluster]; j++){
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] Visit the " << i << "th task in the queues, visit the " << j << "th core in cluster " << best_cluster << std::endl;
              LOCK_RELEASE(output_lck);
#endif 
              if(worker_ready_q[j].size() > i){
                std::list<PolyTask *>::iterator it = worker_ready_q[j].begin();
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] The queue size of " << j << "th core > 0. Now point to the " << i << "th task. " << std::endl;
                LOCK_RELEASE(output_lck);
#endif 
                if(i > 0){
                  std::advance(it, i);
                }
                if((*it)->granularity_fine == true && (*it)->tasktype == assembly->tasktype){
                  task_of_this_round++;
                  consecutive_fine_grained++; /* Another same type of task */
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] The pointing task *it is also a same type + fine-grained task as the assembly task. consecutive_fine_grained = " << consecutive_fine_grained << std::endl;
                  LOCK_RELEASE(output_lck);
#endif 
                  if(consecutive_fine_grained * assembly->get_timetable(cur_ddr_freq_index, cur_freq_index[best_cluster], best_cluster, best_width-1) * (end_coreid[best_cluster] - start_coreid[best_cluster]) / best_width > FINE_GRAIN_THRESHOLD){
                    // find out the best frequency here
#ifdef DEBUG
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "[DEBUG] Enough same type of tasks! Find out the best frequency now for task " << assembly->taskid << std::endl;
                    LOCK_RELEASE(output_lck);
#endif                    
                    float idleP_cluster = 0.0f;
                    float shortest_exec = 100000.0f;
                    float energy_pred = 0.0f;
                    int sum_cluster_active = std::accumulate(status+start_coreid[1-best_cluster], status+end_coreid[1-best_cluster], 0); /* If there is any active cores in another cluster */
// #ifdef DEBUG
//                     LOCK_ACQUIRE(output_lck);
//                     std::cout << "[DEBUG] Number of active cores in cluster " << 1-best_cluster << ": " << sum_cluster_active << ". status[0] = " << status[0] \
//                     << ", status[1] = " << status[1] << ", status[2] = " << status[2] << ", status[3] = " << status[3] << ", status[4] = " << status[4] << ", status[5] = " << status[5] << std::endl;
//                     LOCK_RELEASE(output_lck);
// #endif
                    for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                        if(sum_cluster_active == 0){ /* the number of active cores is zero in another cluster */
                          idleP_cluster = idle_power[ddr_freq_indx][freq_indx][best_cluster] + idle_power[ddr_freq_indx][freq_indx][1-best_cluster]; /* Then equals idle power of whole chip */
// #ifdef DEBUG
//                           LOCK_ACQUIRE(output_lck);
//                           std::cout << "[DEBUG] Cluster " << 1-best_cluster << " no active cores. Therefore, the idle power of cluster " << best_cluster << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
//                           LOCK_RELEASE(output_lck);
// #endif 
                        }else{
                          idleP_cluster = idle_power[ddr_freq_indx][freq_indx][best_cluster]; /* otherwise, equals idle power of the cluster */
// #ifdef DEBUG
//                           LOCK_ACQUIRE(output_lck);
//                           std::cout << "[DEBUG] Cluster " << 1-best_cluster << " has active cores. Therefore, the idle power of cluster " << best_cluster << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
//                           LOCK_RELEASE(output_lck);
// #endif 
                        }
                        sum_cluster_active = (sum_cluster_active < best_width)? best_width : sum_cluster_active;
                        float idleP = idleP_cluster * best_width / sum_cluster_active;
                        float CPUPowerP = assembly->get_cpupowertable(ddr_freq_indx, freq_indx, best_cluster, best_width-1);
                        float DDRPowerP = assembly->get_ddrpowertable(ddr_freq_indx, freq_indx, best_cluster, best_width-1);
                        float timeP = assembly->get_timetable(ddr_freq_indx, freq_indx, best_cluster, best_width-1);
                        energy_pred = timeP * (CPUPowerP - idleP_cluster + idleP + DDRPowerP); 
#ifdef DEBUG
                        LOCK_ACQUIRE(output_lck);
                        std::cout << "[DEBUG] For the fine-grained tasks, Memory frequency: " << avail_ddr_freq[ddr_freq_indx] <<  ", CPU frequency: " << avail_freq[best_cluster][freq_indx] << " on cluster " << best_cluster << " with width "<< best_width \
                        << ", CPU power " << CPUPowerP- idleP_cluster + idleP << ", Memory power " << DDRPowerP << ", execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
                        LOCK_RELEASE(output_lck);
#endif 
                        if(energy_pred < shortest_exec){
                          shortest_exec = energy_pred;
                          assembly->set_best_cpu_freq(freq_indx);
                          assembly->set_best_ddr_freq(ddr_freq_indx);
                        }
                      }
                    }
#ifdef DEBUG
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "[DEBUG] For the fine-grained tasks, get the optimal CPU and Memory frequency: " << avail_freq[best_cluster][assembly->get_best_cpu_freq()] << ", " << avail_ddr_freq[assembly->get_best_ddr_freq()] << std::endl;
                    LOCK_RELEASE(output_lck);
#endif 
                    (*it)->set_enable_cpu_freq_change(true); // Set the frequency change state for the current testing fine-grained task
                    (*it)->set_enable_ddr_freq_change(true);
                    goto consecutive_true; // No more searching
                    // assembly->set_enable_cpu_freq_change(true); // Set the frequency change state for the current fine-grained task that to be scheduled
                    // assembly->set_enable_ddr_freq_change(true);
                    // break;
                  }else{
                    continue;
                  }
                }else{
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] Encounter other type of tasks! Task " << assembly->taskid << " is not allowed to do frequency change (both)." << std::endl;
                  LOCK_RELEASE(output_lck);
#endif 
                  goto consecutive_false;
//                   assembly->set_enable_cpu_freq_change(false); 
//                   assembly->set_enable_ddr_freq_change(false);
//                   break;
                }
              }else{
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] No (more) task in " << j << "th core's queue." << std::endl;
                LOCK_RELEASE(output_lck);
#endif 
                continue;
              } 
            }
            if(consecutive_fine_grained == 1 || task_of_this_round == 0){ // if the beginning tasks of the cluster cores are not the same task OR no tasks at all
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] Loop for a round in the cluster and find no same type of tasks. So task " << assembly->taskid << " is not allowed to do frequency change (both)." << std::endl;
              LOCK_RELEASE(output_lck);
#endif 
              // assembly->set_enable_cpu_freq_change(false); 
              // assembly->set_enable_ddr_freq_change(false);
              // break; // exit, no need to loop for another round
              goto consecutive_false;
            }
          }
          consecutive_true: {
            assembly->set_enable_cpu_freq_change(true); 
            assembly->set_enable_ddr_freq_change(true);
          } /* TBD Problem: should mark traversed tasks, next following tasks should do the process again, since the DAG might include other type of tasks */
          consecutive_false: {
            assembly->set_enable_cpu_freq_change(false); 
            assembly->set_enable_ddr_freq_change(false);
          }
        }
          
        /*Tune CPU frequency if required != current, for both fine-grained and coarse-grained tasks */
        if(assembly->get_enable_cpu_freq_change() == true){ /* Allow to do frequency tuning for the tasks */
          assembly->best_cpu_freq = avail_freq[best_cluster][assembly->get_best_cpu_freq()];
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " is allowed to do CPU Frequency scaling. \n";
          LOCK_RELEASE(output_lck);
#endif
#ifndef FineStrategyTest /* The aim is to test energy difference when the DVFS is also applied to fine-grained tasks, need to enable #define threshold 0.0001 in include/config.h */
          if(assembly->best_cpu_freq != cur_freq[best_cluster]){ /* check if the required frequency equals the current frequency! */
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] For " << assembly->kernel_name << " task " << assembly->taskid << ": current frequency " << cur_freq[best_cluster] << " != required frequency " << assembly->best_cpu_freq << ". \n";
            LOCK_RELEASE(output_lck);
#endif
#else
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DVFSforFineGrained] For " << assembly->kernel_name << " task " << assembly->taskid << ": current frequency " << cur_freq[best_cluster] << ", required frequency " << assembly->best_cpu_freq << ". \n";
            LOCK_RELEASE(output_lck);
#endif
#endif
            if(best_width == (end_coreid[best_cluster] - start_coreid[best_cluster])){  /* ==> Strategy for tuning the frequency: if the best width = number of cores in cluster, just change frequency! */
              assembly->cpu_frequency_tuning(nthread, best_cluster, assembly->get_best_cpu_freq());
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] Best width: " << best_width << " equals the number of cores in the cluster. Change the frequency now! =====> current frequency[Denver] = " \
              << cur_freq[0] << ", current frequency[A57] = " << cur_freq[1] << std::endl;
              LOCK_RELEASE(output_lck);
#endif
            }else{
              // int freq_change_cluster_active = std::accumulate(status_working + start_coreid[best_cluster], status_working + end_coreid[best_cluster], 0); 
              if(std::accumulate(status_working + start_coreid[best_cluster], status_working + end_coreid[best_cluster], 0) > 0){ /* Check if there are any concurrent tasks? Yes, take the average, no, change the frequency to the required! */
                int new_freq_index = (cur_freq_index[best_cluster] + assembly->get_best_cpu_freq()) / 2;   /* Method 1: take the average of two frequencies */
                assembly->cpu_frequency_tuning(nthread, best_cluster, new_freq_index);
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] CPU frequency tuning: The cluster has concurrent tasks running at the same time! Strategy: take the average, tune the frequency to " << avail_freq[best_cluster][new_freq_index] \
                << " =====> current frequency[Denver] = " << cur_freq[0] << ", current frequency[A57] = " << cur_freq[1] << std::endl;
                LOCK_RELEASE(output_lck);
#endif 
              }else{
                assembly->cpu_frequency_tuning(nthread, best_cluster, assembly->get_best_cpu_freq());
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] CPU frequency tuning: The cluster has no tasks running now! Change the cluster frequency now! =====> current frequency[Denver] = " << cur_freq[0] << ", current frequency[A57] = " << cur_freq[1] << std::endl;
                LOCK_RELEASE(output_lck);
#endif
              }
            }
#ifndef FineStrategyTest
          }
#endif
        }
        /* Tune DDR frequency if required != current, for both fine-grained and coarse-grained tasks */
        if(assembly->get_enable_ddr_freq_change() == true){
          // assembly->best_ddr_freq = avail_ddr_freq[assembly->get_best_ddr_freq()];
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " is allowed to do Memory Frequency Scaling. Its best DDR freq index = " << assembly->get_best_ddr_freq() << ". Current DDR frequency is = " << cur_ddr_freq << ", index = " << cur_ddr_freq_index << ". \n";
          LOCK_RELEASE(output_lck);
#endif
          if(assembly->get_best_ddr_freq() != cur_ddr_freq_index){
            if(std::accumulate(std::begin(status_working), status_working + assembly->leader, 0) + std::accumulate(status_working + assembly->leader + assembly->width, std::end(status_working), 0) > 0){  /* Check if there are any concurrent tasks? No, change the frequency to the required! */
              int new_ddr_freq_index = (cur_ddr_freq_index + assembly->get_best_ddr_freq()) / 2;   /* Method 1: take the average of two frequencies */
              assembly->ddr_frequency_tuning(nthread, new_ddr_freq_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] DDR frequency tuning: concurrent tasks running at the same time! Strategy: take the average, tune the frequency to " << avail_ddr_freq[new_ddr_freq_index] \
              << " =====> current memory frequency = " << cur_ddr_freq << std::endl;
              LOCK_RELEASE(output_lck);
#endif 
            }else{ /* Check if there are any concurrent tasks? No, change the frequency to the required! */
              assembly->ddr_frequency_tuning(nthread, assembly->get_best_ddr_freq());
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] DDR frequency tuning: No other tasks running now! Change the memory frequency now! =====> current memory frequency = " << cur_ddr_freq << std::endl;
              LOCK_RELEASE(output_lck);
#endif
            }
          }
        }
      }

        for(int i = assembly->leader; i < assembly->leader + assembly->width; i++){
	        LOCK_ACQUIRE(worker_assembly_lock[i]);
          worker_assembly_q[i].push_back(st);
#ifdef NUMTASKS_MIX
#ifdef ONLYCRITICAL
          int pr = assembly->if_prio(nthread, assembly);
          if(pr == 1){
#endif
            num_task[assembly->tasktype][assembly->width * gotao_nthreads + i]++;
#ifdef ONLYCRITICAL
          }
#endif
#endif
          // LOCK_RELEASE(worker_assembly_lock[i]);
        }
        for(int i = assembly->leader; i < assembly->leader + assembly->width; i++){
          LOCK_RELEASE(worker_assembly_lock[i]);
        }
        st = nullptr;
      }
      continue;
    }

    // 1. check for assemblies
    if(!worker_assembly_q[nthread].pop_front(&st)){
      st = nullptr;
    }
  // assemblies are inlined between two barriers
    if(st) {
      int _final = 0; // remaining
      assembly = (AssemblyTask *) st;
#ifdef NEED_BARRIER
      if(assembly->width > 1){
        assembly->barrier->wait(assembly->width);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout <<"[BARRIER-Before] For Task " << assembly->taskid << ", thread "<< nthread << " arrives." << std::endl;
        LOCK_RELEASE(output_lck);
#endif
      }
#endif
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] Thread "<< nthread << " starts executing " << assembly->kernel_name << " task " << assembly->taskid << "!\n";
      LOCK_RELEASE(output_lck);
#endif
      // if(Sched == 1 && nthread == assembly->leader){
      if(Sched == 1 && assembly->start_running == false){ /* for tasks with wider width, if this is the first thread, set the start running frequency, this is meant to avoid updating the performance table when frequency changing happens between threads execution */
#if defined Performance_Model_Cycle 
        ioctl(fd1, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fd1, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
        assembly->start_running = true;
        int clus_id = (nthread < START_CLUSTER_B)? 0:1;
        assembly->start_running_freq = cur_freq[clus_id];
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Task "<< assembly->taskid << " starts from frequency " << assembly->start_running_freq << ", assembly->start_running = " << assembly->start_running << "!\n";
        LOCK_RELEASE(output_lck);
#endif
      }
      status_working[nthread] = 1; // The core/thread is going to work on task, so set status as 1 
      std::chrono::time_point<std::chrono::system_clock> t1,t2;
      t1 = std::chrono::system_clock::now();
      auto start1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(t1);
      auto epoch1 = start1_ms.time_since_epoch();
      
      assembly->execute(nthread);

      t2 = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = t2-t1;
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " execution time on thread " << nthread << ": " << elapsed_seconds.count() << "\n";
      LOCK_RELEASE(output_lck);
#endif 
      status_working[nthread] = 0;  // The core/thread finished task, so set status as 0
      // if(Sched == 1 && nthread == assembly->leader){
#if defined Performance_Model_Cycle 
      if(Sched == 1){
        ioctl(fd1, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        read(fd1, buf, sizeof(buf));
        for (int i = 0; i < rf->nr; i++){
          if (rf->values[i].id == id1) {
              val1 = rf->values[i].value;
          }
        }
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " execution time on thread " << nthread << ": " << elapsed_seconds.count() << ", cycles: " << val1 << "\n";
        LOCK_RELEASE(output_lck);
#endif 
      }
#endif
#ifdef PowerProfiling
      auto end1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(t2);
      auto epoch1_end = end1_ms.time_since_epoch();
      out << assembly->kernel_name << "\t" << epoch1.count() << "\t" << epoch1_end.count() << "\n";
      out.flush();
#endif
#ifdef NEED_BARRIER
      if(assembly->width > 1){
        assembly->barrier->wait(assembly->width);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout <<"[BARRIER-After] For Task " << assembly->taskid << " thread  "<< nthread << " arrives." << std::endl;
        LOCK_RELEASE(output_lck);
#endif
      }
#endif
#ifdef EXECTIME
      elapsed_exe += t2-t1;
#endif
      double ticks = elapsed_seconds.count(); 
      if(Sched == 1 && ticks > 0.00001 && assembly->get_timetable_state(2) == false && assembly->tasktype < num_kernels){  /* Only leader core update the PTT entries */ // Make sure that Null tasks do not used to update PTT tables
        int width_index = assembly->width - 1;
        /* (1) Was running on Denver (2) Denver PTT table hasn't finished training     (3) if the frequency changing is happening during the task execution, do not update the table*/
        if(assembly->leader < START_CLUSTER_B){
        if(assembly->get_timetable_state(0) == false && assembly->start_running_freq == cur_freq[0]){
          /* Step 1: Update PTT table values */
          if(ptt_freq_index[0] == 0){ /*2.04GHz*/ 
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(0, 0, 0, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = float(std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0)) / float(assembly->width);
                if(oldticks == 0.0f || (newticks < oldticks && fabs(newticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, 0, 0, newticks, width_index);
                }else{
                  assembly->set_timetable(0, 0, 0, (newticks+oldticks)/2, width_index);
                }
              }else{
                if(oldticks == 0.0f || (ticks < oldticks && fabs(ticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, 0, 0, ticks, width_index);
                }else{
                  assembly->set_timetable(0, 0, 0, (ticks+oldticks)/2, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(0, 0, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[1.866GHz, 2.04GHz, Denver, width = " \
              << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(0,0, width_index) << ". Current time: " << assembly->get_timetable(0, 0, 0, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              //++assembly->threads_out_tao;
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[Jing] assembly->threads_out_tao = " << assembly->threads_out_tao << ". Store " << ticks << " to assembly->temp_ticks[" << nthread - assembly->leader << "]. \n";
//               LOCK_RELEASE(output_lck);
// #endif
            }
//             float oldticks = assembly->get_timetable(0, 0, width_index);
//             if(oldticks == 0.0f || ticks < oldticks){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
//               assembly->set_timetable(0, 0, ticks, width_index);
// #if defined Performance_Model_Cycle   
//               assembly->set_cycletable(0, 0, val1, width_index);
// #endif
//             }
//             // else{
//               // assembly->set_timetable(0, 0, ((oldticks + ticks)/2), width_index);  
//             // }
//             // if(nthread == assembly->leader){ 
//             assembly->increment_PTT_UpdateFinish(0, 0, width_index);
//             // }
          }else{ /*1.11GHz*/
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(0, NUM_AVAIL_FREQ/2, 0, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0) / assembly->width;
                if(oldticks == 0.0f || (newticks < oldticks && fabs(newticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, NUM_AVAIL_FREQ/2, 0, newticks, width_index);
                }else{
                  assembly->set_timetable(0, NUM_AVAIL_FREQ/2, 0, (newticks+oldticks)/2, width_index);
                }
              }else{
                if(oldticks == 0.0f || (ticks < oldticks && fabs(ticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, NUM_AVAIL_FREQ/2, 0, ticks, width_index);
                }else{
                  assembly->set_timetable(0, NUM_AVAIL_FREQ/2, 0, (ticks+oldticks)/2, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(1, 0, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[1.866GHz, 1.11GHz, Denver, width = " \
               << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(1,0, width_index) << ". Current time: " << assembly->get_timetable(0, NUM_AVAIL_FREQ/2, 0, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              //++assembly->threads_out_tao;
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
            }
//             float oldticks = assembly->get_timetable(NUM_AVAIL_FREQ/2, 0, width_index);
//             if(oldticks == 0.0f || ticks < oldticks){
//               assembly->set_timetable(NUM_AVAIL_FREQ/2, 0, ticks, width_index); 
// #if defined Performance_Model_Cycle  
//               assembly->set_cycletable(1, 0, val1, width_index);
// #endif
//             }
//             // else{
//             //   assembly->set_timetable(NUM_AVAIL_FREQ/2, 0, ((oldticks + ticks)/2), width_index);  
//             // }
//             // if(nthread == assembly->leader){ 
//             // assembly->PTT_UpdateFinish[1][0][width_index]++;
//             assembly->increment_PTT_UpdateFinish(1, 0, width_index);
//             // }
          }
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_UpdateFinish[" << ptt_freq_index[0] << ", Denver, width = " << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(ptt_freq_index[0],0, width_index) << ".\n";
//           LOCK_RELEASE(output_lck);
// #endif
//           /* Step 2: Update cycle table values */
//           uint64_t oldcycles = assembly->get_cycletable(ptt_freq_index[0], 0, width_index);
//           if(oldcycles == 0){
//             assembly->set_cycletable(ptt_freq_index[0], 0, val1, width_index);
//           }else{
//             assembly->set_cycletable(ptt_freq_index[0], 0, ((oldcycles + val1)/2), width_index);
//           }
//           if(assembly->get_timetable_state(0) == false){ /* PTT training hasn't finished */   

            if (ptt_freq_index[0] == 0){          /* Current frequency is 2.04GHz */
              if(assembly->get_PTT_UpdateFinish(0, 0, 0) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 0, 1) >= NUM_TRAIN_TASKS){ /* First dimention: 2.04GHz, second dimention: Denver, third dimention: width_index */
                PTT_finish_state[0][0][assembly->tasktype] = 1; /* First dimention: 2.04GHz, second dimention: Denver, third dimention: tasktype */
                if(std::accumulate(std::begin(PTT_finish_state[0][0]), std::begin(PTT_finish_state[0][0])+num_kernels, 0) == num_kernels){ /* Check all kernels have finished the training of Denver at 2.04GHz */
                  std::ofstream ClusterA("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[START_CLUSTER_A]) + "/cpufreq/scaling_setspeed"); // edit Denver cluster frequency
                  if (!ClusterA.is_open()){
                    std::cerr << "[DEBUG] Somthing failed while opening the file! " << std::endl;
                    return 0;
                  }
                  ClusterA << std::to_string(TRAIN_MED_A_FREQ) << std::endl;
                  ClusterA.close();
                  ptt_freq_index[0] = 1;
                  cur_freq[0] = TRAIN_MED_A_FREQ;
                  cur_freq_index[0] = NUM_AVAIL_FREQ/2;
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] Denver cluster completed PTT training at 2.04GHz, turn to 1.11GHz. \n";
                  LOCK_RELEASE(output_lck);
#endif
                }
              }
            }else{ /* Current frequency is 1.11GHz */
              int ptt_check = 0;            
              for(auto&& width : ptt_layout[START_CLUSTER_A]){ 
                float check_ticks = assembly->get_timetable(0, NUM_AVAIL_FREQ/2, 0, width - 1); /* First parameter is ptt_freq_index[0] = 12/2 = 6, 1.11GHz */
                if(assembly->get_PTT_UpdateFinish(1, 0, width-1) >= NUM_TRAIN_TASKS && check_ticks > 0.0f){ /* PTT_UpdateFinish first dimention is 1 => 1.11GHz */
                  ptt_check++;
                  if(assembly->get_mbtable(0, width-1) == 0.0f){ // If the memory-boundness of this config hasn't been computed yet
                    float memory_boundness = 0.0f;
#if defined Performance_Model_Cycle                  /* Method 1: Calculate Memory-boundness using cycles = (1 - cycle2/cycle1) / (1- f2/f1) */
                    uint64_t check_cycles = assembly->get_cycletable(1, 0, width - 1); /* get_cycletable first dimention is 1 => 1.11GHz */
                    uint64_t cycles_high = assembly->get_cycletable(0, 0, width-1); /* First parameter is 0, means 2.04GHz, check_cycles is at 1.11Ghz */
                    float a = 1 - float(check_cycles) / float(cycles_high);
                    float b = 1 - float(avail_freq[0][NUM_AVAIL_FREQ/2]) / float(avail_freq[0][0]);
                    memory_boundness = a/b;
#endif
#if defined Performance_Model_Time                   /* Method 2: Calculate Memory-boundness (using execution time only) = ((T2f2/T1)-f1) / (f2-f1) */
                    float highest_ticks = assembly->get_timetable(0, 0, 0, width - 1);
                    float a = float(avail_freq[0][0]) / float(avail_freq[0][NUM_AVAIL_FREQ/2]);
                    float b = check_ticks / highest_ticks;
                    memory_boundness = (b-a) / (1-a);
                    // LOCK_ACQUIRE(output_lck);
                    // std::cout << assembly->kernel_name << ": Memory-boundness Calculation (Denver, width " << width << ") = " << memory_boundness << ". a = " << a << ", b = " << b << std::endl;
                    // LOCK_RELEASE(output_lck);
#endif                    
                    if(memory_boundness > 1){
                      LOCK_ACQUIRE(output_lck);
                      std::cout << "[Warning]" << assembly->kernel_name << "->Memory-boundness (Denver) is greater than 1!" << std::endl;
                      LOCK_RELEASE(output_lck);
                      memory_boundness = 1;
                    }else{
                      if(memory_boundness <= 0){ /* Execution time and power prediction according to the computed memory-boundness level */
                        LOCK_ACQUIRE(output_lck);
                        std::cout << "[Warning]" << assembly->kernel_name << "->Memory-boundness (Denver) is smaller than 0!" << std::endl;
                        LOCK_RELEASE(output_lck);
                        memory_boundness = 0.00001;
                      }
                    }
                    assembly->set_mbtable(0, memory_boundness, width-1); /*first parameter: cluster 0 - Denver, second parameter: update value, third value: width_index */
                    // std::cout << "Set the MB table with the computation. Then start with the model prediction! \n"; 
#if defined Model_Computation_Overhead
                    // std::chrono::time_point<std::chrono::system_clock> Denver_model_start;
                    // Denver_model_start = std::chrono::system_clock::now();
                    // auto Denver_model_start_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(Denver_model_start);
                    // auto Denver_model_start_epoch = Denver_model_start_ms.time_since_epoch();
                    // LOCK_ACQUIRE(output_lck);
                    // std::cout << "[Overhead] Model calculation (Denver) starts from " << Denver_model_start_epoch.count() << ". " << std::endl;
                    // LOCK_RELEASE(output_lck);
                    struct timespec Denver_start, Denver_finish, Denver_delta;
                    clock_gettime(CLOCK_REALTIME, &Denver_start);
#endif
                    for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){ /* Compute Predictions according to Memory-boundness Values */
                      float ddr_freq_scaling = float(avail_ddr_freq[0]) / float(avail_ddr_freq[ddr_freq_indx]);
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                        if((ddr_freq_indx == 0 && freq_indx == 0) || (ddr_freq_indx == 0 && freq_indx == NUM_AVAIL_FREQ/2)){ /* if DDR freq = 1.866 and CPU freq = 2.04 or 1.11 GHz, then skip since they are sampled */
                          continue;
                        }else{
#if defined Performance_Model_Cycle 
                        uint64_t new_cycles = cycles_high * float(avail_freq[0][freq_indx])/float(avail_freq[0][0]);
                        float ptt_value_newfreq = float(new_cycles) / float(avail_freq[0][freq_indx]*1000);
#ifdef DEBUG
                        LOCK_ACQUIRE(output_lck);  
                        std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[1.11GHz, Denver, " << width << "] = " << check_ticks << ". Cycles(1.11GHz) = " << check_cycles << ", Cycles(2.04GHz) = " << cycles_high <<\
                        ". Memory-boundness(Denver, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
                        LOCK_RELEASE(output_lck);
#endif
#endif
#if defined Performance_Model_Time
                        float ptt_value_newfreq = 0.0;
                        float cpu_freq_scaling = float(avail_freq[0][0]) / float(avail_freq[0][freq_indx]);
                        if(width == 1){ /*Denver, width=1*/                     
#if defined Performance_Model_1
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 1.7250611 * memory_boundness + 0.0410749 * cpu_freq_scaling + 0.1339562 * ddr_freq_scaling - 0.2918719); 
#endif
#if defined Performance_Model_2
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling - 0.3217646 * memory_boundness + 0.0777825 * cpu_freq_scaling \
                          + 0.0979088 * ddr_freq_scaling + 0.2310173 * memory_boundness * cpu_freq_scaling + 0.9953363 * memory_boundness * ddr_freq_scaling \
                          - 0.0447902 * cpu_freq_scaling * ddr_freq_scaling - 0.1645463);
#endif
#if defined Performance_Model_3
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling - 0.3738012 * memory_boundness - 0.0719465 * cpu_freq_scaling \
                          - 0.0923319 * ddr_freq_scaling + 0.1895604 * pow(memory_boundness, 2) + 0.2310173 * memory_boundness * cpu_freq_scaling \
                          + 0.0229213 * pow(cpu_freq_scaling, 2) + 0.9953363 * memory_boundness * ddr_freq_scaling \
                          - 0.0447902 * cpu_freq_scaling * ddr_freq_scaling + 0.0567653 * pow(ddr_freq_scaling, 2) + 0.1594567);
#endif
// #ifdef DEBUG
//                           LOCK_ACQUIRE(output_lck);
//                           std::cout << "[DEBUG] " << assembly->kernel_name << "(Denver, 1): " << avail_ddr_freq[ddr_freq_indx] << ", " << avail_freq[freq_indx] << ", execution time prediction = " << ptt_value_newfreq << ".\n";
//                           LOCK_RELEASE(output_lck);
// #endif
                        }
                        if(width == 2){ /*Denver, width=2*/
#if defined Performance_Model_1
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 2.1204276 * memory_boundness + 0.0167277 * cpu_freq_scaling + 0.0903234 * ddr_freq_scaling - 0.1620744); 
#endif
#if defined Performance_Model_2
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling - 1.8822725 * memory_boundness + 0.1079102 * cpu_freq_scaling \
                          + 0.007308 * ddr_freq_scaling - 0.6263244 * memory_boundness * cpu_freq_scaling + 3.5389818 * memory_boundness * ddr_freq_scaling \
                          - 0.039597 * cpu_freq_scaling * ddr_freq_scaling - 0.1040418);
#endif
#if defined Performance_Model_3
                          // ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 1.1308933 * memory_boundness - 0.0018929 * cpu_freq_scaling \
                          // - 0.2420771 * ddr_freq_scaling - 30.7739177 * pow(memory_boundness, 2) - 0.6263244 * memory_boundness * cpu_freq_scaling \
                          // + 0.0168092 * pow(cpu_freq_scaling, 2) + 3.5389818 * memory_boundness * ddr_freq_scaling \
                          // - 0.039597 * cpu_freq_scaling * ddr_freq_scaling + 0.0744132 * pow(ddr_freq_scaling, 2) + 0.1485836);
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling - 1.3227937 * memory_boundness + 0.0945313 * cpu_freq_scaling \
                          +0.0407899 * ddr_freq_scaling - 0.328293 * memory_boundness * cpu_freq_scaling + 2.7087046 * memory_boundness * ddr_freq_scaling \
                          - 0.039641 * cpu_freq_scaling * ddr_freq_scaling - 0.1284175);
#endif
// #ifdef DEBUG
//                           LOCK_ACQUIRE(output_lck);
//                           std::cout << "[DEBUG] " << assembly->kernel_name << "(Denver, 2): " << avail_ddr_freq[ddr_freq_indx] << ", " << avail_freq[freq_indx] << ", execution time prediction = " << ptt_value_newfreq << ".\n";
//                           LOCK_RELEASE(output_lck);
// #endif
                        }
#endif
                        assembly->set_timetable(ddr_freq_indx, freq_indx, 0, ptt_value_newfreq, width-1);
                        }
                      }
                    }
                    for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){ /* (2) CPU, DDR Power value prediction */
                      float ddrfreq = float(avail_ddr_freq[ddr_freq_indx])/1000000000.0;
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ 
                        float cpupower = 0.0;
                        float ddrpower = 0.0;
                        float cpufreq = float(avail_freq[0][freq_indx])/1000000.0;                        
#if defined CPU_Power_Model_6
                        if(width == 1){ /*Denver, width=1*/
                          cpupower = 0.5134403 + 0.1343821 * memory_boundness - 0.3752914 * cpufreq + 0.8371783 * memory_boundness * cpufreq + 0.5086778 * pow(cpufreq, 2);
                        }
                        if(width == 2){ /*Denver, width=2*/
                          cpupower = 0.6918615 + 1.2556321 * memory_boundness - 0.5476326 * cpufreq - 0.0728049 * memory_boundness * cpufreq + 0.8590242 * pow(cpufreq, 2);
                        }
#endif                        
                        assembly->set_cpupowertable(ddr_freq_indx, freq_indx, 0, cpupower, width-1); 
#if defined DDR_Power_Model_1
                        if(width == 1){ /*Denver, width=1*/
                          ddrpower = 4.0040504 * memory_boundness + 0.306568 * cpufreq + 0.7431409 * ddrfreq - 0.8051031;
                        }
                        if(width == 2){ /*Denver, width=2*/
                          ddrpower = 17.830587 * memory_boundness + 0.3768498 * cpufreq + 0.7559968 * ddrfreq - 0.8784249;
                        }
#endif  
#if defined DDR_Power_Model_2
                        if(width == 1){ /*Denver, width=1*/
                          ddrpower = 2.5701454 * memory_boundness + 0.0517271 * cpufreq + 0.8294422 * ddrfreq + 1.8753657 * memory_boundness * cpufreq - 0.5995333 * memory_boundness * ddrfreq - 0.0029895* cpufreq * ddrfreq - 0.6119471;
                        }
                        if(width == 2){ /*Denver, width=2*/
                          ddrpower = 10.075838 * memory_boundness - 0.0133797 * cpufreq + 0.8017427 * ddrfreq + 8.7131834 * memory_boundness * cpufreq - 1.9651515 * memory_boundness * ddrfreq + 0.0243026 * cpufreq * ddrfreq - 0.5452121;
                        }
#endif
#if defined DDR_Power_Model_3
                        if(width == 1){ /*Denver, width=1*/
                          ddrpower = 2.7306748 * memory_boundness + 0.1950916 * cpufreq - 1.3051822 * ddrfreq - 0.584781 * pow(memory_boundness, 2) + 1.8753657 * memory_boundness * cpufreq - 0.0602169 * pow(cpufreq, 2) \
                          - 0.5995333 * memory_boundness * ddrfreq - 0.0029895 * cpufreq * ddrfreq + 0.8006888 * pow(ddrfreq, 2) + 0.6209039;
                        }
                        if(width == 2){ /*Denver, width=2*/
                          //ddrpower = 11.8840022 * memory_boundness + 0.2207538 * cpufreq - 1.1475948 * ddrfreq - 23.7916345 * pow(memory_boundness, 2) + 8.7131834 * memory_boundness * cpufreq - 0.0871027 * pow(cpufreq, 2) \
                          //- 1.9651515 * memory_boundness * ddrfreq + 0.0243026 * cpufreq * ddrfreq + 0.7311884 * pow(ddrfreq, 2) + 0.5285202;
			                    ddrpower = 11.9390059 * memory_boundness + 0.224722 * cpufreq - 1.1527272 * ddrfreq - 24.7881921 * pow(memory_boundness, 2) + 8.704204 * memory_boundness * cpufreq - 0.0864551 * pow(cpufreq, 2) \
                          - 1.9467908 * memory_boundness * ddrfreq + 0.0211395 * cpufreq * ddrfreq + 0.7337044 * pow(ddrfreq, 2) + 0.5295545;
			                    //ddrpower = 10.075838 * memory_boundness - 0.0133797 * cpufreq + 0.8017427 * ddrfreq + 8.7131834 * memory_boundness * cpufreq - 1.9651515 * memory_boundness * ddrfreq + 0.0243026 * cpufreq * ddrfreq - 0.5452121;
                        }
#endif
#if defined DDR_Power_Model_4
                        if(width == 1){ /*Denver, width=1*/
                          ddrpower = 4.1645798 * memory_boundness + 0.4499325 * cpufreq - 1.3914835 * ddrfreq - 0.584781 * pow(memory_boundness, 2) - 0.0602169 * pow(cpufreq, 2) + 0.8006888 * pow(ddrfreq, 2) + 0.4277479;
                        }
                        if(width == 2){ /*Denver, width=2*/
                          ddrpower = 19.6387512	* memory_boundness + 0.584224 * cpufreq - 1.1933407 * ddrfreq - 23.7916345 * pow(memory_boundness, 2) - 0.0871027 * pow(cpufreq, 2) + 0.7311884 * pow(ddrfreq, 2) + 0.1953074;
                        }
#endif
                        assembly->set_ddrpowertable(ddr_freq_indx, freq_indx, 0, ddrpower, width-1); 
                      }
                    }
#if defined Model_Computation_Overhead
                    // std::chrono::time_point<std::chrono::system_clock> Denver_model_end;
                    // Denver_model_end = std::chrono::system_clock::now();
                    // auto Denver_model_end_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(Denver_model_end);
                    // auto Denver_model_end_epoch = Denver_model_end_ms.time_since_epoch();
                    // LOCK_ACQUIRE(output_lck);
                    // std::cout << "[Overhead] Model calculation (Denver) ends " << Denver_model_end_epoch.count() << ". " << std::endl;
                    // LOCK_RELEASE(output_lck);
                    clock_gettime(CLOCK_REALTIME, &Denver_finish);
                    sub_timespec(Denver_start, Denver_finish, &Denver_delta);
                    LOCK_ACQUIRE(output_lck);
                    printf("[Overhead] Model calculation (Denver): %d.%.9ld\n", (int)Denver_delta.tv_sec, Denver_delta.tv_nsec);
                    LOCK_RELEASE(output_lck);
#endif
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
#if defined Performance_Model_Cycle    
                  std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[1.11GHz, Denver, " << width << "] = " << check_ticks << ". Cycles(1.11GHz) = " << check_cycles << ", Cycles(2.04GHz) = " << cycles_high <<\
                  ". Memory-boundness(Denver, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
#endif
#if defined Performance_Model_Time
                  std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[1.866GHz, 2.04GHz, Denver, " << width << "] = " << highest_ticks << ", PTT_Value[1.866GHz, 1.11GHz, Denver, " << width << "] = " \
                  << check_ticks << ". Memory-boundness(Denver, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
#endif
                  LOCK_RELEASE(output_lck);
#endif
                  }
                  continue;
                }else{
                  break;
                }
              }
              if(ptt_check == inclusive_partitions[START_CLUSTER_A].size()){ /* PTT at 1.11GHz are filled out */ 
                PTT_finish_state[1][0][assembly->tasktype] = 1;
                assembly->set_timetable_state(0, true); /* Finish the PTT training in Denver part, set state to true */
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] " << assembly->kernel_name << ": Denver cluster completed PTT training at 2.04GHz and 1.11GHz. \n";
                LOCK_RELEASE(output_lck);
#endif
              }
            }
          }else{ 
            // mtx.lock();
            _final = (++assembly->threads_out_tao == assembly->width);
            // mtx.unlock();
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "Task " << assembly->taskid << "->_final = " << _final << ", assembly->get_timetable_state(Denver) = " << assembly->get_timetable_state(0) << ", assembly->start_running_freq = " \
            << assembly->start_running_freq<< ", cur_freq[Denver] = " << cur_freq[0] << "\n";
            LOCK_RELEASE(output_lck);
#endif        
          }    
        }

        /* (1) Was running on A57 (2) A57 PTT table hasn't finished training      (3) if the frequency changing is happening during the task execution, do not update the table */
        if(assembly->leader >= START_CLUSTER_B){
          if(assembly->get_timetable_state(1) == false && assembly->start_running_freq == cur_freq[1]){ 
          if(ptt_freq_index[1] == 0){ /*2.04GHz*/
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(0, 0, 1, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0) / assembly->width;
                if(oldticks == 0.0f ||  (newticks < oldticks && fabs(newticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, 0, 1, newticks, width_index);
                }else{
                  assembly->set_timetable(0, 0, 1, (newticks+oldticks)/2, width_index);
                }
              }else{
                if(oldticks == 0.0f || (ticks < oldticks && fabs(ticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, 0, 1, ticks, width_index);
                }else{
                  assembly->set_timetable(0, 0, 1, (ticks+oldticks)/2, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(0, 1, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[1.866GHz, 2.04GHz, A57, width = " \
              << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(0,1, width_index) << ". Current time: " << assembly->get_timetable(0, 0, 1, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
              //++assembly->threads_out_tao;
            }
//             float oldticks = assembly->get_timetable(0, 1, width_index);
//             if(oldticks == 0.0f || ticks < oldticks){
//               assembly->set_timetable(0, 1, ticks, width_index); 
// #if defined Performance_Model_Cycle              
//               assembly->set_cycletable(0, 1, val1, width_index);
// #endif             
//             }
//             // else{
//             //   assembly->set_timetable(0, 1, ((oldticks + ticks)/2), width_index);  
//             // }
//             // if(nthread == assembly->leader){ 
//             // assembly->PTT_UpdateFinish[0][1][width_index]++;
//             assembly->increment_PTT_UpdateFinish(0, 1, width_index);
//             // }
          }else{ /*1.11GHz*/
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(0, NUM_AVAIL_FREQ/2, 1, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0) / assembly->width;
                if(oldticks == 0.0f || (newticks < oldticks && fabs(newticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, NUM_AVAIL_FREQ/2, 1, newticks, width_index);
                }else{
                  assembly->set_timetable(0, NUM_AVAIL_FREQ/2, 1, (newticks+oldticks)/2, width_index);
                }
              }else{
                if(oldticks == 0.0f || (ticks < oldticks && fabs(ticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, NUM_AVAIL_FREQ/2, 1, ticks, width_index);
                }else{
                  assembly->set_timetable(0, NUM_AVAIL_FREQ/2, 1, (ticks+oldticks)/2, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(1, 1, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[1.866GHz, 1.11GHz, A57, width = " \
              << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(1,1, width_index) << ". Current time: " << assembly->get_timetable(0, NUM_AVAIL_FREQ/2, 1, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              //++assembly->threads_out_tao;
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
            }
//             float oldticks = assembly->get_timetable(NUM_AVAIL_FREQ/2, 1, width_index);
//             if(oldticks == 0.0f || ticks < oldticks){
//               assembly->set_timetable(NUM_AVAIL_FREQ/2, 1, ticks, width_index); 
// #if defined Performance_Model_Cycle                 
//               assembly->set_cycletable(1, 1, val1, width_index);
// #endif              
//             }
//             // else{
//             //   assembly->set_timetable(NUM_AVAIL_FREQ/2, 1, ((oldticks + ticks)/2), width_index);  
//             // }
//             // if(nthread == assembly->leader){ 
//             // assembly->PTT_UpdateFinish[1][1][width_index]++;
//             assembly->increment_PTT_UpdateFinish(1, 1, width_index);
//             // }
          }
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_UpdateFinish[" << ptt_freq_index[1] << ", A57, width = " << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(ptt_freq_index[1], 1, width_index) << ".\n";
//             LOCK_RELEASE(output_lck);
// #endif
          // float oldticks = assembly->get_timetable(ptt_freq_index[1], 1, width_index);
          // if(oldticks == 0.0f){
          //   assembly->set_timetable(ptt_freq_index[1], 1, ticks, width_index); 
          // }else{
          //   assembly->set_timetable(ptt_freq_index[1], 1, ((oldticks + ticks)/2), width_index);  
          // }
          // //if(nthread == assembly->leader){
          //   PTT_UpdateFinish[ptt_freq_index[1]][1][width_index]++;

          //}
          /* Step 2: Update cycle table values */
          // uint64_t oldcycles = assembly->get_cycletable(ptt_freq_index[1], 1, width_index);
          // if(oldcycles == 0){
          //   assembly->set_cycletable(ptt_freq_index[1], 1, val1, width_index);
          // }else{
          //   assembly->set_cycletable(ptt_freq_index[1], 1, ((oldcycles + val1)/2), width_index);
          // }
          // if(assembly->get_timetable_state(1) == false){ /* PTT training hasn't finished */

          // if(nthread == assembly->leader){
            if (ptt_freq_index[1] == 0) {             /* Current A57 frequency is 2.04GHz */
              if(assembly->get_PTT_UpdateFinish(0, 1, 0) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 1, 1) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 1, 3) >= NUM_TRAIN_TASKS){ /* First dimention: 2.04GHz, second dimention: Denver, third dimention: width_index */
                PTT_finish_state[0][1][assembly->tasktype] = 1; /* First dimention: 2.04GHz, second dimention: Denver, third dimention: tasktype */
                if(std::accumulate(std::begin(PTT_finish_state[0][1]), std::begin(PTT_finish_state[0][1])+num_kernels, 0) == num_kernels){ /* Check all kernels have finished the training of Denver at 2.04GHz */
                  std::ofstream ClusterB("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[START_CLUSTER_B]) + "/cpufreq/scaling_setspeed");
                  if (!ClusterB.is_open()){
                    std::cerr << "[DEBUG] Somthing failed while opening the file! " << std::endl;
                    return 0;
                  }
                  ClusterB << std::to_string(TRAIN_MED_B_FREQ) << std::endl;
                  ClusterB.close();
                  ptt_freq_index[1] = 1;
                  cur_freq[1] = TRAIN_MED_B_FREQ;
                  cur_freq_index[1] = NUM_AVAIL_FREQ/2;
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] A57 cluster completed all kernels' PTT training at 2.04GHz, turn to 1.11GHz. \n";
                  LOCK_RELEASE(output_lck);
#endif
                }
              }
            }else{
              int ptt_check = 0; /* Step 2: Check if PTT values of A57 are filled out */
              for(auto&& width : ptt_layout[START_CLUSTER_B]) { 
                float check_ticks = assembly->get_timetable(0, NUM_AVAIL_FREQ/2, 1, width - 1);
                if(check_ticks > 0.0f && assembly->get_PTT_UpdateFinish(1, 1, width-1) >=NUM_TRAIN_TASKS){
                  ptt_check++;
                  if(assembly->get_mbtable(1, width-1) == 0.0f){
                    float memory_boundness = 0.0f;
#if defined Performance_Model_Cycle                  /* Calculate Memory-boundness = (1 - cycle2/cycle1) / (1- f2/f1) */
                    uint64_t check_cycles = assembly->get_cycletable(1, 1, width - 1);
                    uint64_t cycles_high = assembly->get_cycletable(0, 1, width-1); /* First parameter is 0, means 2.04GHz, check_cycles is at 1.11Ghz */
                    float a = 1 - float(check_cycles) / float(cycles_high);
                    float b = 1 - float(avail_freq[1][NUM_AVAIL_FREQ/2]) / float(avail_freq[1][0]);
                    memory_boundness = a/b;
#endif
#if defined Performance_Model_Time
                    float highest_ticks = assembly->get_timetable(0, 0, 1, width - 1);
                    float a = float(avail_freq[1][0]) / float(avail_freq[1][NUM_AVAIL_FREQ/2]);
                    float b = check_ticks / highest_ticks;
                    memory_boundness = (b-a) / (1-a);
                    // LOCK_ACQUIRE(output_lck);
                    // std::cout << assembly->kernel_name << ": Memory-boundness Calculation (A57, width " << width << ") = " << memory_boundness << ". a = " << a << ", b = " << b << std::endl;
                    // LOCK_RELEASE(output_lck);
#endif
                    if(memory_boundness > 1){
                      LOCK_ACQUIRE(output_lck);
                      std::cout << "[Warning] Memory-boundness Calculation (A57) is greater than 1!" << std::endl;
                      LOCK_RELEASE(output_lck);
                      memory_boundness = 1;
                    }else{
                      if(memory_boundness <= 0){ /* Execution time and power prediction according to the computed memory-boundness level */
                        LOCK_ACQUIRE(output_lck);
                        std::cout << "[Warning] Memory-boundness Calculation (A57) is smaller than 0!" << std::endl;
                        LOCK_RELEASE(output_lck);
                        memory_boundness = 0.00001;
                      }
                    }
                    assembly->set_mbtable(1, memory_boundness, width-1); /*first parameter: cluster 0 - Denver, second parameter: update value, third value: width_index */
#if defined Model_Computation_Overhead
                    // std::chrono::time_point<std::chrono::system_clock> A57_model_start;
                    // A57_model_start = std::chrono::system_clock::now();
                    // auto A57_model_start_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(A57_model_start);
                    // auto A57_model_start_epoch = A57_model_start_ms.time_since_epoch();
                    // LOCK_ACQUIRE(output_lck);
                    // std::cout << "[Overhead] Model calculation (A57) starts from " << A57_model_start_epoch.count() << ". " << std::endl;
                    // LOCK_RELEASE(output_lck);
                    struct timespec A57_start, A57_finish, A57_delta;
                    clock_gettime(CLOCK_REALTIME, &A57_start);
#endif
                    for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){ /* Compute Predictions according to Memory-boundness Values */
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                        if((ddr_freq_indx == 0 && freq_indx == 0) || (ddr_freq_indx == 0 && freq_indx == NUM_AVAIL_FREQ/2)){ /* if DDR freq = 1.866 and CPU freq = 2.04 or 1.11 GHz, then skip */
                          continue;
                        }else{
#if defined Performance_Model_Cycle 
                        uint64_t new_cycles = cycles_high * float(avail_freq[1][freq_indx])/float(avail_freq[1][0]);
                        float ptt_value_newfreq = float(new_cycles) / float(avail_freq[1][freq_indx]*1000);
#endif
#if defined Performance_Model_Time
                        float ptt_value_newfreq = 0.0;
                        float cpu_freq_scaling = float(avail_freq[1][0]) / float(avail_freq[1][freq_indx]);
                        float ddr_freq_scaling = float(avail_ddr_freq[0]) / float(avail_ddr_freq[ddr_freq_indx]);
                        if(width == 1){ /* A57, width=1 */                     
#if defined Performance_Model_1
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 1.3711753 * memory_boundness + 0.0660749 * cpu_freq_scaling + 0.1744398 * ddr_freq_scaling - 0.3950029); 
#endif
#if defined Performance_Model_2
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 0.328 * memory_boundness + 0.0562 * cpu_freq_scaling \
                          + 0.0612 * ddr_freq_scaling + 0.1358 * memory_boundness * cpu_freq_scaling + 0.4805 * memory_boundness * ddr_freq_scaling \
                          - 0.0248 * cpu_freq_scaling * ddr_freq_scaling - 0.1133);
#endif
#if defined Performance_Model_3
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 0.2880999 * memory_boundness - 0.0518217 * cpu_freq_scaling \
                          + 0.0261097 * ddr_freq_scaling + 0.0563669 * pow(memory_boundness, 2) + 0.1358472 * memory_boundness * cpu_freq_scaling \
                          + 0.0165384 * pow(cpu_freq_scaling, 2) + 0.4805453 * memory_boundness * ddr_freq_scaling \
                          - 0.0248303 * cpu_freq_scaling * ddr_freq_scaling + 0.0104742 * pow(ddr_freq_scaling, 2) + 0.0452253);
#endif
// #ifdef DEBUG
//                           LOCK_ACQUIRE(output_lck);
//                           std::cout << "[DEBUG] " << assembly->kernel_name << "(A57, 1): " << avail_ddr_freq[ddr_freq_indx] << "GHz, " << avail_freq[freq_indx] << "GHz, execution time prediction = " << ptt_value_newfreq << ".\n";
//                           LOCK_RELEASE(output_lck);
// #endif
                        }
                        if(width == 2){ /* A57, width=2*/
#if defined Performance_Model_1
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 1.3807643 * memory_boundness + 0.0538466 * cpu_freq_scaling + 0.1897099 * ddr_freq_scaling - 0.4061276);
#endif
#if defined Performance_Model_2
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 0.2873954 * memory_boundness + 0.0708212 * cpu_freq_scaling \
                          + 0.0875247 * ddr_freq_scaling + 0.1251056 * memory_boundness * cpu_freq_scaling + 0.5291735 * memory_boundness * ddr_freq_scaling \
                          - 0.0412743 * cpu_freq_scaling * ddr_freq_scaling - 0.1450613);
#endif
#if defined Performance_Model_3
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 0.2954466 * memory_boundness - 0.0591274 * cpu_freq_scaling \
                          + 0.0076495 * ddr_freq_scaling - 0.0109684 * pow(memory_boundness, 2) + 0.1251056 * memory_boundness * cpu_freq_scaling \
                          + 0.0198932 * pow(cpu_freq_scaling, 2) + 0.5291735 * memory_boundness * ddr_freq_scaling \
                          - 0.0412743 * cpu_freq_scaling * ddr_freq_scaling + 0.0238337 * pow(ddr_freq_scaling, 2) + 0.0679118);
#endif
// #ifdef DEBUG
//                           LOCK_ACQUIRE(output_lck);
//                           std::cout << "[DEBUG] " << assembly->kernel_name << "(A57, 2): " << avail_ddr_freq[ddr_freq_indx] << "GHz, " << avail_freq[freq_indx] << "GHz, execution time prediction = " << ptt_value_newfreq << ".\n";
//                           LOCK_RELEASE(output_lck);
// #endif
                        }
                        if(width == 4){ /* A57, width = 4*/
#if defined Performance_Model_1
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 1.2547213 * memory_boundness + 0.0675267 * cpu_freq_scaling + 0.1870626 * ddr_freq_scaling - 0.3802896);
#endif
#if defined Performance_Model_2
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 0.4627277 * memory_boundness + 0.1460303 * cpu_freq_scaling \
                          + 0.1370799 * ddr_freq_scaling + 0.0066989 * memory_boundness * cpu_freq_scaling + 0.5072814 * memory_boundness * ddr_freq_scaling \
                          - 0.0527228 * cpu_freq_scaling * ddr_freq_scaling - 0.2986913);
#endif
#if defined Performance_Model_3
                          ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 0.424071 * memory_boundness - 0.0597007 * cpu_freq_scaling \
                          + 0.0558782 * ddr_freq_scaling + 0.0574955 * pow(memory_boundness, 2) + 0.0066989 * memory_boundness * cpu_freq_scaling \
                          + 0.0314944 * pow(cpu_freq_scaling, 2) + 0.5072814 * memory_boundness * ddr_freq_scaling \
                          - 0.0527228 * cpu_freq_scaling * ddr_freq_scaling + 0.0242295 * pow(ddr_freq_scaling, 2) + 0.0093365);
#endif
// #ifdef DEBUG
//                           LOCK_ACQUIRE(output_lck);
//                           std::cout << "[DEBUG] " << assembly->kernel_name << "(A57, 4): " << avail_ddr_freq[ddr_freq_indx] << "GHz, " << avail_freq[freq_indx] << "GHz, execution time prediction = " << ptt_value_newfreq << ".\n";
//                           LOCK_RELEASE(output_lck);
// #endif
                        }
#endif
                        assembly->set_timetable(ddr_freq_indx, freq_indx, 1, ptt_value_newfreq, width-1);
                        }
                      }
                    }
                    for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){ /* (2) Power value prediction */
                      float ddrfreq = float(avail_ddr_freq[ddr_freq_indx])/1000000000.0;
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ 
                        float cpupower = 0.0;
                        float ddrpower = 0.0;
                        float cpufreq = float(avail_freq[1][freq_indx])/1000000.0;
#if defined CPU_Power_Model_6
                        if(width == 1){ /*A57, width=1*/
                          cpupower = 0.3035816 + 0.1363508 * memory_boundness - 0.1067623 * cpufreq - 0.1471331 * memory_boundness * cpufreq + 0.2793643 * pow(cpufreq, 2);
                        }
                        if(width == 2){ /*A57, width=2*/
                          cpupower = 0.3463124 + 0.2209002 * memory_boundness - 0.0078078 * cpufreq - 0.4175843 * memory_boundness * cpufreq + 0.3754007 * pow(cpufreq, 2);
                        }
                        if(width == 4){ /*A57, width=4*/
                          cpupower = 0.4496991 + 0.287165 * memory_boundness - 0.1324573 * cpufreq - 0.5858516 * memory_boundness * cpufreq + 0.5907194 * pow(cpufreq, 2);
                        }
#endif                        
                        assembly->set_cpupowertable(ddr_freq_indx, freq_indx, 1, cpupower, width-1); 
#if defined DDR_Power_Model_1
                        if(width == 1){ /*A57, width=1*/
                          ddrpower = 0.9316994 * memory_boundness + 0.1218638 * cpufreq + 0.7876816 * ddrfreq - 0.6563012;
                        }
                        if(width == 2){ /*A57, width=2*/
                          ddrpower = 1.2216713 * memory_boundness + 0.1532258 * cpufreq + 0.7893731 * ddrfreq - 0.7134712;
                        }
                        if(width == 4){ /*A57, width=4*/
                          ddrpower = 1.2637439 * memory_boundness + 0.1509104 * cpufreq + 0.7944365 * ddrfreq - 0.7112166;
                        }
#endif  
#if defined DDR_Power_Model_2
                        if(width == 1){ /*A57, width=1*/
                          ddrpower = 0.9138285 * memory_boundness + 0.0399894 * cpufreq + 0.8386321 * ddrfreq + 0.2044974 * memory_boundness * cpufreq - 0.1693516 * memory_boundness * ddrfreq + 0.0073483 * cpufreq * ddrfreq - 0.6383507;
                        }
                        if(width == 2){ /*A57, width=2*/
                          ddrpower = 1.080931 * memory_boundness + 0.0597889 * cpufreq + 0.8335765 * ddrfreq + 0.2524553 * memory_boundness * cpufreq - 0.119964 * memory_boundness * ddrfreq + 0.0001079 * cpufreq * ddrfreq - 0.6612905;
                        }
                        if(width == 4){ /*A57, width=4*/
                          ddrpower = 1.0329543 * memory_boundness + 0.0385068 * cpufreq + 0.8373515 * ddrfreq + 0.3373493 * memory_boundness * cpufreq - 0.1282292 * memory_boundness * ddrfreq - 0.0001117 * cpufreq * ddrfreq - 0.6343936;
                        }
#endif
#if defined DDR_Power_Model_3
                        if(width == 1){ /*A57, width=1*/
                          ddrpower = 0.7957444 * memory_boundness + 0.1425522 * cpufreq - 1.6483112 * ddrfreq + 0.1669195 * pow(memory_boundness, 2) + 0.2044974 * memory_boundness * cpufreq - 0.0389682 * pow(cpufreq, 2) \
                          - 0.1693516 * memory_boundness * ddrfreq + 0.9361233 * pow(ddrfreq, 2) + 0.8491373;
                        }
                        if(width == 2){ /*A57, width=2*/
                          ddrpower = 0.8780419 * memory_boundness + 0.2063387 * cpufreq - 1.4751842 * ddrfreq + 0.2764038 * pow(memory_boundness, 2) + 0.2524553 * memory_boundness * cpufreq - 0.0615549 * pow(cpufreq, 2) \
                          - 0.119964 * memory_boundness * ddrfreq + 0.0001079 * cpufreq * ddrfreq + 0.8660066 * pow(ddrfreq, 2) + 0.7093737;
                        }
                        if(width == 4){ /*A57, width=4*/
                          ddrpower = 0.4613634 * memory_boundness + 0.133786 * cpufreq - 1.5223486 * ddrfreq + 0.8501488 * pow(memory_boundness, 2) + 0.3373493 * memory_boundness * cpufreq - 0.0400198 * pow(cpufreq, 2) \
                          - 0.1282292 * memory_boundness * ddrfreq - 0.0001117 * cpufreq * ddrfreq + 0.8851138 * pow(ddrfreq, 2) + 0.8238755;
                        }
#endif
#if defined DDR_Power_Model_4
                        if(width == 1){ /*A57, width=1*/
                          ddrpower = 0.8136153 * memory_boundness + 0.2146393 * cpufreq - 1.7080091 * ddrfreq + 0.1669195 * pow(memory_boundness, 2) - 0.0389682 * pow(cpufreq, 2) + 0.9361233 * pow(ddrfreq, 2) + 0.8428377;
                        }
                        if(width == 2){ /*A57, width=2*/
                          ddrpower = 1.0187822 * memory_boundness + 0.2997757 * cpufreq - 1.5193876 * ddrfreq + 0.2764038 * pow(memory_boundness, 2) - 0.0615549 * pow(cpufreq, 2) + 0.8660066 * pow(ddrfreq, 2) + 0.657193;
                        }
                        if(width == 4){ /*A57, width=4*/
                          ddrpower = 0.692153 * memory_boundness + 0.2461896 * cpufreq - 1.5652636 * ddrfreq + 0.8501488 * pow(memory_boundness, 2) - 0.0400198 * pow(cpufreq, 2) + 0.8851138 * pow(ddrfreq, 2) + 0.7470525;
                        }
#endif
                        assembly->set_ddrpowertable(ddr_freq_indx, freq_indx, 1, ddrpower, width-1); 
                      }
                    }
#if defined Model_Computation_Overhead
                    // std::chrono::time_point<std::chrono::system_clock> A57_model_end;
                    // A57_model_end = std::chrono::system_clock::now();
                    // auto A57_model_end_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(A57_model_end);
                    // auto A57_model_end_epoch = A57_model_end_ms.time_since_epoch();
                    // LOCK_ACQUIRE(output_lck);
                    // std::cout << "[Overhead] Model calculation (A57) ends " << A57_model_end_epoch.count() << ". " << std::endl;
                    // LOCK_RELEASE(output_lck);
                    clock_gettime(CLOCK_REALTIME, &A57_finish);
                    sub_timespec(A57_start, A57_finish, &A57_delta);
                    LOCK_ACQUIRE(output_lck);
                    printf("[Overhead] Model calculation (A57): %d.%.9ld\n", (int)A57_delta.tv_sec, A57_delta.tv_nsec);
                    LOCK_RELEASE(output_lck);
#endif
                      // return -1;
                    
//                     }else{
//                       if(memory_boundness <= 0){ /* Execution time and power prediction according to the computed memory-boundness level */
//                         memory_boundness = 0.001;
//                         for(int freq_indx = 1; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* Fill out PTT[other freqs]. Start from 1, which is next freq after 2.04, also skip 1.11GHz*/
//                           if(freq_indx == NUM_AVAIL_FREQ/2){
//                             continue;
//                           }else{
// #if defined Performance_Model_Cycle 
//                             float ptt_value_newfreq = float(cycles_high) / float(avail_freq[freq_indx]*1000); /* (1) Execution time prediction */
// #endif
// #if defined Performance_Model_Time
//                             float ptt_value_newfreq = highest_ticks * (float(avail_freq[0])/float(avail_freq[freq_indx]));  /* (1) Execution time prediction */
// #endif
//                             assembly->set_timetable(freq_indx, 1, ptt_value_newfreq, width-1);
//                           }
//                         }
//                         for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ /* (2) Power value prediction (now memory-boundness is 0, cluster is A57 and width is known) */
//                           // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
//                             assembly->set_powertable(freq_indx, 1, runtime_power[0][freq_indx][1][width-1], width-1); /*Power: first parameter 0 is memory-boundness index*/
//                           // }
//                         }
//                       }else{
//                         for(int freq_indx = 1; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
//                           if(freq_indx == NUM_AVAIL_FREQ/2){
//                             continue;
//                           }else{
// #if defined Performance_Model_Cycle 
//                             uint64_t new_cycles = cycles_high * (1 - memory_boundness + memory_boundness * float(avail_freq[freq_indx])/float(avail_freq[0]));
//                             float ptt_value_newfreq = float(new_cycles) / float(avail_freq[freq_indx]*1000);
// #endif
// #if defined Performance_Model_Time
//                             // float ptt_value_newfreq = highest_ticks * (memory_boundness + (1-memory_boundness) * float(avail_freq[0]) / float(avail_freq[freq_indx])); /* No consideration of DDR frequency */
//                             float ptt_value_newfreq = highest_ticks * /*2022 Aug 14th: consideration of CPU frequency + DDR frequency*/
// #endif                         
//                             assembly->set_timetable(freq_indx, 1, ptt_value_newfreq, width-1);
//                           }
//                         }
//                         int mb_bound = floor(memory_boundness/0.1);  /* (2) Power value prediction */
//                         for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
//                           assembly->set_powertable(freq_indx, 1, runtime_power[mb_bound][freq_indx][1][width-1], width-1); 
//                         }
//                       }
//                     }
#if (defined DEBUG)
                    LOCK_ACQUIRE(output_lck);
#if (defined Performance_Model_Cycle)
                    std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[1.11GHz, A57, " << width << "] = " << check_ticks << ". Cycles(1.11GHz) = " << check_cycles << ", Cycles(2.04GHz) = " << cycles_high <<\
                    ". Memory-boundness(A57, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
#endif
#if (defined Performance_Model_Time)
                    std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[1.866GHz, 2.04GHz, A57, " << width << "] = " << highest_ticks << ", PTT_Value[1.866GHz, 1.11GHz, A57, " << width << "] = " << check_ticks << ". Memory-boundness(A57, width=" << width << ") = " << memory_boundness \
                    << ". ptt_check = " << ptt_check << ".\n";
#endif
                    LOCK_RELEASE(output_lck);
#endif
                  }
                continue;
              }else{
                break;
              }
            }
            if(ptt_check == inclusive_partitions[START_CLUSTER_B].size()){ /* All ptt values of A57 are positive => visited at least once, then configure the frequency to 1.11GHz */
              PTT_finish_state[1][1][assembly->tasktype] = 1;
              assembly->set_timetable_state(1, true); /* Finish the PTT training in A57 part, set state to true */
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] " << assembly->kernel_name << ": A57 cluster completed PTT training at 2.04GHz and 1.11GHz. \n";
              LOCK_RELEASE(output_lck);
#endif
            }
          }
        // }
          }else{
            // mtx.lock();
            _final = (++assembly->threads_out_tao == assembly->width);
            // mtx.unlock();
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "Task " << assembly->taskid << "->_final = " << _final << ", assembly->get_timetable_state(A57) = " << assembly->get_timetable_state(1) << ", assembly->start_running_freq = " \
            << assembly->start_running_freq<< ", cur_freq[A57] = " << cur_freq[1] << "\n";
            LOCK_RELEASE(output_lck);
#endif  
          }
        }

        if(assembly->get_timetable_state(2) == false && assembly->get_timetable_state(0) == true && assembly->get_timetable_state(1) == true){
          assembly->set_timetable_state(2, true);
          global_training_state[assembly->tasktype] = 1;
// #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << assembly->kernel_name << ": Training Phase finished. Predicted execution time and power results for the kernel tasks: \n";
          std::cout << "\n---------- Execution Time Predictions ---------- \n";
          for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){
            std::cout << "Memory Frequency: " << avail_ddr_freq[ddr_freq_indx] << ": " << std::endl;
            for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
              std::cout << "Cluster " << clus_id << ": " << std::endl;
              for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                std::cout << "CPU_Freq[" << avail_freq[clus_id][freq_indx] << "]: ";
                for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
                  std::cout << assembly->get_timetable(ddr_freq_indx, freq_indx, clus_id, wid-1) << "\t";
                }
                std::cout << std::endl;
              }
            }
            // for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
            //   std::cout << "CPU Frequency: " << avail_freq[freq_indx] << ": " << std::endl;
            //   for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
            //     std::cout << "Cluster " << clus_id << ": ";
            //     for(int wid = 1; wid <= 4; wid *= 2){
            //       std::cout << assembly->get_timetable(ddr_freq_indx, freq_indx, clus_id, wid-1) << "\t";
            //     }
            //     std::cout << std::endl;
            //   }
            // }
          }
          std::cout << "\n---------- CPU Power Predictions ---------- \n";
          for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){
            std::cout << "Memory Frequency: " << avail_ddr_freq[ddr_freq_indx] << ": " << std::endl;
            for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
              std::cout << "Cluster " << clus_id << ": " << std::endl;
              for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                std::cout << "CPU_Freq[" << avail_freq[clus_id][freq_indx] << "]: ";
                for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
                  std::cout << assembly->get_timetable(ddr_freq_indx, freq_indx, clus_id, wid-1) << "\t";
                }
                std::cout << std::endl;
              }
            }
            // for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
            //   std::cout << "CPU Frequency: " << avail_freq[freq_indx] << ": " << std::endl;
            //   for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
            //     std::cout << "Cluster " << clus_id << ": ";
            //     for(int wid = 1; wid <= 4; wid *= 2){
            //       std::cout << assembly->get_cpupowertable(ddr_freq_indx, freq_indx, clus_id, wid-1) << "\t";
            //     }
            //     std::cout << std::endl;
            //   }
            // }
          }
          std::cout << "\n---------- Memory Power Predictions ---------- \n";
          for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){
            std::cout << "Memory Frequency: " << avail_ddr_freq[ddr_freq_indx] << ": " << std::endl;
            for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
              std::cout << "Cluster " << clus_id << ": " << std::endl;
              for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                std::cout << "CPU_Freq[" << avail_freq[clus_id][freq_indx] << "]: ";
                for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
                  std::cout << assembly->get_timetable(ddr_freq_indx, freq_indx, clus_id, wid-1) << "\t";
                }
                std::cout << std::endl;
              }
            }
            // for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
            //   std::cout << "CPU Frequency: " << avail_freq[freq_indx] << ": " << std::endl;
            //   for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
            //     std::cout << "Cluster " << clus_id << ": ";
            //     for(int wid = 1; wid <= 4; wid *= 2){
            //       std::cout << assembly->get_ddrpowertable(ddr_freq_indx, freq_indx, clus_id, wid-1) << "\t";
            //     }
            //     std::cout << std::endl;
            //   }
            // }
          }
          std::cout << std::endl;
          LOCK_RELEASE(output_lck);
// #endif
        }
        if(std::accumulate(std::begin(global_training_state), std::begin(global_training_state) + num_kernels, 0) == num_kernels) {// Check if all kernels have finished the training phase.
          global_training = true;
          std::chrono::time_point<std::chrono::system_clock> train_end;
          train_end = std::chrono::system_clock::now();
          auto train_end_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(train_end);
          auto train_end_epoch = train_end_ms.time_since_epoch();
          LOCK_ACQUIRE(output_lck);
          std::cout << "[Congratulations!] All the training Phase finished. Training finished time: " << train_end_epoch.count() << ". " << std::endl;
          LOCK_RELEASE(output_lck);
        }
      }
      else{ /* Other No training tasks or Other kinds of schedulers*/
        // mtx.lock();
        _final = (++assembly->threads_out_tao == assembly->width);
        // mtx.unlock();
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[DEBUG] Task " << assembly->taskid << ", _final = " << _final << "\n";
//         LOCK_RELEASE(output_lck);
// #endif        
      }
    st = nullptr;
    if(_final){ // the last exiting thread updates
      task_completions[nthread].tasks++;
      if(task_completions[nthread].tasks > 0){
        PolyTask::pending_tasks -= task_completions[nthread].tasks;
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Thread " << nthread << " completed " << task_completions[nthread].tasks << " task " << assembly->taskid <<". Pending tasks = " << PolyTask::pending_tasks << "\n";
        LOCK_RELEASE(output_lck);
#endif
        task_completions[nthread].tasks = 0;
      }
      
#ifdef OVERHEAD_PTT
      //st = assembly->commit_and_wakeup(nthread, elapsed_ptt);
      assembly->commit_and_wakeup(nthread, elapsed_ptt);
#else
      //st = assembly->commit_and_wakeup(nthread);
      assembly->commit_and_wakeup(nthread);
#endif
      assembly->cleanup();
    }
    idle_try = 0;
    idle_times = 0;
    continue;
  }

    // 2. check own queue
    if(!stop){
      LOCK_ACQUIRE(worker_lock[nthread]);
      if(!worker_ready_q[nthread].empty()){
        st = worker_ready_q[nthread].front(); 
        worker_ready_q[nthread].pop_front();
        LOCK_RELEASE(worker_lock[nthread]);
        continue;
      }     
      LOCK_RELEASE(worker_lock[nthread]);        
    }

#ifdef WORK_STEALING
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[Test] Thread " << nthread << " goes out of work stealing.\n";
//       LOCK_RELEASE(output_lck);          
// #endif

    // 3. try to steal rand_r(&seed)
// #ifdef ERASE_target_edp
//     if((rand() % A57_best_edp_width == 0) && !stop)
// #else
    if((rand() % STEAL_ATTEMPTS == 0) && !stop)
// #endif
    {
      // status_working[nthread] = 0;
      int attempts = gotao_nthreads;
#ifdef SLEEP
#if (defined RWSS_SLEEP)
      if(Sched == 3){
        idle_try++;
      }
#endif
#if (defined FCAS_SLEEP)
      if(Sched == 0){
        idle_try++;
      }
#endif
#if (defined EAS_SLEEP)
      if(Sched == 1){
        idle_try++;
      }
#endif
#endif
      do{
        if(Sched == 2){
          if(DtoA <= maySteal_DtoA){
            do{
              random_core = (rand_r(&seed) % gotao_nthreads);
            } while(random_core == nthread);
           
          }
          else{
            if(nthread < START_CLUSTER_B){
              do{
                random_core = (rand_r(&seed) % 2);
              } while(random_core == nthread);
            }else{
         	    do{
                random_core = 2 + (rand_r(&seed) % 4);
              }while(random_core == nthread); 
            }
          }
        }
        if(Sched == 1){ /* ERASE && STEER && JOSS */
// #if defined(TX2)
// #ifndef MultipleKernels
//         	if(!ptt_full){
          	// do{
            // 	random_core = (rand_r(&seed) % gotao_nthreads);
          	// } while(random_core == nthread);
//         	}else
// #endif
          // {
// #if (defined ERASE_target_perf) || (defined ERASE_target_edp_method1)
//           if(D_give_A == 0 || steal_DtoA < D_give_A){
//             int Denver_workload = worker_ready_q[0].size() + worker_ready_q[1].size();
//             int A57_workload = worker_ready_q[2].size() + worker_ready_q[3].size() + worker_ready_q[4].size() + worker_ready_q[5].size();
//           // If there is more workload to share with A57 
//             D_give_A = (Denver_workload-A57_workload) > 0? floor((Denver_workload-A57_workload) * 1 / (D_A+1)) : 0; 
//             //if((Denver_workload-A57_workload) > 0 && D_give_A > 0){
//             if(D_give_A > 0){
//               random_core = rand_r(&seed) % 2;
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] There is more workload that can be shared with A57. Size =  " << D_give_A << ". \n";
//               LOCK_RELEASE(output_lck);          
// #endif
//             }
//           }else
// #endif
          // {          	
#ifdef JOSS_RWS  /* [JOSS - EDP test: Energy Minimization per task + Random work stealing] */
            if(global_training == true){
              do{
                random_core = (rand_r(&seed) % gotao_nthreads);
              } while(random_core == nthread);
            }else{
              if(nthread < START_CLUSTER_B){
                do{
                  random_core = (rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A));
                } while(random_core == nthread);
              }else{
                do{
                  random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
                }while(random_core == nthread); 
              }
            }          
#else /* [JOSS - default] Only steal tasks from same cluster */
            if(nthread < START_CLUSTER_B){
            	do{
              	random_core = (rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A));
            	} while(random_core == nthread);
          	}else{
         	  	do{
              	random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
            	}while(random_core == nthread); 
          	}
#endif
					// }    
        // }
// #endif
#if defined(Haswell)
          // if(nthread < gotao_nthreads/NUMSOCKETS){
          //   do{
          //     random_core = (rand_r(&seed) % (gotao_nthreads/NUMSOCKETS));
          //   } while(random_core == nthread);
          // }else{
          //   do{
          //     random_core = (gotao_nthreads/NUMSOCKETS) + (rand_r(&seed) % (gotao_nthreads/NUMSOCKETS));
          //   } while(random_core == nthread);
          // }
          do{
            random_core = (rand_r(&seed) % gotao_nthreads);
          } while(random_core == nthread);
#endif
        }

        if(Sched == 0){
#ifdef CATS
          if(nthread > 1){
            do{
              random_core = 2 + (rand_r(&seed) % 4);
            } while(random_core == nthread);
          }else{
            do{
              random_core = (rand_r(&seed) % gotao_nthreads);
            } while(random_core == nthread);
          }
#else
				  do{
            random_core = (rand_r(&seed) % gotao_nthreads);
          } while(random_core == nthread);
#endif
        }

        if(Sched == 3){
	  //do{
          //  random_core = (rand_r(&seed) % gotao_nthreads);
          //} while(random_core == nthread);
	  if(nthread < START_CLUSTER_B){
            	do{
              	random_core = (rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A));
            	} while(random_core == nthread);
          	}else{
         	  	do{
              	random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
            	}while(random_core == nthread); 
          	}
				}

        LOCK_ACQUIRE(worker_lock[random_core]);
        if(!worker_ready_q[random_core].empty()){
          st = worker_ready_q[random_core].back();
					if((Sched == 1) || (Sched == 2)){
            // [EAS] Not steal tasks from same pair, e.g, thread 0 does not steal the task width=2 from thread 1.
            // [EDP] Not steal tasks when ready queue task size is only 1.
            if((st->width >= abs(random_core-nthread)+1)) {
              st = NULL;
              LOCK_RELEASE(worker_lock[random_core]);
              continue;
            }
            else{
              // if(Sched == 2 && DtoA <= maySteal_DtoA){
              //   if(random_core > 1 && nthread < START_CLUSTER_B){
              //     std::atomic_fetch_sub(&DtoA, 1);
              //   }
              //   else{
              //     if(random_core < 2 && nthread > 1){
              //       if(worker_ready_q[random_core].size() <= 4){
              //         st = NULL;
              //         LOCK_RELEASE(worker_lock[random_core]);
              //         continue;
              //       }
              //       std::atomic_fetch_add(&DtoA, 1);
              //     }
              //   }
              //   //std::cout << "Steal D to A is " << DtoA << "\n";
              // }
              worker_ready_q[random_core].pop_back();
              
// #if (defined ERASE_target_edp_method1)
//               if(ptt_full==true && nthread > 1 && random_core < 2){
//                 if((steal_DtoA++) == D_give_A){
//                   steal_DtoA = 0;
//                 }
//                 st->width = A57_best_edp_width;
//                 if(st->width == 4){
//                   st->leader = 2 + (nthread-2) / st->width;
//                 }
//                 if(st->width <= 2){
//                   st->leader = nthread /st->width * st->width;
//                 }
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Full PTT: Thread " << nthread << " steal task " << st->taskid << " from " << random_core << " successfully. Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
//               LOCK_RELEASE(output_lck);          
// #endif	
//               }else{
//                 st->leader = nthread /st->width * st->width;
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Other: Thread " << nthread << " steal task " << st->taskid << " from " << random_core << " successfully. Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
//               LOCK_RELEASE(output_lck);          
// #endif	
//               }       
      
// #endif

#if (defined ERASE_target_perf) 
#ifndef MultipleKernels
              // if(ptt_full == true){
                st->history_mold(nthread, st); 
              // }
              // else{
              //   st->eas_width_mold(nthread, st);    
              // }
#endif             
#endif
              if(st->width == 4){
                st->leader = 2 + (nthread-2) / st->width;
              }
              if(st->width <= 2){
                st->leader = nthread /st->width * st->width;
              }
           
              // if(st->get_bestconfig_state() == true && nthread >= 2){ //Test Code: Allow work stealing across clusters
              //   st->width = 4;
              //   st->leader = 2;
              // }
              tao_total_steals++;  
            }
// #ifdef Energyaccuracy
//             if(st->finalenergypred == 0.0f){
//               st->finalenergypred = st->finalpowerpred * st->get_timetable(st->leader, st->width - 1);
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] Task " << st->taskid << " prediction energy = " << st->finalenergypred << ". \n";
//               LOCK_RELEASE(output_lck);          
// #endif
//             }
// #endif
          }
          else{
            if((Sched == 0) || (Sched == 3)){
            // if((st->criticality == 0 && Sched == 0) ){
              worker_ready_q[random_core].pop_back();
#ifdef ALLOWSTEALING
              if(nthread >= 2){
                st->leader = 2;
                st->width = 2;
              }
#else
              st->leader = nthread /st->width * st->width;
#endif
              tao_total_steals++;
            }else{
              st = NULL;
              LOCK_RELEASE(worker_lock[random_core]);
              continue;
            }
          }

#ifndef CATS				
          if(Sched == 0){
            st->history_mold(nthread, st);    
#ifdef DEBUG
          	LOCK_ACQUIRE(output_lck);
						std::cout << "[DEBUG] Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
						LOCK_RELEASE(output_lck);
#endif
          }
#endif
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Thread " << nthread << " steal task " << st->taskid << " from " << random_core << " successfully. Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
          LOCK_RELEASE(output_lck);          
#endif	
//           if(Sched == 1){
//             st->eas_width_mold(nthread, st);    
// #ifdef DEBUG
//           	LOCK_ACQUIRE(output_lck);
// 						std::cout << "[EAS-STEAL] Task " << st->taskid << " leader is " << st->leader << ", width is " << st->width << std::endl;
// 						LOCK_RELEASE(output_lck);
// #endif
//           }
          
        }
        LOCK_RELEASE(worker_lock[random_core]);  
      }while(!st && (attempts-- > 0));
      if(st){
#ifdef SLEEP
#if (defined RWSS_SLEEP)
        if(Sched == 3){
          idle_try = 0;
          idle_times = 0;
        }
#endif
#if (defined FCAS_SLEEP)
        if(Sched == 0){
          idle_try = 0;
          idle_times = 0;
        }
#endif
#if (defined EAS_SLEEP)
        if(Sched == 1){
          idle_try = 0;
          idle_times = 0;
        }
#endif
#endif
        continue;
      }
    }
#endif
    
/*
    if(Sched == 1){
      if(idle_try >= idle_sleep){
        if(!stop){
			  idle_times++;
        usleep( 100000 * idle_times);
        if(idle_times >= forever_sleep){
          // Step1: disable PTT entries of the thread
          for (int j = 0; j < inclusive_partitions[nthread].size(); ++j){
					  PTT_flag[inclusive_partitions[nthread][j].second - 1][inclusive_partitions[nthread][j].first] = 0;
					  //std::cout << "PTT_flag[" << inclusive_partitions[nthread][j].second - 1 << "]["<<inclusive_partitions[nthread][j].first << "] = 0.\n";
    		  }
          // Step 2: Go back to main work loop to check AQ 
          stop = true;
          continue;
        }
        }
      }
      if(stop){
        // Step 3: Go to sleep forever 
        std::cout << "Thread " << nthread << " go to sleep forever!\n";
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, []{return finish;});
        lk.unlock();
        break;
      }
    }
*/
#if (defined SLEEP) 
      if(idle_try >= IDLE_SLEEP){
        long int limit = (SLEEP_LOWERBOUND * pow(2,idle_times) < SLEEP_UPPERBOUND) ? SLEEP_LOWERBOUND * pow(2,idle_times) : SLEEP_UPPERBOUND;  
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);      
//         std::cout << "Thread " << nthread << " sleep for " << limit/1000000 << " ms.\n";
//         LOCK_RELEASE(output_lck);
// #endif
        status[nthread] = 0;
        // status_working[nthread] = 0;
        tim.tv_sec = 0;
        tim.tv_nsec = limit;
        nanosleep(&tim , &tim2);
        //SleepNum++;
        AccumTime += limit/1000000;
        idle_times++;
        idle_try = 0;
        status[nthread] = 1;
      }
#endif
    // 4. It may be that there are no more tasks in the flow
    // this condition signals termination of the program
    // First check the number of actual tasks that have completed
//     if(task_completions[nthread].tasks > 0){
//       PolyTask::pending_tasks -= task_completions[nthread].tasks;
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[DEBUG] Thread " << nthread << " completed " << task_completions[nthread].tasks << " tasks. Pending tasks = " << PolyTask::pending_tasks << "\n";
//       LOCK_RELEASE(output_lck);
// #endif
//       task_completions[nthread].tasks = 0;
//     }
    LOCK_ACQUIRE(worker_lock[nthread]);
    // Next remove any virtual tasks from the per-thread task pool
    if(task_pool[nthread].tasks > 0){
      PolyTask::pending_tasks -= task_pool[nthread].tasks;
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] Thread " << nthread << " removed " << task_pool[nthread].tasks << " virtual tasks. Pending tasks = " << PolyTask::pending_tasks << "\n";
      LOCK_RELEASE(output_lck);
#endif
      task_pool[nthread].tasks = 0;
    }
    LOCK_RELEASE(worker_lock[nthread]);
    
    // Finally check if the program has terminated
    if(gotao_can_exit && (PolyTask::pending_tasks == 0)){
#ifdef PowerProfiling
      out.close();
#endif
      
#ifdef SLEEP
      LOCK_ACQUIRE(output_lck);
      std::cout << "Thread " << nthread << " sleeps for " << AccumTime << " ms. \n";
      LOCK_RELEASE(output_lck);
#endif

#ifdef PTTaccuracy
      LOCK_ACQUIRE(output_lck);
      std::cout << "Thread " << nthread << " 's MAE = " << MAE << ". \n";
      LOCK_RELEASE(output_lck);
      PTT.close();
#endif

#ifdef Energyaccuracy
      LOCK_ACQUIRE(output_lck);
      std::cout << "Thread " << nthread << " 's Energy Prediction = " << EnergyPrediction << ". \n";
      LOCK_RELEASE(output_lck);
#endif

#ifdef NUMTASKS_MIX
      LOCK_ACQUIRE(output_lck);
#if defined(TX2)
      for(int b = 0;b < XITAO_MAXTHREADS; b++){
        for(int a = 1; a < gotao_nthreads; a = a*2){
          std::cout << "Task type: " << b << ": Thread " << nthread << " with width " << a << " completes " << num_task[b][a * gotao_nthreads + nthread] << " tasks.\n";
          num_task[b][a * gotao_nthreads + nthread] = 0;
        }
      }
      // NUM_WIDTH_TASK[1] += num_task[1 * gotao_nthreads + nthread];
      // NUM_WIDTH_TASK[2] += num_task[2 * gotao_nthreads + nthread];
      // NUM_WIDTH_TASK[4] += num_task[4 * gotao_nthreads + nthread];
      // num_task[1 * gotao_nthreads + nthread] = 0;
      // num_task[2 * gotao_nthreads + nthread] = 0;
      // num_task[4 * gotao_nthreads + nthread] = 0;
#endif
      LOCK_RELEASE(output_lck);
#endif
#ifdef EXECTIME
      LOCK_ACQUIRE(output_lck);
      std::cout << "The total execution time of thread " << nthread << " is " << elapsed_exe.count() << " s.\n";
      LOCK_RELEASE(output_lck);
#endif
#ifdef OVERHEAD_PTT
      LOCK_ACQUIRE(output_lck);
      std::cout << "PTT overhead of thread " << nthread << " is " << elapsed_ptt.count() << " s.\n";
      LOCK_RELEASE(output_lck);
#endif
      // if(Sched == 0){
      //break;
      // }else{
      return 0;
      // }
    }
  }
  // pmc.close();
  return 0;
}
