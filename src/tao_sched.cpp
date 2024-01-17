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

std::chrono::time_point<std::chrono::system_clock> interval_t1{}, interval_t2{};
int ptt_freq_index[NUMSOCKETS] = {0};
int start_coreid[NUMSOCKETS] = {0, START_CLUSTER_B};
int end_coreid[NUMSOCKETS] = {START_CLUSTER_B, XITAO_MAXTHREADS};
int PTT_finish_state[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS] = {0}; // First: 2.04 or 1.11; Second: two clusters; third: the number of kernels (assume < XITAO_MAXTHREADS)
int global_training_state[XITAO_MAXTHREADS] = {0}; // array size: the number of kernels (assume < XITAO_MAXTHREADS)
bool global_training = false;
bool across_cluster_stealing = false; // Paper 4: the signal of whether the stealing is across clusters
float percent_cluster[XITAO_MAXTHREADS][NUMSOCKETS] = {0.0f}; // Paper 4: the percentage of tasks allocated to each cluster, first parameter is task type, second parameter is cluster id
int DOP_detection[XITAO_MAXTHREADS] = {0}; // Paper 4: The total number of tasks, inc ready tasks and tasks being executed currently. Array size: assume an application has XITAO_MAXTHREADS different kernels at most 
// int DOP_executing[NUMSOCKETS] = {0}; // Paper 4: The number of tasks being executed currently in each cluster
int remaining_distri_num[XITAO_MAXTHREADS][NUMSOCKETS][XITAO_MAXTHREADS] = {0}; // Paper 4: first parameter is task type, second parameter is thread id, third parameter is cluster id
int LP_best_width[XITAO_MAXTHREADS][NUMSOCKETS] = {1}; // Paper 4: first parameter is task type, second parameter is cluster id
int HP_best_width[XITAO_MAXTHREADS][NUMSOCKETS] = {1};
bool interval_distri_state = false; // Paper 4: whether the interval distribution is finished
int round_robin_counter[XITAO_MAXTHREADS][XITAO_MAXTHREADS] = {0}; // Paper 4: parameter is task type and core id
bool history_LP_visited[XITAO_MAXTHREADS][10] = {false}; // The first parameter is task type, the second parameter is the index of DOP
int history_LP_numTask[XITAO_MAXTHREADS][NUMSOCKETS][10] = {0}; // The first parameter is task type, the second parameter is the index of cluster id, the third parameter is the index of DOP
int history_LP_bestwid[XITAO_MAXTHREADS][NUMSOCKETS][10] = {1}; // The default resource width value is 1
int HP_cpu_freq[XITAO_MAXTHREADS][NUMSOCKETS] = {0}; // The default cpu frequency setting for high paralellism regions is 0. The first parameter is task type, the second parameter is the index of cluster id.
int HP_ddr_freq[XITAO_MAXTHREADS] = {0}; // parameter is task type
int history_LP_cpu_freq[XITAO_MAXTHREADS][NUMSOCKETS][10] = {0}; // The default cpu frequency setting for low paralellism regions is 0. The first parameter is task type, the second parameter is the index of cluster id, the third parameter is the index of DOP
int LP_cpu_freq[XITAO_MAXTHREADS][NUMSOCKETS] = {0}; 
int history_LP_ddr_freq[XITAO_MAXTHREADS][10] = {0}; // first parameter is task type, second parameter is the index of DOP
int LP_ddr_freq[XITAO_MAXTHREADS] = {0}; // parameter is task type
bool interval_HP[XITAO_MAXTHREADS][XITAO_MAXTHREADS] = {false}; // first parameter is task type, second parameter is thread id
bool interval_LP[XITAO_MAXTHREADS][XITAO_MAXTHREADS] = {false}; // first parameter is task type, second parameter is thread id
int status[XITAO_MAXTHREADS];
int status_working[XITAO_MAXTHREADS];
int Sched, num_kernels;
int maySteal_DtoA, maySteal_AtoD;
int distri_thread = -1;
std::atomic<int> DtoA(0);
std::atomic<int> temp_counter(0);
// define the topology
int gotao_sys_topo[5] = TOPOLOGY;

#ifdef NUMTASKS
int NUM_WIDTH_TASK[XITAO_MAXTHREADS] = {0};
#endif

#ifdef EXECTIME
float exe_time[XITAO_MAXTHREADS] = {0.0};
#endif

#ifdef SWEEP_Overhead
std::chrono::duration<double> elapsed_overhead;
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
#endif

float idle_cpu_power[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS] = {0.0};
float perf_alpha[NUMSOCKETS][XITAO_MAXTHREADS][10] = {0.0};  // 10 coefficients for models
float CPUPower_alpha[NUMSOCKETS][XITAO_MAXTHREADS][10] = {0.0};  // 10 coefficients for models
#ifdef DDR_FREQ_TUNING
float DDRPower_alpha[NUMSOCKETS][XITAO_MAXTHREADS][10] = {0.0};  // 10 coefficients for models
float idle_ddr_power[NUM_DDR_AVAIL_FREQ] = {0.0};
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
          if(line_count == 0){
            int thread_id = 0;
            while ((pos = line.find(",")) != std::string::npos) {
              token = line.substr(0, pos);
              int val = stoi(token);
              cluster_mapper[thread_id++] = val;
              // std::cout << "thread_id: " << thread_id << " val: " << val << ", cluster_mapper = " << cluster_mapper[thread_id] << std::endl;
              line.erase(0, pos + 1);
            }
            token = line.substr(0, line.size());
            int val = stoi(token);
            cluster_mapper[thread_id] = val;
            line_count++;
            continue;
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
          if(!init_affinity) { 
            gotao_nthreads = thread_count; 
            init_affinity = true;
          }
          current_thread_id++; 
          line_count++;     
        }
        myfile.close();
      } else {
        std::cout << "Fatal error: could not open hardware layout path " << layout_file << std::endl;    
        exit(0);
      }
    }else {
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
	std::cout << "[DEBUG] XiTAO initialized with " << gotao_nthreads << " threads and configured with " << XITAO_MAXTHREADS << " max threads " << std::endl;
  std::cout << "[DEBUG] The hardware includes " << NUMSOCKETS << " core types.\n";
  for(int i = 0; i < cluster_mapper.size(); i++){
    std::cout << "[DEBUG] Thread " << i << " belongs to cluster " << cluster_mapper[i] << ".\n";
  }
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
  for(int i = 0; i < XITAO_MAXTHREADS; ++i) {
    FreqReader.open("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[i]) + "/cpufreq/scaling_cur_freq");
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
// #ifdef AAWS_CASE
//   SupportedCPUFreq.open("AAWS_Test_CPU_Freq");
//   SupportedDDRFreq.open("AAWS_Test_DDR_Freq");
// #else
  SupportedCPUFreq.open("/home/nvidia/Work_1/JOSS_AL/HW_Supported_CPU_Freq");
  SupportedDDRFreq.open("/home/nvidia/Work_1/JOSS_AL/HW_Supported_DDR_Freq");
// #endif
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

#if (defined TX2) && (defined DDR_FREQ_TUNING) 
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
#else
  cur_ddr_freq = avail_ddr_freq[0];
#endif

// #if (defined ADD_CPU_FREQ_TUNING) 
  // for(int i = 0; i < XITAO_MAXTHREADS; ++i) {
  //   if(cur_freq[i] != avail_freq[cluster_mapper[i]][0]){ // If the starting frequency is not the highest
  //     std::cout << "Detected frequency of core " << i << ": " << cur_freq[i] << ". Throttle to the highest now! \n";
  //     std::ofstream Throttle("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[i]) + "/cpufreq/scaling_setspeed");
  //     if (!Throttle.is_open()){
  //       std::cerr << "[DEBUG] failed while opening the scaling_setspeed file! " << std::endl;
  //       return 0;
  //     }
  //     Throttle << std::to_string(avail_freq[cluster_mapper[i]][0]) << std::endl;
  //     Throttle.close();
  //     cur_freq[i] = avail_freq[cluster_mapper[i]][0];
  //     std::cout << "New frequency of core " << i << ": " << cur_freq[i] << std::endl;
  //   }
  // }
// #endif

// #if (defined DDR_FREQ_TUNING) 
//   if(cur_ddr_freq != avail_ddr_freq[0]){
//     std::cout << "Detected frequency of DDR: " << cur_ddr_freq << ". Throttle to the highest now! \n";
//     std::ofstream EMC("/sys/kernel/debug/bpmp/debug/clk/emc/rate"); // edit chip memory frequency - TX2 specific
//     if (!EMC.is_open()){
//       std::cerr << "[DEBUG] failed while opening the DDR setspeed file! " << std::endl;
//       return 0;
//     }
//     EMC << std::to_string(avail_ddr_freq[0]) << std::endl;
//     EMC.close();
//     cur_ddr_freq = avail_ddr_freq[0];
//   }
// #endif
  cur_ddr_freq_index = 0; // Start by the highest DDR frequency
  cur_freq_index[XITAO_MAXTHREADS] = {0}; // Start by the highest CPU frequency

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
  std::ifstream infile1;
  infile1.open("/home/nvidia/Work_1/JOSS_AL/HW_idle_CPUPower"); /* Step 1: Read idle CPU power */
  if(infile1.fail()){
    std::cout << "Failed to open power profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token4;
  while(std::getline(infile1, token4)) {
    std::istringstream line(token4);
    int ii = 0; // Column index of power files
    int ddrfreq = 0;  // DDR Frequency Index ranging from 0 to 4
    int cpufreq = 0; // CPU Frequency Index ranging from 0 to 11
    while(line >> token4) {
      if(ii == 0){
        ddrfreq = stoi(token4); // first column is DDR frequency index
      }
      if(ii == 1){
        cpufreq = stoi(token4); // Second column is CPU frequency index
      }      
      if(ii > 1){
        float idlep = stof(token4);
        idle_cpu_power[ddrfreq][cpufreq][ii-2] = idlep;  // Third and fourth columns are the idle power of Denver and A57 clusters
      }
      ii++;
    }
  }
  infile1.close();

#ifdef DDR_FREQ_TUNING
  infile1.open("/home/nvidia/Work_1/JOSS_AL/HW_idle_DDRPower"); /* Step 2: Read idle DDR power */
  if(infile1.fail()){
    std::cout << "Failed to open power profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token3;
  while(std::getline(infile1, token3)) {
    std::istringstream line(token3);
    int ii = 0; 
    while(line >> token3) {
      float idlep = stof(token3);
      idle_ddr_power[ii] = idlep;  
      ii++;
    }
  }
  infile1.close();
#endif

#ifdef TX2
  infile1.open("/home/nvidia/Work_1/JOSS_AL/HW_TX2_Perf_Model_Alpha"); /* Read performance model coefficients */
#elif defined ALDERLAKE
  infile1.open("HW_ALDERLAKE_Perf_Model_Alpha"); /* Read performance model coefficients */  
#endif
  if(infile1.fail()){
    std::cout << "Failed to open performance model coefficients profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token5;
  while(std::getline(infile1, token5)) {
    std::istringstream line(token5);
    int ii = 0; 
    int clusterid = 0;
    int width = 0;
    while(line >> token5) {
      if(ii == 0){ // first column is cluster index
        clusterid = stoi(token5); 
      }
      if(ii == 1){ // Second column is available width
        width = stoi(token5); 
      }
      if(ii > 1){
        float coeff = stof(token5);
        perf_alpha[clusterid][width][ii-2] = coeff;
      }
      ii++;
    }
  }
  infile1.close();

#ifdef TX2
  infile1.open("/home/nvidia/Work_1/JOSS_AL/HW_TX2_CPUPower_Model_Alpha"); /* Read CPU Power model coefficients */
#elif defined ALDERLAKE
  infile1.open("HW_ALDERLAKE_CPUPower_Model_Alpha"); /* Read CPU Power model coefficients */  
#endif
  if(infile1.fail()){
    std::cout << "Failed to open CPU power model coefficients profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token6;
  while(std::getline(infile1, token6)) {
    std::istringstream line(token6);
    int ii = 0; 
    int clusterid = 0;
    int width = 0;
    while(line >> token6) {
      if(ii == 0){ // first column is cluster index
        clusterid = stoi(token6); 
      }
      if(ii == 1){ // Second column is available width
        width = stoi(token6); 
      }
      if(ii > 1){
        float coeff = stof(token6);
        CPUPower_alpha[clusterid][width][ii-2] = coeff;
        // std::cout << "CPUPower_alpha[" << clusterid << "][" << width << "][" << ii-2 << "] = " << CPUPower_alpha[clusterid][width][ii-2] << std::endl;
      }
      ii++;
    }
  }
  infile1.close();

#ifdef DDR_FREQ_TUNING
#ifdef TX2
  infile1.open("/home/nvidia/Work_1/JOSS_AL/HW_TX2_DDRPower_Model_Alpha"); /* Read DDR Power model coefficients */
#elif defined ALDERLAKE
  infile1.open("HW_ALDERLAKE_DDRPower_Model_Alpha"); /* Read CPU Power model coefficients */  
#endif
  if(infile1.fail()){
    std::cout << "Failed to open DDR power model coefficients profile file!" << std::endl;
    std::cin.get();
    return 0;
  }
  std::string token7;
  while(std::getline(infile1, token7)) {
    std::istringstream line(token7);
    int ii = 0; 
    int clusterid = 0;
    int width = 0;
    while(line >> token7) {
      if(ii == 0){ // first column is cluster index
        clusterid = stoi(token7); 
      }
      if(ii == 1){ // Second column is available width
        width = stoi(token7); 
      }
      if(ii > 1){
        float coeff = stof(token7);
        DDRPower_alpha[clusterid][width][ii-2] = coeff;
        // std::cout << "DDRPower_alpha[" << clusterid << "][" << width << "][" << ii-2 << "] = " << DDRPower_alpha[clusterid][width][ii-2] << std::endl;
      }
      ii++;
    }
  }
  infile1.close();
#endif
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
  // DOP_detection[pt->tasktype]++;
  LOCK_ACQUIRE(worker_lock[queue]);
  worker_ready_q[queue].push_front(pt);
  LOCK_RELEASE(worker_lock[queue]);
#ifdef DOP_TRACE
  std::chrono::time_point<std::chrono::system_clock> task_para_update;
  task_para_update = std::chrono::system_clock::now();
  auto task_para_update_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(task_para_update);
  auto task_para_update_epoch = task_para_update_ms.time_since_epoch();
  PolyTask::task_counter.fetch_add(1); // The total number of ready tasks (inc. task in queues and just released) increases by 1
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << task_para_update_epoch.count() << ", Currrent ready tasks = " << PolyTask::task_counter.load() << std::endl;
  LOCK_RELEASE(output_lck);
#endif 
#endif
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
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] nthread: " << nthread << " mapped to physical core: "<< phys_core << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif  
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
// #ifdef EXECTIME
  //std::chrono::time_point<std::chrono::system_clock> idle_start, idle_end;
  // std::chrono::duration<double> elapsed_exe;
//   double exe_time = 0.0;
// #endif

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
#if defined(Haswell) || defined(CATS)
        assembly->leader = nthread / assembly->width * assembly->width;
#endif
        /* In some applications (e.g., Dot Product), pretty loose task dependencies (very high parallelism), so here we need to check if tasks are going to run with best config.
        Otherwise, tasks will be executed with the original setting config */
        if(global_training == true && assembly->tasktype < num_kernels){ 
#ifdef ERASE_target_energy_method2  // JOSS design
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
#endif
#if (defined Target_EPTO) && (defined Loose_Parallelism)
// #ifdef INTERVAL_RECOMPUTATION  
          // interval_t1 = std::chrono::system_clock::now();
          // std::chrono::duration<double> time_interval = interval_t1 - interval_t2;
          // if(time_interval.count() >= COMP_INTERVAL) // if interval between last task distribution computation and now is larger than e.g., 1 second
// #endif
          if(temp_counter.load() >= TASK_NUM_INTERVAL){  
            interval_distri_state = false;
            assembly->set_lp_task_distri_state(false);  // set task distribution state to false and recompute the task distribution again
            assembly->set_hp_task_distri_state(false);
            temp_counter.store(0);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            // std::cout << "[DEBUG] Interval between last task distribution computation and now is larger than " << COMP_INTERVAL << " seconds. " << std::endl;
            std::cout << "[DEBUG] Interval between last task distribution computation and now is larger than 100 tasks. " << std::endl;
            LOCK_RELEASE(output_lck); 
#endif
          }else{
            if(distri_thread == -1){
              int temp_cnt = 0;
              for(int a = 0; a < gotao_nthreads; a++){
                for(int b = 0; b < NUMSOCKETS; b++){
                  temp_cnt += remaining_distri_num[assembly->tasktype][a][b];
                  if(temp_cnt > 0){
                    interval_distri_state = true;
                    break;
                  }
                }
              }
            }else{
              interval_distri_state = true;
            }
          }
          if(interval_distri_state == false){
            interval_HP[assembly->tasktype][nthread] = false;
            interval_LP[assembly->tasktype][nthread] = false;
            if(PolyTask::task_counter.load() >= HP_LP_Threshold * gotao_nthreads){ // if the number of ready tasks are more than 4 * 6 = 24, then we consider it as high parallelism
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck); 
              std::cout << "[LOOSE-P] " << assembly->kernel_name << " task " << assembly->taskid << ". Current total ready tasks: " << PolyTask::task_counter.load() << " >= " << HP_LP_Threshold * gotao_nthreads << " ===> High parallelism." << std::endl;
              LOCK_RELEASE(output_lck);
#endif
              across_cluster_stealing = true;
              // assembly->task_distri_HP(nthread, assembly, PolyTask::task_counter.load());
              assembly->task_distri_HP(nthread, assembly, TASK_NUM_INTERVAL);
              interval_HP[assembly->tasktype][nthread] = true;
              distri_thread = nthread;
              temp_counter.fetch_add(1);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] temp_counter = " << temp_counter.load() << std::endl;
              LOCK_RELEASE(output_lck); 
#endif
            }else{
              float clus_part[NUMSOCKETS] = {0}; /* Here we decide the current DOP is high parallelism or low parallelism */
              float suppose_load_balance = 0.0f;
              for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
                clus_part[clus_id] = assembly->get_timetable(0, 0, 1, 0) / assembly->get_timetable(0, 0, clus_id, 0) * (end_coreid[clus_id]-start_coreid[clus_id]);
                suppose_load_balance += clus_part[clus_id];
              } 
              if(PolyTask::task_counter.load() >= suppose_load_balance){ /* High Parallelism */ // old: dop >= suppose_load_balance
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck); 
                std::cout << "[LOOSE-P] " << assembly->kernel_name << " task " << assembly->taskid << ". Current total ready tasks: " << PolyTask::task_counter.load() << " < " << HP_LP_Threshold * gotao_nthreads << ". DOP for load balancing is " \
                << suppose_load_balance << ". Current total number of READY tasks of the type is " << PolyTask::task_counter.load() << " ===> High parallelism." << std::endl;
                LOCK_RELEASE(output_lck);
#endif
                // assembly->task_distri_HP(nthread, assembly, PolyTask::task_counter.load());
                assembly->task_distri_HP(nthread, assembly, TASK_NUM_INTERVAL);
                across_cluster_stealing = true;
                interval_HP[assembly->tasktype][nthread] = true;
                distri_thread = nthread;
                temp_counter.fetch_add(1);
              }else{
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[LOOSE-P] " << assembly->kernel_name << " task " << assembly->taskid << ". Current total ready tasks: " << PolyTask::task_counter.load() << " < " << HP_LP_Threshold * gotao_nthreads << ". DOP for load balancing is " \
                << suppose_load_balance << ". Current total number of READY tasks of the type is " << PolyTask::task_counter.load() << " ===> Low parallelism." << std::endl;
                LOCK_RELEASE(output_lck);
#endif
                // assembly->task_distri_LP(nthread, assembly, PolyTask::task_counter.load());
                assembly->task_distri_LP(nthread, assembly, TASK_NUM_INTERVAL);
                across_cluster_stealing = false;
                interval_LP[assembly->tasktype][nthread] = true;
                distri_thread = nthread;
                temp_counter.fetch_add(1);
              }
            }
          }else{ /* After the task distribution computation, get the percentage for each cluster, based on the value to assign the following tasks */  
            assembly->assign_config(distri_thread, assembly);
            temp_counter.fetch_add(1);
          }
#endif
        }

#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Distributing " << assembly->kernel_name << " task " << assembly->taskid << " with width " << assembly->width << " to workers [" << assembly->leader << "," << assembly->leader + assembly->width << ")" << std::endl;
        LOCK_RELEASE(output_lck);
#endif

        /* JOSS: After getting the best config, and before distributing to AQs: 
        (1) Coarse-grained task: check if it is needed to tune the frequency; 
        (2) Fine-grained task: check the WQs of the cluster include N consecutive same tasks, that the total execution time of these N tasks > threshold, then search for the best frequency and then tune the frequency */
#ifdef ERASE_target_energy_method2
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
                          idleP_cluster = idle_cpu_power[ddr_freq_indx][freq_indx][best_cluster] + idle_cpu_power[ddr_freq_indx][freq_indx][1-best_cluster]; /* Then equals idle power of whole chip */
// #ifdef DEBUG
//                           LOCK_ACQUIRE(output_lck);
//                           std::cout << "[DEBUG] Cluster " << 1-best_cluster << " no active cores. Therefore, the idle power of cluster " << best_cluster << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
//                           LOCK_RELEASE(output_lck);
// #endif 
                        }else{
                          idleP_cluster = idle_cpu_power[ddr_freq_indx][freq_indx][best_cluster]; /* otherwise, equals idle power of the cluster */
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
          for(int ii = assembly->leader; ii < assembly->leader + best_width; ii++){
            if(assembly->best_cpu_freq != cur_freq[ii]){ /* check if the required frequency equals the current frequency! */
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] For " << assembly->kernel_name << " task " << assembly->taskid << ": current core frequency " << cur_freq[ii] << " != required frequency " << assembly->best_cpu_freq << ". \n";
              LOCK_RELEASE(output_lck);
#endif
#else
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DVFSforFineGrained] For " << assembly->kernel_name << " task " << assembly->taskid << ": current core frequency " << cur_freq[ii] << ", required frequency " << assembly->best_cpu_freq << ". \n";
              LOCK_RELEASE(output_lck);
#endif
#endif
              if(best_width == (end_coreid[best_cluster] - start_coreid[best_cluster])){  /* ==> Strategy for tuning the frequency: if the best width = number of cores in cluster, just change frequency! */
                assembly->cpu_frequency_tuning(nthread, best_cluster, assembly->leader, best_width, assembly->get_best_cpu_freq());
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] Best width: " << best_width << " equals the number of cores in cluster " << best_cluster << ". Change the frequency now!" << std::endl;
                LOCK_RELEASE(output_lck);
#endif
              }else{
#ifdef ALDERLAKE
                if(ii < START_CLUSTER_B){ /* if the task is scheduled on Alder lake platform + P-cores ==> Per core DVFS */
                  assembly->cpu_frequency_tuning(nthread, best_cluster, assembly->leader, best_width, assembly->get_best_cpu_freq());
                }else{ /* if the task is scheduled on platform with cluster-level DVFS */
#endif
                // int freq_change_cluster_active = std::accumulate(status_working + start_coreid[best_cluster], status_working + end_coreid[best_cluster], 0); 
                  if(std::accumulate(status_working + start_coreid[best_cluster], status_working + end_coreid[best_cluster], 0) > 0){ /* Check if there are any concurrent tasks? Yes, take the average, no, change the frequency to the required! */
                    int new_freq_index = (cur_freq_index[ii] + assembly->get_best_cpu_freq()) / 2;   /* Method 1: take the average of two frequencies */
                    assembly->cpu_frequency_tuning(nthread, best_cluster, assembly->leader, best_width, new_freq_index);
#ifdef DEBUG
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "[DEBUG] CPU frequency tuning: The cluster has concurrent tasks running at the same time! Strategy: take the average, tune the frequency to " << avail_freq[best_cluster][new_freq_index] << std::endl;
                    LOCK_RELEASE(output_lck);
#endif 
                  }else{
                    assembly->cpu_frequency_tuning(nthread, best_cluster, assembly->leader, best_width, assembly->get_best_cpu_freq());
#ifdef DEBUG
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "[DEBUG] CPU frequency tuning: The cluster has no tasks running now! Change the cluster frequency now!" << std::endl;
                    LOCK_RELEASE(output_lck);
#endif
                  }
#ifdef ALDERLAKE
                }
#endif
              }
#ifndef FineStrategyTest
            }
          }
#endif
        }
#ifdef DDR_FREQ_TUNING        /* Tune DDR frequency if required != current, for both fine-grained and coarse-grained tasks */
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
#endif
      }
#endif
#if (defined Target_EPTO) 
#if (defined ADD_CPU_FREQ_TUNING)
      if(assembly->get_enable_cpu_freq_change() == true){ /* Allow to do frequency tuning for the tasks */
        int clus0_freq = assembly->get_clus0_cpu_freq(); 
        int clus1_freq = assembly->get_clus1_cpu_freq();
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " is allowed to do CPU Frequency scaling => required frequency index is <" << clus0_freq << ", " << clus1_freq << ">\n";
        LOCK_RELEASE(output_lck);
#endif
        if(clus0_freq != cur_freq_index[START_CLUSTER_A]){ 
          if(clus0_freq == -1){ // If the task has no requirement on cluster 0,
            if(std::accumulate(status_working + start_coreid[0], status_working + end_coreid[0], 0) == 0){ // and there is no concurrent task running on cluster 0,
              assembly->cpu_frequency_tuning(nthread, 0, start_coreid[0], end_coreid[0] - start_coreid[0], NUM_AVAIL_FREQ-1); // set the best frequency setting for the kernel tasks
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] No concurrent task, can be any freq, cluster 0 change frequency to " << avail_freq[0][NUM_AVAIL_FREQ-1] << std::endl;
              LOCK_RELEASE(output_lck);
#endif
            }
          }else{
          if(num_kernels > 1 && std::accumulate(status_working + start_coreid[0], status_working + end_coreid[0], 0) > 0){
            int new_freq_index = (cur_freq_index[START_CLUSTER_A] + clus0_freq + 1) / 2;   /* Method 1: take the average of two frequencies, rounding up */
            assembly->cpu_frequency_tuning(nthread, 0, start_coreid[0], end_coreid[0] - start_coreid[0], new_freq_index); // set the best frequency setting for the kernel tasks
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] CPU frequency tuning: cluster 0 has concurrent tasks running at the same time! Strategy: take the average, tune the frequency to " << avail_freq[0][new_freq_index] << std::endl;
            LOCK_RELEASE(output_lck);
#endif 
          }else{
            assembly->cpu_frequency_tuning(nthread, 0, start_coreid[0], end_coreid[0] - start_coreid[0], clus0_freq); // set the best frequency setting for the kernel tasks
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] CPU frequency tuning: cluster 0 change frequency to " << avail_freq[0][clus0_freq] << std::endl;
            LOCK_RELEASE(output_lck);
#endif
          }
        }
        }
        if(clus1_freq != cur_freq_index[START_CLUSTER_B]){ 
          if(clus1_freq == -1){
            if(std::accumulate(status_working + start_coreid[1], status_working + end_coreid[1], 0) == 0){
              assembly->cpu_frequency_tuning(nthread, 1, start_coreid[1], end_coreid[1] - start_coreid[1], NUM_AVAIL_FREQ-1); // set the best frequency setting for the kernel tasks
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] No concurrent task, can be any freq, cluster 1 change frequency to " << avail_freq[1][NUM_AVAIL_FREQ-1] << std::endl;
              LOCK_RELEASE(output_lck);
#endif
            }
          }else{
          if(num_kernels > 1 && std::accumulate(status_working + start_coreid[1], status_working + end_coreid[1], 0) > 0){
            int new_freq_index = (cur_freq_index[START_CLUSTER_B] + clus1_freq + 1) / 2;   /* Method 1: take the average of two frequencies, rounding up*/
            assembly->cpu_frequency_tuning(nthread, 1, start_coreid[1], end_coreid[1] - start_coreid[1], new_freq_index); // set the best frequency setting for the kernel tasks
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] CPU frequency tuning: cluster 1 has concurrent tasks running at the same time! Strategy: take the average, tune the frequency to " << avail_freq[1][new_freq_index] << std::endl;
            LOCK_RELEASE(output_lck);
#endif 
          }else{
            assembly->cpu_frequency_tuning(nthread, 1, start_coreid[1], end_coreid[1] - start_coreid[1], clus1_freq);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] CPU frequency tuning: cluster 1 change frequency to " << avail_freq[1][clus1_freq] << std::endl;
            LOCK_RELEASE(output_lck);
#endif
          }
        }
        }
      }
#endif
#if defined ADD_DDR_FREQ_TUNING
      if(assembly->get_enable_ddr_freq_change() == true){ /* Allow to do frequency tuning for the tasks */
        int ddr_freq = assembly->get_best_ddr_freq();
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " is allowed to do Memory Frequency Scaling. Its best DDR freq index = " << ddr_freq << ". Current DDR frequency is = " << cur_ddr_freq << ", index = " << cur_ddr_freq_index << ". \n";
        LOCK_RELEASE(output_lck);
#endif
        if(ddr_freq != cur_ddr_freq_index){
          if(num_kernels > 1 && std::accumulate(std::begin(status_working), status_working + assembly->leader, 0) + std::accumulate(status_working + assembly->leader + assembly->width, std::end(status_working), 0) > 0){  /* Check if there are any concurrent tasks? No, change the frequency to the required! */
            int new_ddr_freq_index = (cur_ddr_freq_index + ddr_freq + 1) / 2;   /* Method 1: take the average of two frequencies */
            assembly->ddr_frequency_tuning(nthread, new_ddr_freq_index);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] DDR frequency tuning: concurrent tasks running at the same time! Strategy: take the average, tune the frequency to " << avail_ddr_freq[new_ddr_freq_index] \
            << " =====> current memory frequency = " << cur_ddr_freq << std::endl;
            LOCK_RELEASE(output_lck);
#endif 
          }else{ 
            assembly->ddr_frequency_tuning(nthread, ddr_freq);
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] DDR frequency tuning: Change the memory frequency now! =====> current memory frequency = " << cur_ddr_freq << std::endl;
            LOCK_RELEASE(output_lck);
#endif
          }
        }
      }
#endif
#endif
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
        }
        for(int i = assembly->leader; i < assembly->leader + assembly->width; i++){
          LOCK_RELEASE(worker_assembly_lock[i]);
        }
#ifdef DOP_TRACE          
        PolyTask::task_counter.fetch_sub(1); // inserted into assembly queue. The total number of ready tasks decreased by 1
        std::chrono::time_point<std::chrono::system_clock> task_para_update;
        task_para_update = std::chrono::system_clock::now();
        auto task_para_update_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(task_para_update);
        auto task_para_update_epoch = task_para_update_ms.time_since_epoch();
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << task_para_update_epoch.count() << ", Currrent ready tasks = " << PolyTask::task_counter.load() << std::endl;
        LOCK_RELEASE(output_lck);
#endif
#endif
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
      if(Sched == 1 && nthread == assembly->leader){
      // if(Sched == 1 && assembly->start_running == false){ /* for tasks with wider width, if this is the first thread, set the start running frequency, this is meant to avoid updating the performance table when frequency changing happens between threads execution */
#if defined Performance_Model_Cycle 
        ioctl(fd1, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fd1, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
        assembly->start_running = true;
        // int clus_id = (nthread < START_CLUSTER_B)? 0:1;
        // DOP_executing[clus_id]++; // Paper 4: increase the number of executing tasks on this cluster
        assembly->start_running_freq = cur_freq[nthread];
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
      status_working[nthread] = 0;  // The core/thread finished task, so set status as 0
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] " << assembly->kernel_name << " task " << assembly->taskid << " execution time on thread " << nthread << ": " << elapsed_seconds.count() << "\n";
      LOCK_RELEASE(output_lck);
#endif 
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
      double ticks = elapsed_seconds.count(); 
#ifdef EXECTIME
      if(ticks > 0.00001){
        // elapsed_exe += elapsed_seconds;
        exe_time[nthread] += ticks;
      }
#endif
      
      int width_index = assembly->width - 1;
      /* Paper 4: Update the PTT table after every subtask */
      if(global_training == true && assembly->start_running_freq == cur_freq[nthread]){
        int clus_id = (nthread < START_CLUSTER_B)? 0:1;
        float oldticks = assembly->get_timetable(cur_ddr_freq_index, cur_freq_index[nthread], clus_id, width_index);
        assembly->set_timetable(cur_ddr_freq_index, cur_freq_index[nthread], clus_id, (4 * oldticks + ticks)/5.0, width_index);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Task " << assembly->taskid << " execution time on thread " << nthread << " updates the performance table, current time<" << cur_ddr_freq_index << ", " << cur_freq_index[nthread] << ", " \
        << clus_id << ", " << width_index << ">: " << assembly->get_timetable(cur_ddr_freq_index, cur_freq_index[nthread], clus_id, width_index) << std::endl;
        LOCK_RELEASE(output_lck);
#endif
      }

      if(Sched == 1 && ticks > 0.00001 && assembly->get_timetable_state(2) == false && assembly->tasktype < num_kernels){  /* Only leader core update the PTT entries */ // Make sure that Null tasks do not used to update PTT tables
        /* (1) Was running on Denver (2) Denver PTT table hasn't finished training     (3) if the frequency changing is happening during the task execution, do not update the table*/
        if(assembly->leader < START_CLUSTER_B){
        if(assembly->get_timetable_state(0) == false && assembly->start_running_freq == cur_freq[nthread]){
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
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[" << avail_ddr_freq[0] << ", " << TRAIN_MAX_A_FREQ << ", ClusterA, width = " \
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
              float oldticks = assembly->get_timetable(0, TRAIN_MED_A_FREQ_idx, 0, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0) / assembly->width;
                if(oldticks == 0.0f || (newticks < oldticks && fabs(newticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, TRAIN_MED_A_FREQ_idx, 0, newticks, width_index);
                }else{
                  assembly->set_timetable(0, TRAIN_MED_A_FREQ_idx, 0, (newticks+oldticks)/2, width_index);
                }
              }else{
                if(oldticks == 0.0f || (ticks < oldticks && fabs(ticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, TRAIN_MED_A_FREQ_idx, 0, ticks, width_index);
                }else{
                  assembly->set_timetable(0, TRAIN_MED_A_FREQ_idx, 0, (ticks+oldticks)/2, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(1, 0, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[" << avail_ddr_freq[0] << ", " << TRAIN_MED_A_FREQ << ", CLusterA, width = " \
               << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(1,0, width_index) << ". Current time: " << assembly->get_timetable(0, TRAIN_MED_A_FREQ_idx, 0, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              //++assembly->threads_out_tao;
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
            }
          }

          if (ptt_freq_index[0] == 0){          /* Current frequency is 2.04GHz */
            int Sampling = 0;
            for(auto&& ii : ptt_layout[start_coreid[0]]){
              if (assembly->get_PTT_UpdateFinish(0, 0, ii-1) >= NUM_TRAIN_TASKS){
                Sampling++;
              }
            }
            if(Sampling == ptt_layout[start_coreid[0]].size()){
            // if(assembly->get_PTT_UpdateFinish(0, 0, 0) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 0, 1) >= NUM_TRAIN_TASKS){ /* First dimention: 2.04GHz, second dimention: Denver, third dimention: width_index */
              PTT_finish_state[0][0][assembly->tasktype] = 1; /* First dimention: 2.04GHz, second dimention: Denver, third dimention: tasktype */
              if(std::accumulate(std::begin(PTT_finish_state[0][0]), std::begin(PTT_finish_state[0][0])+num_kernels, 0) == num_kernels){ /* Check all kernels have finished the training of Denver at 2.04GHz */
                for(int ii = START_CLUSTER_A; ii < START_CLUSTER_B; ii++){
                  std::ofstream ClusterA("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[ii]) + "/cpufreq/scaling_setspeed"); // edit Denver cluster frequency
                  if (!ClusterA.is_open()){
                    std::cerr << "[DEBUG] Somthing failed while opening the file! " << std::endl;
                    return 0;
                  }
                  ClusterA << std::to_string(TRAIN_MED_A_FREQ) << std::endl;
                  ClusterA.close();                      
                  cur_freq[ii] = TRAIN_MED_A_FREQ;
                  cur_freq_index[ii] = TRAIN_MED_A_FREQ_idx;
                }
                ptt_freq_index[0] = 1;
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] ClusterA completed Sampling phase at highest frequency " << TRAIN_MAX_A_FREQ << ", now throttle the frequency to the medium " << TRAIN_MED_A_FREQ << "\n";
                LOCK_RELEASE(output_lck);
#endif
              }
            }
          }else{ /* Current frequency is 1.11GHz */
              int ptt_check = 0;            
              for(auto&& width : ptt_layout[START_CLUSTER_A]){ 
                float check_ticks = assembly->get_timetable(0, TRAIN_MED_A_FREQ_idx, 0, width - 1); /* First parameter is ptt_freq_index[0] = 12/2 = 6, 1.11GHz */
                if(assembly->get_PTT_UpdateFinish(1, 0, width-1) >= NUM_TRAIN_TASKS && check_ticks > 0.0f){ /* PTT_UpdateFinish first dimention is 1 => 1.11GHz */
                  ptt_check++;
                  if(assembly->get_mbtable(0, width-1) == 0.0f){ // If the memory-boundness of this config hasn't been computed yet
                    float memory_boundness = 0.0f;
#if defined Performance_Model_Cycle                  /* Method 1: Calculate Memory-boundness using cycles = (1 - cycle2/cycle1) / (1- f2/f1) */
                    uint64_t check_cycles = assembly->get_cycletable(1, 0, width - 1); /* get_cycletable first dimention is 1 => 1.11GHz */
                    uint64_t cycles_high = assembly->get_cycletable(0, 0, width-1); /* First parameter is 0, means 2.04GHz, check_cycles is at 1.11Ghz */
                    float a = 1 - float(check_cycles) / float(cycles_high);
                    float b = 1 - float(avail_freq[0][TRAIN_MED_A_FREQ_idx]) / float(avail_freq[0][0]);
                    memory_boundness = a/b;
#endif
#if defined Performance_Model_Time                   /* Method 2: Calculate Memory-boundness (using execution time only) = ((T2f2/T1)-f1) / (f2-f1) */
                    float highest_ticks = assembly->get_timetable(0, 0, 0, width - 1);
                    float a = float(avail_freq[0][0]) / float(avail_freq[0][TRAIN_MED_A_FREQ_idx]);
                    float b = check_ticks / highest_ticks;
                    memory_boundness = (b-a) / (1-a);
#endif                    
                    if(memory_boundness > 1){
                      // LOCK_ACQUIRE(output_lck);
                      // std::cout << "[Warning] " << assembly->kernel_name << "->Memory-boundness (ClusterA) is greater than 1!" << std::endl;
                      // LOCK_RELEASE(output_lck);
                      memory_boundness = 1;
                    }else{
                      if(memory_boundness <= 0){ /* Execution time and power prediction according to the computed memory-boundness level */
                        // LOCK_ACQUIRE(output_lck);
                        // std::cout << "[Warning] " << assembly->kernel_name << "->Memory-boundness (ClusterA) is smaller than 0!" << std::endl;
                        // LOCK_RELEASE(output_lck);
                        memory_boundness = 0.00001;
                      }
                    }
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "Memory-boundness of " << assembly->kernel_name <<" tasks on <Cluster0, " << width << ">: " << memory_boundness << std::endl;
                    LOCK_RELEASE(output_lck);
                    assembly->set_mbtable(0, memory_boundness, width-1); /*first parameter: cluster 0 - Denver, second parameter: update value, third value: width_index */
#if defined Model_Computation_Overhead
                    struct timespec Denver_start, Denver_finish, Denver_delta;
                    clock_gettime(CLOCK_REALTIME, &Denver_start);
#endif
                    for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){ /* Compute Predictions according to Memory-boundness Values */
                      float ddr_freq_scaling = float(avail_ddr_freq[0]) / float(avail_ddr_freq[ddr_freq_indx]);
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                        if((ddr_freq_indx == 0 && freq_indx == 0) || (ddr_freq_indx == 0 && freq_indx == TRAIN_MED_A_FREQ_idx)){ /* if DDR freq = 1.866 and CPU freq = 2.04 or 1.11 GHz, then skip since they are sampled */
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
                        ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + perf_alpha[0][width][0] * memory_boundness + perf_alpha[0][width][1] * cpu_freq_scaling \
                        + perf_alpha[0][width][2] * ddr_freq_scaling + perf_alpha[0][width][3] * pow(memory_boundness, 2) + perf_alpha[0][width][4] * memory_boundness * cpu_freq_scaling \
                        + perf_alpha[0][width][5] * pow(cpu_freq_scaling, 2) + perf_alpha[0][width][6] * memory_boundness * ddr_freq_scaling + perf_alpha[0][width][7] * cpu_freq_scaling * ddr_freq_scaling \
                        + perf_alpha[0][width][8] * pow(ddr_freq_scaling, 2) + perf_alpha[0][width][9]);
//                         if(width == 1){ /*Denver, width=1*/                     
// #if defined Performance_Model_1
//                           ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 1.7250611 * memory_boundness + 0.0410749 * cpu_freq_scaling + 0.1339562 * ddr_freq_scaling - 0.2918719); 
// #endif
// #if defined Performance_Model_2
//                           ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling - 0.3217646 * memory_boundness + 0.0777825 * cpu_freq_scaling \
//                           + 0.0979088 * ddr_freq_scaling + 0.2310173 * memory_boundness * cpu_freq_scaling + 0.9953363 * memory_boundness * ddr_freq_scaling \
//                           - 0.0447902 * cpu_freq_scaling * ddr_freq_scaling - 0.1645463);
// #endif
// #if defined JOSS_Perf_Model
//                           ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling - 0.3738012 * memory_boundness - 0.0719465 * cpu_freq_scaling \
//                           - 0.0923319 * ddr_freq_scaling + 0.1895604 * pow(memory_boundness, 2) + 0.2310173 * memory_boundness * cpu_freq_scaling \
//                           + 0.0229213 * pow(cpu_freq_scaling, 2) + 0.9953363 * memory_boundness * ddr_freq_scaling \
//                           - 0.0447902 * cpu_freq_scaling * ddr_freq_scaling + 0.0567653 * pow(ddr_freq_scaling, 2) + 0.1594567);
// #endif
// // #ifdef DEBUG
// //                           LOCK_ACQUIRE(output_lck);
// //                           std::cout << "[DEBUG] " << assembly->kernel_name << "(Denver, 1): " << avail_ddr_freq[ddr_freq_indx] << ", " << avail_freq[freq_indx] << ", execution time prediction = " << ptt_value_newfreq << ".\n";
// //                           LOCK_RELEASE(output_lck);
// // #endif
//                         }
//                         if(width == 2){ /*Denver, width=2*/
// #if defined Performance_Model_1
//                           ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 2.1204276 * memory_boundness + 0.0167277 * cpu_freq_scaling + 0.0903234 * ddr_freq_scaling - 0.1620744); 
// #endif
// #if defined Performance_Model_2
//                           ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling - 1.8822725 * memory_boundness + 0.1079102 * cpu_freq_scaling \
//                           + 0.007308 * ddr_freq_scaling - 0.6263244 * memory_boundness * cpu_freq_scaling + 3.5389818 * memory_boundness * ddr_freq_scaling \
//                           - 0.039597 * cpu_freq_scaling * ddr_freq_scaling - 0.1040418);
// #endif
// #if defined JOSS_Perf_Model
//                           // ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + 1.1308933 * memory_boundness - 0.0018929 * cpu_freq_scaling \
//                           // - 0.2420771 * ddr_freq_scaling - 30.7739177 * pow(memory_boundness, 2) - 0.6263244 * memory_boundness * cpu_freq_scaling \
//                           // + 0.0168092 * pow(cpu_freq_scaling, 2) + 3.5389818 * memory_boundness * ddr_freq_scaling \
//                           // - 0.039597 * cpu_freq_scaling * ddr_freq_scaling + 0.0744132 * pow(ddr_freq_scaling, 2) + 0.1485836);
//                           ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling - 1.3227937 * memory_boundness + 0.0945313 * cpu_freq_scaling \
//                           +0.0407899 * ddr_freq_scaling - 0.328293 * memory_boundness * cpu_freq_scaling + 2.7087046 * memory_boundness * ddr_freq_scaling \
//                           - 0.039641 * cpu_freq_scaling * ddr_freq_scaling - 0.1284175);
// #endif
// // #ifdef DEBUG
// //                           LOCK_ACQUIRE(output_lck);
// //                           std::cout << "[DEBUG] " << assembly->kernel_name << "(Denver, 2): " << avail_ddr_freq[ddr_freq_indx] << ", " << avail_freq[freq_indx] << ", execution time prediction = " << ptt_value_newfreq << ".\n";
// //                           LOCK_RELEASE(output_lck);
// // #endif
//                         }
#endif
                        assembly->set_timetable(ddr_freq_indx, freq_indx, 0, ptt_value_newfreq, width-1);
                        }
                      }
                    }
                    for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){ /* (2) CPU, DDR Power value prediction */
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ 
                        float cpupower = 0.0;
                        float ddrpower = 0.0;
                        float cpufreq = float(avail_freq[0][freq_indx])/1000000.0;   
                        cpupower = CPUPower_alpha[0][width][0] * memory_boundness + CPUPower_alpha[0][width][1] * cpufreq + CPUPower_alpha[0][width][2] * memory_boundness * cpufreq \
                        + CPUPower_alpha[0][width][3] * pow(cpufreq, 2) + CPUPower_alpha[0][width][4];                    
                        assembly->set_cpupowertable(ddr_freq_indx, freq_indx, 0, cpupower, width-1); 
#ifdef DDR_FREQ_TUNING
                        float ddrfreq = float(avail_ddr_freq[ddr_freq_indx])/1000000000.0;
                        ddrpower = DDRPower_alpha[0][width][0] * memory_boundness + DDRPower_alpha[0][width][1] * cpufreq + DDRPower_alpha[0][width][2] * ddrfreq \
                        + DDRPower_alpha[0][width][3] * pow(memory_boundness, 2) + DDRPower_alpha[0][width][4] * memory_boundness * cpufreq + DDRPower_alpha[0][width][5] * pow(cpufreq, 2) \
                        + DDRPower_alpha[0][width][6] * memory_boundness * ddrfreq + DDRPower_alpha[0][width][7] * cpufreq * ddrfreq + DDRPower_alpha[0][width][8] * pow(ddrfreq, 2) \
                        + DDRPower_alpha[0][width][9]; 
// #if defined DDR_Power_Model_1
//                         if(width == 1){ /*Denver, width=1*/
//                           ddrpower = 4.0040504 * memory_boundness + 0.306568 * cpufreq + 0.7431409 * ddrfreq - 0.8051031;
//                         }
//                         if(width == 2){ /*Denver, width=2*/
//                           ddrpower = 17.830587 * memory_boundness + 0.3768498 * cpufreq + 0.7559968 * ddrfreq - 0.8784249;
//                         }
// #endif  
// #if defined DDR_Power_Model_2
//                         if(width == 1){ /*Denver, width=1*/
//                           ddrpower = 2.5701454 * memory_boundness + 0.0517271 * cpufreq + 0.8294422 * ddrfreq + 1.8753657 * memory_boundness * cpufreq - 0.5995333 * memory_boundness * ddrfreq - 0.0029895* cpufreq * ddrfreq - 0.6119471;
//                         }
//                         if(width == 2){ /*Denver, width=2*/
//                           ddrpower = 10.075838 * memory_boundness - 0.0133797 * cpufreq + 0.8017427 * ddrfreq + 8.7131834 * memory_boundness * cpufreq - 1.9651515 * memory_boundness * ddrfreq + 0.0243026 * cpufreq * ddrfreq - 0.5452121;
//                         }
// #endif
// #if defined DDR_Power_Model_3
//                         if(width == 1){ /*Denver, width=1*/
//                           ddrpower = 2.7306748 * memory_boundness + 0.1950916 * cpufreq - 1.3051822 * ddrfreq - 0.584781 * pow(memory_boundness, 2) + 1.8753657 * memory_boundness * cpufreq - 0.0602169 * pow(cpufreq, 2) \
//                           - 0.5995333 * memory_boundness * ddrfreq - 0.0029895 * cpufreq * ddrfreq + 0.8006888 * pow(ddrfreq, 2) + 0.6209039;
//                         }
//                         if(width == 2){ /*Denver, width=2*/
//                           //ddrpower = 11.8840022 * memory_boundness + 0.2207538 * cpufreq - 1.1475948 * ddrfreq - 23.7916345 * pow(memory_boundness, 2) + 8.7131834 * memory_boundness * cpufreq - 0.0871027 * pow(cpufreq, 2) \
//                           //- 1.9651515 * memory_boundness * ddrfreq + 0.0243026 * cpufreq * ddrfreq + 0.7311884 * pow(ddrfreq, 2) + 0.5285202;
// 			                    ddrpower = 11.9390059 * memory_boundness + 0.224722 * cpufreq - 1.1527272 * ddrfreq - 24.7881921 * pow(memory_boundness, 2) + 8.704204 * memory_boundness * cpufreq - 0.0864551 * pow(cpufreq, 2) \
//                           - 1.9467908 * memory_boundness * ddrfreq + 0.0211395 * cpufreq * ddrfreq + 0.7337044 * pow(ddrfreq, 2) + 0.5295545;
// 			                    //ddrpower = 10.075838 * memory_boundness - 0.0133797 * cpufreq + 0.8017427 * ddrfreq + 8.7131834 * memory_boundness * cpufreq - 1.9651515 * memory_boundness * ddrfreq + 0.0243026 * cpufreq * ddrfreq - 0.5452121;
//                         }
// #endif
// #if defined DDR_Power_Model_4
//                         if(width == 1){ /*Denver, width=1*/
//                           ddrpower = 4.1645798 * memory_boundness + 0.4499325 * cpufreq - 1.3914835 * ddrfreq - 0.584781 * pow(memory_boundness, 2) - 0.0602169 * pow(cpufreq, 2) + 0.8006888 * pow(ddrfreq, 2) + 0.4277479;
//                         }
//                         if(width == 2){ /*Denver, width=2*/
//                           ddrpower = 19.6387512	* memory_boundness + 0.584224 * cpufreq - 1.1933407 * ddrfreq - 23.7916345 * pow(memory_boundness, 2) - 0.0871027 * pow(cpufreq, 2) + 0.7311884 * pow(ddrfreq, 2) + 0.1953074;
//                         }
// #endif
                        assembly->set_ddrpowertable(ddr_freq_indx, freq_indx, 0, ddrpower, width-1); 
#endif
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
                    printf("[Overhead] Model calculation (ClusterA): %d.%.9ld\n", (int)Denver_delta.tv_sec, Denver_delta.tv_nsec);
                    LOCK_RELEASE(output_lck);
#endif
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
#if defined Performance_Model_Cycle    
                  std::cout << "[DEBUG] " << assembly->kernel_name << "->PTT_Value[1.11GHz, Denver, " << width << "] = " << check_ticks << ". Cycles(1.11GHz) = " << check_cycles << ", Cycles(2.04GHz) = " << cycles_high <<\
                  ". Memory-boundness(Denver, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
#endif
#if defined Performance_Model_Time
                  std::cout << "[DEBUG] Memory-boundness(CLusterA, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
                  // " << assembly->kernel_name << "->PTT_Value[HIGHEST_DDR_FREQ, Highest_CPU_FREQ, CLusterA, " << width << "] = " << highest_ticks << ", PTT_Value[HIGHEST_DDR_FREQ, MED_CPU_FREQ, CLusterA, " << width << "] = " \
                  << check_ticks << ". Memory-boundness(CLusterA, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
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
                assembly->set_timetable_state(0, true); /* Finish the PTT training in Cluster A part, set state to true */
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] " << assembly->kernel_name << ": Cluster A completed PTT training at " << TRAIN_MAX_A_FREQ << " and " << TRAIN_MED_A_FREQ << "\n";
                LOCK_RELEASE(output_lck);
#endif
                if(std::accumulate(std::begin(PTT_finish_state[1][0]), std::begin(PTT_finish_state[1][0])+num_kernels, 0) == num_kernels){
#ifdef AAWS_CASE
                  assembly->cpu_frequency_tuning(nthread, 0, start_coreid[0], end_coreid[0] - start_coreid[0], 10); // All kernels finished training on Cluster A, then tune cluster A frequency to 0.5GHz
#else
                  assembly->cpu_frequency_tuning(nthread, 0, start_coreid[0], end_coreid[0] - start_coreid[0], 0); // All kernels finished training on Cluster A, then tune cluster A frequency to highest frequency
#endif
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] All kernels finished training on Cluster A at both " << TRAIN_MAX_A_FREQ << " and " << TRAIN_MED_A_FREQ << ". Now tune to the highest.\n";
                  LOCK_RELEASE(output_lck);
#endif                  
                }
              }
            }
          }else{ 
            // mtx.lock();
            _final = (++assembly->threads_out_tao == assembly->width);
            // mtx.unlock();
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Task " << assembly->taskid << "->_final = " << _final << ", assembly->get_timetable_state(clusterA) = " << assembly->get_timetable_state(0) << ", assembly->start_running_freq = " \
            << assembly->start_running_freq << ", cur_freq[" << nthread << "] = " << cur_freq[nthread] << "\n";
            LOCK_RELEASE(output_lck);
#endif        
          }    
        }

        /* (1) Was running on A57 (2) A57 PTT table hasn't finished training      (3) if the frequency changing is happening during the task execution, do not update the table */
        if(assembly->leader >= START_CLUSTER_B){
          if(assembly->get_timetable_state(1) == false && assembly->start_running_freq == cur_freq[nthread]){ 
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
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[" << avail_ddr_freq[0] << ", "  << TRAIN_MAX_B_FREQ << ", ClusterB, width = " \
              << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(0,1, width_index) << ". Current time: " << assembly->get_timetable(0, 0, 1, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
              //++assembly->threads_out_tao;
            }
          }else{ /*1.11GHz*/
            if(++assembly->threads_out_tao == assembly->width){ /* All threads finished the execution */
              _final = 1;
              float oldticks = assembly->get_timetable(0, TRAIN_MED_B_FREQ_idx, 1, width_index);
              if(assembly->width > 1){
                assembly->temp_ticks[nthread - assembly->leader] = ticks;
                float newticks = std::accumulate(std::begin(assembly->temp_ticks), std::begin(assembly->temp_ticks) + assembly->width, 0.0) / assembly->width;
                if(oldticks == 0.0f || (newticks < oldticks && fabs(newticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, TRAIN_MED_B_FREQ_idx, 1, newticks, width_index);
                }else{
                  assembly->set_timetable(0, TRAIN_MED_B_FREQ_idx, 1, (newticks+oldticks)/2, width_index);
                }
              }else{
                if(oldticks == 0.0f || (ticks < oldticks && fabs(ticks - oldticks)/oldticks > 0.1)){   /*Only update the PTT tables when (1) the entry hasn't been trained; (2) when new execution time is smaller*/
                  assembly->set_timetable(0, TRAIN_MED_B_FREQ_idx, 1, ticks, width_index);
                }else{
                  assembly->set_timetable(0, TRAIN_MED_B_FREQ_idx, 1, (ticks+oldticks)/2, width_index);
                }
              }
              assembly->increment_PTT_UpdateFinish(1, 1, width_index);
#ifdef DEBUG
              LOCK_ACQUIRE(output_lck);
              std::cout << "[DEBUG] _final = " << _final << ", task " << assembly->taskid << ", " << assembly->kernel_name << "->PTT_UpdateFinish[" << avail_ddr_freq[0] << ", " << TRAIN_MED_B_FREQ << ", ClusterB, width = " \
              << assembly->width << "] = " << assembly->get_PTT_UpdateFinish(1,1, width_index) << ". Current time: " << assembly->get_timetable(0, TRAIN_MED_B_FREQ_idx, 1, width_index) <<"\n";
              LOCK_RELEASE(output_lck);
#endif
            }else{ /* Has't finished the execution */
              //++assembly->threads_out_tao;
              assembly->temp_ticks[nthread - assembly->leader] = ticks;
            }
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
            if (ptt_freq_index[1] == 0) {             /* Current ClusterB frequency is 2.04GHz */
              int Sampling = 0;
              for(auto&& ii : ptt_layout[start_coreid[1]]){
                if (assembly->get_PTT_UpdateFinish(0, 1, ii-1) >= NUM_TRAIN_TASKS){
                  Sampling++;
                }
              }
              if(Sampling == ptt_layout[start_coreid[1]].size()){
              // if(assembly->get_PTT_UpdateFinish(0, 1, 0) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 1, 1) >= NUM_TRAIN_TASKS && assembly->get_PTT_UpdateFinish(0, 1, 3) >= NUM_TRAIN_TASKS){ /* First dimention: 2.04GHz, second dimention: Denver, third dimention: width_index */
                PTT_finish_state[0][1][assembly->tasktype] = 1; /* First dimention: 2.04GHz, second dimention: Denver, third dimention: tasktype */
                if(std::accumulate(std::begin(PTT_finish_state[0][1]), std::begin(PTT_finish_state[0][1])+num_kernels, 0) == num_kernels){ /* Check all kernels have finished the training of Denver at 2.04GHz */
                  for(int ii = START_CLUSTER_B; ii < gotao_nthreads; ii++){
                    std::ofstream ClusterB("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[ii]) + "/cpufreq/scaling_setspeed");
                    if (!ClusterB.is_open()){
                      std::cerr << "[DEBUG] Somthing failed while opening the file! " << std::endl;
                      return 0;
                    }
                    ClusterB << std::to_string(TRAIN_MED_B_FREQ) << std::endl;
                    ClusterB.close();
                    cur_freq[ii] = TRAIN_MED_B_FREQ;
                    cur_freq_index[ii] = TRAIN_MED_B_FREQ_idx;
                  }                  
                  ptt_freq_index[1] = 1;
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] ClusterB completed Sampling phase at highest frequency " << TRAIN_MAX_B_FREQ << ", now throttle the frequency to the medium " << TRAIN_MED_B_FREQ << "\n";
                  LOCK_RELEASE(output_lck);
#endif
                }
              }
            }else{
              int ptt_check = 0; /* Step 2: Check if PTT values of ClusterB are filled out */
              for(auto&& width : ptt_layout[START_CLUSTER_B]) { 
                float check_ticks = assembly->get_timetable(0, TRAIN_MED_B_FREQ_idx, 1, width - 1);
                if(check_ticks > 0.0f && assembly->get_PTT_UpdateFinish(1, 1, width-1) >=NUM_TRAIN_TASKS){
                  ptt_check++;
                  if(assembly->get_mbtable(1, width-1) == 0.0f){
                    float memory_boundness = 0.0f;
#if defined Performance_Model_Cycle                  /* Calculate Memory-boundness = (1 - cycle2/cycle1) / (1- f2/f1) */
                    uint64_t check_cycles = assembly->get_cycletable(1, 1, width - 1);
                    uint64_t cycles_high = assembly->get_cycletable(0, 1, width-1); /* First parameter is 0, means 2.04GHz, check_cycles is at 1.11Ghz */
                    float a = 1 - float(check_cycles) / float(cycles_high);
                    float b = 1 - float(avail_freq[1][TRAIN_MED_B_FREQ_idx]) / float(avail_freq[1][0]);
                    memory_boundness = a/b;
#endif
#if defined Performance_Model_Time
                    float highest_ticks = assembly->get_timetable(0, 0, 1, width - 1);
                    float a = float(avail_freq[1][0]) / float(avail_freq[1][TRAIN_MED_B_FREQ_idx]);
                    float b = check_ticks / highest_ticks;
                    memory_boundness = (b-a) / (1-a);
                    // LOCK_ACQUIRE(output_lck);
                    // std::cout << assembly->kernel_name << ": Memory-boundness Calculation (ClusterB, width " << width << ") = " << memory_boundness << ". a = " << a << ", b = " << b << std::endl;
                    // LOCK_RELEASE(output_lck);
#endif
                    if(memory_boundness > 1){
                      memory_boundness = 1;
                    }else{
                      if(memory_boundness <= 0){ /* Execution time and power prediction according to the computed memory-boundness level */
                        memory_boundness = 0.00001;
                      }
                    }
                    LOCK_ACQUIRE(output_lck);
                    std::cout << "Memory-boundness of " << assembly->kernel_name <<" tasks on <Cluster1, " << width << ">: " << memory_boundness << std::endl;
                    LOCK_RELEASE(output_lck);
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
                        if((ddr_freq_indx == 0 && freq_indx == 0) || (ddr_freq_indx == 0 && freq_indx == TRAIN_MED_B_FREQ_idx)){ /* if DDR freq = 1.866 and CPU freq = 2.04 or 1.11 GHz, then skip */
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
                        ptt_value_newfreq = highest_ticks * ((1-memory_boundness) * cpu_freq_scaling + perf_alpha[1][width][0] * memory_boundness + perf_alpha[1][width][1] * cpu_freq_scaling \
                        + perf_alpha[1][width][2] * ddr_freq_scaling + perf_alpha[1][width][3] * pow(memory_boundness, 2) + perf_alpha[1][width][4] * memory_boundness * cpu_freq_scaling \
                        + perf_alpha[1][width][5] * pow(cpu_freq_scaling, 2) + perf_alpha[1][width][6] * memory_boundness * ddr_freq_scaling + perf_alpha[1][width][7] * cpu_freq_scaling * ddr_freq_scaling \
                        + perf_alpha[1][width][8] * pow(ddr_freq_scaling, 2) + perf_alpha[1][width][9]);
#endif
                        assembly->set_timetable(ddr_freq_indx, freq_indx, 1, ptt_value_newfreq, width-1);
                        }
                      }
                    }
                    for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){ /* (2) Power value prediction */
                      for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ 
                        float cpupower = 0.0;
                        float ddrpower = 0.0;
                        float cpufreq = float(avail_freq[1][freq_indx])/1000000.0;
                        cpupower = CPUPower_alpha[1][width][0] * memory_boundness + CPUPower_alpha[1][width][1] * cpufreq + CPUPower_alpha[1][width][2] * memory_boundness * cpufreq \
                        + CPUPower_alpha[1][width][3] * pow(cpufreq, 2) + CPUPower_alpha[1][width][4];                     
                        assembly->set_cpupowertable(ddr_freq_indx, freq_indx, 1, cpupower, width-1); 
#ifdef DDR_FREQ_TUNING
                        float ddrfreq = float(avail_ddr_freq[ddr_freq_indx])/1000000000.0;
                        ddrpower = DDRPower_alpha[1][width][0] * memory_boundness + DDRPower_alpha[1][width][1] * cpufreq + DDRPower_alpha[1][width][2] * ddrfreq \
                        + DDRPower_alpha[1][width][3] * pow(memory_boundness, 2) + DDRPower_alpha[1][width][4] * memory_boundness * cpufreq + DDRPower_alpha[1][width][5] * pow(cpufreq, 2) \
                        + DDRPower_alpha[1][width][6] * memory_boundness * ddrfreq + DDRPower_alpha[1][width][7] * cpufreq * ddrfreq + DDRPower_alpha[1][width][8] * pow(ddrfreq, 2) \
                        + DDRPower_alpha[1][width][9];
                        assembly->set_ddrpowertable(ddr_freq_indx, freq_indx, 1, ddrpower, width-1); 
#endif
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
                    std::cout << "[DEBUG] Memory-boundness(ClusterB, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
                    // " << assembly->kernel_name << "->PTT_Value[Highest_DDR_Freq, Highest_CPU_Freq, ClusterB, " << width << "] = " << highest_ticks << ", PTT_Value[Highest_DDR_Freq, MED_CPU_Freq, ClusterB, " \
                    << width << "] = " << check_ticks << ". Memory-boundness(ClusterB, width=" << width << ") = " << memory_boundness << ". ptt_check = " << ptt_check << ".\n";
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
              std::cout << "[DEBUG] " << assembly->kernel_name << ": Cluster B completed PTT training at " << TRAIN_MAX_B_FREQ << " and " << TRAIN_MED_B_FREQ << "\n";
              LOCK_RELEASE(output_lck);
#endif
              if(std::accumulate(std::begin(PTT_finish_state[1][1]), std::begin(PTT_finish_state[1][1])+num_kernels, 0) == num_kernels){
#ifdef AAWS_CASE
                assembly->cpu_frequency_tuning(nthread, 1, start_coreid[1], end_coreid[1] - start_coreid[1], 6); // All kernels finished training on Cluster A, then tune cluster A frequency to 0.65GHz
#else
                assembly->cpu_frequency_tuning(nthread, 1, start_coreid[1], end_coreid[1] - start_coreid[1], 0); // All kernels finished training on Cluster A, then tune cluster A frequency to highest frequency
#endif
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] All kernels finished training on Cluster B at both " << TRAIN_MAX_A_FREQ << " and " << TRAIN_MED_A_FREQ << ". Now tune to the highest.\n";
                LOCK_RELEASE(output_lck);
#endif                  
              }
            }
          }
        // }
          }else{
            // mtx.lock();
            _final = (++assembly->threads_out_tao == assembly->width);
            // mtx.unlock();
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[DEBUG] Task " << assembly->taskid << "->_final = " << _final << ", assembly->get_timetable_state(ClusterB) = " << assembly->get_timetable_state(1) << ", assembly->start_running_freq = " \
            << assembly->start_running_freq << ", cur_freq[" << nthread << "] = " << cur_freq[nthread] << "\n";
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
                  std::cout << assembly->get_cpupowertable(ddr_freq_indx, freq_indx, clus_id, wid-1) << "\t";
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
#ifdef DDR_FREQ_TUNING
          std::cout << "\n---------- Memory Power Predictions ---------- \n";
          for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){
            std::cout << "Memory Frequency: " << avail_ddr_freq[ddr_freq_indx] << ": " << std::endl;
            for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
              std::cout << "Cluster " << clus_id << ": " << std::endl;
              for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
                std::cout << "CPU_Freq[" << avail_freq[clus_id][freq_indx] << "]: ";
                for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
                  std::cout << assembly->get_ddrpowertable(ddr_freq_indx, freq_indx, clus_id, wid-1) << "\t";
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
#endif
          std::cout << std::endl;
          LOCK_RELEASE(output_lck);
// #endif
        }
        if(std::accumulate(std::begin(global_training_state), std::begin(global_training_state) + num_kernels, 0) == num_kernels) {// Check if all kernels have finished the training phase.
          global_training = true;
          // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ 
          //   assembly->cpu_frequency_tuning(nthread, clus_id, start_coreid[clus_id], end_coreid[clus_id] - start_coreid[clus_id], 0); // After training phase, set the CPU frequency of all clusters back to the highest frequency
          // }
          std::chrono::time_point<std::chrono::system_clock> train_end;
          train_end = std::chrono::system_clock::now();
          auto train_end_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(train_end);
          auto train_end_epoch = train_end_ms.time_since_epoch();
          LOCK_ACQUIRE(output_lck);
          std::cout << "[Congratulations!] All the training Phase finished. Training finished time: " << train_end_epoch.count() << std::endl; 
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
// #ifdef AAWS_CASE // Replicate AAWS case to test "if lowest execution time leads to lowest EDP?": alpha is larger than beta, where on TX2 Denver frequency is 0.65GHz and A57 frequency is 1.11GHz
//       _final = (++assembly->threads_out_tao == assembly->width);
//       if(global_training == false){
//         int clus_id = (nthread < START_CLUSTER_B)? 0:1;
//         float oldticks = assembly->get_timetable(cur_ddr_freq_index, cur_freq_index[nthread], clus_id, width_index);
//         if(oldticks == 0.0){
//           assembly->set_timetable(cur_ddr_freq_index, cur_freq_index[nthread], clus_id, ticks,width_index);  
//         }
//         else{
//           assembly->set_timetable(cur_ddr_freq_index, cur_freq_index[nthread], clus_id, (4 * oldticks + ticks)/5.0, width_index);
//         }
//         int AAWS_Test_counter[XITAO_MAXTHREADS] = {0};
//         for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ 
//           for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
//             if(assembly->get_timetable(cur_ddr_freq_index, cur_freq_index[nthread], clus_id, wid-1) != 0){
//               AAWS_Test_counter[assembly->tasktype]++;
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[AAWS] Task " << assembly->taskid << ", cluster " << clus_id << ", wid " << wid << ", execution time: " << assembly->get_timetable(cur_ddr_freq_index, cur_freq_index[nthread], clus_id, wid-1) << std::endl;
//               LOCK_RELEASE(output_lck);
//             }
//           }
//         }
//         if(AAWS_Test_counter[assembly->tasktype] == 5){
//           global_training_state[assembly->tasktype] = 1;
//         }
//         if(std::accumulate(std::begin(global_training_state), std::begin(global_training_state) + num_kernels, 0) == num_kernels) {
//           global_training = true;
//           std::chrono::time_point<std::chrono::system_clock> train_end;
//           train_end = std::chrono::system_clock::now();
//           auto train_end_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(train_end);
//           auto train_end_epoch = train_end_ms.time_since_epoch();
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[Congratulations!] Sampling finished. Time: " << train_end_epoch.count() << std::endl; 
//           LOCK_RELEASE(output_lck);
//         }
//       }
// #endif
      st = nullptr;
      if(_final){ // the last exiting thread updates
        task_completions[nthread].tasks++;
        if(task_completions[nthread].tasks > 0){
          PolyTask::pending_tasks -= task_completions[nthread].tasks;
          task_completions[nthread].tasks = 0;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Thread " << nthread << " completed " << task_completions[nthread].tasks << " task " << assembly->taskid <<". Pending tasks = " << PolyTask::pending_tasks << std::endl;
          LOCK_RELEASE(output_lck);
#endif
          // DOP_detection[assembly->tasktype]--; // The total DOP tasks is decreased by 1
          // int clus_id = (nthread < START_CLUSTER_B)? 0:1;
          // DOP_executing[clus_id]--; // executing tasks in the cluster is decreased by 1
        }
      
#ifdef OVERHEAD_PTT
        //st = assembly->commit_and_wakeup(nthread, elapsed_ptt);
        assembly->commit_and_wakeup(nthread, elapsed_ptt);
#else
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
#ifdef JOSS_RWS  /* [JOSS - default] Only steal tasks from same cluster. [JOSS - EDP test: Energy Minimization per task + Random work stealing] */
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
#endif
#ifdef Target_EPTO  
            if(global_training == false){
              do{
                random_core = (rand_r(&seed) % gotao_nthreads);
              } while(random_core == nthread);
            }else{
            if(across_cluster_stealing == false)
            // if((global_training == false) || (global_training == true && across_cluster_stealing == false))
            {  // (1). During training phase, do not steal tasks from other clusters
              if(nthread < START_CLUSTER_B){ // (2). During low parallelism phase, do not steal tasks from other clusters
                do{
                  random_core = START_CLUSTER_A + (rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A));
                } while(random_core == nthread);
              }else{
                do{
                  random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
                }while(random_core == nthread); 
              }
            }else{ // During task scheduling phase, steal tasks from other clusters in HP region
             int j = 0; 
              if(nthread < START_CLUSTER_B){ 
                // for(int coreid = START_CLUSTER_A; coreid < START_CLUSTER_B; coreid++){
                //   j += worker_ready_q[coreid].size();
                // }
                if(idle_try < IDLE_SLEEP){ //|| j > 0 IDLE_SLEEP
                  do{
                    random_core = rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A);
                  } while(random_core == nthread);
                }else{ 
                  random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
                }
              }else{
                // for(int coreid = START_CLUSTER_B; coreid < gotao_nthreads; coreid++){
                //   j += worker_ready_q[coreid].size();
                // }
                if(idle_try < IDLE_SLEEP){ //  || j > 0 MM (1). A57 is idle, within idle sleep, they only steal from each other, otherwise, they steal from Denver
                // if(idle_try < IDLE_SLEEP || j > 0){ // MC, SLU (2). 2 * IDLE_SLEEP: since A57 have 2X cores than Denver, within 2 * idle sleep, they only steal from each other
                // if(j > 0){
                  do{  // (3). Or only allow A57 to steal from each other
                    random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
                  }while(random_core == nthread); 
                }else{// should I steal it? Since steal Denver tasks to A57 is not good for performance
                  random_core = rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A);
                }
              }
            }
          }
// #ifdef WITHIN_CLUSTER
//             if(nthread < START_CLUSTER_B){
//               do{
//                 random_core = rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A);
//               } while(random_core == nthread);
//             }else{
//               do{
//                 random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
//               }while(random_core == nthread); 
//             }
// #endif
// #ifdef ACROSS_CLUSTER_AFTER_IDLE
// // #ifdef DEBUG
// //             LOCK_ACQUIRE(output_lck);
// //             std::cout << "Thread" << nthread << " trying to steal" << std::endl;
// //             LOCK_RELEASE(output_lck);
// // #endif
//             int j = 0; 
//             if(nthread < START_CLUSTER_B){ 
//               for(int coreid = START_CLUSTER_A; coreid < START_CLUSTER_B; coreid++){
//                 j += worker_ready_q[coreid].size();
//               }
// // #ifdef DEBUG
// //             LOCK_ACQUIRE(output_lck);
// //             std::cout << "Thread" << nthread << " trying to steal, j = " << j << std::endl;
// //             LOCK_RELEASE(output_lck);
// // #endif
//               // if(j > 0){
//               if(idle_try < IDLE_SLEEP || j > 0){
//                 do{
//                   random_core = (rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A));
//                 } while(random_core == nthread);
//               }else{ 
//                 random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
//               }
//           	}else{
//               for(int coreid = START_CLUSTER_B; coreid < gotao_nthreads; coreid++){
//                 j += worker_ready_q[coreid].size();
//               }
// // #ifdef DEBUG
// //             LOCK_ACQUIRE(output_lck);
// //             std::cout << "Thread" << nthread << " trying to steal, j = " << j << std::endl;
// //             LOCK_RELEASE(output_lck);
// // #endif
//               if(idle_try <  2 * IDLE_SLEEP || j > 0){ // (1). A57 is idle, within idle sleep, they only steal from each other, otherwise, they steal from Denver
//               // if(idle_try < 2 * IDLE_SLEEP){ // (2). 2 * IDLE_SLEEP: since A57 have 2X cores than Denver, within 2 * idle sleep, they only steal from each other
//               // if(j > 0){
//                 do{  // (3). Or only allow A57 to steal from each other
//                   random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
//                 }while(random_core == nthread); 
//               }else{// should I steal it? Since steal Denver tasks to A57 is not good for performance
//                 random_core = (rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A));
//               }
//           	}
// #endif
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
	        do{
           random_core = (rand_r(&seed) % gotao_nthreads);
          } while(random_core == nthread);
          // if(nthread < START_CLUSTER_B){
          //   do{
          //     random_core = (rand_r(&seed) % (START_CLUSTER_B - START_CLUSTER_A));
          //   } while(random_core == nthread);
          // }
          // else{
          //   // break;
          //   do{
          //     random_core = START_CLUSTER_B + (rand_r(&seed) % (gotao_nthreads - START_CLUSTER_B));
          //   }while(random_core == nthread); 
          // }
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
            }else{
              worker_ready_q[random_core].pop_back();
// #if (defined Target_EPTO) && (defined ACROSS_CLUSTER_AFTER_IDLE) // Paper 4: for high parallelism region, it is better to steal from other cluster to ease the load imbalance
              if(global_training == true && across_cluster_stealing == true){
                int clus_thief = (nthread < START_CLUSTER_B) ? 0 : 1;
                int clus_victim = (random_core < START_CLUSTER_B) ? 0 : 1;
                if(HP_best_width[st->tasktype][clus_thief] != 0){
                  st->width = HP_best_width[st->tasktype][clus_thief];
                }
                if(clus_thief != clus_victim){ // steal from other cluster
                  tao_total_across_steals++; 
#ifdef DEBUG
          	      LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] Work stealing from cluster " << clus_victim << " to cluster " << clus_thief << std::endl;
                  LOCK_RELEASE(output_lck);          
#endif
                }
              }
// #endif
              // st->leader = start_coreid[clus] + (rand() % ((end_coreid[clus]-start_coreid[clus])/st->width)) * st->width;
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
#ifdef TX2
              if(st->width == 4){
                st->leader = 2 + (nthread-2) / st->width;
              }
              if(st->width <= 2){
                st->leader = nthread /st->width * st->width;
              }
#endif         
              tao_total_steals++;  
            }
          }else{
            if((Sched == 0) || (Sched == 3)){
              if((st->width >= abs(random_core-nthread)+1)) {
                st = NULL;
                LOCK_RELEASE(worker_lock[random_core]);
                continue;
              }else{
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
              }
            }else{ // Wrong scheduler id
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
    
#if (defined SLEEP) 
      if(idle_try >= IDLE_SLEEP){
        long int limit = (SLEEP_LOWERBOUND * pow(2,idle_times) < SLEEP_UPPERBOUND) ? SLEEP_LOWERBOUND * pow(2,idle_times) : SLEEP_UPPERBOUND;  
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "idle_try = " << idle_try << ", Thread " << nthread << " sleep for " << limit/1000000 << " ms...\n";
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
#endif
      LOCK_RELEASE(output_lck);
#endif
#ifdef EXECTIME
      LOCK_ACQUIRE(output_lck);
      // std::cout << "The total execution time of thread " << nthread << " is " << elapsed_exe.count() << " s.\n";
      std::cout << "The total execution time of thread " << nthread << " is " << exe_time[nthread] << " s.\n";
      LOCK_RELEASE(output_lck);
#endif

#if defined SWEEP_Overhead
  if(nthread == 0){
    LOCK_ACQUIRE(output_lck);
    std::cout << "The timing overhead of running SWEEP scheduler: " <<  elapsed_overhead.count() << "\n";
    LOCK_RELEASE(output_lck);
  }
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
