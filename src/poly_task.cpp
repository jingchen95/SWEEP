#include "poly_task.h"
// #include "tao.h"
#include <errno.h> 
#include <cstring>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iterator>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include "xitao_workspace.h"
using namespace xitao;

extern int ptt_freq_index[NUMSOCKETS];
extern bool global_training;
extern bool across_cluster_stealing;
extern int num_kernels;


#if (defined DynaDVFS)
// extern int freq_dec;
// extern int freq_inc;
extern int env;
extern int current_freq;
// extern int dynamic_time_change;
#endif

#if defined(Haswell)
extern int num_sockets;
#endif

#ifdef OVERHEAD_PTT
extern std::chrono::duration<double> elapsed_ptt;
#endif

#ifdef NUMTASKS_MIX
extern int num_task[XITAO_MAXTHREADS][XITAO_MAXTHREADS * XITAO_MAXTHREADS];
#endif

extern int start_coreid[NUMSOCKETS];
extern int end_coreid[NUMSOCKETS];
extern float percent_cluster[XITAO_MAXTHREADS][NUMSOCKETS]; // the percentage of tasks allocated to each cluster
extern int DOP_detection[XITAO_MAXTHREADS];
// extern int DOP_executing[NUMSOCKETS];
extern int remaining_distri_num[XITAO_MAXTHREADS][XITAO_MAXTHREADS][NUMSOCKETS]; // first parameter is task type, second parameter is thread id, third parameter is cluster id
extern int LP_best_width[XITAO_MAXTHREADS][NUMSOCKETS]; // Low task parallelism: first parameter is task type, second parameter is cluster id
extern int HP_best_width[XITAO_MAXTHREADS][NUMSOCKETS]; // High task parallelism
extern bool interval_HP[XITAO_MAXTHREADS][XITAO_MAXTHREADS]; // first parameter is task type, second parameter is thread id
extern bool interval_LP[XITAO_MAXTHREADS][XITAO_MAXTHREADS]; // first parameter is task type, second parameter is thread id
extern int round_robin_counter[XITAO_MAXTHREADS][XITAO_MAXTHREADS];
extern bool interval_distri_state;
extern bool history_LP_visited[XITAO_MAXTHREADS][10];
extern int history_LP_numTask[XITAO_MAXTHREADS][NUMSOCKETS][10];
extern int history_LP_bestwid[XITAO_MAXTHREADS][NUMSOCKETS][10];
extern int HP_cpu_freq[XITAO_MAXTHREADS][NUMSOCKETS]; // first parameter is task type, second parameter is cluster id
extern int HP_ddr_freq[XITAO_MAXTHREADS]; // parameter is task type
extern int history_LP_cpu_freq[XITAO_MAXTHREADS][NUMSOCKETS][10]; // first parameter is task type, second parameter is cluster id, third parameter is DOP index
extern int LP_cpu_freq[XITAO_MAXTHREADS][NUMSOCKETS]; // first parameter is task type, second parameter is cluster id
extern int history_LP_ddr_freq[XITAO_MAXTHREADS][10];
extern int LP_ddr_freq[XITAO_MAXTHREADS];

#if (defined DVFS) && (defined TX2)
extern float compute_bound_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS];
extern float memory_bound_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS];
extern float cache_intensive_power[NUMSOCKETS][FREQLEVELS][XITAO_MAXTHREADS];
#elif (defined DynaDVFS) // Currently only consider 4 combinations: max&max, max&min, min&max, min&min
extern float compute_bound_power[4][NUMSOCKETS][XITAO_MAXTHREADS];
extern float memory_bound_power[4][NUMSOCKETS][XITAO_MAXTHREADS];
extern float cache_intensive_power[4][NUMSOCKETS][XITAO_MAXTHREADS];
#elif (defined ERASE)
extern float compute_bound_power[NUMSOCKETS][XITAO_MAXTHREADS];
extern float memory_bound_power[NUMSOCKETS][XITAO_MAXTHREADS];
extern float cache_intensive_power[NUMSOCKETS][XITAO_MAXTHREADS];
#endif

extern float idle_cpu_power[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS];
#ifdef DDR_FREQ_TUNING
extern float idle_ddr_power[NUM_DDR_AVAIL_FREQ];
#endif
extern int status[XITAO_MAXTHREADS];
extern int status_working[XITAO_MAXTHREADS];
extern int Parallelism;
extern int Sched;
extern int TABLEWIDTH;

extern std::chrono::time_point<std::chrono::system_clock> interval_t1, interval_t2;

#ifdef SWEEP_Overhead
extern std::chrono::duration<double> elapsed_overhead;
#endif

// Task parallelism counter
std::atomic<int> PolyTask::task_counter;

//The pending PolyTasks count 
std::atomic<int> PolyTask::pending_tasks;

// need to declare the static class memeber
// #if defined(DEBUG)
std::atomic<int> PolyTask::created_tasks;
// #endif

PolyTask::PolyTask(int t, int _nthread=0) : type(t){
  refcount = 0;
#define GOTAO_NO_AFFINITY (1.0)
  affinity_relative_index = GOTAO_NO_AFFINITY;
  affinity_queue = -1;
// #if defined(DEBUG) 
  taskid = created_tasks += 1;
// #endif
  LOCK_ACQUIRE(worker_lock[_nthread]);
  if(task_pool[_nthread].tasks == 0){
    pending_tasks += TASK_POOL;
    task_pool[_nthread].tasks = TASK_POOL-1;
#ifdef DEBUG
    std::cout << "[DEBUG] _nthread = " << _nthread << ". Requested: " << TASK_POOL << " tasks. Pending is now: " << pending_tasks << "\n";
#endif
  }
  else {
    task_pool[_nthread].tasks--;
// #ifdef DEBUG
//     std::cout << "[Jing] task_pool[" << _nthread << "]--: " << task_pool[_nthread].tasks << "\n";
// #endif
  }
  LOCK_RELEASE(worker_lock[_nthread]);
/*	LOCK_ACQUIRE(worker_lock[0]);
  if(task_pool[0].tasks == 0){
    pending_tasks += TASK_POOL;
    task_pool[0].tasks = TASK_POOL-1;
#ifdef DEBUG
    std::cout << "[DEBUG] Requested: " << TASK_POOL << " tasks. Pending is now: " << pending_tasks << "\n";
#endif
  }
  else {
    task_pool[0].tasks--;
#ifdef DEBUG
    std::cout << "[Jing] task_pool[0]--: " << task_pool[0].tasks << "\n";
#endif
  }
  LOCK_RELEASE(worker_lock[0]);*/
  threads_out_tao = 0;
  criticality=0;
  marker = 0;
}

// Internally, GOTAO works only with queues, not stas
int PolyTask::sta_to_queue(float x){
  if(x >= GOTAO_NO_AFFINITY){ 
    affinity_queue = -1;
  }
    else if (x < 0.0) return 1;  // error, should it be reported?
    else affinity_queue = (int) (x*gotao_nthreads);
    return 0; 
  }
int PolyTask::set_sta(float x){    
  affinity_relative_index = x;  // whenever a sta is changed, it triggers a translation
  return sta_to_queue(x);
} 
float PolyTask::get_sta(){             // return sta value
  return affinity_relative_index; 
}    
int PolyTask::clone_sta(PolyTask *pt) { 
  affinity_relative_index = pt->affinity_relative_index;    
  affinity_queue = pt->affinity_queue; // make sure to copy the exact queue
  return 0;
}
void PolyTask::make_edge(PolyTask *t){
  out.push_back(t);
  t->refcount++;
}

enum { NS_PER_SECOND = 1000000000 };
void poly_sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
{
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0)
    {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0)
    {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}

//History-based molding
//#if defined(CRIT_PERF_SCHED)
int PolyTask::history_mold(int _nthread, PolyTask *it){
  int new_width = 1; 
  int new_leader = -1;
  float shortest_exec = 1000.0f;
  float comp_perf = 0.0f; 
  auto&& partitions = inclusive_partitions[_nthread];
// #ifndef ERASE_target_perf
//   if(rand()%10 != 0) {
// #endif 
    for(auto&& elem : partitions) {
      int leader = elem.first;
      int width  = elem.second;
      auto&& ptt_val = 0.0f;
#ifdef DVFS
#else
      ptt_val = it->get_timetable(0, 0, leader, width - 1);
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout <<"[DEBUG] Priority=0, ptt value("<< leader << "," << width << ") = " << ptt_val << std::endl;
      LOCK_RELEASE(output_lck);
#endif
#endif
      if(ptt_val == 0.0f) {
        new_width = width;
        new_leader = leader;       
        break;
      }
#ifdef CRI_COST
      //For Cost
      comp_perf = width * ptt_val;
#endif
#if (defined CRI_PERF) || (defined ERASE_target_perf) 
      //For Performance
      comp_perf = ptt_val;
#endif
      if (comp_perf < shortest_exec) {
        shortest_exec = comp_perf;
        new_width = width;
        new_leader = leader;      
      }
    }
// #ifndef ERASE_target_perf
//   } else { 
//     auto&& rand_partition = partitions[rand() % partitions.size()];
//     new_leader = rand_partition.first;
//     new_width  = rand_partition.second;
//   }
// #endif
  if(new_leader != -1) {
    it->width  = new_width;
    it->leader = new_leader;
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout <<"[DEBUG] History Mold, task "<< it->taskid << " leader: " << it->leader << ", width: " << it->width << std::endl;
    LOCK_RELEASE(output_lck);
#endif
  }
  return it->leader;
} 
  //Recursive function assigning criticality
int PolyTask::set_criticality(){
  if((criticality)==0){
    int max=0;
    for(std::list<PolyTask *>::iterator edges = out.begin();edges != out.end();++edges){
      int new_max =((*edges)->set_criticality());
      max = ((new_max>max) ? (new_max) : (max));
    }
    criticality=++max;
  } 
  return criticality;
}

int PolyTask::set_marker(int i){
  for(std::list<PolyTask *>::iterator edges = out.begin(); edges != out.end(); ++edges){
    if((*edges) -> criticality == critical_path - i){
      (*edges) -> marker = 1;
      i++;
      (*edges) -> set_marker(i);
      break;
    }
  }
  return marker;
}

//Determine if task is critical task
int PolyTask::if_prio(int _nthread, PolyTask * it){
#ifdef EAS_NoCriticality
	if((Sched == 1) || (Sched == 2)){
	  return 0;
  }
	if(Sched == 0){
#endif
    return it->criticality;
#ifdef EAS_NoCriticality
  }
#endif
}

// #ifdef DVFS
// void PolyTask::print_ptt(float table[][XITAO_MAXTHREADS][XITAO_MAXTHREADS], const char* table_name) { 
// #else
void PolyTask::print_ptt(float table[][XITAO_MAXTHREADS], const char* table_name) { 
// #endif
  std::cout << std::endl << table_name <<  " PTT Stats: " << std::endl;
  for(int leader = 0; leader < ptt_layout.size() && leader < gotao_nthreads; ++leader) {
    auto row = ptt_layout[leader];
    std::sort(row.begin(), row.end());
    std::ostream time_output (std::cout.rdbuf());
    std::ostream scalability_output (std::cout.rdbuf());
    std::ostream width_output (std::cout.rdbuf());
    std::ostream empty_output (std::cout.rdbuf());
    time_output  << std::scientific << std::setprecision(3);
    scalability_output << std::setprecision(3);    
    empty_output << std::left << std::setw(5);
    std::cout << "Thread = " << leader << std::endl;    
    std::cout << "==================================" << std::endl;
    std::cout << " | " << std::setw(5) << "Width" << " | " << std::setw(9) << std::left << "Time" << " | " << "Scalability" << std::endl;
    std::cout << "==================================" << std::endl;
    for (int i = 0; i < row.size(); ++i) {
      int curr_width = row[i];
      if(curr_width <= 0) continue;
      auto comp_perf = table[curr_width - 1][leader];
      std::cout << " | ";
      width_output << std::left << std::setw(5) << curr_width;
      std::cout << " | ";      
      time_output << comp_perf; 
      std::cout << " | ";
      if(i == 0) {        
        empty_output << " - ";
      } else if(comp_perf != 0.0f) {
        auto scaling = table[row[0] - 1][leader]/comp_perf;
        auto efficiency = scaling / curr_width;
        if(efficiency  < 0.6 || efficiency > 1.3) {
          scalability_output << "(" <<table[row[0] - 1][leader]/comp_perf << ")";  
        } else {
          scalability_output << table[row[0] - 1][leader]/comp_perf;
        }
      }
      std::cout << std::endl;  
    }
    std::cout << std::endl;
  }
}  

void PolyTask::cpu_frequency_tuning(int nthread, int best_core_type, int leader, int width, int freq_index){
  uint64_t best_cpu_freq = avail_freq[best_core_type][freq_index];
  for(int i = leader; i< leader + width; i++){
    std::ofstream Throttle("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[i]) + "/cpufreq/scaling_setspeed");
    if (!Throttle.is_open()){
      std::cerr << "[DEBUG] failed while opening the scaling_setspeed file! " << std::endl;
      return;
    }
    Throttle << std::to_string(best_cpu_freq) << std::endl;
    cur_freq[i] = best_cpu_freq; /* Update the current CPU frequency */
    cur_freq_index[i] = freq_index; /* Update the current CPU frequency index */
    Throttle.close();
  }
  // std::ofstream ClusterB("/sys/devices/system/cpu/cpu" + std::to_string(static_resource_mapper[START_CLUSTER_B]) + "/cpufreq/scaling_setspeed");
  // if(best_cluster == 0){ //Denver
  //   ClusterA << std::to_string(best_cpu_freq) << std::endl;
  //   /* If the other cluster is totally idle, here it should set the frequency of the other cluster to the same */
  //   int cluster_active = std::accumulate(status_working + start_coreid[1], status_working + end_coreid[1], 0);   
  //   if(cluster_active == 0 && cur_freq_index[1] > cur_freq_index[0]){ /* No working cores on A57 cluster and the current frequency of A57 is higher than working Denver, then tune the frequency */
  //     ClusterB << std::to_string(best_cpu_freq) << std::endl;
  //     cur_freq[1] = best_cpu_freq; /* Update the current frequency */
  //     cur_freq_index[1] = freq_index;
  //   }  
  // }else{
  //   ClusterB << std::to_string(best_cpu_freq) << std::endl;
  //   /* If the other cluster is totally idle, here it should set the frequency of the other cluster to the same */
  //   int cluster_active = std::accumulate(status_working + start_coreid[0], status_working + end_coreid[0], 0);   
  //   if(cluster_active == 0 && cur_freq_index[0] > cur_freq_index[1]){ /* No working cores on Denver cluster and the current frequency of Denver is higher than working A57, then tune the frequency */
  //     ClusterA << std::to_string(best_cpu_freq) << std::endl;
  //     cur_freq[0] = best_cpu_freq; /* Update the current frequency */
  //     cur_freq_index[0] = freq_index;
  //   }  
  // }
  // ClusterB.close();
}

/* Tune the Memory Frequency (shared resources of whole chip) */
void PolyTask::ddr_frequency_tuning(int nthread, int ddr_freq_index){
  uint64_t best_ddr_freq = avail_ddr_freq[ddr_freq_index];
  std::ofstream EMC("/sys/kernel/debug/bpmp/debug/clk/emc/rate"); // edit chip memory frequency - TX2 specific
  if (!EMC.is_open()){
    std::cerr << "[DEBUG] failed while opening the DDR setspeed file! " << std::endl;
    return;
  }
  EMC << std::to_string(best_ddr_freq) << std::endl;
  cur_ddr_freq_index = ddr_freq_index; /* Update the current Memory frequency index */
  cur_ddr_freq = best_ddr_freq; /* Update the current Memory frequency */
  EMC.close();
}

int PolyTask::optimized_search(int nthread, PolyTask * it){
  // float Energy_L_B[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0f};
  // float Energy_L_T[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0f};
  // float Energy_R_B[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0f};
  // float Energy_R_T[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0f};
#if defined Search_Overhead    
  struct timespec start1, finish1, delta1;
  clock_gettime(CLOCK_REALTIME, &start1);
#endif
  int best_cluster;
  int x[8] = {-1, -1, -1, 0, 0, 1, 1, 1}; // x and y arrays are meant to define 8 directions
  int y[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  float Energy[NUMSOCKETS][XITAO_MAXTHREADS][NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0.0f};
  float idleP_cluster[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0.0f};
  float idleP[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0.0f};
  int execution_place_winner[NUMSOCKETS][XITAO_MAXTHREADS] = {0};
  float shortest_exec_L_B = 100000.0f;
  float shortest_exec_L_T = 100000.0f;
  float shortest_exec_R_B = 100000.0f;
  float shortest_exec_R_T = 100000.0f;
  int previous_winner_cluster = 0; 
  int previous_winner_width = 1;
  float idleP_cluster_fine_grain = 0.0f;
  int sum_cluster_active[NUMSOCKETS] = {0};
  // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ /* Get the number of active cores in each cluster */
  //   sum_cluster_active[clus_id] = std::accumulate(status+start_coreid[clus_id], status+end_coreid[clus_id], 0); 
// #ifdef DEBUG
//     LOCK_ACQUIRE(output_lck);
//     std::cout << "[DEBUG] Number of active cores in cluster " << clus_id << ": " << sum_cluster_active[clus_id] << ". status[0] = " << status[0] \
//     << ", status[1] = " << status[1] << ", status[2] = " << status[2] << ", status[3] = " << status[3] << ", status[4] = " << status[4] << ", status[5] = " << status[5] << std::endl;
//     LOCK_RELEASE(output_lck);
// #endif 
  // }
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[Fine-Test] " << it->get_timetable(cur_ddr_freq_index, cur_freq_index[0], 0, 1) << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif  
  /* Step 1: first standard: check if the execution time of (current frequency of Denver, Denver, 2) < 1 ms? --- define as Fine-grained tasks --- Find out best core type, number of cores, doesn't change frequency */
  if(it->get_timetable(cur_ddr_freq_index, cur_freq_index[0], 0, 1) < FINE_GRAIN_THRESHOLD){
    float shortest_exec = 100000.0f;
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
      sum_cluster_active[clus_id] = std::accumulate(status+start_coreid[clus_id], status+end_coreid[clus_id], 0); /* Get the number of active cores in each cluster */
      if(sum_cluster_active[1-clus_id] == 0){ /* the number of active cores is zero in another cluster */
        idleP_cluster_fine_grain = idle_cpu_power[cur_ddr_freq_index][cur_freq_index[clus_id]][clus_id] + idle_cpu_power[cur_ddr_freq_index][cur_freq_index[1-clus_id]][1-clus_id]; /* Then equals idle power of whole chip */
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Cluster " << 1-clus_id << " no active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of whole chip " \
        << idleP_cluster_fine_grain << std::endl;
        LOCK_RELEASE(output_lck);
#endif 
      }else{
        idleP_cluster_fine_grain = idle_cpu_power[cur_ddr_freq_index][cur_freq_index[clus_id]][clus_id]; /* otherwise, equals idle power of the cluster */
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Cluster " << 1-clus_id << " has active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
        LOCK_RELEASE(output_lck);
#endif 
      }
      for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
        sum_cluster_active[clus_id] = (sum_cluster_active[clus_id] < wid)? wid : sum_cluster_active[clus_id];
        float idleP = idleP_cluster_fine_grain * wid / sum_cluster_active[clus_id];
        float CPUPowerP = it->get_cpupowertable(cur_ddr_freq_index, cur_freq_index[clus_id], clus_id, wid-1);
        float DDRPowerP = it->get_ddrpowertable(cur_ddr_freq_index, cur_freq_index[clus_id], clus_id, wid-1);
        float timeP = it->get_timetable(cur_ddr_freq_index, cur_freq_index[clus_id], clus_id, wid-1);
        float energy_pred = timeP * (CPUPowerP - idleP_cluster_fine_grain + idleP + DDRPowerP);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Frequency: " << cur_freq[clus_id] << ", cluster " << clus_id << ", width "<< wid << ", sum_cluster_active = " << sum_cluster_active[clus_id] \
          << ", CPU power " << CPUPowerP - idleP_cluster_fine_grain + idleP << ", memory power " << DDRPowerP << ", execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
          LOCK_RELEASE(output_lck);
#endif 
          if(energy_pred < shortest_exec){
            shortest_exec = energy_pred;
            it->set_best_core_type(clus_id);
            it->set_best_numcores(wid);
          }
        }
      }
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] For current frequency: " << cur_freq[it->get_best_cluster()] << ", best cluster: " << it->get_best_cluster() << ", best width: " << it->get_best_numcores() << std::endl;
      LOCK_RELEASE(output_lck);
#endif 
      // if(it->get_timetable(cur_freq_index[best_cluster], best_cluster, best_width) < FINE_GRAIN_THRESHOLD){ /* Double confirm that the task using the best config is still fine-grained */
      it->set_enable_cpu_freq_change(false); /* No CPU frequency scaling */
      it->set_enable_ddr_freq_change(false); /* No memory frequency scaling */
      it->granularity_fine = true; /* Mark it as fine-grained task */
  }else{ /* Coarse-grained tasks --- Find out best CPU and Memory frequency, core type, number of cores */
    int i = 0;
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ // for each cluster
      sum_cluster_active[clus_id] = std::accumulate(status+start_coreid[clus_id], status+end_coreid[clus_id], 0); /* Get the number of active cores in each cluster */
      if(sum_cluster_active[1-clus_id] == 0){ /* the number of active cores is zero in another cluster */
        idleP_cluster[0][0] = idle_cpu_power[0][0][clus_id] + idle_cpu_power[0][0][1-clus_id]; /* Then equals idle power of whole chip */
        idleP_cluster[NUM_DDR_AVAIL_FREQ-1][0] = idle_cpu_power[NUM_DDR_AVAIL_FREQ-1][0][clus_id] + idle_cpu_power[NUM_DDR_AVAIL_FREQ-1][0][1-clus_id];
        idleP_cluster[0][NUM_AVAIL_FREQ-1] = idle_cpu_power[0][NUM_AVAIL_FREQ-1][clus_id] + idle_cpu_power[0][NUM_AVAIL_FREQ-1][1-clus_id];
        idleP_cluster[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] = idle_cpu_power[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1][clus_id] + idle_cpu_power[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1][1-clus_id];
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[DEBUG] Cluster " << 1-clus_id << " no active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif 
      }else{
        idleP_cluster[0][0] = idle_cpu_power[0][0][clus_id]; /* otherwise, equals idle power of the cluster */
        idleP_cluster[NUM_DDR_AVAIL_FREQ-1][0] = idle_cpu_power[NUM_DDR_AVAIL_FREQ-1][0][clus_id];
        idleP_cluster[0][NUM_AVAIL_FREQ-1] = idle_cpu_power[0][NUM_AVAIL_FREQ-1][clus_id];
        idleP_cluster[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] = idle_cpu_power[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1][clus_id];
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] Cluster " << 1-clus_id << " has active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif 
      }
      for(auto&& wid : ptt_layout[start_coreid[clus_id]]){ // for each width in the cluster 
        /* CPU power prediction */
        float CPUPower_L_B = it->get_cpupowertable(0, 0, clus_id, wid-1); // Left Bottom: highest CPU frequency, highest DDR frequency
        float CPUPower_L_T = it->get_cpupowertable(NUM_DDR_AVAIL_FREQ-1, 0, clus_id, wid-1); // Left Top: highest CPU frequency, lowest DDR frequency
        float CPUPower_R_B = it->get_cpupowertable(0, NUM_AVAIL_FREQ-1, clus_id, wid-1); // Right Bottom: highest CPU frequency, lowest DDR frequency
        float CPUPower_R_T = it->get_cpupowertable(NUM_DDR_AVAIL_FREQ-1, NUM_AVAIL_FREQ-1, clus_id, wid-1); // Right Top: lowest CPU frequency, lowest DDR frequency
        /* Memory power prediction */
        float DDRPower_L_B = it->get_ddrpowertable(0, 0, clus_id, wid-1); // Left Bottom: highest CPU frequency, highest DDR frequency
        float DDRPower_L_T = it->get_ddrpowertable(NUM_DDR_AVAIL_FREQ-1, 0, clus_id, wid-1); // Left Top: highest CPU frequency, lowest DDR frequency
        float DDRPower_R_B = it->get_ddrpowertable(0, NUM_AVAIL_FREQ-1, clus_id, wid-1); // Right Bottom: highest CPU frequency, lowest DDR frequency
        float DDRPower_R_T = it->get_ddrpowertable(NUM_DDR_AVAIL_FREQ-1, NUM_AVAIL_FREQ-1, clus_id, wid-1); // Right Top: lowest CPU frequency, lowest DDR frequency
        /* Execution time prediction */
        float ExecTime_L_B = it->get_timetable(0, 0, clus_id, wid-1); // Left Bottom: highest CPU frequency, highest DDR frequency
        float ExecTime_L_T = it->get_timetable(NUM_DDR_AVAIL_FREQ-1, 0, clus_id, wid-1); // Left Top: highest CPU frequency, lowest DDR frequency
        float ExecTime_R_B = it->get_timetable(0, NUM_AVAIL_FREQ-1, clus_id, wid-1); // Right Bottom: highest CPU frequency, lowest DDR frequency
        float ExecTime_R_T = it->get_timetable(NUM_DDR_AVAIL_FREQ-1, NUM_AVAIL_FREQ-1, clus_id, wid-1); // Right Top: lowest CPU frequency, lowest DDR frequency
        /* Idle power computation */
        sum_cluster_active[clus_id] = (sum_cluster_active[clus_id] < wid)? wid : sum_cluster_active[clus_id];
        idleP[0][0] = idleP_cluster[0][0] * wid / sum_cluster_active[clus_id];
        idleP[NUM_DDR_AVAIL_FREQ-1][0] = idleP_cluster[NUM_DDR_AVAIL_FREQ-1][0] * wid / sum_cluster_active[clus_id];
        idleP[0][NUM_AVAIL_FREQ-1] = idleP_cluster[0][NUM_AVAIL_FREQ-1] * wid / sum_cluster_active[clus_id];
        idleP[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] = idleP_cluster[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] * wid / sum_cluster_active[clus_id];
        /* Energy prediction */
        Energy[clus_id][wid-1][0][0] = ExecTime_L_B * (CPUPower_L_B - idleP_cluster[0][0] + idleP[0][0] + DDRPower_L_B);
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[Optimized_Search] execution place: <" << clus_id << ", " << wid << "> --- Left-bottom corner: CPU power=" << CPUPower_L_B << ", memory power=" << DDRPower_L_B << ", execution time=" << ExecTime_L_B << ", idle power=" << idleP[0][0] << ". Energy = " << Energy[0][0][clus_id][wid-1] << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif
        Energy[clus_id][wid-1][NUM_DDR_AVAIL_FREQ-1][0] = ExecTime_L_T * (CPUPower_L_T - idleP_cluster[NUM_DDR_AVAIL_FREQ-1][0] + idleP[NUM_DDR_AVAIL_FREQ-1][0] + DDRPower_L_T);
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[Optimized_Search] execution place: <" << clus_id << ", " << wid << "> --- Left-top corner: CPU power=" << CPUPower_L_T << ", memory power=" << DDRPower_L_T << ", execution time=" << ExecTime_L_T << ", idle power=" << idleP[NUM_DDR_AVAIL_FREQ-1][0] << ". Energy = " << Energy[NUM_DDR_AVAIL_FREQ-1][0][clus_id][wid-1] << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif
        Energy[clus_id][wid-1][0][NUM_AVAIL_FREQ-1] = ExecTime_R_B * (CPUPower_R_B - idleP_cluster[0][NUM_AVAIL_FREQ-1] + idleP[0][NUM_AVAIL_FREQ-1] + DDRPower_R_B);
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[Optimized_Search] execution place: <" << clus_id << ", " << wid << "> --- Right-bottom corner: CPU power=" << CPUPower_R_B << ", memory power=" << DDRPower_R_B << ", execution time=" << ExecTime_R_B << ", idle power=" << idleP[0][NUM_AVAIL_FREQ-1] << ". Energy = " << Energy[0][NUM_AVAIL_FREQ-1][clus_id][wid-1] << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif
        Energy[clus_id][wid-1][NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] = ExecTime_R_T * (CPUPower_R_T - idleP_cluster[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] + idleP[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] + DDRPower_R_T);
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[Optimized_Search] execution place: <" << clus_id << ", " << wid << "> --- Right-top corner: CPU power=" << CPUPower_R_T << ", memory power=" << DDRPower_R_T << ", execution time=" << ExecTime_R_T << ", idle power=" << idleP[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] << ". Energy = " << Energy[NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1][clus_id][wid-1] << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif
        if(i == 0){ // starting point as the initialization
          shortest_exec_L_B = Energy[clus_id][wid-1][0][0];
          shortest_exec_L_T = Energy[clus_id][wid-1][NUM_DDR_AVAIL_FREQ-1][0];
          shortest_exec_R_B = Energy[clus_id][wid-1][0][NUM_AVAIL_FREQ-1];
          shortest_exec_R_T = Energy[clus_id][wid-1][NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1];
          execution_place_winner[clus_id][wid-1] = 4; // 4 corners
          previous_winner_cluster = clus_id;
          previous_winner_width = wid;
        }else{
        /* Compare the four corner values, lead to the winner execution places */
        if(Energy[clus_id][wid-1][0][0] < shortest_exec_L_B){
          if(previous_winner_cluster != clus_id || previous_winner_width != wid){
            execution_place_winner[previous_winner_cluster][previous_winner_width-1]--;
            previous_winner_cluster = clus_id;
            previous_winner_width = wid;
          }
          shortest_exec_L_B = Energy[clus_id][wid-1][0][0];
          execution_place_winner[clus_id][wid-1]++;
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[Optimized_Search] Left bottom: execution_place_winner: <" << clus_id << ", " << wid << ">,  value = " << execution_place_winner[clus_id][wid-1] << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif
        }
        if(Energy[clus_id][wid-1][NUM_DDR_AVAIL_FREQ-1][0] < shortest_exec_L_T){
          if(previous_winner_cluster != clus_id || previous_winner_width != wid){
            execution_place_winner[previous_winner_cluster][previous_winner_width-1]--;
            previous_winner_cluster = clus_id;
            previous_winner_width = wid;
          }
          shortest_exec_L_T = Energy[clus_id][wid-1][NUM_DDR_AVAIL_FREQ-1][0];
          execution_place_winner[clus_id][wid-1]++;
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[Optimized_Search] Left top: execution_place_winner: <" << clus_id << ", " << wid << ">,  value = " << execution_place_winner[clus_id][wid-1] << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif        
        }
        if(Energy[clus_id][wid-1][0][NUM_AVAIL_FREQ-1] < shortest_exec_R_B){
         if(previous_winner_cluster != clus_id || previous_winner_width != wid){
            execution_place_winner[previous_winner_cluster][previous_winner_width-1]--;
            previous_winner_cluster = clus_id;
            previous_winner_width = wid;
          }
          shortest_exec_R_B = Energy[clus_id][wid-1][0][NUM_AVAIL_FREQ-1];
          execution_place_winner[clus_id][wid-1]++;
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[Optimized_Search] Right bottom: execution_place_winner: <" << clus_id << ", " << wid << ">,  value = " << execution_place_winner[clus_id][wid-1] << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif
        }
        if(Energy[clus_id][wid-1][NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1] < shortest_exec_R_T){
          if(previous_winner_cluster != clus_id || previous_winner_width != wid){
            execution_place_winner[previous_winner_cluster][previous_winner_width-1]--;
            previous_winner_cluster = clus_id;
            previous_winner_width = wid;
          }
          shortest_exec_R_T = Energy[clus_id][wid-1][NUM_DDR_AVAIL_FREQ-1][NUM_AVAIL_FREQ-1];
          execution_place_winner[clus_id][wid-1]++;
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[Optimized_Search] Right top: execution_place_winner: <" << clus_id << ", " << wid << ">,  value = " << execution_place_winner[clus_id][wid-1] << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif
        }
        }
        i++;
      }
    }
    /* Find out the best execution place */
    int winner = 0;
#ifdef ALLOWSTEALING
    int current_best_width[NUMSOCKETS] = {1};
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ // for each cluster
      winner = 0;
      for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[Optimized_Search] Cluster " << clus_id << ", width " << wid << ", execution_place_winner = " << execution_place_winner[clus_id][wid-1]<< std::endl;
        LOCK_RELEASE(output_lck);
#endif
        if(execution_place_winner[clus_id][wid-1] > winner){
          winner = execution_place_winner[clus_id][wid-1];
          current_best_width[clus_id] = wid;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          //std::cout << "[Optimized_Search] Cluster " << clus_id << ", width " << wid << ", execution_place_winner = " << execution_place_winner[clus_id][wid-1] << ", current best width = " << wid << std::endl;
          std::cout << "[Optimized_Search] Cluster " << clus_id << ", current best width = " << wid << std::endl;
          LOCK_RELEASE(output_lck);
#endif
        }
      }
    }
    // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
    //   if(execution_place_winner[clus_id][current_best_width[clus_id]-1] > winner){
    //     winner = execution_place_winner[clus_id][current_best_width[clus_id]-1];
    //     it->set_best_core_type(clus_id);
    //     it->set_best_numcores(current_best_width[clus_id]);
    //   }
    // }
    if(execution_place_winner[0][current_best_width[0]-1] > execution_place_winner[1][current_best_width[1]-1]){
      it->set_best_core_type(0);
      it->set_best_numcores(current_best_width[0]);
      it->set_second_best_cluster(1);
      // it->set_second_best_numcores(current_best_width[1]);
      it->set_second_best_numcores(4);
    }else{
      it->set_best_core_type(1);
      it->set_best_numcores(current_best_width[1]);
      it->set_second_best_cluster(0);
      it->set_second_best_numcores(current_best_width[0]);
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[Optimized_Search] The best execution place is (" << it->get_best_cluster() << ", " << it->get_best_numcores() << "). Second best execution place is (" \
    << it->get_second_best_cluster() << ", " << it->get_second_best_numcores() << ")." << std::endl;
    LOCK_RELEASE(output_lck);
#endif
#else
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ // for each cluster
      for(auto&& wid : ptt_layout[start_coreid[clus_id]]){ 
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[Optimized_Search] The execution place winner of " << clus_id << "," << wid << " = " << execution_place_winner[clus_id][wid-1] << ". current best execution place: " << it->get_best_cluster() << ", " << it->get_best_numcores() << ")." << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif
        if(execution_place_winner[clus_id][wid-1] > winner){
          winner = execution_place_winner[clus_id][wid-1];
          it->set_best_core_type(clus_id);
          it->set_best_numcores(wid);
// #ifdef DEBUG
//     LOCK_ACQUIRE(output_lck);
//     std::cout << "[Optimized_Search] The best execution place is (" << it->get_best_cluster() << ", " << it->get_best_numcores() << ")." << std::endl;
//     LOCK_RELEASE(output_lck);
// #endif
        }
      }
    }
#endif
    it->width = it->get_best_numcores();
    best_cluster = it->get_best_cluster();
    float energy_mini = 100000.0f;
    int starting_ddr_freq_idx = 0;
    int starting_cpu_freq_idx = 0;
    // float energy[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0.0f};
    for(int i = 0; i < NUM_DDR_AVAIL_FREQ; i += NUM_DDR_AVAIL_FREQ-1){ // DDR Frequency index, 0 is highest DDR frequency (bottom), 1 is lowest DDR frequency (top)
      for(int j = 0; j < NUM_AVAIL_FREQ; j += NUM_AVAIL_FREQ-1){ // CPU Frequency, 0 is highest CPU frequency (left), 1 is lowest CPU frequency (right) 
        // energy[i][j] = Energy[i][j][it->get_best_cluster()][it->get_best_numcores()-1];
        if(Energy[best_cluster][it->width-1][i][j] < energy_mini){
          energy_mini = Energy[best_cluster][it->width-1][i][j];
          starting_ddr_freq_idx = i;
          starting_cpu_freq_idx = j;
        }
      }
    }
// #ifdef DEBUG
//     LOCK_ACQUIRE(output_lck);
//     std::cout << "[Optimized_Search] The best execution place is (" << best_cluster << ", " << it->width << "). Searching route starts from " << starting_ddr_freq_idx \
//     << " (DDR frequency index) and " << starting_cpu_freq_idx << " (CPU frequency index). " << std::endl;
//     LOCK_RELEASE(output_lck);
// #endif
    // float energy_thres = energy[starting_ddr_freq_idx][starting_cpu_freq_idx]; // starting comparison energy point
    float energy_thres = Energy[best_cluster][it->width-1][starting_ddr_freq_idx][starting_cpu_freq_idx];
    int visit_flag[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0}; // if the position has been visited before, it is -1,this is meant to avoid recalculate the energy for the same positions in two steps
    visit_flag[starting_ddr_freq_idx][starting_cpu_freq_idx] = 1; // the starting position has been computed, so set 1
    int step_progress;
    int cur_best_ddr_freq_idx = starting_ddr_freq_idx;
    int cur_best_cpu_freq_idx = starting_cpu_freq_idx;
    do{
      step_progress = 0;
      starting_ddr_freq_idx = cur_best_ddr_freq_idx;
      starting_cpu_freq_idx = cur_best_cpu_freq_idx;
      for(int i = 0; i < 8; i++){
        if(starting_ddr_freq_idx + x[i] >= 0 && starting_cpu_freq_idx + y[i] >= 0 && starting_ddr_freq_idx + x[i] < NUM_DDR_AVAIL_FREQ && starting_cpu_freq_idx + y[i] < NUM_AVAIL_FREQ){
          if(visit_flag[starting_ddr_freq_idx + x[i]][starting_cpu_freq_idx + y[i]] == 0) { // If this position hasn't been visited yet
            /* Idle power computation */
            float idleP_clus = 0.0f;
            if(sum_cluster_active[1 - best_cluster] == 0){ /* the number of active cores is zero in another cluster */
              idleP_clus = idle_cpu_power[starting_ddr_freq_idx + x[i]][starting_cpu_freq_idx + y[i]][best_cluster] \
              + idle_cpu_power[starting_ddr_freq_idx + x[i]][starting_cpu_freq_idx + y[i]][1-best_cluster]; /* Then equals idle power of whole chip */
            }else{
              idleP_clus = idle_cpu_power[starting_ddr_freq_idx + x[i]][starting_cpu_freq_idx + y[i]][best_cluster];
            }
            sum_cluster_active[best_cluster] = (sum_cluster_active[best_cluster] < it->width)? it->width : sum_cluster_active[best_cluster];
            float idle_P = idleP_clus * it->width / sum_cluster_active[best_cluster];
            float CPUPower_P = it->get_cpupowertable(starting_ddr_freq_idx + x[i], starting_cpu_freq_idx + y[i], best_cluster, it->width-1); 
            float DDRPower_P = it->get_ddrpowertable(starting_ddr_freq_idx + x[i], starting_cpu_freq_idx + y[i], best_cluster, it->width-1); 
            float Time_P = it->get_timetable(starting_ddr_freq_idx + x[i], starting_cpu_freq_idx + y[i], best_cluster, it->width-1); 
            float Energy_P = Time_P * (CPUPower_P -idleP_clus + idle_P + DDRPower_P);
            visit_flag[starting_ddr_freq_idx + x[i]][starting_cpu_freq_idx + y[i]] = 1; // visited
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[Optimized_Search] Position: (" << starting_ddr_freq_idx + x[i] << ", " << starting_cpu_freq_idx + y[i] << "): CPU power=" << CPUPower_P - idleP_clus + idle_P << ", memory power=" << DDRPower_P << ", execution time=" << Time_P << ". Energy = " << Energy_P << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif
            if(Energy_P < energy_thres){
              energy_thres = Energy_P;
              cur_best_ddr_freq_idx = starting_ddr_freq_idx + x[i]; // Update new starting frequency index per step
              cur_best_cpu_freq_idx = starting_cpu_freq_idx + y[i];
              step_progress = 1;
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[Optimized_Search] Position: (" << cur_best_ddr_freq_idx << ", " << cur_best_cpu_freq_idx << ") becomes bew winner!" << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif
            }
          }
        }
      }
    }while(step_progress == 1);
    it->set_best_ddr_freq(cur_best_ddr_freq_idx);
    it->set_best_cpu_freq(cur_best_cpu_freq_idx);
// #ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << ": the optimal cluster Memory and CPU frequency: " << avail_ddr_freq[starting_ddr_freq_idx] << ", " << avail_freq[best_cluster][starting_cpu_freq_idx] << ", best cluster: " \
    << best_cluster << ", best width: " << it->width << std::endl;
    LOCK_RELEASE(output_lck);
// #endif 
#ifdef ALLOWSTEALING
    int second_best_cluster = it->get_second_best_cluster();
    float second_energy_mini = 100000.0f;
    int second_starting_ddr_freq_idx = 0;
    int second_starting_cpu_freq_idx = 0;
    // float energy[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0.0f};
    for(int i = 0; i < NUM_DDR_AVAIL_FREQ; i += NUM_DDR_AVAIL_FREQ-1){ // DDR Frequency index, 0 is highest DDR frequency (bottom), 1 is lowest DDR frequency (top)
      for(int j = 0; j < NUM_AVAIL_FREQ; j += NUM_AVAIL_FREQ-1){ // CPU Frequency, 0 is highest CPU frequency (left), 1 is lowest CPU frequency (right) 
        // energy[i][j] = Energy[i][j][it->get_best_cluster()][it->get_best_numcores()-1];
        if(Energy[second_best_cluster][it->get_second_best_numcores()-1][i][j] < second_energy_mini){
          second_energy_mini = Energy[second_best_cluster][it->get_second_best_numcores()-1][i][j];
          second_starting_ddr_freq_idx = i;
          second_starting_cpu_freq_idx = j;
        }
      }
    }
    float second_energy_thres = Energy[second_best_cluster][it->get_second_best_numcores()-1][second_starting_ddr_freq_idx][second_starting_cpu_freq_idx];
    int second_visit_flag[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0}; // if the position has been visited before, it is -1,this is meant to avoid recalculate the energy for the same positions in two steps
    second_visit_flag[second_starting_ddr_freq_idx][second_starting_cpu_freq_idx] = 1; // the starting position has been computed, so set 1
    int second_step_progress;
    int second_cur_best_ddr_freq_idx = second_starting_ddr_freq_idx;
    int second_cur_best_cpu_freq_idx = second_starting_cpu_freq_idx;
    do{
      second_step_progress = 0;
      second_starting_ddr_freq_idx = second_cur_best_ddr_freq_idx;
      second_starting_cpu_freq_idx = second_cur_best_cpu_freq_idx;
      for(int i = 0; i < 8; i++){
        if(second_starting_ddr_freq_idx + x[i] >= 0 && second_starting_cpu_freq_idx + y[i] >= 0 && second_starting_ddr_freq_idx + x[i] < NUM_DDR_AVAIL_FREQ && second_starting_cpu_freq_idx + y[i] < NUM_AVAIL_FREQ){
          if(second_visit_flag[second_starting_ddr_freq_idx + x[i]][second_starting_cpu_freq_idx + y[i]] == 0) { // If this position hasn't been visited yet
            /* Idle power computation */
            float idleP_clus = 0.0f;
            if(sum_cluster_active[1 - second_best_cluster] == 0){ /* the number of active cores is zero in another cluster */
              idleP_clus = idle_cpu_power[second_starting_ddr_freq_idx + x[i]][second_starting_cpu_freq_idx + y[i]][second_best_cluster] \
              + idle_cpu_power[second_starting_ddr_freq_idx + x[i]][second_starting_cpu_freq_idx + y[i]][1-second_best_cluster]; /* Then equals idle power of whole chip */
            }else{
              idleP_clus = idle_cpu_power[second_starting_ddr_freq_idx + x[i]][second_starting_cpu_freq_idx + y[i]][second_best_cluster];
            }
            sum_cluster_active[second_best_cluster] = (sum_cluster_active[second_best_cluster] < it->get_second_best_numcores())? it->get_second_best_numcores() : sum_cluster_active[second_best_cluster];
            float idle_P = idleP_clus * it->get_second_best_numcores() / sum_cluster_active[second_best_cluster];
            float CPUPower_P = it->get_cpupowertable(second_starting_ddr_freq_idx + x[i], second_starting_cpu_freq_idx + y[i], second_best_cluster, it->get_second_best_numcores()-1); 
            float DDRPower_P = it->get_ddrpowertable(second_starting_ddr_freq_idx + x[i], second_starting_cpu_freq_idx + y[i], second_best_cluster, it->get_second_best_numcores()-1); 
            float Time_P = it->get_timetable(second_starting_ddr_freq_idx + x[i], second_starting_cpu_freq_idx + y[i], second_best_cluster, it->get_second_best_numcores()-1); 
            float Energy_P = Time_P * (CPUPower_P -idleP_clus + idle_P + DDRPower_P);
            second_visit_flag[second_starting_ddr_freq_idx + x[i]][second_starting_cpu_freq_idx + y[i]] = 1; // visited
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[Optimized_Search] Position: (" << second_starting_ddr_freq_idx + x[i] << ", " << starting_cpu_freq_idx + y[i] << "): CPU power=" << CPUPower_P - idleP_clus + idle_P << ", memory power=" << DDRPower_P << ", execution time=" << Time_P << ". Energy = " << Energy_P << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif
            if(Energy_P < second_energy_thres){
              second_energy_thres = Energy_P;
              second_cur_best_ddr_freq_idx = second_starting_ddr_freq_idx + x[i]; // Update new starting frequency index per step
              second_cur_best_cpu_freq_idx = second_starting_cpu_freq_idx + y[i];
              second_step_progress = 1;
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[Optimized_Search] Position: (" << second_cur_best_ddr_freq_idx << ", " << second_cur_best_cpu_freq_idx << ") becomes bew winner!" << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif
            }
          }
        }
      }
    }while(second_step_progress == 1);
    it->set_second_best_ddr_freq(second_cur_best_ddr_freq_idx);
    it->set_second_best_cpu_freq(second_cur_best_cpu_freq_idx);
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << ": the SECOND optimal cluster Memory and CPU frequency: " << avail_ddr_freq[second_starting_ddr_freq_idx] << ", " << avail_freq[second_best_cluster][second_starting_cpu_freq_idx] << ", best cluster: " \
    << second_best_cluster << ", best width: " << it->get_second_best_numcores() << std::endl;
    LOCK_RELEASE(output_lck);
#endif 
#endif
    it->set_enable_cpu_freq_change(true);
    it->set_enable_ddr_freq_change(true);
  }
#if defined Search_Overhead
  clock_gettime(CLOCK_REALTIME, &finish1);
  poly_sub_timespec(start1, finish1, &delta1);
  printf("[Overhead] This part is: %d.%.9ld\n", (int)delta1.tv_sec, delta1.tv_nsec);
#endif  
  it->set_bestconfig_state(true);
  it->get_bestconfig = true; // the reason of adding this: check if task itself gets the best config or not, since not all tasks are released because of dependencies (e.g., alya, k-means, dot product)
  it->width = it->get_best_numcores();
  best_cluster = it->get_best_cluster();
  it->leader = start_coreid[best_cluster] + (rand() % ((end_coreid[best_cluster] - start_coreid[best_cluster])/it->width)) * it->width;
  return it->leader;
}

int PolyTask::AAWS_LP(int nthread, PolyTask * it, int DOP){ 
  int CPU_freq_AAWS[NUMSOCKETS] = {10, 6}; // For AAWS - Test, Denver - 0.5GHz, A57 - 1.11GHz
  if(history_LP_visited[it->tasktype][DOP] == false || it->get_lp_task_distri_state() == false){ // If this is the first time to compute the task distribution for the specific DOP
    float time_pertask[XITAO_MAXTHREADS] = {0.0f};
    float time_DOP[XITAO_MAXTHREADS] = {0.0f};
    float CPUPowerP[XITAO_MAXTHREADS] = {0.0f};
    float DDRPowerP[XITAO_MAXTHREADS] = {0.0f};
    float EDP_Cluster0[XITAO_MAXTHREADS] = {0.0f};
    float lowest_edp = 100000000.0f;
    int num_cores[NUMSOCKETS] = {0};
    for(int i = 0; i < NUMSOCKETS; i++){
      num_cores[i] = end_coreid[i] - start_coreid[i];
    }
    for(auto&& wid : ptt_layout[start_coreid[0]]) { /* Step 1: first compute EDP of using single fastest cluster, i.e., cluster 0 */
      time_pertask[wid-1] = it->get_timetable(0, CPU_freq_AAWS[0], 0, wid-1);
      time_DOP[wid-1] = (wid * DOP <= num_cores[0])? time_pertask[wid-1] : time_pertask[wid-1] * ceil((wid * DOP) / num_cores[0]);
      CPUPowerP[wid-1] = (wid * DOP < num_cores[0])? it->get_cpupowertable(0, CPU_freq_AAWS[0], 0, (wid * DOP)-1) : it->get_cpupowertable(0, CPU_freq_AAWS[0], 0, num_cores[0]-1);
      DDRPowerP[wid-1] = (wid * DOP < num_cores[0])? it->get_ddrpowertable(0, CPU_freq_AAWS[0], 0, (wid * DOP)-1) : it->get_ddrpowertable(0, CPU_freq_AAWS[0], 0, num_cores[0]-1);
      EDP_Cluster0[wid-1] = powf(time_DOP[wid-1],2.0) * (CPUPowerP[wid-1] + DDRPowerP[wid-1]);
      if(EDP_Cluster0[wid-1] < lowest_edp){
        lowest_edp = EDP_Cluster0[wid-1];
        LP_best_width[it->tasktype][0] = wid;
      }
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] " << DOP << " tasks running on fastest Cluster-0: with resource width = " << wid << ", time prediction = " << time_DOP[wid-1] << ", CPU power prediction = " \
      << CPUPowerP[wid-1] << ", DDR power prediction = " << DDRPowerP[wid-1] << ", EDP prediction = " << EDP_Cluster0[wid-1] << std::endl; 
      LOCK_RELEASE(output_lck);
#endif
    }
    remaining_distri_num[it->tasktype][nthread][0] = DOP;
    remaining_distri_num[it->tasktype][nthread][1] = 0;
    float extra_power = idle_cpu_power[0][0][0] + idle_cpu_power[0][0][1] + idle_ddr_power[0];
    for(int N = 1; N <= DOP; N++){ /* Step 2: compute EDP of when allocating N tasks to the other cluster */
      float NewEnergy = 0.0f; 
      float NewEDP = 0.0f; 
      float Clus0_perf = ceil(float((DOP - N) * LP_best_width[it->tasktype][0])/float(num_cores[0])) * it->get_timetable(0, CPU_freq_AAWS[0], 0, LP_best_width[it->tasktype][0]-1); // execution time prediction on cluster 0 when allowing N tasks on the other cluster
      float Clus0_perf_for_energy = (DOP - N) * LP_best_width[it->tasktype][0] < num_cores[0]? Clus0_perf : float((DOP - N) * LP_best_width[it->tasktype][0])/float(num_cores[0]) * it->get_timetable(0, CPU_freq_AAWS[0], 0, LP_best_width[it->tasktype][0]-1);
      float cpu_power_clus0 = (LP_best_width[it->tasktype][0] * (DOP - N) < num_cores[0])? it->get_cpupowertable(0, CPU_freq_AAWS[0], 0, (LP_best_width[it->tasktype][0] * (DOP - N)) - 1) : it->get_cpupowertable(0, CPU_freq_AAWS[0], 0, num_cores[0]-1);
      float ddr_power_clus0 = (LP_best_width[it->tasktype][0] * (DOP - N) < num_cores[0])? it->get_ddrpowertable(0, CPU_freq_AAWS[0], 0, (LP_best_width[it->tasktype][0] * (DOP - N)) - 1) : it->get_ddrpowertable(0, CPU_freq_AAWS[0], 0, num_cores[0]-1);
      for(auto&& wid : ptt_layout[start_coreid[1]]){
        float Clus1_perf = ceil(float(N * wid)/float(num_cores[1])) * it->get_timetable(0, CPU_freq_AAWS[1], 1, wid-1); // execution time prediction on cluster 1 when allowing N tasks on cluster 1
        float Clus1_perf_for_energy = (N * wid) < num_cores[1]? Clus1_perf : float(N * wid)/float(num_cores[1]) * it->get_timetable(0, CPU_freq_AAWS[1], 1, wid-1);
        float cpu_power_clus1 = 0.0f;
        float ddr_power_clus1 = 0.0f;
        if((N * wid) % 2 != 0 && (N * wid < num_cores[1]) && it->get_cpupowertable(0, CPU_freq_AAWS[1], 1, (N * wid) - 1) == 0){ // e.g., when three tasks are running on 3 A57 with width 1, the default power table does not provide power prediction for those odd numbers
          float prev_wid_cpu_power = it->get_cpupowertable(0, CPU_freq_AAWS[1], 1, (N * wid) - ((N * wid) % 2) - 1);
          float next_wid_cpu_power = it->get_cpupowertable(0, CPU_freq_AAWS[1], 1, (N * wid) + ((N * wid) % 2) - 1);
          float prev_wid_ddr_power = it->get_ddrpowertable(0, CPU_freq_AAWS[1], 1, (N * wid) - ((N * wid) % 2) - 1); 
          float next_wid_ddr_power = it->get_ddrpowertable(0, CPU_freq_AAWS[1], 1, (N * wid) + ((N * wid) % 2) - 1);
          cpu_power_clus1 = prev_wid_cpu_power + (next_wid_cpu_power - prev_wid_cpu_power) / 2;
          ddr_power_clus1 = prev_wid_ddr_power + (next_wid_ddr_power - prev_wid_ddr_power) / 2;
        }else{
          cpu_power_clus1 = (N * wid < num_cores[1])? it->get_cpupowertable(0, CPU_freq_AAWS[1], 1, (N * wid) - 1) : it->get_cpupowertable(0, CPU_freq_AAWS[1], 1, num_cores[1]-1);
          ddr_power_clus1 = (N * wid < num_cores[1])? it->get_ddrpowertable(0, CPU_freq_AAWS[1], 1, (N * wid) - 1) : it->get_ddrpowertable(0, CPU_freq_AAWS[1], 1, num_cores[1]-1);
        }
        NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0) + Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1);
        if(Clus0_perf > Clus1_perf){
          if(N < DOP){
            NewEnergy -= Clus1_perf * extra_power;
          }
          // NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0) + Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1) - Clus1_perf * extra_power;
          // NewEnergy = Clus0_perf * (it->get_cpupowertable(0, 0, 0, LP_best_width[it->tasktype][0]-1) + it->get_ddrpowertable(0, 0, 0, LP_best_width[it->tasktype][0]-1)) \
          + Clus1_perf * (it->get_cpupowertable(0, 0, 1, wid-1) + it->get_ddrpowertable(0, 0, 1, wid-1)) - Clus1_perf * (idle_cpu_power[0][0][0] + idle_cpu_power[0][0][1] + idle_ddr_power[0]);
          // + (Clus0_perf - Clus1_perf) * (idle_cpu_power[0][0][1] + idle_ddr_power[0]);
          NewEDP = NewEnergy * Clus0_perf; 
        }else{
          if(N < DOP){
            NewEnergy -= Clus0_perf * extra_power;
          }
          // NewEnergy = Clus0_perf * (it->get_cpupowertable(0, 0, 0, LP_best_width[it->tasktype][0]-1) + it->get_ddrpowertable(0, 0, 0, LP_best_width[it->tasktype][0]-1)) \
          + Clus1_perf * (it->get_cpupowertable(0, 0, 1, wid-1) + it->get_ddrpowertable(0, 0, 1, wid-1)) - Clus0_perf * (idle_cpu_power[0][0][0] + idle_cpu_power[0][0][1] + idle_ddr_power[0]);
          // + (Clus1_perf - Clus0_perf) * (idle_cpu_power[0][0][0]+ idle_ddr_power[0]);
          NewEDP = NewEnergy * Clus1_perf; 
        }
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] When allocating " << N << " tasks to cluster 1 (with width " << wid << "), Cluster 0 exec time = " << Clus0_perf << ". Cluster 1 exec time = " << Clus1_perf \
        << ". CPU Power[cluster0] = " << cpu_power_clus0 << ", DDR Power[cluster0] = " << ddr_power_clus0 << ". CPU Power[cluster1] = " << cpu_power_clus1 << ", DDR Power[cluster1] = " << ddr_power_clus1 \
        << ", extra power = " << extra_power << ". Energy prediction = " << NewEnergy << ". EDP prediction = " << NewEDP << std::endl;
        LOCK_RELEASE(output_lck); 
#endif 
        if(NewEDP < lowest_edp){
          lowest_edp = NewEDP;
          remaining_distri_num[it->tasktype][nthread][0] = DOP - N;
          remaining_distri_num[it->tasktype][nthread][1] = N;
          LP_best_width[it->tasktype][1] = wid;
        }
      }
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] For " << DOP << " tasks, the best task distribution across clusters for reducing EDP is " << remaining_distri_num[it->tasktype][nthread][0] << " in cluster 0 with width " \
    << LP_best_width[it->tasktype][0] << ", and " << remaining_distri_num[it->tasktype][nthread][1] << " tasks in cluster 1 with width " << LP_best_width[it->tasktype][1] << std::endl;
    LOCK_RELEASE(output_lck); 
#endif
    for(int i=0; i < NUMSOCKETS; i++){
      history_LP_numTask[it->tasktype][i][DOP] = remaining_distri_num[it->tasktype][nthread][i];
      history_LP_bestwid[it->tasktype][i][DOP] = LP_best_width[it->tasktype][i];
    }
    history_LP_visited[it->tasktype][DOP] = true;
    it->set_lp_task_distri_state(true); 
#ifdef INTERVAL_RECOMPUTATION  
    interval_t2 = std::chrono::system_clock::now();
#endif
  }else{ // if the DOP scanario has been visited before, use the previous best task distribution
    for(int i=0; i < NUMSOCKETS; i++){
      remaining_distri_num[it->tasktype][nthread][i] = history_LP_numTask[it->tasktype][i][DOP];
      LP_best_width[it->tasktype][i] = history_LP_bestwid[it->tasktype][i][DOP];
    }
  }
  if(remaining_distri_num[it->tasktype][nthread][0] > 0){
    it->width = LP_best_width[it->tasktype][0];
    it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    remaining_distri_num[it->tasktype][nthread][0]--;
  }else{ // otherwise, schedule tasks to cluster B
    it->width = LP_best_width[it->tasktype][1];
    it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    remaining_distri_num[it->tasktype][nthread][1]--;
    // round_robin_counter[it->tasktype][nthread] = 1;
  }
  round_robin_counter[it->tasktype][nthread] = 0;
  DOP_detection[it->tasktype]--;
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] remaining_distri_num[" << it->kernel_name << "][" << nthread << "][0] = " << remaining_distri_num[it->tasktype][nthread][0] << ", remaining_distri_num[" << it->kernel_name \
  << "][" << nthread << "][1] = " << remaining_distri_num[it->tasktype][nthread][1] << ". DOP_detection[" << it->kernel_name << "] = " << DOP_detection[it->tasktype] << ". Round_robin_counter[" \
  << it->kernel_name << "][" << nthread << "] = " << round_robin_counter[it->tasktype][nthread] << ". For task " << it->taskid << ", it will be scheduled with leader core " << it->leader << " and width " << it->width << std::endl; 
  LOCK_RELEASE(output_lck);
#endif
  return it->leader;
}


int PolyTask::AAWS_HP(int nthread, PolyTask * it, int DOP){ /* Goal: minimize EDP => decide the resource width per task type and distribute DOP task among clusters */ 
  // int DOP = std::accumulate(std::begin(DOP_detection), std::begin(DOP_detection) + XITAO_MAXTHREADS, 0);  // DOP is the total number of tasks (inc multiple task types) that are currently ready and executing
  if(it->get_hp_task_distri_state() == false){
    /* Step 1: decide the resource width per task type for each cluster */
    int CPU_freq_AAWS[NUMSOCKETS] = {10, 6}; // For AAWS - Test, Denver - 0.5GHz, A57 - 1.11GHz
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ 
      HP_best_width[it->tasktype][clus_id] = {1}; // initialize the best width for each cluster
      int i = 0;
      float timeP[XITAO_MAXTHREADS] = {0.0f};
      for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
        timeP[wid-1] = it->get_timetable(0, CPU_freq_AAWS[clus_id], clus_id, wid-1); // the first two parameters are both zero, since we only check the highest frequency
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[DEBUG] for cluster " << clus_id << ", timeP[" << wid-1 << "] = " << timeP[wid-1] << std::endl; 
//       LOCK_RELEASE(output_lck);
// #endif       
        if(i > 0){
          if(timeP[0] / timeP[wid-1] > (wid)){ // if the execution time of width 1 is wid/1 times of execution time of the current width, then it achieves linearity
            HP_best_width[it->tasktype][clus_id] = wid; // best width is with moldability
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] Now achieves the linear speedup with more cores, the new best width becomes " << wid << std::endl; 
//           LOCK_RELEASE(output_lck);
// #endif   
          }
        }
        i++;
      }
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] The best resource width for " << it->kernel_name << " tasks on cluster 1 and 2 are <" << HP_best_width[it->tasktype][0] << "> and <" << HP_best_width[it->tasktype][1] << ">" << std::endl; 
    // std::cout << "[DEBUG] Task exec_time on cluster 1: " << it->get_timetable(0, 0, 0, HP_best_width[it->tasktype][0]-1) << ", task exec_time on cluster 2: " << it->get_timetable(0, 0, 1, HP_best_width[it->tasktype][1]-1) << std::endl;
    LOCK_RELEASE(output_lck);
#endif
    /* Step 2: start from using all cores and compute the corresonding EDP value */
    float timeP[XITAO_MAXTHREADS+1] = {0};
    float CPUPowerP[XITAO_MAXTHREADS+1] = {0};
    float DDRPowerP[XITAO_MAXTHREADS+1] = {0};
    float EDP[XITAO_MAXTHREADS+1] = {0};
    int edp_best_nc[NUMSOCKETS] = {0};
    float lowest_edp = 100000000.0f;
    for(int i = end_coreid[0] - start_coreid[0]; i > 0; i=i/2){  // i is cluster 0, e.g., on TX2, i = 2, 1
      if(i < HP_best_width[it->tasktype][0]){ // if the number of cores is less than the best width, then skip => at least one faster core is used
        break;
      }else{
        for(int j = end_coreid[1] - start_coreid[1]; j >= 0; j=j/2){ // j is cluster 1, e.g., on TX2, j = 4, 2, 1, 0
          timeP[i+j] = DOP / ( (i / (HP_best_width[it->tasktype][0] * it->get_timetable(0, CPU_freq_AAWS[0], 0, HP_best_width[it->tasktype][0]-1))) + (j / (HP_best_width[it->tasktype][1] * it->get_timetable(0, CPU_freq_AAWS[1], 1, HP_best_width[it->tasktype][1]-1))) );
          if(j > 0){
            CPUPowerP[i+j] = it->get_cpupowertable(0, CPU_freq_AAWS[0], 0, i-1) + it->get_cpupowertable(0, CPU_freq_AAWS[1], 1, j-1) - (idle_cpu_power[0][CPU_freq_AAWS[0]][0] + idle_cpu_power[0][CPU_freq_AAWS[1]][1]); // The total CPU power consumption when using i+j cores
            DDRPowerP[i+j] = it->get_ddrpowertable(0, CPU_freq_AAWS[0], 0, i-1) + it->get_ddrpowertable(0, CPU_freq_AAWS[1], 1, j-1) - idle_ddr_power[0];
          }else{
            CPUPowerP[i+j] = it->get_cpupowertable(0, CPU_freq_AAWS[0], 0, i-1);
            DDRPowerP[i+j] = it->get_ddrpowertable(0, CPU_freq_AAWS[0], 0, i-1);
          }
          EDP[i+j] = powf(timeP[i+j], 2.0) * (CPUPowerP[i+j] + DDRPowerP[i+j]);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Using " << i << " cores in cluster 1 and " << j << " cores in cluster 2: predicted execution time = " << timeP[i+j] << ", predicted CPU power = " << CPUPowerP[i+j] \
          << ", predicted DDR power = " << DDRPowerP[i+j] << ", predicted EDP = " << EDP[i+j] << std::endl; 
          LOCK_RELEASE(output_lck);
#endif 
          if(EDP[i+j] < lowest_edp){
            lowest_edp = EDP[i+j];
            edp_best_nc[0] = i;
            edp_best_nc[1] = j;
          }
          if(j == 0){
            break;
          }
        }
      }
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] Using <" << edp_best_nc[0] << "> cores in cluster 1 and <" << edp_best_nc[1] << "> cores in cluster 2 achieves the lowest predicted EDP." << std::endl; 
    LOCK_RELEASE(output_lck);
#endif
    
    // it->set_stealing_range(end_coreid[0] - start_coreid[0], end_coreid[1] - start_coreid[1], edp_best_nc[0], edp_best_nc[1]);   // set the stealing core range for the kernel task
    /* Step 3: find out the best frequency setting that further reduces EDP */
    float freq_throttle_timeP[NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0};
    float freq_throttle_CPUPowerP[NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0};
    float freq_throttle_edpP[NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0};
    int edp_best_freq_idx[NUMSOCKETS] = {10, 6};
//     for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
//       int k = 0;
//       for(int i = 0; i < 2; i++){
//         for(int j = 0; j < 2; j++){
//           if(i+j == 0){
//             continue; 
//           }else{
//             freq_throttle_timeP[freq_indx+i][freq_indx+j] = DOP / ( (edp_best_nc[0] / (HP_best_width[it->tasktype][0] * it->get_timetable(0, freq_indx+i, 0, HP_best_width[it->tasktype][0]-1))) + (edp_best_nc[1] / (HP_best_width[it->tasktype][1] * it->get_timetable(0, freq_indx+j, 1, HP_best_width[it->tasktype][1]-1))) );
//             if(edp_best_nc[1] > 0){
//               // freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] = it->get_cpupowertable(0, freq_indx+i, 0, HP_best_width[it->tasktype][0]-1) + it->get_cpupowertable(0, freq_indx+j, 1, HP_best_width[it->tasktype][1]-1)- (idle_cpu_power[0][freq_indx+i][0] + idle_cpu_power[0][freq_indx+j][1]);
//               freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] = it->get_cpupowertable(0, freq_indx+i, 0, edp_best_nc[0]-1) + it->get_cpupowertable(0, freq_indx+j, 1, edp_best_nc[1]-1) - (idle_cpu_power[0][freq_indx+i][0] + idle_cpu_power[0][freq_indx+j][1]);
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);        
//               std::cout << "[DEBUG] When tuning CPU frequency to <" << avail_freq[0][freq_indx+i] << ", " << avail_freq[1][freq_indx+j] << ">, the predicted CPU power consumption is " << it->get_cpupowertable(0, freq_indx+i, 0, edp_best_nc[0]-1) << " + " << it->get_cpupowertable(0, freq_indx+j, 1, edp_best_nc[1]-1) << " - " << \
//               idle_cpu_power[0][freq_indx+i][0] << " - " << idle_cpu_power[0][freq_indx+j][1] << " = " << freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] << std::endl;
//               LOCK_RELEASE(output_lck);
// #endif
//             }else{
//               freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] = it->get_cpupowertable(0, freq_indx+i, 0, HP_best_width[it->tasktype][0]-1);
//             }
//             freq_throttle_edpP[freq_indx+i][freq_indx+j] = powf(freq_throttle_timeP[freq_indx+i][freq_indx+j], 2.0) * freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j];
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[DEBUG] When tuning CPU frequency to <" << avail_freq[0][freq_indx+i] << ", " << avail_freq[1][freq_indx+j] << ">, the predicted execution time is " << freq_throttle_timeP[freq_indx+i][freq_indx+j] << ", the predicted CPU power consumption is " << \
//             freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] << ", the predicted EDP is " << freq_throttle_edpP[freq_indx+i][freq_indx+j] << std::endl; 
//             LOCK_RELEASE(output_lck);
// #endif            
//             if(freq_throttle_edpP[freq_indx+i][freq_indx+j] < lowest_edp){
//               k++;
//               lowest_edp = freq_throttle_edpP[freq_indx+i][freq_indx+j];
//               edp_best_freq_idx[0] = freq_indx+i;
//               edp_best_freq_idx[1] = freq_indx+j;
//             }
//           }
//         }
//       }
//       if(k == 0 || freq_indx == NUM_AVAIL_FREQ-1){
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[DEBUG] The best frequency setting for " << it->kernel_name << " tasks are <" << avail_freq[0][edp_best_freq_idx[0]] << ", " << avail_freq[1][edp_best_freq_idx[1]] << ">" << std::endl; 
//         LOCK_RELEASE(output_lck);
// #endif
//         break;
//       }
//     }
    /* Step 4: probability distribution of all parallel tasks */
    float clus_part[NUMSOCKETS] = {0};
    float suppose_load_balance = 0.0f;
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
      clus_part[clus_id] = (it->get_timetable(0, edp_best_freq_idx[1], 1, HP_best_width[it->tasktype][1]-1) / it->get_timetable(0, edp_best_freq_idx[clus_id], clus_id, HP_best_width[it->tasktype][clus_id]-1)) * float(edp_best_nc[clus_id] / HP_best_width[it->tasktype][clus_id]);
      suppose_load_balance += clus_part[clus_id];
    }
    // if(DOP >= suppose_load_balance){ // Crateria for deciding if it is high parallelism or not (temporary method)
      for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
        percent_cluster[it->tasktype][clus_id] = clus_part[clus_id] / suppose_load_balance;
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] For " << DOP << " " << it->kernel_name << " tasks, " << percent_cluster[it->tasktype][clus_id] * 100 << "% of tasks should be scheduled to cluster " << clus_id << std::endl; 
        LOCK_RELEASE(output_lck);
#endif
      }
    // } // TBD: what if DOP is lower than the value? 
    // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ 
    //   it->cpu_frequency_tuning(nthread, clus_id, start_coreid[clus_id], end_coreid[clus_id] - start_coreid[clus_id], edp_best_freq_idx[clus_id]); // set the best frequency setting for the kernel tasks
    // }
    it->set_hp_task_distri_state(true); /* Comment it and do the computation every DOP tasks. Otherwise, we only calculate all the parameters once, then next time just use the parameters */
#ifdef INTERVAL_RECOMPUTATION  
    interval_t2 = std::chrono::system_clock::now();
#endif
  }
  /* Step 5: try to schedule tasks based on the DOP and percentage of task distribution */
  // int task_distri_proposal[XITAO_MAXTHREADS][NUMSOCKETS] = {0};
  // task_distri_proposal[it->tasktype][0] = floor(DOP * percent_cluster[it->tasktype][0] * (DOP_detection[it->tasktype] / DOP) + 0.5);
  // task_distri_proposal[it->tasktype][1] = floor(DOP * percent_cluster[it->tasktype][1] * (DOP_detection[it->tasktype] / DOP) + 0.5);
  // task_distri_proposal[it->tasktype][0] = ceil(DOP * percent_cluster[it->tasktype][0] * (DOP_detection[it->tasktype] / DOP));
  // task_distri_proposal[it->tasktype][1] = DOP_detection[it->tasktype] - task_distri_proposal[it->tasktype][0];
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] The total: Task distribution across clusters for " << it->kernel_name << " tasks are <" << task_distri_proposal[it->tasktype][0] << "> for cluster 1 and <" << task_distri_proposal[it->tasktype][1] << "> for cluster 2" << std::endl; 
//   LOCK_RELEASE(output_lck);
// #endif
  remaining_distri_num[it->tasktype][nthread][0] = floor(DOP_detection[it->tasktype] * percent_cluster[it->tasktype][0] + 0.5); // ceil(DOP * percent_cluster[it->tasktype][0] * float(DOP_detection[it->tasktype] / DOP));
  remaining_distri_num[it->tasktype][nthread][1] = DOP_detection[it->tasktype] - remaining_distri_num[it->tasktype][nthread][0];
  // remaining_distri_num[it->tasktype][0] = floor(DOP * percent_cluster[it->tasktype][0] * (DOP_detection[it->tasktype] / DOP) + 0.5);
  // remaining_distri_num[it->tasktype][1] = floor(DOP * percent_cluster[it->tasktype][1] * (DOP_detection[it->tasktype] / DOP) + 0.5);
  // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
    // remaining_distri_num[it->tasktype][clus_id] = (DOP_executing[clus_id] > 0) ? task_distri_proposal[it->tasktype][clus_id] - DOP_executing[clus_id] : task_distri_proposal[it->tasktype][clus_id];
    // remaining_distri_num[it->tasktype][clus_id] = task_distri_proposal[it->tasktype][clus_id];
  // }
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] To be scheduled: Task distribution for " << DOP_detection[it->tasktype] << " " << it->kernel_name << " tasks are <" << remaining_distri_num[it->tasktype][nthread][0] << "> for cluster 1 and <" << remaining_distri_num[it->tasktype][nthread][1] << "> for cluster 2" << std::endl; 
  LOCK_RELEASE(output_lck);
#endif
  // if(DOP_executing[0] < task_distri_proposal[it->tasktype][0]){ // if currently executing tasks are less than the expected number of tasks, then schedule more tasks to cluster A
  if(remaining_distri_num[it->tasktype][nthread][0] > 0){
    it->width = HP_best_width[it->tasktype][0];
    it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    remaining_distri_num[it->tasktype][nthread][0]--;
  }else{ // otherwise, schedule tasks to cluster B
    it->width = HP_best_width[it->tasktype][1];
    it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    remaining_distri_num[it->tasktype][nthread][1]--;
    // round_robin_counter[it->tasktype][nthread] = 1;
  }
  round_robin_counter[it->tasktype][nthread] = 0;
  DOP_detection[it->tasktype]--;
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] remaining_distri_num[" << it->kernel_name << "][" << nthread << "][0] = " << remaining_distri_num[it->tasktype][nthread][0] << ", remaining_distri_num[" << it->kernel_name \
  << "][" << nthread << "][1] = " << remaining_distri_num[it->tasktype][nthread][1] << ". DOP_detection[" << it->kernel_name << "] = " << DOP_detection[it->tasktype] << ". Round_robin_counter[" \
  << it->kernel_name << "][" << nthread << "] = " << round_robin_counter[it->tasktype][nthread] << ". For task " << it->taskid << ", it will be scheduled with leader core " << it->leader << " and width " << it->width << std::endl; 
  LOCK_RELEASE(output_lck);
#endif
  return it->leader;
}

int PolyTask::task_distri_LP(int nthread, PolyTask * it, int DOP){ 
  // int DOP = DOP_detection[it->tasktype]; // haven't consider the multple task types
  if(history_LP_visited[it->tasktype][DOP] == false || it->get_lp_task_distri_state() == false){ // If this is the first time to compute the task distribution for the specific DOP
    float time_pertask[XITAO_MAXTHREADS] = {0.0f};
    float time_DOP[XITAO_MAXTHREADS] = {0.0f};
    float CPUPowerP[XITAO_MAXTHREADS] = {0.0f};
    float DDRPowerP[XITAO_MAXTHREADS] = {0.0f};
    float EDP_Cluster0[XITAO_MAXTHREADS] = {0.0f};
    float lowest_edp = 100000000.0f;
    int num_cores[NUMSOCKETS] = {0};
    for(int i = 0; i < NUMSOCKETS; i++){
      num_cores[i] = end_coreid[i] - start_coreid[i];
    }
    for(auto&& wid : ptt_layout[start_coreid[0]]) { /* Step 1: first compute EDP of using single fastest cluster, i.e., cluster 0 */
      time_pertask[wid-1] = it->get_timetable(0, 0, 0, wid-1);
      time_DOP[wid-1] = (wid * DOP <= num_cores[0])? time_pertask[wid-1] : time_pertask[wid-1] * ceil(float(wid * DOP) / num_cores[0]);
      CPUPowerP[wid-1] = (wid * DOP < num_cores[0])? it->get_cpupowertable(0, 0, 0, (wid * DOP)-1) : it->get_cpupowertable(0, 0, 0, num_cores[0]-1);
#ifdef Pro_fix_D
      DDRPowerP[wid-1] = (wid * DOP < num_cores[0])? it->get_ddrpowertable(0, 0, 1, (wid * DOP)-1) : it->get_ddrpowertable(0, 0, 1, num_cores[0]-1);
#else
      DDRPowerP[wid-1] = (wid * DOP < num_cores[0])? it->get_ddrpowertable(0, 0, 0, (wid * DOP)-1) : it->get_ddrpowertable(0, 0, 0, num_cores[0]-1);
#endif
      EDP_Cluster0[wid-1] = powf(time_DOP[wid-1], DELAY_METRIC) * powf(CPUPowerP[wid-1] + DDRPowerP[wid-1], POWER_METRIC);
      if(EDP_Cluster0[wid-1] < lowest_edp){
        lowest_edp = EDP_Cluster0[wid-1];
        LP_best_width[it->tasktype][0] = wid;
      }
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] " << DOP << " tasks running on fastest Cluster-0: with resource width = " << wid << ", time prediction = " << time_DOP[wid-1] << ", CPU power prediction = " \
      << CPUPowerP[wid-1] << ", DDR power prediction = " << DDRPowerP[wid-1] << ", EDP prediction = " << EDP_Cluster0[wid-1] << std::endl; 
      LOCK_RELEASE(output_lck);
#endif
    }
    remaining_distri_num[it->tasktype][nthread][0] = DOP;
    remaining_distri_num[it->tasktype][nthread][1] = 0;
    // int edp_best_freq_idx[NUMSOCKETS] = {0};
// #ifdef ADD_CPU_FREQ_TUNING
//     float freq_throttle_timeP[NUM_AVAIL_FREQ+1] = {0};
//     float freq_throttle_CPUPowerP[NUM_AVAIL_FREQ+1] = {0};
//     float freq_throttle_DDRPowerP[NUM_AVAIL_FREQ+1] = {0};
//     float freq_throttle_edpP[NUM_AVAIL_FREQ+1] = {0};
//     for(int freq_indx = 1; freq_indx <= NUM_AVAIL_FREQ; freq_indx++){ // step 1: CPU frequency tuning of cluster 0
//       float timepertask = it->get_timetable(0, freq_indx, 0, LP_best_width[it->tasktype][0]-1);
//       freq_throttle_timeP[freq_indx] = (LP_best_width[it->tasktype][0] * DOP < num_cores[0])?  timepertask : timepertask * ceil((LP_best_width[it->tasktype][0] * DOP) / num_cores[0]);
//       freq_throttle_CPUPowerP[freq_indx] = (LP_best_width[it->tasktype][0] * DOP < num_cores[0])? it->get_cpupowertable(0, freq_indx, 0, (LP_best_width[it->tasktype][0] * DOP)-1) : it->get_cpupowertable(0, freq_indx, 0, num_cores[0]-1);
//       freq_throttle_DDRPowerP[freq_indx] = (LP_best_width[it->tasktype][0] * DOP < num_cores[0])? it->get_ddrpowertable(0, freq_indx, 0, (LP_best_width[it->tasktype][0] * DOP)-1) : it->get_ddrpowertable(0, freq_indx, 0, num_cores[0]-1);
//       freq_throttle_edpP[freq_indx] = powf(freq_throttle_timeP[freq_indx],2.0) * (freq_throttle_CPUPowerP[freq_indx] + freq_throttle_DDRPowerP[freq_indx]);
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[DEBUG] When first tuning Cluster 0 CPU frequency to <" << avail_freq[0][freq_indx] << ", Cluster 0 exec time = " << freq_throttle_timeP[freq_indx] \
//       << ". CPU Power = " << freq_throttle_CPUPowerP[freq_indx] << ", DDR Power = " << freq_throttle_DDRPowerP[freq_indx] << ". EDP prediction = " << freq_throttle_edpP[freq_indx] << std::endl;
//       LOCK_RELEASE(output_lck);
// #endif
//       if(freq_throttle_edpP[freq_indx] < lowest_edp){
//         lowest_edp = freq_throttle_edpP[freq_indx];
//         edp_best_freq_idx[0] = freq_indx;
//       }else{
//         break; // stop when EDP of the current frequency setting now is increasing
//       }
//     }
// // #ifdef DEBUG
//     LOCK_ACQUIRE(output_lck);
//     std::cout << "[DEBUG] Best frequency setting for " << it->kernel_name << " tasks are <" << avail_freq[0][edp_best_freq_idx[0]] << ", " << avail_freq[1][edp_best_freq_idx[1]] << ">" << std::endl; 
//     LOCK_RELEASE(output_lck);
// // #endif
// #endif
#ifdef EXTRA_CPUPWR
    float extra_power = idle_cpu_power[0][0][0] + idle_cpu_power[0][0][1] + idle_ddr_power[0];
#else
    float extra_power = idle_ddr_power[0];
#endif
    for(int N = 1; N <= DOP; N++){ /* Step 2: compute EDP of when allocating N tasks to the other cluster */
      float NewEnergy = 0.0f; 
      float NewEDP = 0.0f; 
      float Clus0_perf = ceil(float((DOP - N) * LP_best_width[it->tasktype][0])/float(num_cores[0])) * it->get_timetable(0, 0, 0, LP_best_width[it->tasktype][0]-1); // execution time prediction on cluster 0 when allowing N tasks on the other cluster
      float Clus0_perf_for_energy = (DOP - N) * LP_best_width[it->tasktype][0] < num_cores[0]? Clus0_perf : float((DOP - N) * LP_best_width[it->tasktype][0])/float(num_cores[0]) * it->get_timetable(0, 0, 0, LP_best_width[it->tasktype][0]-1);
      float cpu_power_clus0 = (LP_best_width[it->tasktype][0] * (DOP - N) < num_cores[0])? it->get_cpupowertable(0, 0, 0, (LP_best_width[it->tasktype][0] * (DOP - N)) - 1) : it->get_cpupowertable(0, 0, 0, num_cores[0]-1);
#ifdef Pro_fix_D
      float ddr_power_clus0 = (LP_best_width[it->tasktype][0] * (DOP - N) < num_cores[0])? it->get_ddrpowertable(0, 0, 1, (LP_best_width[it->tasktype][0] * (DOP - N)) - 1) : it->get_ddrpowertable(0, 0, 1, num_cores[0]-1);
#else
      float ddr_power_clus0 = (LP_best_width[it->tasktype][0] * (DOP - N) < num_cores[0])? it->get_ddrpowertable(0, 0, 0, (LP_best_width[it->tasktype][0] * (DOP - N)) - 1) : it->get_ddrpowertable(0, 0, 0, num_cores[0]-1);
#endif
      for(auto&& wid : ptt_layout[start_coreid[1]]){
        float Clus1_perf = ceil(float(N * wid)/float(num_cores[1])) * it->get_timetable(0, 0, 1, wid-1); // execution time prediction on cluster 1 when allowing N tasks on cluster 1
        float Clus1_perf_for_energy = (N * wid) < num_cores[1]? Clus1_perf : float(N * wid)/float(num_cores[1]) * it->get_timetable(0, 0, 1, wid-1);
        float cpu_power_clus1 = 0.0f;
        float ddr_power_clus1 = 0.0f;
        if((N * wid) % 2 != 0 && (N * wid < num_cores[1]) && it->get_cpupowertable(0, 0, 1, (N * wid) - 1) == 0){ // e.g., when three tasks are running on 3 A57 with width 1, the default power table does not provide power prediction for those odd numbers
          float prev_wid_cpu_power = it->get_cpupowertable(0, 0, 1, (N * wid) - ((N * wid) % 2) - 1);
          float next_wid_cpu_power = it->get_cpupowertable(0, 0, 1, (N * wid) + ((N * wid) % 2) - 1);
          float prev_wid_ddr_power = it->get_ddrpowertable(0, 0, 1, (N * wid) - ((N * wid) % 2) - 1); 
          float next_wid_ddr_power = it->get_ddrpowertable(0, 0, 1, (N * wid) + ((N * wid) % 2) - 1);
          cpu_power_clus1 = prev_wid_cpu_power + (next_wid_cpu_power - prev_wid_cpu_power) / 2;
          ddr_power_clus1 = prev_wid_ddr_power + (next_wid_ddr_power - prev_wid_ddr_power) / 2;
        }else{
          cpu_power_clus1 = (N * wid < num_cores[1])? it->get_cpupowertable(0, 0, 1, (N * wid) - 1) : it->get_cpupowertable(0, 0, 1, num_cores[1]-1);
          ddr_power_clus1 = (N * wid < num_cores[1])? it->get_ddrpowertable(0, 0, 1, (N * wid) - 1) : it->get_ddrpowertable(0, 0, 1, num_cores[1]-1);
        }
        if(Clus0_perf > Clus1_perf){
          NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0) + Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1) - Clus1_perf * extra_power;
          // NewEnergy = Clus0_perf * (it->get_cpupowertable(0, 0, 0, LP_best_width[it->tasktype][0]-1) + it->get_ddrpowertable(0, 0, 0, LP_best_width[it->tasktype][0]-1)) \
          + Clus1_perf * (it->get_cpupowertable(0, 0, 1, wid-1) + it->get_ddrpowertable(0, 0, 1, wid-1)) - Clus1_perf * (idle_cpu_power[0][0][0] + idle_cpu_power[0][0][1] + idle_ddr_power[0]);
          // + (Clus0_perf - Clus1_perf) * (idle_cpu_power[0][0][1] + idle_ddr_power[0]);
          NewEDP = powf(NewEnergy, POWER_METRIC) * powf(Clus0_perf, DELAY_METRIC - POWER_METRIC); 
        }else{
          NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0) + Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1) - Clus0_perf * extra_power;
          // NewEnergy = Clus0_perf * (it->get_cpupowertable(0, 0, 0, LP_best_width[it->tasktype][0]-1) + it->get_ddrpowertable(0, 0, 0, LP_best_width[it->tasktype][0]-1)) \
          + Clus1_perf * (it->get_cpupowertable(0, 0, 1, wid-1) + it->get_ddrpowertable(0, 0, 1, wid-1)) - Clus0_perf * (idle_cpu_power[0][0][0] + idle_cpu_power[0][0][1] + idle_ddr_power[0]);
          // + (Clus1_perf - Clus0_perf) * (idle_cpu_power[0][0][0]+ idle_ddr_power[0]);
          NewEDP = powf(NewEnergy, POWER_METRIC) * powf(Clus1_perf, DELAY_METRIC - POWER_METRIC);
        }
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] When allocating " << N << " tasks to cluster 1 (with width " << wid << "), Cluster 0 exec time = " << Clus0_perf << ". Cluster 1 exec time = " << Clus1_perf \
        << ". CPU Power[cluster0] = " << cpu_power_clus0 << ", DDR Power[cluster0] = " << ddr_power_clus0 << ". CPU Power[cluster1] = " << cpu_power_clus1 << ", DDR Power[cluster1] = " << ddr_power_clus1 \
        << ", extra power = " << extra_power << ". Energy prediction = " << NewEnergy << ". EDP prediction = " << NewEDP << std::endl;
        LOCK_RELEASE(output_lck); 
#endif 
        if(NewEDP < lowest_edp){
          lowest_edp = NewEDP;
          remaining_distri_num[it->tasktype][nthread][0] = DOP - N;
          remaining_distri_num[it->tasktype][nthread][1] = N;
          LP_best_width[it->tasktype][1] = wid;
        }
      }
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] For " << DOP << " tasks, the best task distribution across clusters for reducing EDP is " << remaining_distri_num[it->tasktype][nthread][0] << " in cluster 0 with width " \
    << LP_best_width[it->tasktype][0] << ", and " << remaining_distri_num[it->tasktype][nthread][1] << " tasks in cluster 1 with width " << LP_best_width[it->tasktype][1] << std::endl;
    LOCK_RELEASE(output_lck); 
#endif
#ifdef ADD_CPU_FREQ_TUNING
    int visit_flag[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0}; // if the position has been visited before, it is -1,this is meant to avoid recalculate the EDP for the same positions in two steps
    visit_flag[0][0][0] = 1; // Current starting CPU frequency for cluster 0 and cluster 1 are highest frequency, the starting position has been computed, so set 1
    int step_progress = 0, cur_bestcpufreq_clus0 = 0, cur_bestcpufreq_clus1 = 0, starting_cpufreq_clus0 = 0, starting_cpufreq_clus1 = 0, cur_bestDDRfreq = 0, starting_DDRfreq = 0;
    int z[2] = {0};
    if(remaining_distri_num[it->tasktype][nthread][0] > 0 && remaining_distri_num[it->tasktype][nthread][1] > 0){ // If there are tasks allocated to cluster 1, i.e., N > 0 from the above for loop
      const int N = 9;
      int x[N] = {0, -1, -1, -1, 0, 0, 1, 1, 1}; // x and y arrays are meant to define 8 directions
      int y[N] = {0, -1, 0, 1, -1, 1, -1, 0, 1};
#ifdef ADD_DDR_FREQ_TUNING
      z[1] = 1;
#endif
      // int x[3] = {0, 1, 1}; // x and y arrays are meant to define 3 directions
      // int y[3] = {1, 0, 1};
      do{
        step_progress = 0;
        starting_DDRfreq = cur_bestDDRfreq;
        for(int j = 0; j < 2; j++){
          if(starting_DDRfreq + z[j] >=0 && starting_DDRfreq + z[j] < NUM_DDR_AVAIL_FREQ){
            starting_cpufreq_clus0 = cur_bestcpufreq_clus0;
            starting_cpufreq_clus1 = cur_bestcpufreq_clus1;
            for(int i = 0; i < N; i++){
              if(starting_cpufreq_clus0 + x[i] >= 0 && starting_cpufreq_clus1 + y[i] >= 0 && starting_cpufreq_clus0 + x[i] < NUM_AVAIL_FREQ && starting_cpufreq_clus1 + y[i] < NUM_AVAIL_FREQ){
                if(visit_flag[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] == 0) { // If this position hasn't been visited yet
                  float Clus0_perf = ceil(float(remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0])/float(num_cores[0])) * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, LP_best_width[it->tasktype][0]-1); // execution time prediction on cluster 0 when allowing N tasks on the other cluster
                  float Clus0_perf_for_energy = remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0] < num_cores[0]? Clus0_perf : float(remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0])/float(num_cores[0]) * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, LP_best_width[it->tasktype][0]-1);
                  float cpu_power_clus0 = (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0] < num_cores[0])? it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0]) - 1) : it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, num_cores[0]-1);
#ifdef Pro_fix_D
                  float ddr_power_clus0 = (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0] < num_cores[0])? it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 1, (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0]) - 1) : it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 1, num_cores[0]-1);
#else                  
                  float ddr_power_clus0 = (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0] < num_cores[0])? it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0]) - 1) : it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, num_cores[0]-1);
#endif
                  int occupied_cores_clus1 = remaining_distri_num[it->tasktype][nthread][1] * LP_best_width[it->tasktype][1];
                  float Clus1_perf = ceil(float(occupied_cores_clus1)/float(num_cores[1])) * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, LP_best_width[it->tasktype][1]-1); // execution time prediction on cluster 1 when allowing N tasks on cluster 1
                  float Clus1_perf_for_energy = (occupied_cores_clus1) < num_cores[1]? Clus1_perf : float(occupied_cores_clus1)/float(num_cores[1]) * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, LP_best_width[it->tasktype][1]-1);
                  float cpu_power_clus1 = 0.0f;
                  float ddr_power_clus1 = 0.0f;
                  if(occupied_cores_clus1 % 2 != 0 && (occupied_cores_clus1 < num_cores[1]) && it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, occupied_cores_clus1 - 1) == 0){ // e.g., when three tasks are running on 3 A57 with width 1, the default power table does not provide power prediction for those odd numbers
                    float prev_wid_cpu_power = it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, (occupied_cores_clus1) - ((occupied_cores_clus1) % 2) - 1);
                    float next_wid_cpu_power = it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, (occupied_cores_clus1) + ((occupied_cores_clus1) % 2) - 1);
                    float prev_wid_ddr_power = it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, (occupied_cores_clus1) - ((occupied_cores_clus1) % 2) - 1); 
                    float next_wid_ddr_power = it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, (occupied_cores_clus1) + ((occupied_cores_clus1) % 2) - 1);
                    cpu_power_clus1 = prev_wid_cpu_power + (next_wid_cpu_power - prev_wid_cpu_power) / 2.0;
                    ddr_power_clus1 = prev_wid_ddr_power + (next_wid_ddr_power - prev_wid_ddr_power) / 2.0;
                  }else{
                    cpu_power_clus1 = (occupied_cores_clus1 < num_cores[1])? it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, occupied_cores_clus1 - 1) : it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, num_cores[1]-1);
                    ddr_power_clus1 = (occupied_cores_clus1 < num_cores[1])? it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, occupied_cores_clus1 - 1) : it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, num_cores[1]-1);
                  }
#ifdef EXTRA_CPUPWR
                  float throttle_extra_power = idle_cpu_power[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][0] + idle_cpu_power[starting_DDRfreq + z[j]][starting_cpufreq_clus1 + y[i]][1] + idle_ddr_power[starting_DDRfreq + z[j]];
#else
                  float throttle_extra_power = idle_ddr_power[starting_DDRfreq + z[j]];
#endif                  
                  float throttle_NewEnergy = 0.0f;
                  float throttle_NewEDP = 0.0f;
                  if(Clus0_perf > Clus1_perf){
                    throttle_NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0) + Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1) - Clus1_perf * throttle_extra_power;
                    throttle_NewEDP = powf(throttle_NewEnergy, POWER_METRIC) * powf(Clus0_perf_for_energy, DELAY_METRIC - POWER_METRIC); 
                  }else{
                    throttle_NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0) + Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1) - Clus0_perf * throttle_extra_power;
                    throttle_NewEDP = powf(throttle_NewEnergy, POWER_METRIC) * powf(Clus1_perf_for_energy, DELAY_METRIC - POWER_METRIC); 
                  }
                  visit_flag[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = 1; // visited
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] When tuning Mem frequency to " << avail_ddr_freq[starting_DDRfreq + z[j]] << " and CPU frequency to <" << avail_freq[0][starting_cpufreq_clus0 + x[i]] << ", " << avail_freq[1][starting_cpufreq_clus1 + y[i]] << ">, Cluster 0 exec time = " << Clus0_perf << ". Cluster 1 exec time = " << Clus1_perf \
                  << ". CPU Power[cluster0] = " << cpu_power_clus0 << ", DDR Power[cluster0] = " << ddr_power_clus0 << ". CPU Power[cluster1] = " << cpu_power_clus1 << ", DDR Power[cluster1] = " << ddr_power_clus1 \
                  << ", throttle_extra_power = " << throttle_extra_power << ". Energy prediction = " << throttle_NewEnergy << ". EDP prediction = " << throttle_NewEDP << std::endl;
                  LOCK_RELEASE(output_lck);
#endif       
                  if(throttle_NewEDP < lowest_edp){
                    lowest_edp = throttle_NewEDP;
                    cur_bestcpufreq_clus0 = starting_cpufreq_clus0 + x[i]; // Update new starting frequency index per step
                    cur_bestcpufreq_clus1 = starting_cpufreq_clus1 + y[i];
                    cur_bestDDRfreq = starting_DDRfreq + z[j];
                    step_progress = 1;
                  }
                }
              }
            }
          }
        }
      }while(step_progress == 1);
// #ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] Best frequency setting for " << it->kernel_name << " tasks are <" << avail_ddr_freq[cur_bestDDRfreq] << ", " << avail_freq[0][cur_bestcpufreq_clus0] << ", " << avail_freq[1][cur_bestcpufreq_clus1] << ">" << std::endl; 
      LOCK_RELEASE(output_lck);
// #endif
//       for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
//         int k = 0;
//         for(int i = 0; i < 2; i++){
//           for(int j = 0; j < 2; j++){
//             if(i+j == 0){
//               continue; 
//             }else{
//               float Clus0_perf = ceil(float(remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0])/float(num_cores[0])) * it->get_timetable(0, freq_indx+i, 0, LP_best_width[it->tasktype][0]-1); // execution time prediction on cluster 0 when allowing N tasks on the other cluster
//               float Clus0_perf_for_energy = remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0] < num_cores[0]? Clus0_perf : float(remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0])/float(num_cores[0]) * it->get_timetable(0, freq_indx+i, 0, LP_best_width[it->tasktype][0]-1);
//               float cpu_power_clus0 = (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0] < num_cores[0])? it->get_cpupowertable(0, freq_indx+i, 0, (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0]) - 1) : it->get_cpupowertable(0, freq_indx+i, 0, num_cores[0]-1);
//               float ddr_power_clus0 = (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0] < num_cores[0])? it->get_ddrpowertable(0, freq_indx+i, 0, (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0]) - 1) : it->get_ddrpowertable(0, freq_indx+i, 0, num_cores[0]-1);
//               int occupied_cores_clus1 = remaining_distri_num[it->tasktype][nthread][1] * LP_best_width[it->tasktype][1];
//               float Clus1_perf = ceil(float(occupied_cores_clus1)/float(num_cores[1])) * it->get_timetable(0, freq_indx+j, 1, LP_best_width[it->tasktype][1]-1); // execution time prediction on cluster 1 when allowing N tasks on cluster 1
//               float Clus1_perf_for_energy = (occupied_cores_clus1) < num_cores[1]? Clus1_perf : float(occupied_cores_clus1)/float(num_cores[1]) * it->get_timetable(0, freq_indx+j, 1, LP_best_width[it->tasktype][1]-1);
//               float cpu_power_clus1 = 0.0f;
//               float ddr_power_clus1 = 0.0f;
//               if(occupied_cores_clus1 % 2 != 0 && (occupied_cores_clus1 < num_cores[1]) && it->get_cpupowertable(0, freq_indx+j, 1, occupied_cores_clus1 - 1) == 0){ // e.g., when three tasks are running on 3 A57 with width 1, the default power table does not provide power prediction for those odd numbers
//                 float prev_wid_cpu_power = it->get_cpupowertable(0, freq_indx+j, 1, (occupied_cores_clus1) - ((occupied_cores_clus1) % 2) - 1);
//                 float next_wid_cpu_power = it->get_cpupowertable(0, freq_indx+j, 1, (occupied_cores_clus1) + ((occupied_cores_clus1) % 2) - 1);
//                 float prev_wid_ddr_power = it->get_ddrpowertable(0, freq_indx+j, 1, (occupied_cores_clus1) - ((occupied_cores_clus1) % 2) - 1); 
//                 float next_wid_ddr_power = it->get_ddrpowertable(0, freq_indx+j, 1, (occupied_cores_clus1) + ((occupied_cores_clus1) % 2) - 1);
//                 cpu_power_clus1 = prev_wid_cpu_power + (next_wid_cpu_power - prev_wid_cpu_power) / 2.0;
//                 ddr_power_clus1 = prev_wid_ddr_power + (next_wid_ddr_power - prev_wid_ddr_power) / 2.0;
//               }else{
//                 cpu_power_clus1 = (occupied_cores_clus1 < num_cores[1])? it->get_cpupowertable(0, freq_indx+j, 1, occupied_cores_clus1 - 1) : it->get_cpupowertable(0, freq_indx+j, 1, num_cores[1]-1);
//                 ddr_power_clus1 = (occupied_cores_clus1 < num_cores[1])? it->get_ddrpowertable(0, freq_indx+j, 1, occupied_cores_clus1 - 1) : it->get_ddrpowertable(0, freq_indx+j, 1, num_cores[1]-1);
//               }
//               float throttle_extra_power = idle_cpu_power[0][freq_indx+i][0] + idle_cpu_power[0][freq_indx+j][1] + idle_ddr_power[0];
//               float throttle_NewEnergy = 0.0f;
//               float throttle_NewEDP = 0.0f;
//               if(Clus0_perf > Clus1_perf){
//                 throttle_NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0) + Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1) - Clus1_perf * throttle_extra_power;
//                 throttle_NewEDP = throttle_NewEnergy * Clus0_perf_for_energy; 
//               }else{
//                 throttle_NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0) + Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1) - Clus0_perf * throttle_extra_power;
//                 throttle_NewEDP = throttle_NewEnergy * Clus1_perf_for_energy; 
//               }
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] When tuning CPU frequency to <" << avail_freq[0][freq_indx+i] << ", " << avail_freq[1][freq_indx+j] << ">, Cluster 0 exec time = " << Clus0_perf << ". Cluster 1 exec time = " << Clus1_perf \
//               << ". CPU Power[cluster0] = " << cpu_power_clus0 << ", DDR Power[cluster0] = " << ddr_power_clus0 << ". CPU Power[cluster1] = " << cpu_power_clus1 << ", DDR Power[cluster1] = " << ddr_power_clus1 \
//               << ", throttle_extra_power = " << throttle_extra_power << ". Energy prediction = " << throttle_NewEnergy << ". EDP prediction = " << throttle_NewEDP << std::endl;
//               LOCK_RELEASE(output_lck);
// #endif       
//               if(throttle_NewEDP < lowest_edp){
//                 k++;
//                 lowest_edp = throttle_NewEDP;
//                 edp_best_freq_idx[0] = freq_indx+i;
//                 edp_best_freq_idx[1] = freq_indx+j;
//               }
//             }
//           }
//         }
//         if(k == 0 || freq_indx == NUM_AVAIL_FREQ-1){
//           break;
//         }
//       }
    }else{ // One of the clusters does not allocate any tasks 
      // int x = 0, y = 0;
      if(remaining_distri_num[it->tasktype][nthread][0] > 0 && remaining_distri_num[it->tasktype][nthread][1] == 0){
        int x[2] = {0, 1};
#ifdef ADD_DDR_FREQ_TUNING
        z[1] = 1;
#endif
        do{
          step_progress = 0;
          starting_DDRfreq = cur_bestDDRfreq;
          for(int j = 0; j < 2; j++){
            if(starting_DDRfreq + z[j] >=0 && starting_DDRfreq + z[j] < NUM_DDR_AVAIL_FREQ){
              starting_cpufreq_clus0 = cur_bestcpufreq_clus0;
              for(int i = 0; i < 2; i++){
              if(starting_cpufreq_clus0 + x[i] < NUM_AVAIL_FREQ && visit_flag[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][cur_bestcpufreq_clus1] == 0){
                float Clus0_perf = ceil(float(remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0])/float(num_cores[0])) * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, LP_best_width[it->tasktype][0]-1); // execution time prediction on cluster 0 when allowing N tasks on the other cluster
                float Clus0_perf_for_energy = remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0] < num_cores[0]? Clus0_perf : float(remaining_distri_num[it->tasktype][nthread][0] * LP_best_width[it->tasktype][0])/float(num_cores[0]) * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, LP_best_width[it->tasktype][0]-1);
                float cpu_power_clus0 = (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0] < num_cores[0])? it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0]) - 1) : it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, num_cores[0]-1);
#ifdef Pro_fix_D
                float ddr_power_clus0 = (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0] < num_cores[0])? it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 1, (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0]) - 1) : it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 1, num_cores[0]-1);
#else
                float ddr_power_clus0 = (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0] < num_cores[0])? it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, (LP_best_width[it->tasktype][0] * remaining_distri_num[it->tasktype][nthread][0]) - 1) : it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, num_cores[0]-1);
#endif
                // float throttle_NewEnergy = Clus0_perf_for_energy * (cpu_power_clus0 + ddr_power_clus0);
                float throttle_NewEDP = powf(Clus0_perf_for_energy, DELAY_METRIC) * powf((cpu_power_clus0 + ddr_power_clus0), POWER_METRIC);
                visit_flag[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][cur_bestcpufreq_clus1] = 1; // visited
    #ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] When tuning Mem frequency to " << avail_ddr_freq[starting_DDRfreq + z[j]] << " and CPU frequency of cluster 0 to " << avail_freq[0][starting_cpufreq_clus0 + x[i]] << ", Cluster 0 exec time = " << Clus0_perf \
                << ". CPU Power[cluster0] = " << cpu_power_clus0 << ", DDR Power[cluster0] = " << ddr_power_clus0 << ". EDP prediction = " << throttle_NewEDP << std::endl;
                LOCK_RELEASE(output_lck);
    #endif    
                if(throttle_NewEDP < lowest_edp){
                  lowest_edp = throttle_NewEDP;
                  cur_bestcpufreq_clus0 = starting_cpufreq_clus0 + x[i]; // Update new starting frequency index per step
                  cur_bestDDRfreq = starting_DDRfreq + z[j];
                  step_progress = 1;
                }
              }
              }
            }
          }
        }while(step_progress == 1);
        cur_bestcpufreq_clus1 = -1; // this means cluster 1's frequency can be any number
// #ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Best frequency setting for " << it->kernel_name << " tasks are <" << avail_ddr_freq[cur_bestDDRfreq] << ", " << avail_freq[0][cur_bestcpufreq_clus0] << ", ANY>" << std::endl; 
        LOCK_RELEASE(output_lck);
// #endif
      }else{
        if(remaining_distri_num[it->tasktype][nthread][0] == 0 && remaining_distri_num[it->tasktype][nthread][1] > 0){
          int y[2] = {0, 1};
#ifdef ADD_DDR_FREQ_TUNING
          z[1] = 1;
#endif
          do{
            step_progress = 0;
            starting_DDRfreq = cur_bestDDRfreq;
            for(int j = 0; j < 2; j++){
              if(starting_DDRfreq + z[j] >=0 && starting_DDRfreq + z[j] < NUM_DDR_AVAIL_FREQ){
                starting_cpufreq_clus1 = cur_bestcpufreq_clus1;
                for(int i = 0; i < 2; i++){
                if(starting_cpufreq_clus1 + y[i] < NUM_AVAIL_FREQ && visit_flag[starting_DDRfreq + z[j]][cur_bestcpufreq_clus0][starting_cpufreq_clus1 + y[i]] == 0){
                  int occupied_cores_clus1 = remaining_distri_num[it->tasktype][nthread][1] * LP_best_width[it->tasktype][1];
                  float Clus1_perf = ceil(float(occupied_cores_clus1)/float(num_cores[1])) * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, LP_best_width[it->tasktype][1]-1); // execution time prediction on cluster 1 when allowing N tasks on cluster 1
                  float Clus1_perf_for_energy = (occupied_cores_clus1) < num_cores[1]? Clus1_perf : float(occupied_cores_clus1)/float(num_cores[1]) * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, LP_best_width[it->tasktype][1]-1);
                  float cpu_power_clus1 = 0.0f;
                  float ddr_power_clus1 = 0.0f;
                  if(occupied_cores_clus1 % 2 != 0 && (occupied_cores_clus1 < num_cores[1]) && it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, occupied_cores_clus1 - 1) == 0){ // e.g., when three tasks are running on 3 A57 with width 1, the default power table does not provide power prediction for those odd numbers
                    float prev_wid_cpu_power = it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, (occupied_cores_clus1) - ((occupied_cores_clus1) % 2) - 1);
                    float next_wid_cpu_power = it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, (occupied_cores_clus1) + ((occupied_cores_clus1) % 2) - 1);
                    float prev_wid_ddr_power = it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, (occupied_cores_clus1) - ((occupied_cores_clus1) % 2) - 1); 
                    float next_wid_ddr_power = it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, (occupied_cores_clus1) + ((occupied_cores_clus1) % 2) - 1);
                    cpu_power_clus1 = prev_wid_cpu_power + (next_wid_cpu_power - prev_wid_cpu_power) / 2.0;
                    ddr_power_clus1 = prev_wid_ddr_power + (next_wid_ddr_power - prev_wid_ddr_power) / 2.0;
                  }else{
                    cpu_power_clus1 = (occupied_cores_clus1 < num_cores[1])? it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, occupied_cores_clus1 - 1) : it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, num_cores[1]-1);
                    ddr_power_clus1 = (occupied_cores_clus1 < num_cores[1])? it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, occupied_cores_clus1 - 1) : it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, num_cores[1]-1);
                  }
                  // float throttle_NewEnergy = Clus1_perf_for_energy * (cpu_power_clus1 + ddr_power_clus1);
                  float throttle_NewEDP = powf(Clus1_perf_for_energy, DELAY_METRIC) * (cpu_power_clus1 + ddr_power_clus1);
                  visit_flag[starting_DDRfreq + z[j]][cur_bestcpufreq_clus0][starting_cpufreq_clus1 + y[i]] = 1; // visited
#ifdef DEBUG
                  LOCK_ACQUIRE(output_lck);
                  std::cout << "[DEBUG] When tuning Mem frequency to " << avail_ddr_freq[starting_DDRfreq + z[j]] << " and CPU frequency of cluster 1 to " << avail_freq[1][starting_cpufreq_clus1 + y[i]] << ">, Cluster 1 exec time = " << Clus1_perf \
                  << ". CPU Power[cluster1] = " << cpu_power_clus1 << ", DDR Power[cluster1] = " << ddr_power_clus1 << ". EDP prediction = " << throttle_NewEDP << std::endl;
                  LOCK_RELEASE(output_lck);
#endif    
                  if(throttle_NewEDP < lowest_edp){
                    lowest_edp = throttle_NewEDP;
                    cur_bestcpufreq_clus1 = starting_cpufreq_clus1 + y[i]; // Update new starting frequency index per step
                    cur_bestDDRfreq = starting_DDRfreq + z[j];
                    step_progress = 1;
                  }
                }
              }
              }
            }
          }while(step_progress == 1);
          cur_bestcpufreq_clus0 = -1; 
// #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Best frequency setting for " << it->kernel_name << " tasks are <" << avail_ddr_freq[cur_bestDDRfreq] << ", ANY, " << avail_freq[1][cur_bestcpufreq_clus1] << ">" << std::endl; 
          LOCK_RELEASE(output_lck);
// #endif
        }
      }
    }
    history_LP_cpu_freq[it->tasktype][0][DOP] = cur_bestcpufreq_clus0;
    history_LP_cpu_freq[it->tasktype][1][DOP] = cur_bestcpufreq_clus1;
    history_LP_ddr_freq[it->tasktype][DOP] = cur_bestDDRfreq;
    // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
    //   LP_cpu_freq[it->tasktype][clus_id][DOP] = edp_best_freq_idx[clus_id];
    // }
    // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ // should tune frequency before executing tasks  
    //   it->cpu_frequency_tuning(nthread, clus_id, start_coreid[clus_id], end_coreid[clus_id] - start_coreid[clus_id], edp_best_freq_idx[clus_id]); // set the best frequency setting for the kernel tasks
    // }
#endif
    for(int i=0; i < NUMSOCKETS; i++){
      history_LP_numTask[it->tasktype][i][DOP] = remaining_distri_num[it->tasktype][nthread][i];
      history_LP_bestwid[it->tasktype][i][DOP] = LP_best_width[it->tasktype][i];
      LP_cpu_freq[it->tasktype][i] = history_LP_cpu_freq[it->tasktype][i][DOP];
    }
    LP_ddr_freq[it->tasktype] = history_LP_ddr_freq[it->tasktype][DOP]; 
    history_LP_visited[it->tasktype][DOP] = true;
    it->set_lp_task_distri_state(true); 
#ifdef INTERVAL_RECOMPUTATION  
    interval_t2 = std::chrono::system_clock::now();
#endif
  }else{ // if the DOP scanario has been visited before, use the previous best task distribution
    for(int i=0; i < NUMSOCKETS; i++){
      remaining_distri_num[it->tasktype][nthread][i] = history_LP_numTask[it->tasktype][i][DOP];
      LP_best_width[it->tasktype][i] = history_LP_bestwid[it->tasktype][i][DOP];
      LP_cpu_freq[it->tasktype][i] = history_LP_cpu_freq[it->tasktype][i][DOP];
    }
  }
  if(remaining_distri_num[it->tasktype][nthread][0] > 0){
    it->width = LP_best_width[it->tasktype][0];
    it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    remaining_distri_num[it->tasktype][nthread][0]--;
  }else{ // otherwise, schedule tasks to cluster B
    it->width = LP_best_width[it->tasktype][1];
    it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    remaining_distri_num[it->tasktype][nthread][1]--;
    // round_robin_counter[it->tasktype][nthread] = 1;
  }
#ifdef ADD_CPU_FREQ_TUNING
  it->set_enable_cpu_freq_change(true);
  it->set_clus0_cpu_freq(LP_cpu_freq[it->tasktype][0]);
  it->set_clus1_cpu_freq(LP_cpu_freq[it->tasktype][1]);
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] The CPU frequency scheduled for task " << it->taskid << ": cluster 0 is " << LP_cpu_freq[it->tasktype][0] << ", cluster 1 is " << LP_cpu_freq[it->tasktype][1] << std::endl;
  LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef ADD_DDR_FREQ_TUNING
  it->set_enable_ddr_freq_change(true);
  it->set_best_ddr_freq(LP_ddr_freq[it->tasktype]);
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] DDR frequency scheduled for task " << it->taskid << ": " << avail_ddr_freq[LP_ddr_freq[it->tasktype]] << std::endl;
  LOCK_RELEASE(output_lck);
#endif
#endif
  round_robin_counter[it->tasktype][nthread] = 0;
  DOP_detection[it->tasktype]--;
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] remaining_distri_num[" << it->kernel_name << "][" << nthread << "][0] = " << remaining_distri_num[it->tasktype][nthread][0] << ", remaining_distri_num[" << it->kernel_name \
  << "][" << nthread << "][1] = " << remaining_distri_num[it->tasktype][nthread][1] << ". DOP_detection[" << it->kernel_name << "] = " << DOP_detection[it->tasktype] << ". Round_robin_counter[" \
  << it->kernel_name << "][" << nthread << "] = " << round_robin_counter[it->tasktype][nthread] << ". For task " << it->taskid << ", it will be scheduled with leader core " << it->leader << " and width " << it->width << std::endl; 
  LOCK_RELEASE(output_lck);
#endif
  return it->leader;
}

int PolyTask::task_distri_HP(int nthread, PolyTask * it, int DOP){ /* Goal: minimize EDP => decide the resource width per task type and distribute DOP task among clusters */ 
  // int DOP = std::accumulate(std::begin(DOP_detection), std::begin(DOP_detection) + XITAO_MAXTHREADS, 0);  // DOP is the total number of tasks (inc multiple task types) that are currently ready and executing
  if(it->get_hp_task_distri_state() == false){
    /* Step 1: decide the resource width per task type for each cluster */
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ 
      HP_best_width[it->tasktype][clus_id] = 1; // initialize the best width for each cluster
      int i = 0;
      float timeP[XITAO_MAXTHREADS] = {0.0f};
      for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
        timeP[wid-1] = it->get_timetable(0, 0, clus_id, wid-1); // the first two parameters are both zero, since we only check the highest frequency
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[DEBUG] for cluster " << clus_id << ", timeP[" << i << "] = " << timeP[i] << std::endl; 
//       LOCK_RELEASE(output_lck);
// #endif       
        if(i > 0){
          if(timeP[0] / timeP[wid-1] > (wid)){ // if the execution time of width 1 is wid/1 times of execution time of the current width, then it achieves linearity
            HP_best_width[it->tasktype][clus_id] = wid; // best width is with moldability
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] Now achieves the linear speedup with more cores, the new best width becomes " << wid << std::endl; 
//           LOCK_RELEASE(output_lck);
// #endif   
          }
        }
        i++;
      }
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] The best resource width for " << it->kernel_name << " tasks on cluster 1 and 2 are <" << HP_best_width[it->tasktype][0] << "> and <" << HP_best_width[it->tasktype][1] << ">" << std::endl; 
    // std::cout << "[DEBUG] Task exec_time on cluster 1: " << it->get_timetable(0, 0, 0, HP_best_width[it->tasktype][0]-1) << ", task exec_time on cluster 2: " << it->get_timetable(0, 0, 1, HP_best_width[it->tasktype][1]-1) << std::endl;
    LOCK_RELEASE(output_lck);
#endif
    /* Step 2: start from using all cores and compute the corresonding EDP value */
    float timeP[XITAO_MAXTHREADS+1] = {0};
    float CPUPowerP[XITAO_MAXTHREADS+1] = {0};
    float DDRPowerP[XITAO_MAXTHREADS+1] = {0};
    float EDP[XITAO_MAXTHREADS+1] = {0};
    int edp_best_nc[NUMSOCKETS] = {0};
    float lowest_edp = 100000000.0f;
    for(int i = end_coreid[0] - start_coreid[0]; i > 0; i=i/2){  // i is cluster 0, e.g., on TX2, i = 2, 1
      if(i < HP_best_width[it->tasktype][0]){ // if the number of cores is less than the best width, then skip => at least one faster core is used
        break;
      }else{
        for(int j = end_coreid[1] - start_coreid[1]; j >= 0; j=j/2){ // j is cluster 1, e.g., on TX2, j = 4, 2, 1, 0
          timeP[i+j] = DOP / ( (i / (HP_best_width[it->tasktype][0] * it->get_timetable(0, 0, 0, HP_best_width[it->tasktype][0]-1))) + (j / (HP_best_width[it->tasktype][1] * it->get_timetable(0, 0, 1, HP_best_width[it->tasktype][1]-1))) );
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] i = " << i  << ", HP_best_width[it->tasktype][0] * it->get_timetable(0, 0, 0, HP_best_width[it->tasktype][0]-1) = " << HP_best_width[it->tasktype][0] * it->get_timetable(0, 0, 0, HP_best_width[it->tasktype][0]-1) << ", division = " << i / (HP_best_width[it->tasktype][0] * it->get_timetable(0, 0, 0, HP_best_width[it->tasktype][0]-1))\
//           << ", j = " << j << ", HP_best_width[it->tasktype][1] * it->get_timetable(0, 0, 1, HP_best_width[it->tasktype][1]-1 = " << HP_best_width[it->tasktype][1] * it->get_timetable(0, 0, 1, HP_best_width[it->tasktype][1]-1) << ", division = " << j / (HP_best_width[it->tasktype][1] * it->get_timetable(0, 0, 1, HP_best_width[it->tasktype][1]-1)) << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif                     
          if(j > 0){
            CPUPowerP[i+j] = it->get_cpupowertable(0, 0, 0, i-1) + it->get_cpupowertable(0, 0, 1, j-1) - (idle_cpu_power[0][0][0] + idle_cpu_power[0][0][1]); // The total CPU power consumption when using i+j cores
#ifdef Pro_fix_D
            DDRPowerP[i+j] = it->get_ddrpowertable(0, 0, 1, i-1) + it->get_ddrpowertable(0, 0, 1, j-1) - idle_ddr_power[0];
#else
            DDRPowerP[i+j] = it->get_ddrpowertable(0, 0, 0, i-1) + it->get_ddrpowertable(0, 0, 1, j-1) - idle_ddr_power[0];
#endif
          }else{
            CPUPowerP[i+j] = it->get_cpupowertable(0, 0, 0, i-1);
#ifdef Pro_fix_D
            DDRPowerP[i+j] = it->get_ddrpowertable(0, 0, 1, i-1);
#else
            DDRPowerP[i+j] = it->get_ddrpowertable(0, 0, 0, i-1);
#endif
          }

          EDP[i+j] = powf(timeP[i+j], DELAY_METRIC) * powf(CPUPowerP[i+j] + DDRPowerP[i+j], POWER_METRIC);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Using " << i << " cores in cluster 1 and " << j << " cores in cluster 2: predicted execution time = " << timeP[i+j] << ", predicted CPU power = " << CPUPowerP[i+j] \
          << ", predicted DDR power = " << DDRPowerP[i+j] << ", predicted Trade-off = " << EDP[i+j] << std::endl; 
          LOCK_RELEASE(output_lck);
#endif 
          if(EDP[i+j] < lowest_edp){
            lowest_edp = EDP[i+j];
            edp_best_nc[0] = i;
            edp_best_nc[1] = j;
          }

          if(j == 0){
            break;
          }
        }
      }
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] Using <" << edp_best_nc[0] << "> cores in cluster 1 and <" << edp_best_nc[1] << "> cores in cluster 2 achieves the lowest predicted Trade-off." << std::endl; 
    LOCK_RELEASE(output_lck);
#endif
    
    // it->set_stealing_range(end_coreid[0] - start_coreid[0], end_coreid[1] - start_coreid[1], edp_best_nc[0], edp_best_nc[1]);   // set the stealing core range for the kernel task
    /* Step 3: find out the best frequency setting that further reduces EDP */
    int edp_best_freq_idx[NUMSOCKETS] = {0};
    int edp_best_DDRfreq_idx = 0;
#ifdef ADD_CPU_FREQ_TUNING
    float freq_throttle_timeP[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0};
    float freq_throttle_CPUPowerP[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0};
    float freq_throttle_DDRPowerP[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0};
    float freq_throttle_edpP[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0};
    int visit_flag[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUM_AVAIL_FREQ] = {0}; // if the position has been visited before, it is -1,this is meant to avoid recalculate the EDP for the same positions in two steps
    visit_flag[0][0][0] = 1; // Current starting CPU frequency for cluster 0 and cluster 1 are highest frequency, the starting position has been computed, so set 1
    const int N = 9;
    int x[N] = {0, -1, -1, -1, 0, 0, 1, 1, 1}; // x and y arrays are meant to define 8 directions
    int y[N] = {0, -1, 0, 1, -1, 1, -1, 0, 1};
    int z[2] = {0};
    // int x[3] = {0, 1, 1}; // x and y arrays are meant to define 8 directions
    // int y[3] = {1, 0, 1};
    int step_progress = 0, cur_bestcpufreq_clus0 = 0, cur_bestcpufreq_clus1 = 0, starting_cpufreq_clus0 = 0, starting_cpufreq_clus1 = 0, cur_bestDDRfreq = 0, starting_DDRfreq = 0;
#ifdef ADD_DDR_FREQ_TUNING
    z[1] = 1;
#endif
    do{
      step_progress = 0;
      starting_DDRfreq = cur_bestDDRfreq; // starting DDR frequency is the best DDR frequency in the previous step
      for(int j = 0; j < 2; j++){
        if(starting_DDRfreq + z[j] >=0 && starting_DDRfreq + z[j] < NUM_DDR_AVAIL_FREQ){
          starting_cpufreq_clus0 = cur_bestcpufreq_clus0;
          starting_cpufreq_clus1 = cur_bestcpufreq_clus1;
          for(int i = 0; i < N; i++){
            if(starting_cpufreq_clus0 + x[i] >= 0 && starting_cpufreq_clus1 + y[i] >= 0 && starting_cpufreq_clus0 + x[i] < NUM_AVAIL_FREQ && starting_cpufreq_clus1 + y[i] < NUM_AVAIL_FREQ){
              if(visit_flag[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] == 0) { // If this position hasn't been visited yet
                freq_throttle_timeP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = \
                DOP / ( (edp_best_nc[0] / (HP_best_width[it->tasktype][0] * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, HP_best_width[it->tasktype][0]-1))) + (edp_best_nc[1] / (HP_best_width[it->tasktype][1] * it->get_timetable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, HP_best_width[it->tasktype][1]-1))) );
                if(edp_best_nc[1] > 0){
                // freq_throttle_CPUPowerP[starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = it->get_cpupowertable(0, starting_cpufreq_clus0 + x[i], 0, HP_best_width[it->tasktype][0]-1) + it->get_cpupowertable(0, starting_cpufreq_clus1 + y[i], 1, HP_best_width[it->tasktype][1]-1)- (idle_cpu_power[0][starting_cpufreq_clus0 + x[i]][0] + idle_cpu_power[0][starting_cpufreq_clus1 + y[i]][1]);
                  freq_throttle_CPUPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, edp_best_nc[0]-1) + it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, edp_best_nc[1]-1) - (idle_cpu_power[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][0] + idle_cpu_power[starting_DDRfreq + z[j]][starting_cpufreq_clus1 + y[i]][1]);
#ifdef Pro_fix_D
                  freq_throttle_DDRPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 1, edp_best_nc[0]-1) + it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, edp_best_nc[1]-1) - idle_ddr_power[starting_DDRfreq + z[j]];
#else                  
                  freq_throttle_DDRPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, edp_best_nc[0]-1) + it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus1 + y[i], 1, edp_best_nc[1]-1) - idle_ddr_power[starting_DDRfreq + z[j]];
#endif
    // #ifdef DEBUG
    //               LOCK_ACQUIRE(output_lck);        
    //               std::cout << "[DEBUG] When tuning CPU frequency to <" << avail_freq[0][starting_cpufreq_clus0 + x[i]] << ", " << avail_freq[1][starting_cpufreq_clus1 + y[i]] << ">, the predicted CPU power consumption is " << it->get_cpupowertable(0, starting_cpufreq_clus0 + x[i], 0, edp_best_nc[0]-1) << " + " << it->get_cpupowertable(0, starting_cpufreq_clus1 + y[i], 1, edp_best_nc[1]-1) << " - " << \
    //               idle_cpu_power[0][starting_cpufreq_clus0 + x[i]][0] << " - " << idle_cpu_power[0][starting_cpufreq_clus1 + y[i]][1] << " = " << freq_throttle_CPUPowerP[starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]]  << std::endl;
    //               LOCK_RELEASE(output_lck);
    // #endif
                }else{
                  freq_throttle_CPUPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = it->get_cpupowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, HP_best_width[it->tasktype][0]-1);
#ifdef Pro_fix_D
                  freq_throttle_DDRPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 1, HP_best_width[it->tasktype][0]-1);
#else
                  freq_throttle_DDRPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = it->get_ddrpowertable(starting_DDRfreq + z[j], starting_cpufreq_clus0 + x[i], 0, HP_best_width[it->tasktype][0]-1);
#endif
                }
                freq_throttle_edpP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] =\
                powf(freq_throttle_timeP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]], DELAY_METRIC) * \
                powf((freq_throttle_CPUPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] + freq_throttle_DDRPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]]), POWER_METRIC);
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] When tuning Mem frequency to " << avail_ddr_freq[starting_DDRfreq + z[j]] << " and CPU frequency to <" << avail_freq[0][starting_cpufreq_clus0 + x[i]] << ", " << avail_freq[1][starting_cpufreq_clus1 + y[i]] << ">, the predicted execution time is " << freq_throttle_timeP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] << ", the predicted CPU power consumption is " << \
                freq_throttle_CPUPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] << ", the predicted DDR power consumption is " << freq_throttle_DDRPowerP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] << ", the predicted trade-off is " << freq_throttle_edpP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] << std::endl; 
                LOCK_RELEASE(output_lck);
#endif            
                visit_flag[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] = 1; // visited
                if(freq_throttle_edpP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]] < lowest_edp){
                  lowest_edp = freq_throttle_edpP[starting_DDRfreq + z[j]][starting_cpufreq_clus0 + x[i]][starting_cpufreq_clus1 + y[i]];
                  cur_bestcpufreq_clus0 = starting_cpufreq_clus0 + x[i]; // Update new starting frequency index per step
                  cur_bestcpufreq_clus1 = starting_cpufreq_clus1 + y[i];
                  cur_bestDDRfreq = starting_DDRfreq + z[j];
                  step_progress = 1;
                }
              }
            }
          }
        }
      }
    }while(step_progress == 1);
    edp_best_freq_idx[0] = cur_bestcpufreq_clus0;
    edp_best_freq_idx[1] = cur_bestcpufreq_clus1;
    edp_best_DDRfreq_idx = cur_bestDDRfreq;
//     for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){ // old approach
//       int k = 0;
//       for(int i = 0; i < 2; i++){
//         for(int j = 0; j < 2; j++){
//           if(i+j == 0){
//             continue; 
//           }else{
//             freq_throttle_timeP[freq_indx+i][freq_indx+j] = DOP / ( (edp_best_nc[0] / (HP_best_width[it->tasktype][0] * it->get_timetable(0, freq_indx+i, 0, HP_best_width[it->tasktype][0]-1))) + (edp_best_nc[1] / (HP_best_width[it->tasktype][1] * it->get_timetable(0, freq_indx+j, 1, HP_best_width[it->tasktype][1]-1))) );
//             if(edp_best_nc[1] > 0){
//               // freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] = it->get_cpupowertable(0, freq_indx+i, 0, HP_best_width[it->tasktype][0]-1) + it->get_cpupowertable(0, freq_indx+j, 1, HP_best_width[it->tasktype][1]-1)- (idle_cpu_power[0][freq_indx+i][0] + idle_cpu_power[0][freq_indx+j][1]);
//               freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] = it->get_cpupowertable(0, freq_indx+i, 0, edp_best_nc[0]-1) + it->get_cpupowertable(0, freq_indx+j, 1, edp_best_nc[1]-1) - (idle_cpu_power[0][freq_indx+i][0] + idle_cpu_power[0][freq_indx+j][1]);
//               freq_throttle_DDRPowerP[freq_indx+i][freq_indx+j] = it->get_ddrpowertable(0, freq_indx+i, 0, edp_best_nc[0]-1) + it->get_ddrpowertable(0, freq_indx+j, 1, edp_best_nc[1]-1) - idle_ddr_power[0];
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);        
//               std::cout << "[DEBUG] When tuning CPU frequency to <" << avail_freq[0][freq_indx+i] << ", " << avail_freq[1][freq_indx+j] << ">, the predicted CPU power consumption is " << it->get_cpupowertable(0, freq_indx+i, 0, edp_best_nc[0]-1) << " + " << it->get_cpupowertable(0, freq_indx+j, 1, edp_best_nc[1]-1) << " - " << \
//               idle_cpu_power[0][freq_indx+i][0] << " - " << idle_cpu_power[0][freq_indx+j][1] << " = " << freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j]  << std::endl;
//               LOCK_RELEASE(output_lck);
// #endif
//             }else{
//               freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] = it->get_cpupowertable(0, freq_indx+i, 0, HP_best_width[it->tasktype][0]-1);
//               freq_throttle_DDRPowerP[freq_indx+i][freq_indx+j] = it->get_ddrpowertable(0, freq_indx+i, 0, HP_best_width[it->tasktype][0]-1);
//             }
//             freq_throttle_edpP[freq_indx+i][freq_indx+j] = powf(freq_throttle_timeP[freq_indx+i][freq_indx+j], 2.0) * (freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] + freq_throttle_DDRPowerP[freq_indx+i][freq_indx+j]);
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[DEBUG] When tuning CPU frequency to <" << avail_freq[0][freq_indx+i] << ", " << avail_freq[1][freq_indx+j] << ">, the predicted execution time is " << freq_throttle_timeP[freq_indx+i][freq_indx+j] << ", the predicted CPU power consumption is " << \
//             freq_throttle_CPUPowerP[freq_indx+i][freq_indx+j] << ", the predicted DDR power consumption is " << freq_throttle_DDRPowerP[freq_indx+i][freq_indx+j] << ", the predicted EDP is " << freq_throttle_edpP[freq_indx+i][freq_indx+j] << std::endl; 
//             LOCK_RELEASE(output_lck);
// #endif            
//             if(freq_throttle_edpP[freq_indx+i][freq_indx+j] < lowest_edp){
//               k++;
//               lowest_edp = freq_throttle_edpP[freq_indx+i][freq_indx+j];
//               edp_best_freq_idx[0] = freq_indx+i;
//               edp_best_freq_idx[1] = freq_indx+j;
//             }
//           }
//         }
//       }
//       if(k == 0 || freq_indx == NUM_AVAIL_FREQ-1){
//         break;
//       }
//     }
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Best frequency setting for " << it->kernel_name << " tasks are <" << avail_ddr_freq[edp_best_DDRfreq_idx] << ", " << avail_freq[0][edp_best_freq_idx[0]] << ", " << avail_freq[1][edp_best_freq_idx[1]] << ">" << std::endl; 
        LOCK_RELEASE(output_lck);
#endif
#endif
    /* Step 4: probability distribution of all parallel tasks */
    float clus_part[NUMSOCKETS] = {0.0f};
    float suppose_load_balance = 0.0f;
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
      clus_part[clus_id] = (it->get_timetable(edp_best_DDRfreq_idx, edp_best_freq_idx[1], 1, HP_best_width[it->tasktype][1]-1) / it->get_timetable(edp_best_DDRfreq_idx, edp_best_freq_idx[clus_id], clus_id, HP_best_width[it->tasktype][clus_id]-1)) * (float(edp_best_nc[clus_id]) / HP_best_width[it->tasktype][clus_id]);
      suppose_load_balance += clus_part[clus_id];
    }
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
      percent_cluster[it->tasktype][clus_id] = clus_part[clus_id] / suppose_load_balance;
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      // std::cout << "[JING] cluster 0 time: " << it->get_timetable(edp_best_DDRfreq_idx, edp_best_freq_idx[0], 0, HP_best_width[it->tasktype][0]-1) << ", cluster 1 time: " \
      << it->get_timetable(edp_best_DDRfreq_idx, edp_best_freq_idx[1], 1, HP_best_width[it->tasktype][1]-1) << ", clus_part[0] = " << clus_part[0] << ", clus_part[1] = " << clus_part[1] << std::endl;
      std::cout << "[DEBUG] For " << DOP << " " << it->kernel_name << " tasks, " << percent_cluster[it->tasktype][clus_id] * 100 << "% of tasks should be scheduled to cluster " << clus_id << std::endl; 
      LOCK_RELEASE(output_lck);
#endif
    }
#ifdef ADD_CPU_FREQ_TUNING
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
      HP_cpu_freq[it->tasktype][clus_id] = edp_best_freq_idx[clus_id];
    }
    // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ // should tune frequency before executing tasks  
    //   it->cpu_frequency_tuning(nthread, clus_id, start_coreid[clus_id], end_coreid[clus_id] - start_coreid[clus_id], edp_best_freq_idx[clus_id]); // set the best frequency setting for the kernel tasks
    // }
#endif
#ifdef ADD_DDR_FREQ_TUNING
    HP_ddr_freq[it->tasktype] = edp_best_DDRfreq_idx;
#endif
    it->set_hp_task_distri_state(true); /* Comment it and do the computation every DOP tasks. Otherwise, we only calculate all the parameters once, then next time just use the parameters */
#ifdef INTERVAL_RECOMPUTATION  
    interval_t2 = std::chrono::system_clock::now();
#endif
  }
  /* Step 5: try to schedule tasks based on the DOP and percentage of task distribution */
  // int task_distri_proposal[XITAO_MAXTHREADS][NUMSOCKETS] = {0};
  // task_distri_proposal[it->tasktype][0] = floor(DOP * percent_cluster[it->tasktype][0] * (DOP_detection[it->tasktype] / DOP) + 0.5);
  // task_distri_proposal[it->tasktype][1] = floor(DOP * percent_cluster[it->tasktype][1] * (DOP_detection[it->tasktype] / DOP) + 0.5);
  // task_distri_proposal[it->tasktype][0] = ceil(DOP * percent_cluster[it->tasktype][0] * (DOP_detection[it->tasktype] / DOP));
  // task_distri_proposal[it->tasktype][1] = DOP_detection[it->tasktype] - task_distri_proposal[it->tasktype][0];
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[DEBUG] The total: Task distribution across clusters for " << it->kernel_name << " tasks are <" << task_distri_proposal[it->tasktype][0] << "> for cluster 1 and <" << task_distri_proposal[it->tasktype][1] << "> for cluster 2" << std::endl; 
//   LOCK_RELEASE(output_lck);
// #endif
  remaining_distri_num[it->tasktype][nthread][0] = ceil(DOP * percent_cluster[it->tasktype][0]); // ceil(DOP * percent_cluster[it->tasktype][0] * float(DOP_detection[it->tasktype] / DOP));
  remaining_distri_num[it->tasktype][nthread][1] = DOP - remaining_distri_num[it->tasktype][nthread][0];
  // remaining_distri_num[it->tasktype][0] = floor(DOP * percent_cluster[it->tasktype][0] * (DOP_detection[it->tasktype] / DOP) + 0.5);
  // remaining_distri_num[it->tasktype][1] = floor(DOP * percent_cluster[it->tasktype][1] * (DOP_detection[it->tasktype] / DOP) + 0.5);
  // for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
    // remaining_distri_num[it->tasktype][clus_id] = (DOP_executing[clus_id] > 0) ? task_distri_proposal[it->tasktype][clus_id] - DOP_executing[clus_id] : task_distri_proposal[it->tasktype][clus_id];
    // remaining_distri_num[it->tasktype][clus_id] = task_distri_proposal[it->tasktype][clus_id];
  // }
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] To be scheduled: Task distribution for " << DOP << " " << it->kernel_name << " tasks are <" << remaining_distri_num[it->tasktype][nthread][0] << "> for cluster 1 and <" << remaining_distri_num[it->tasktype][nthread][1] << "> for cluster 2" << std::endl; 
  LOCK_RELEASE(output_lck);
#endif
  // if(DOP_executing[0] < task_distri_proposal[it->tasktype][0]){ // if currently executing tasks are less than the expected number of tasks, then schedule more tasks to cluster A
  if(remaining_distri_num[it->tasktype][nthread][0] > 0){
    it->width = HP_best_width[it->tasktype][0];
    it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    remaining_distri_num[it->tasktype][nthread][0]--;
  }else{ // otherwise, schedule tasks to cluster B
    it->width = HP_best_width[it->tasktype][1];
    it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    remaining_distri_num[it->tasktype][nthread][1]--;
    // round_robin_counter[it->tasktype][nthread] = 1;
  }
#ifdef ADD_CPU_FREQ_TUNING
  it->set_enable_cpu_freq_change(true);
  it->set_clus0_cpu_freq(HP_cpu_freq[it->tasktype][0]);
  it->set_clus1_cpu_freq(HP_cpu_freq[it->tasktype][1]);
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] CPU frequency scheduled for task " << it->taskid << ": <" << avail_freq[0][HP_cpu_freq[it->tasktype][0]] << ", " << avail_freq[1][HP_cpu_freq[it->tasktype][1]] << ">"<< std::endl;
  LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef ADD_DDR_FREQ_TUNING
  it->set_enable_ddr_freq_change(true);
  it->set_best_ddr_freq(HP_ddr_freq[it->tasktype]);
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] DDR frequency scheduled for task " << it->taskid << ": " << avail_ddr_freq[HP_ddr_freq[it->tasktype]] << std::endl;
  LOCK_RELEASE(output_lck);
#endif
#endif
  round_robin_counter[it->tasktype][nthread] = 0;
  DOP_detection[it->tasktype]--;
#ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[DEBUG] remaining_distri_num[" << it->kernel_name << "][" << nthread << "][0] = " << remaining_distri_num[it->tasktype][nthread][0] << ", remaining_distri_num[" << it->kernel_name \
  << "][" << nthread << "][1] = " << remaining_distri_num[it->tasktype][nthread][1] << ". DOP_detection[" << it->kernel_name << "] = " << DOP_detection[it->tasktype] << ". Round_robin_counter[" \
  << it->kernel_name << "][" << nthread << "] = " << round_robin_counter[it->tasktype][nthread] << ". For task " << it->taskid << ", it will be scheduled with leader core " << it->leader << " and width " << it->width << std::endl; 
  LOCK_RELEASE(output_lck);
#endif
  return it->leader;
}

int PolyTask::assign_config(int nthread, PolyTask * it){ // Assign the cluster for current ready tasks in a round robin manner
  for(; ;){
    int cluster = (++round_robin_counter[it->tasktype][nthread]) % NUMSOCKETS;
    if(remaining_distri_num[it->tasktype][nthread][cluster] > 0) {
      if(interval_HP[it->tasktype][nthread] == true){
        it->width = HP_best_width[it->tasktype][cluster];
#ifdef ADD_CPU_FREQ_TUNING
        it->set_enable_cpu_freq_change(true);
        it->set_clus0_cpu_freq(HP_cpu_freq[it->tasktype][0]);
        it->set_clus1_cpu_freq(HP_cpu_freq[it->tasktype][1]);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] CPU frequency scheduled for task " << it->taskid << ": <" << avail_freq[0][HP_cpu_freq[it->tasktype][0]] << ", " << avail_freq[1][HP_cpu_freq[it->tasktype][1]] << ">"<< std::endl;
        LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef ADD_DDR_FREQ_TUNING
        it->set_enable_ddr_freq_change(true);
        it->set_best_ddr_freq(HP_ddr_freq[it->tasktype]);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] DDR frequency scheduled for task " << it->taskid << ": " << avail_ddr_freq[HP_ddr_freq[it->tasktype]] << std::endl;
        LOCK_RELEASE(output_lck);
#endif
#endif
      }else{
        if(interval_LP[it->tasktype][nthread] == true){
          it->width = LP_best_width[it->tasktype][cluster];
#ifdef ADD_CPU_FREQ_TUNING
          it->set_enable_cpu_freq_change(true);
          it->set_clus0_cpu_freq(LP_cpu_freq[it->tasktype][0]);
          it->set_clus1_cpu_freq(LP_cpu_freq[it->tasktype][1]);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] CPU frequency scheduled for task " << it->taskid << ": <" << avail_freq[0][LP_cpu_freq[it->tasktype][0]] << ", " << avail_freq[0][LP_cpu_freq[it->tasktype][1]] << ">" << std::endl;
          LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef ADD_DDR_FREQ_TUNING
          it->set_enable_ddr_freq_change(true);
          it->set_best_ddr_freq(LP_ddr_freq[it->tasktype]);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] DDR frequency scheduled for task " << it->taskid << ": " << avail_ddr_freq[LP_ddr_freq[it->tasktype]] << std::endl;
          LOCK_RELEASE(output_lck);
#endif
#endif
        }
      }
      it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster]-start_coreid[cluster])/it->width)) * it->width;
      remaining_distri_num[it->tasktype][nthread][cluster]--;
      DOP_detection[it->tasktype]--;
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[DEBUG] remaining_distri_num[" << it->kernel_name << "][" << nthread << "][0] = " << remaining_distri_num[it->tasktype][nthread][0] <<", remaining_distri_num[" << it->kernel_name << "][" \
      << nthread << "][1] = " << remaining_distri_num[it->tasktype][nthread][1] << ". DOP_detection[" << it->kernel_name << "] = " << DOP_detection[it->tasktype] << ". Round_robin_counter[" << it->kernel_name \
      << "][" << nthread << "] = " << round_robin_counter[it->tasktype][nthread] << ". For task " << it->taskid << ", it will be scheduled with leader core " << it->leader << " and width " << it->width << std::endl; 
      LOCK_RELEASE(output_lck);
#endif
      break;
    }else{
      continue;
    }
  }
  return it->leader;
}

int PolyTask::find_best_config(int nthread, PolyTask * it){ /* The kernel task hasn't got the best config yet, three loops to search for the best configs. */
#if defined Search_Overhead    
  struct timespec start2, finish2, delta2;
  clock_gettime(CLOCK_REALTIME, &start2);
#endif
  float shortest_exec = 100000.0f;
#ifdef perf_improve
  float optimalconfig_time = 100000.0f; 
  float energy_prediction[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS] = {0.0f}; 
#endif
  float energy_pred = 0.0f;
  float idleP_cluster = 0.0f;
  int sum_cluster_active[NUMSOCKETS] = {0}; // number of active cores in each cluster, meant to compute the static CPU power distribution on each concurrent task
  int sum_active_cores = 0;                 // number of active cores in total, meant to compute the static DDR power distribution on each concurrent task
  // float idle_ddr_power[NUM_DDR_AVAIL_FREQ] = {1.152, 0.73, 0.346, 0.269, 0.211};  // Profile the static DDR power of TX2 through the power reading when system is idle, unit is watt
  for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){ /* Get the number of active cores in each cluster */
    sum_cluster_active[clus_id] = std::accumulate(status+start_coreid[clus_id], status+end_coreid[clus_id], 0); 
    //sum_active_cores += sum_cluster_active[clus_id];
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] Number of active cores in cluster " << clus_id << ": " << sum_cluster_active[clus_id] << std::endl; 
    // << ". status[0] = " << status[0] << ", status[1] = " << status[1] << ", status[2] = " << status[2] << ", status[3] = " << status[3] << ", status[4] = " << status[4] << ", status[5] = " << status[5] 
    LOCK_RELEASE(output_lck);
#endif 
  }

  /* Step 1: first standard: check if the execution time of (current frequency of fastest cluster, fastest cluster, 2) < 1 ms? --- define as Fine-grained tasks --- Find out best core type, number of cores, doesn't change frequency */
  /* After finding out the best config, double check if it is fine-grained? TBD */
  if(it->get_timetable(cur_ddr_freq_index, cur_freq_index[0], 0, 1) < FINE_GRAIN_THRESHOLD){
    // it->previous_tasktype = it->tasktype; /* First fine-grained task goes here and set its task type as the previous task type for next incoming task */
    for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
      if(sum_cluster_active[1-clus_id] == 0){ /* the number of active cores is zero in another cluster */
        idleP_cluster = idle_cpu_power[cur_ddr_freq_index][cur_freq_index[clus_id]][clus_id] + idle_cpu_power[cur_ddr_freq_index][cur_freq_index[1-clus_id]][1-clus_id]; /* Then equals idle power of whole chip */
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Cluster " << 1-clus_id << " no active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
        LOCK_RELEASE(output_lck);
#endif 
      }else{
        idleP_cluster = idle_cpu_power[cur_ddr_freq_index][cur_freq_index[clus_id]][clus_id]; /* otherwise, equals idle power of the cluster */
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Cluster " << 1-clus_id << " has active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
        LOCK_RELEASE(output_lck);
#endif 
      }
      for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
        sum_cluster_active[clus_id] = (sum_cluster_active[clus_id] < wid)? wid : sum_cluster_active[clus_id];
        float idleP = idleP_cluster * wid / sum_cluster_active[clus_id];
        float CPUPowerP = it->get_cpupowertable(cur_ddr_freq_index, cur_freq_index[clus_id], clus_id, wid-1);
        float DDRPowerP = it->get_ddrpowertable(cur_ddr_freq_index, cur_freq_index[clus_id], clus_id, wid-1);
        float timeP = it->get_timetable(cur_ddr_freq_index, cur_freq_index[clus_id], clus_id, wid-1);
        energy_pred = timeP * (CPUPowerP - idleP_cluster + idleP + DDRPowerP);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Fine-grained task " << it->taskid << ": Frequency: " << cur_freq[clus_id] << ", cluster " << clus_id << ", width "<< wid << ", sum_cluster_active = " \
        << sum_cluster_active[clus_id] << ", idle power " << idleP << ", CPU power " << CPUPowerP << ", memory power " << DDRPowerP << ", execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
        LOCK_RELEASE(output_lck);
#endif 
        if(energy_pred < shortest_exec){
          shortest_exec = energy_pred;
          it->set_best_core_type(clus_id);
          it->set_best_numcores(wid);
        }
      }
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] For current frequency: " << cur_freq[it->get_best_cluster()] << ", best cluster: " << it->get_best_cluster() << ", best width: " << it->get_best_numcores() << std::endl;
    LOCK_RELEASE(output_lck);
#endif 
    // if(it->get_timetable(cur_freq_index[best_cluster], best_cluster, best_width) < FINE_GRAIN_THRESHOLD){ /* Double confirm that the task using the best config is still fine-grained */
    it->set_enable_cpu_freq_change(false); /* No DFS */
    it->set_enable_ddr_freq_change(false); /* No DFS */
    it->granularity_fine = true; /* Mark it as fine-grained task */
    // }else{ /* It is not */
    //   it->granularity_fine = true;
    // }
    }else{ /* Coarse-grained tasks --- Find out best frequency, core type, number of cores */
#ifdef perf_contraints
      float ref_perf = it->get_timetable(0, 0, 0, 1); // Four args: ddr_freq_indx = 0: highest DDR frequency, freq_indx = 0: highest CPU frequency, clus_id = 0: Denver cluster, wid-1 = 1: with 2 Denver cores
#endif
#ifdef JOSS_NoMemDVFS
      for(int ddr_freq_indx = 0; ddr_freq_indx < 1; ddr_freq_indx++) // JOSS without memory DVFS tuning knob, thereby fixing the ddr freq at the highest
#else
      for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++)
#endif
      {
        for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
          for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
          if(sum_cluster_active[1-clus_id] == 0){ /* the number of active cores is zero in another cluster */
            idleP_cluster = idle_cpu_power[ddr_freq_indx][freq_indx][clus_id] + idle_cpu_power[ddr_freq_indx][freq_indx][1-clus_id]; /* Then equals idle power of whole chip */
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[DEBUG] Cluster " << 1-clus_id << " no active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of whole chip " << idleP_cluster << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif 
          }else{
            idleP_cluster = idle_cpu_power[ddr_freq_indx][freq_indx][clus_id]; /* otherwise, equals idle power of the cluster */
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "[DEBUG] Cluster " << 1-clus_id << " has active cores. Therefore, the idle power of cluster " << clus_id << " euqals the idle power of the cluster itself " << idleP_cluster << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif 
          }
          for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
	          float timeP = it->get_timetable(ddr_freq_indx, freq_indx, clus_id, wid-1);
#ifdef perf_contraints
            if(timeP > ref_perf * (1 + PERF_SLOWDOWN)){
#ifdef DEBUG
	      LOCK_ACQUIRE(output_lck);
	      std::cout << "[DEBUG] Memory frequency: " << avail_ddr_freq[ddr_freq_indx] <<  ", CPU frequency: " << avail_freq[clus_id][freq_indx] << ", cluster " << clus_id << ", width "<< wid << ", execution time exceeds the constraint! Skip! \n";
	      LOCK_RELEASE(output_lck);
#endif
              continue; // jump to next loop since the performance is over PERF_SLOWDOWN
            }else{    
#endif
            sum_cluster_active[clus_id] = (sum_cluster_active[clus_id] < wid)? wid : sum_cluster_active[clus_id];
            float idleP_cpu = idleP_cluster * wid / sum_cluster_active[clus_id]; // Static CPU power distribution on this task
            float CPUPowerP = it->get_cpupowertable(ddr_freq_indx, freq_indx, clus_id, wid-1); /* includes idle power + runtime power, so when computing energy, we need to remove the replicated idle power from concurrent tasks */
            float CPUPower = (CPUPowerP - idleP_cluster > 0)? CPUPowerP - idleP_cluster + idleP_cpu : idleP_cpu;
            // for(int c_id = 0; c_id < NUMSOCKETS; c_id++){
            //   sum_active_cores += sum_cluster_active[c_id];
            // }
#ifdef DDR_FREQ_TUNING
            float idleP_ddr = idle_ddr_power[ddr_freq_indx] * wid / sum_cluster_active[clus_id]; // Static DDR power distribution on this task
            float DDRPowerP = it->get_ddrpowertable(ddr_freq_indx, freq_indx, clus_id, wid-1);
            float DDRPower = (DDRPowerP - idle_ddr_power[ddr_freq_indx] > 0)? DDRPowerP - idle_ddr_power[ddr_freq_indx] + idleP_ddr : idleP_ddr;
// #ifdef DEBUG
//             LOCK_ACQUIRE(output_lck);
//             std::cout << "idleP_ddr = " << idleP_ddr << ". DDR power from table = " << DDRPowerP << ", Final DDR power = " << DDRPower << std::endl;
//             LOCK_RELEASE(output_lck);
// #endif 
#endif
            float energy_pred = 0.0;
#ifdef EDP_PER_TASK
            energy_pred = timeP * timeP * CPUPower; // JOSS-default to minimize EDP per task
#ifdef DDR_FREQ_TUNING
            energy_pred += timeP * timeP * DDRPower; 
#endif
#else
#ifdef perf_improve
            energy_prediction[ddr_freq_indx][freq_indx][clus_id][wid] = timeP * CPUPower;
#ifdef DDR_FREQ_TUNING
            energy_prediction[ddr_freq_indx][freq_indx][clus_id][wid] += timeP * DDRPower;
#endif
#else
            energy_pred = timeP * CPUPower; // JOSS-default to minimize energy consumption per task
#ifdef DDR_FREQ_TUNING
            energy_pred += timeP * DDRPower;
#endif
#endif
#endif
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[JOSS] Memory frequency: " << avail_ddr_freq[ddr_freq_indx] <<  ", CPU frequency: " << avail_freq[clus_id][freq_indx] << ", cluster " << clus_id << ", width "<< wid << ", sum_cluster_active = " << sum_cluster_active[clus_id] \
            << ", CPU power " << CPUPower << ", ";
#ifdef DDR_FREQ_TUNING            
            std::cout << "Memory power " << DDRPower << ", ";
#endif            
            std::cout << "execution time " << timeP << ", energy prediction: " << energy_pred << std::endl;
            LOCK_RELEASE(output_lck);
#endif 
#ifdef perf_improve
            if(energy_prediction[ddr_freq_indx][freq_indx][clus_id][wid] < shortest_exec){
              optimalconfig_time = timeP; 
              shortest_exec = energy_prediction[ddr_freq_indx][freq_indx][clus_id][wid];
#else
            if(energy_pred < shortest_exec){
              shortest_exec = energy_pred;
#endif
              it->set_best_ddr_freq(ddr_freq_indx);
              it->set_best_cpu_freq(freq_indx);
              it->set_best_core_type(clus_id);
              it->set_best_numcores(wid);
            }
#ifdef perf_contraints
          }
#endif
          }
        }
      }
    }
    it->set_enable_cpu_freq_change(true);
    it->set_enable_ddr_freq_change(true);
// #ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[JOSS] " << it->kernel_name << ": the optimal cluster Memory and CPU frequency: " << avail_ddr_freq[it->get_best_ddr_freq()] << ", " << avail_freq[it->get_best_cluster()][it->get_best_cpu_freq()] \
    << ", best cluster: " << it->get_best_cluster() << ", best width: " << it->get_best_numcores() << std::endl;
    LOCK_RELEASE(output_lck);
// #endif 
#ifdef perf_improve
    float AllowTime = optimalconfig_time / PERF_SPEEDUP;
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << ": Execution time of the most energy efficient config is " << optimalconfig_time << ". With " << PERF_SPEEDUP << "X, execution time should be within " << AllowTime << std::endl;
    LOCK_RELEASE(output_lck);
#endif 
    if(AllowTime < std::min(it->get_timetable(0, 0, 0, 1), it->get_timetable(0, 0, 1, 3))){ // If performance improvenents is too much, then just run with the fastest one
      it->set_best_ddr_freq(0);
      it->set_best_cpu_freq(0);
      it->set_best_core_type(0);
      it->set_best_numcores(2);
    }else{
      shortest_exec = 100000.0f;
      for(int ddr_freq_indx = 0; ddr_freq_indx < NUM_DDR_AVAIL_FREQ; ddr_freq_indx++){
        for(int freq_indx = 0; freq_indx < NUM_AVAIL_FREQ; freq_indx++){
          for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
            for(auto&& wid : ptt_layout[start_coreid[clus_id]]){
              float timeP = it->get_timetable(ddr_freq_indx, freq_indx, clus_id, wid-1);
              if(timeP <= AllowTime && energy_prediction[ddr_freq_indx][freq_indx][clus_id][wid] < shortest_exec){
                shortest_exec = energy_prediction[ddr_freq_indx][freq_indx][clus_id][wid];
                it->set_best_ddr_freq(ddr_freq_indx);
                it->set_best_cpu_freq(freq_indx);
                it->set_best_core_type(clus_id);
                it->set_best_numcores(wid);
              }
            }
          }
        }
      }
    }
// #ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << ": For " << PERF_SPEEDUP << "X performance improvement, the new optimal cluster Memory and CPU frequency: " << avail_ddr_freq[it->get_best_ddr_freq()] << ", " << avail_freq[it->get_best_cpu_freq()] << ", best cluster: " << it->get_best_cluster() << ", best width: " << it->get_best_numcores() << std::endl;
    LOCK_RELEASE(output_lck);
// #endif    
#endif    
  }
#if defined Search_Overhead
  clock_gettime(CLOCK_REALTIME, &finish2);
  poly_sub_timespec(start2, finish2, &delta2);
  printf("[Overhead] This part is: %d.%.9ld\n", (int)delta2.tv_sec, delta2.tv_nsec);
#endif  
  it->set_bestconfig_state(true);
  it->get_bestconfig = true; // the reason of adding this: check if task itself gets the best config or not, since not all tasks are released because of dependencies (e.g., alya, k-means, dot product)
  it->width = it->get_best_numcores();
  int best_core_type = it->get_best_cluster();
  it->leader = start_coreid[best_core_type] + (rand() % ((end_coreid[best_core_type] - start_coreid[best_core_type])/it->width)) * it->width;
  return it->leader;
}

/* After the searching, the kernel task got the best configs: best frequency, best cluster and best width, new incoming tasks directly use the best config. */
int PolyTask::update_best_config(int nthread, PolyTask * it){ 
  it->width = it->get_best_numcores();
  int best_core_type = it->get_best_cluster();
  it->leader = start_coreid[best_core_type] + (rand() % ((end_coreid[best_core_type] - start_coreid[best_core_type])/it->width)) * it->width;
  it->get_bestconfig = true;
// #ifdef DEBUG
//   LOCK_ACQUIRE(output_lck);
//   std::cout << "[FRANZ] it->width: " << it->width << ", best_core_type: " << best_core_type << ", cur_ddr_freq_index: " << cur_ddr_freq_index << ", cur_freq_index[best_core_type]: " << cur_freq_index[best_core_type] << std::endl;
//   LOCK_RELEASE(output_lck);
// #endif
  if(it->get_timetable(cur_ddr_freq_index, cur_freq_index[it->leader], best_core_type, it->width - 1) < FINE_GRAIN_THRESHOLD){ /* Incoming tasks are fine-grained */ 
    it->granularity_fine = true; /* Mark it as fine-grained task */
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    if(it->get_enable_cpu_freq_change() == false){
      std::cout << "[DEBUG] For fine-grained task " << it->taskid << ", current frequency of leader core: " << cur_freq[it->leader] << ", best core type: " << it->get_best_cluster() \
      << ", best width: " << it->get_best_numcores() << ", it->get_timetable = " \
      << it->get_timetable(cur_ddr_freq_index, cur_freq_index[it->leader], best_core_type, it->width - 1) << std::endl;
    }else{
      std::cout << "[DEBUG] For fine-grained task " << it->taskid << ", BEST CPU frequency: " << avail_freq[best_core_type][it->get_best_cpu_freq()] << ", BEST Memory frequency: " << avail_ddr_freq[it->get_best_ddr_freq()] \
      << ", best core type: " << it->get_best_cluster() << ", best width: " << it->get_best_numcores() << ", it->get_timetable = " << it->get_timetable(cur_ddr_freq_index, cur_freq_index[it->leader], best_core_type, it->width - 1) << std::endl;
    }
    LOCK_RELEASE(output_lck);
#endif 
  }
  else{ /* Incoming tasks are coarse-grained */ 
    // it-> = it->get_best_cpu_freq();
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] For task " << it->taskid << ", BEST CPU frequency: " << avail_freq[best_core_type][it->get_best_cpu_freq()] << ", BEST Memory frequency: " << avail_ddr_freq[it->get_best_ddr_freq()] \
      << ", it->leader: " << it->leader << ", it->width: " << it->width << ".\n";
    LOCK_RELEASE(output_lck);
#endif    
  }
  return it->leader;
}

int PolyTask::JOSS_for_Energy(int nthread, PolyTask * it){
  float comp_perf = 0.0f;
  if(it->tasktype < num_kernels){
  // if(it->get_timetable_state(2) == false){     /* PTT training is not finished yet */
  if(global_training == false){
    for(int cluster = 0; cluster < NUMSOCKETS; ++cluster) {
      for(auto&& width : ptt_layout[start_coreid[cluster]]) {
        auto&& ptt_val = 0.0f;
        ptt_val = it->get_timetable(0, ptt_freq_index[cluster], cluster, width - 1);
#ifdef TRAIN_METHOD_1 /* Allow NUM_TRAIN_TASKS tasks to train the same config, pros: training is faster, cons: not apply to memory-bound tasks */
        if(it->get_PTT_UpdateFlag(ptt_freq_index[cluster], cluster, width-1) < NUM_TRAIN_TASKS){
          it->width  = width;
          it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster] - start_coreid[cluster])/width)) * width;
          it->increment_PTT_UpdateFlag(ptt_freq_index[cluster],cluster,width-1);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] " << it->kernel_name <<"->Timetable(" << avail_ddr_freq[0] << ", " << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = " << ptt_val << ". Run with [" << it->leader << ", " << it->width << ")." << std::endl;
          LOCK_RELEASE(output_lck);
#endif
          return it->leader;
        }else{
          continue;
        }  
#endif
#ifdef TRAIN_METHOD_2 /* Allow DOP tasks to train the same config, pros: also apply to memory-bound tasks, cons: training might be slower */
        if(ptt_val == 0.0f){ 
          it->width  = width;
          it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster] - start_coreid[cluster])/width)) * width;
          // it->increment_PTT_UpdateFlag(ptt_freq_index[cluster],cluster,width-1);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] " << it->kernel_name <<"->Timetable(1.866GHz, " << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = 0. Run with [" << it->leader << ", " << it->width << ")." << std::endl;
          LOCK_RELEASE(output_lck);
#endif
          return it->leader;
        }else{
          continue;
        }  
#endif
      }
    }
    /* There have been enough tasks travesed the PTT and train the PTT, but still training phase hasn't been finished yet, for the incoming tasks, execution config goes to here. 
    Method: schedule incoming tasks to random cores with random width */ 
    if(rand() % gotao_nthreads < START_CLUSTER_B){ // Schedule to Denver
      it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_A].size()); // Width: 1 2
      it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    }else{ // Schedule to A57
      it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_B].size()); // Width: 1 2 4
      it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << ". Run with [" << it->leader << ", " << it->width << ")." << std::endl;
    LOCK_RELEASE(output_lck);
#endif
    return it->leader;
  }else{ /* PTT is fully trained */
    if(it->get_bestconfig_state() == false){ /* The kernel task hasn't got the best config yet, three loops to search for the best configs. */
#if defined Exhastive_Search
#if defined Search_Overhead    
      struct timespec search_start, search_finish, search_delta;
      clock_gettime(CLOCK_REALTIME, &search_start);
#endif
      it->find_best_config(nthread, it);
#if defined Search_Overhead
      clock_gettime(CLOCK_REALTIME, &search_finish);
      poly_sub_timespec(search_start, search_finish, &search_delta);
      printf("[Overhead] Exhastive searching time per task is: %d.%.9ld\n", (int)search_delta.tv_sec, search_delta.tv_nsec);
#endif
#endif
#if defined Optimized_Search /* Meant to reduce the overhead of exhaustive search by computing corners firstly and then step-wise moving closer to the minimum value */
#if defined Search_Overhead    
      struct timespec Optimized_search_start, Optimized_search_finish, Optimized_search_delta;
      clock_gettime(CLOCK_REALTIME, &Optimized_search_start);
#endif
      it->optimized_search(nthread, it);
#if defined Search_Overhead
      clock_gettime(CLOCK_REALTIME, &Optimized_search_finish);
      poly_sub_timespec(Optimized_search_start, Optimized_search_finish, &Optimized_search_delta);
      printf("[Overhead] Optimized searching time per task is: %d.%.9ld\n", (int)Optimized_search_delta.tv_sec, Optimized_search_delta.tv_nsec);
#endif
#endif
    }else{ /* After the searching, the kernel task got the best configs: best frequency, best cluster and best width, new incoming tasks directly use the best config. */
      it->update_best_config(nthread, it);
    }
  }
  }else{
    if(rand() % gotao_nthreads < START_CLUSTER_B){ // Schedule to Denver
      it->width = pow(2, rand() % 2); // Width: 1 2
      it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    }else{ // Schedule to A57
      it->width = pow(2, rand() % 3); // Width: 1 2 4
      it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << ". Run with [" << it->leader << ", " << it->width << ")." << std::endl;
    LOCK_RELEASE(output_lck);
#endif
  }
  return it->leader;
}

int PolyTask::EPTO_for_EDP(int nthread, PolyTask * it){ /* Paper 4: targeting energy performance delay product */
  float comp_perf = 0.0f;
  if(it->tasktype < num_kernels){
  // if(it->get_timetable_state(2) == false){     /* PTT training is not finished yet */
  if(global_training == false){
//     for(;;){ // trying to assign the config to tasks with random generator, here logic is not correct, need to be fixed
//       int cluster = rand() % NUMSOCKETS;
//       int width = pow(2, rand() % inclusive_partitions[cluster].size());
//       if(it->get_PTT_UpdateFlag(ptt_freq_index[cluster], cluster, width-1) < NUM_TRAIN_TASKS){
//         it->width  = width;
//         it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster] - start_coreid[cluster])/width)) * width;
//         it->increment_PTT_UpdateFlag(ptt_freq_index[cluster],cluster,width-1);
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[DEBUG] For task " << it->taskid << ", " << it->kernel_name <<"->Timetable(" << avail_ddr_freq[0] << ", " << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = " << ptt_val << ". Run with [" << it->leader << ", " << it->width << ")." << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif
//         return it->leader;
//       }else{
//         continue;
//       }
//     }
    
//     int coreid = 0; 
//     int width = 1;
//     if(it->get_PTT_UpdateFlag(ptt_freq_index[coreid], coreid, width-1) < NUM_TRAIN_TASKS){
//       it->width  = width;
//       it->leader = start_coreid[coreid] + (rand() % ((end_coreid[coreid] - start_coreid[coreid])/width)) * width;
//       it->increment_PTT_UpdateFlag(ptt_freq_index[coreid],coreid,width-1);
// #ifdef DEBUG
//           LOCK_ACQUIRE(output_lck);
//           std::cout << "[DEBUG] For task " << it->taskid << ". Run with [" << it->leader << ", " << it->width << ")." << std::endl;
//           LOCK_RELEASE(output_lck);
// #endif
//           return it->leader;
//     }else{
//       coreid = (coreid++ > gotao_nthreads) ? 0 : coreid;
//     }
//     coreid = (coreid++ > gotao_nthreads) ? 0 : coreid;


    // for(int cluster = 0; cluster < NUMSOCKETS; ++cluster) {
    for(int cluster = NUMSOCKETS-1; cluster >=0; cluster--) {
      for(auto&& width : ptt_layout[start_coreid[cluster]]) {
#ifdef TRAIN_METHOD_1 /* Allow NUM_TRAIN_TASKS tasks to train the same config, pros: training is faster, cons: not apply to memory-bound tasks */
        if(it->get_PTT_UpdateFlag(ptt_freq_index[cluster], cluster, width-1) < NUM_TRAIN_TASKS){
          it->width  = width;
          it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster] - start_coreid[cluster])/width)) * width;
          it->increment_PTT_UpdateFlag(ptt_freq_index[cluster],cluster,width-1);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << " ===> Run with leader " << it->leader << " with width " << it->width << std::endl;
          // << it->kernel_name <<"->Timetable(" << avail_ddr_freq[0] << ", " << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = " << ptt_val 
          LOCK_RELEASE(output_lck);
#endif
          return it->leader;
        }else{
          continue;
        }  
#endif
#ifdef TRAIN_METHOD_2 /* Allow DOP tasks to train the same config, pros: also apply to memory-bound tasks, cons: training might be slower */
        auto&& ptt_val = 0.0f;
        ptt_val = it->get_timetable(0, ptt_freq_index[cluster], cluster, width - 1);
        if(ptt_val == 0.0f){ 
          it->width  = width;
          it->leader = start_coreid[cluster] + (rand() % ((end_coreid[cluster] - start_coreid[cluster])/width)) * width;
          // it->increment_PTT_UpdateFlag(ptt_freq_index[cluster],cluster,width-1);
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] " << it->kernel_name <<"->Timetable(1.866GHz, " << ptt_freq_index[cluster] << ", " << cluster << ", " << width << ") = 0. Run with [" << it->leader << ", " << it->width << ")." << std::endl;
          LOCK_RELEASE(output_lck);
#endif
          return it->leader;
        }else{
          continue;
        }  
#endif
      }
    } 
    /* There have been enough tasks travesed the PTT and train the PTT, but still training phase hasn't been finished yet, for the incoming tasks, execution config goes to here. 
    Method: By default, schedule all incoming tasks to faster cores with width 1 */ 
    // it->width = 1;
    // it->leader = START_CLUSTER_A + rand() % (end_coreid[0]-start_coreid[0]);
    // if(global_training_state[it->tasktype] == 1)
    if(cur_freq_index[0] == 0){ // Core 0 frequency is highest => schedule to Denver core with width 1
      if(rand() % 4 != 0){ // 75% chance to schedule to Denver
        // it->width = 1;
        // it->leader = START_CLUSTER_A + rand() % (end_coreid[0]-start_coreid[0]);
        it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_A].size()); // Width: 1 2
        it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
      }else{ // 25% chance to schedule to A57
        it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_B].size()); // Width: 1 2 4
        it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
      }
    }else{ // Otherwise, schedule to A57 core with width 1
      if(rand() % 4 != 0){ // 75% chance to schedule to A57
        it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_B].size()); // Width: 1 2 4
        it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
      }else{ // 25% chance to schedule to Denver
        it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_A].size()); // Width: 1 2
        it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
      }
    }

    // if(rand() % gotao_nthreads < START_CLUSTER_B){ // Schedule to Denver
    //   it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_A].size()); // Width: 1 2
    //   it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    // }else{ // Schedule to A57
    //   it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_B].size()); // Width: 1 2 4
    //   it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    // }
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << " ===> Run with leader " << it->leader << " with width " << it->width << std::endl;
    LOCK_RELEASE(output_lck);
#endif
    return it->leader;
  }else{ /* After the sapling phase is fully trained */
// #if defined SWEEP_Overhead    
//       // struct timespec search_start, search_finish, search_delta;
//       // clock_gettime(CLOCK_REALTIME, &search_start);
//       std::chrono::time_point<std::chrono::system_clock> t1,t2;
//       t1 = std::chrono::system_clock::now();
//       // auto start1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(t1);
//       // auto epoch1 = start1_ms.time_since_epoch();
// #endif
    int dop = DOP_detection[it->tasktype]; // std::accumulate(std::begin(DOP_detection), std::begin(DOP_detection) + XITAO_MAXTHREADS, 0);  // DOP is the total number of tasks (inc multiple task types) that are currently ready and executing
    if(dop > 0){
      interval_distri_state = (std::accumulate(std::begin(remaining_distri_num[it->tasktype][nthread]), std::end(remaining_distri_num[it->tasktype][nthread]), 0) == 0)? false : true;    
      if(interval_distri_state == false){ /* The kernel task distribution hasn't computed yet for the new interval */
        interval_HP[it->tasktype][nthread] = false;
        interval_LP[it->tasktype][nthread] = false;
        // if(dop == 1){
        //   it->leader = START_CLUSTER_A;
        //   it->width = end_coreid[0] - start_coreid[0];
        //   interval_LP[it->tasktype][nthread] = true;
        // }else{
#ifdef DOP_TRACE
        if(PolyTask::task_counter.load() >= HP_LP_Threshold * gotao_nthreads){ // if the number of ready tasks are more than 4 * 6 = 24, then we consider it as high parallelism
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck); 
          std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << ". Current total ready tasks: " << PolyTask::task_counter.load() << " >= " << HP_LP_Threshold * gotao_nthreads << " ===> High parallelism." << std::endl;
          LOCK_RELEASE(output_lck);
#endif
          across_cluster_stealing = true;
          it->task_distri_HP(nthread, it, dop);
          interval_HP[it->tasktype][nthread] = true;
        }else{ 
#endif
        float clus_part[NUMSOCKETS] = {0}; /* Here we decide the current DOP is high parallelism or low parallelism */
        float suppose_load_balance = 0.0f;
#ifdef AAWS_CASE        
        int CPU_freq_AAWS[NUMSOCKETS] = {10, 6}; // For AAWS - Test, Denver - 0.5GHz, A57 - 1.11GHz
#endif
        for(int clus_id = 0; clus_id < NUMSOCKETS; clus_id++){
#ifdef AAWS_CASE
          clus_part[clus_id] = it->get_timetable(0, CPU_freq_AAWS[1], 1, 0) / it->get_timetable(0, CPU_freq_AAWS[clus_id], clus_id, 0) * (end_coreid[clus_id]-start_coreid[clus_id]);
#else
          clus_part[clus_id] = it->get_timetable(0, 0, 1, 0) / it->get_timetable(0, 0, clus_id, 0) * (end_coreid[clus_id]-start_coreid[clus_id]);
#endif
          suppose_load_balance += clus_part[clus_id];
        } 
        if(PolyTask::task_counter.load() >= suppose_load_balance)
        // if(dop >= suppose_load_balance)
        { /* High Parallelism */ // old: dop >= suppose_load_balance
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck); 
        std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << ". Current total ready tasks: " << PolyTask::task_counter.load() << " < " << HP_LP_Threshold * gotao_nthreads << ". DOP for load balancing is " \
        << suppose_load_balance << ". Current total number of released tasks of the type is " << dop << " ===> High parallelism." << std::endl;
        LOCK_RELEASE(output_lck);
#endif
#ifdef AAWS_CASE   
          it->AAWS_HP(nthread, it, dop);
#else
          it->task_distri_HP(nthread, it, dop);
#endif
          across_cluster_stealing = true;
          interval_HP[it->tasktype][nthread] = true;
        }else{
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck); 
        std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << ". Current total ready tasks: " << PolyTask::task_counter.load() << " < " << HP_LP_Threshold * gotao_nthreads << ". DOP for load balancing is " \
        << suppose_load_balance << ". Current total number of released tasks of the type is " << dop << " ===> Low parallelism." << std::endl;
        LOCK_RELEASE(output_lck);
#endif
#ifdef AAWS_CASE   
          it->AAWS_LP(nthread, it, dop);
#else
          it->task_distri_LP(nthread, it, dop); /* Low Parallelism */
#endif
          across_cluster_stealing = false;
          interval_LP[it->tasktype][nthread] = true;
        }
#ifdef DOP_TRACE
        }
#endif
      // }
#if defined Search_Overhead
      clock_gettime(CLOCK_REALTIME, &search_finish);
      poly_sub_timespec(search_start, search_finish, &search_delta);
      printf("[Overhead] Exhastive searching time per task is: %d.%.9ld\n", (int)search_delta.tv_sec, search_delta.tv_nsec);
#endif
      }else{ /* After the task distribution computation, get the percentage for each cluster, based on the value to assign the following tasks */
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck); 
//         std::cout << "[DEBUG] remaining_distri_num[" << it->kernel_name << "][" << nthread << "][0] = " << remaining_distri_num[it->tasktype][nthread][0] <<\
//         ", remaining_distri_num[" << it->kernel_name << "][" << nthread << "][1] = " << remaining_distri_num[it->tasktype][nthread][1] << ". Thread " << nthread << " still has tasks to distribute! " << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif      
        it->assign_config(nthread, it);
      }
    }else{
      if(rand() % gotao_nthreads < START_CLUSTER_B){ // Schedule to Denver
        it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_A].size()); // Width: 1 2
        it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
      }else{ // Schedule to A57
        it->width = pow(2, rand() % inclusive_partitions[START_CLUSTER_B].size()); // Width: 1 2 4
        it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
      }
    }
// #if defined SWEEP_Overhead
//       t2 = std::chrono::system_clock::now();
//       // auto start2_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(t2);
//       // auto epoch2 = start2_ms.time_since_epoch();
//       std::chrono::duration<double> elapsed = t2 - t1;
//       elapsed_overhead += elapsed;
// // #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[SWEEP-Overhead] Thread " << nthread << ", the phase timing overhead: " << elapsed.count() << ", the accumulated SWEEP overhead: " << elapsed_overhead.count() << std::endl;
//       LOCK_RELEASE(output_lck);
// // #endif      
      
// #endif
  }
  }else{
    if(rand() % gotao_nthreads < START_CLUSTER_B){ // Schedule to Denver
      it->width = pow(2, rand() % 2); // Width: 1 2
      it->leader = START_CLUSTER_A + (rand() % ((end_coreid[0]-start_coreid[0])/it->width)) * it->width;
    }else{ // Schedule to A57
      it->width = pow(2, rand() % 3); // Width: 1 2 4
      it->leader = START_CLUSTER_B + (rand() % ((end_coreid[1]-start_coreid[1])/it->width)) * it->width;
    }
    // it->width = 1;
    // it->leader = rand() % gotao_nthreads;
    // it->set_enable_cpu_freq_change(true);
    // it->set_clus0_cpu_freq(0);
    // it->set_clus1_cpu_freq(0);
#ifdef DEBUG
    LOCK_ACQUIRE(output_lck);
    std::cout << "[DEBUG] " << it->kernel_name << " task " << it->taskid << ". Run with [" << it->leader << ", " << it->width << ")." << std::endl;
    LOCK_RELEASE(output_lck);
#endif
  }
  return it->leader;
}


#ifdef OVERHEAD_PTT
// std::tuple <int, double> 
PolyTask * PolyTask::commit_and_wakeup(int _nthread, std::chrono::duration<double> elapsed_ptt){
#else
PolyTask * PolyTask::commit_and_wakeup(int _nthread){
#endif 
#if defined SWEEP_Overhead    
  std::chrono::time_point<std::chrono::system_clock> t1,t2;
  t1 = std::chrono::system_clock::now();
#endif
#ifdef OVERHEAD_PTT
  std::chrono::time_point<std::chrono::system_clock> start_ptt, end_ptt;
  start_ptt = std::chrono::system_clock::now();
#endif
  PolyTask *ret = nullptr;
//   if((Sched == 1) || (Sched == 2)){
//     int new_layer_leader = -1;
//     int new_layer_width = -1;
//     std::list<PolyTask *>::iterator it = out.begin();
//     ERASE_Target_Energy(_nthread, (*it));
//     LOCK_ACQUIRE(worker_lock[(*it)->leader]);
//     worker_ready_q[(*it)->leader].push_back(*it);
//     LOCK_RELEASE(worker_lock[(*it)->leader]);
//     new_layer_leader = (*it)->leader;
//     new_layer_width = (*it)->width;
//     ++it;
//     for(it; it != out.end(); ++it){
//       (*it)->width = new_layer_width;
//       (*it)->leader = new_layer_leader;
//       LOCK_ACQUIRE(worker_lock[new_layer_leader]);
//       worker_ready_q[new_layer_leader].push_back(*it);
//       LOCK_RELEASE(worker_lock[new_layer_leader]);
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout <<"[DEBUG] Task "<< (*it)->taskid <<" will run on thread "<< (*it)->leader << ", width become " << (*it)->width << std::endl;
//       LOCK_RELEASE(output_lck);
// #endif
//     }
// 	}
//   else{
    // std::cout << "Thread " << _nthread << " out.size = " << out.size() << std::endl;
#ifdef ERASE_target_edp_method1
    D_give_A = (out.size() > 0) && (ptt_full)? ceil(out.size() * 1 / (D_A+1))+1 : 0; 
#endif

#if (defined ERASE_target_edp_method2)
//  || (defined ERASE_target_energy_method2)
    int n = (out.size() > 0) && (ptt_full)? out.size() : 0; 
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[0] N = " << n << std::endl;
//       LOCK_RELEASE(output_lck); 
// #endif
    D_give_A = 0;
    float standard = 1.0;
    float edp_test = 0.0;
    int temp_width = 0;
    if(n > 0){
#ifdef ERASE_target_energy_method2
      float upper = best_power_config[best_cluster_config] * best_perf_config[best_cluster_config] * n;
#ifdef DEBUG
      LOCK_ACQUIRE(output_lck);
      std::cout << "[1] The best energy when only running on cluster " << best_cluster_config << ": " << upper << std::endl;
      LOCK_RELEASE(output_lck); 
#endif
#endif
      // int start_idle[2] = {0,2};
      // int end_idle[2] = {2, gotao_nthreads};
      int num_cores[2] = {2,4};
      for(int allow_steal = 1; allow_steal <= n/NUMSOCKETS; allow_steal++){
        // float energy_increase = float(allow_steal)/float(n) * float(((best_power_config[1]*best_perf_config[1])/(best_power_config[0]*best_perf_config[0]))-1) + 1;
        // float power_increase = 1 + float(best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1])) / float(best_power_config[0] * (end_idle[0]-start_idle[0])/best_width_config[0]);
#ifdef ERASE_target_energy_method2
        float Total_energy = 0.0;
        // ERASE - Target Energy
        //Total_energy = (n-allow_steal) * best_perf_config[best_cluster_config] * best_power_config[best_cluster_config] + allow_steal * best_perf_config[second_best_cluster_config] * best_power_config[second_best_cluster_config];
        Total_energy = (n-allow_steal) * 0.0322308 *(2.70386 + 0.24866) + allow_steal * 0.0543954 * (2.29105+0.283341);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[2] Allow_steal = " << allow_steal << ". Total Energy = " << Total_energy <<  ".\n";
        LOCK_RELEASE(output_lck);          
#endif
        if(Total_energy < upper){
          D_give_A = allow_steal;
          upper = Total_energy;
          // temp_width = new_best_width_config;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[3] Current lowest energy = " << upper << ", D_give_A = " << D_give_A << ".\n";
          LOCK_RELEASE(output_lck);          
#endif
        }
#endif
#ifdef ERASE_target_edp_method2
        // Can use max function
        // if((n-allow_steal)*best_perf_config[0] > allow_steal*best_perf_config[1]){
        //   performance_decline = float(n-allow_steal)/float(n);
        //   // performance_decline = float(n - int(allow_steal * float(best_width_config[0]/(end_idle-start_idle))))/float(n);
        // }else{
        //   performance_decline = (allow_steal*best_perf_config[1])/(n*best_perf_config[0]);
        // }
        float performance_decline = 1.0;
        // Problem Solved: new width can not pass to assembly task (Line 1725-1736)
        if(best_width_config[1] * allow_steal < num_cores[1]){
          int ori_best_width = best_width_config[1];
          int new_best_width_config = ceil(num_cores[1]/allow_steal);
          float new_best_power_config = best_power_config[1] * (new_best_width_config/ori_best_width);
          // int same_type = 0;
          // for (int h = 2; h < gotao_nthreads; h += best_width_config[1]){
          //   average[a][best_width_config[1]] += it->get_timetable(h,testwidth-1); // width = 1
          //    += ptt_value[h];
          //   same_type++;
          // }
          // average[a][testwidth] = average[a][testwidth] / float(same_type);
          float new_best_perf_config = average[1][new_best_width_config];
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[0] new_best_width_config= " << new_best_width_config << ". new_best_power_config =  " << new_best_power_config << ". new_best_perf_config = " << new_best_perf_config << std::endl;
          LOCK_RELEASE(output_lck); 
#endif
          int current_perf = ceil(float((n-allow_steal)*best_width_config[0])/float(num_cores[0]));
          int previous_perf = ceil(float(n*best_width_config[0])/float(num_cores[0]));
          int allow_perf = ceil(float(allow_steal*new_best_width_config)/float(num_cores[1]));
  #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[1] current_perf= " << current_perf << ". previous_perf =  " << previous_perf << ". allow_perf = " << allow_perf << std::endl;
          LOCK_RELEASE(output_lck); 
  #endif
          if(current_perf < previous_perf){
  #ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[2] Denver execution time = " << current_perf*best_perf_config[0] << ". A57 execution time = " << new_best_perf_config*allow_perf << std::endl;
            LOCK_RELEASE(output_lck); 
  #endif     
            if(current_perf*best_perf_config[0] > new_best_perf_config*allow_perf){
              performance_decline = float(current_perf) / float(previous_perf);
            }else{
              performance_decline = (new_best_perf_config* float(allow_perf))/ (best_perf_config[0] * float(previous_perf));
            }
          }
          // float energy_increase = float(current_perf)/float(previous_perf) + float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf)) / float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf));
          // A57 wrong Energy consumption: float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf))
          // Denver's original energy consumption (maybe wrong): float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf))
          // float energy_increase = 1 + ( (best_power_config[1] * allow_steal * best_perf_config[1]) - (best_power_config[0] * allow_steal * best_perf_config[0]))/ float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf));
          float energy_increase = 1 + ( (new_best_power_config * allow_steal * new_best_perf_config) - (best_power_config[0] * allow_steal * best_perf_config[0]))/ (best_power_config[0] * best_perf_config[0] * n);
          // edp_test = power_increase * pow(performance_decline, 2.0);
          edp_test = energy_increase *  performance_decline;
  #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          // std::cout << "[2] Denver's Original Energy consumption: " << float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf)) <<std::endl;
          std::cout << "[3] Denver's Original Energy consumption: " << best_power_config[0] * best_perf_config[0] * n << std::endl;
          std::cout << "[4] Denver reduce energy by " << best_power_config[0] * allow_steal * best_perf_config[0] << std::endl;
          //std::cout << "[4] Additional A57 power consumption: " << best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) << ". A57 consumes energy: " << float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf)) << std::endl; 
          std::cout << "[5] Additional A57 energy consumption: " << (new_best_power_config * allow_steal * new_best_perf_config) << std::endl;
          LOCK_RELEASE(output_lck); 
  #endif        
  #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Thread " << _nthread << " release " << n << " tasks. Allow_steal =  " << allow_steal << ", energy_increase = " << energy_increase << ", perf_decline = " << performance_decline << ". edp_test = " << edp_test << ".\n";
          LOCK_RELEASE(output_lck);          
  #endif
          if(edp_test < standard){
            D_give_A = allow_steal;
            standard = edp_test;
            temp_width = new_best_width_config;
  #ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Current lowest EDP = " << standard << ", D_give_A = " << D_give_A << ".\n";
          LOCK_RELEASE(output_lck);          
  #endif
          }
        }else{
          int current_perf = ceil(float((n-allow_steal)*best_width_config[0])/float(num_cores[0]));
          int previous_perf = ceil(float(n*best_width_config[0])/float(num_cores[0]));
          int allow_perf = ceil(float(allow_steal*best_width_config[1])/float(num_cores[1]));
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[1] current_perf= " << current_perf << ". previous_perf =  " << previous_perf << ". allow_perf = " << allow_perf << std::endl;
          LOCK_RELEASE(output_lck); 
#endif
          if(current_perf < previous_perf){
#ifdef DEBUG
            LOCK_ACQUIRE(output_lck);
            std::cout << "[2] Denver execution time = " << current_perf*best_perf_config[0] << ". A57 execution time = " << best_perf_config[1]*allow_perf << std::endl;
            LOCK_RELEASE(output_lck); 
#endif           

            if(current_perf*best_perf_config[0] > best_perf_config[1]*allow_perf){
              performance_decline = float(current_perf) / float(previous_perf);
            }else{
              performance_decline = (best_perf_config[1]* float(allow_perf))/ (best_perf_config[0] * float(previous_perf));
            }

          }
          // float energy_increase = float(current_perf)/float(previous_perf) + float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf)) / float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf));
          // A57 wrong Energy consumption: float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf))
          // Denver's original energy consumption (maybe wrong): float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf))
          // float energy_increase = 1 + ( (best_power_config[1] * allow_steal * best_perf_config[1]) - (best_power_config[0] * allow_steal * best_perf_config[0]))/ float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf));
          float energy_increase = 1 + ( (best_power_config[1] * allow_steal * best_perf_config[1]) - (best_power_config[0] * allow_steal * best_perf_config[0]))/ (best_power_config[0] * best_perf_config[0] * n);
          // edp_test = power_increase * pow(performance_decline, 2.0);
          edp_test = energy_increase *  performance_decline;
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          // std::cout << "[2] Denver's Original Energy consumption: " << float((best_power_config[0] * std::min(n-allow_steal, (end_idle[0]-start_idle[0])/best_width_config[0]) * best_perf_config[0] * previous_perf)) <<std::endl;
          std::cout << "[3] Denver's Original Energy consumption: " << best_power_config[0] * best_perf_config[0] * n << std::endl;
          std::cout << "[4] Denver reduce energy by " << best_power_config[0] * allow_steal * best_perf_config[0] << std::endl;
          //std::cout << "[4] Additional A57 power consumption: " << best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) << ". A57 consumes energy: " << float((best_power_config[1] * std::min(allow_steal, (end_idle[1]-start_idle[1])/best_width_config[1]) * best_perf_config[1] * allow_perf)) << std::endl; 
          std::cout << "[5] Additional A57 energy consumption: " << (best_power_config[1] * allow_steal * best_perf_config[1]) << std::endl;
          LOCK_RELEASE(output_lck); 
#endif        
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Thread " << _nthread << " release " << n << " tasks. Allow_steal =  " << allow_steal << ", energy_increase = " << energy_increase << ", perf_decline = " << performance_decline << ". edp_test = " << edp_test << ".\n";
          LOCK_RELEASE(output_lck);          
#endif
          if(edp_test < standard){
            D_give_A = allow_steal;
            standard = edp_test;
            temp_width = best_width_config[1];
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << "[DEBUG] Current lowest EDP = " << standard << ", D_give_A = " << D_give_A << ".\n";
          LOCK_RELEASE(output_lck);          
#endif
          }
        }
        best_width_config[1] = temp_width;
#endif
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] D_give_A = " << D_give_A << ". \n";
        LOCK_RELEASE(output_lck);          
#endif
      }
    }
#endif

#ifdef Target_EPTO
    for(std::list<PolyTask *>::iterator it = out.begin(); it != out.end(); ++it){
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[DEBUG] (*it)->refcount = " << (*it)->refcount << std::endl;
//       LOCK_RELEASE(output_lck);
// #endif        
      // int refs = (*it)->refcount.fetch_sub(1);
      if((*it)->refcount == 1){
#ifdef DOP_TRACE
        task_counter.fetch_add(1); // The total number of ready tasks increases by 1
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
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout << "[DEBUG] Thread " << _nthread << ": " << (*it)->kernel_name << " task " << (*it)->taskid << " became ready.";
        LOCK_RELEASE(output_lck);
#endif 
        if(global_training == true && (*it)->tasktype < num_kernels){
          DOP_detection[(*it)->tasktype] += 1; /* Scheduler: Paper 4 for detecting the number of tasks */
          
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << " ===> Current DOP of " << (*it)->kernel_name << " tasks is " << DOP_detection[(*it)->tasktype] << std::endl;
          LOCK_RELEASE(output_lck);
#endif           
        }else{
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout << std::endl;
          LOCK_RELEASE(output_lck);
#endif           
        }
      }
    }
#endif
    int counter = 0;
    for(std::list<PolyTask *>::iterator it = out.begin(); it != out.end(); ++it){
      int refs = (*it)->refcount.fetch_sub(1);
      if(refs == 1){
        counter++;
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[DEBUG] (*it)->refcount = " << (*it)->refcount << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif  
      // if((*it)->refcount == 1){
// #ifdef DEBUG
//         LOCK_ACQUIRE(output_lck);
//         std::cout << "[DEBUG] " << (*it)->kernel_name << " task " << (*it)->taskid << " became ready" << std::endl;
//         LOCK_RELEASE(output_lck);
// #endif  
        if((Sched == 1) || (Sched == 2)){ 
#ifdef Target_EPTO // Paper 4 design
          if(global_training == true && counter == 1 && (*it)->tasktype < num_kernels){
            // interval_distri_state = (std::accumulate(std::begin(remaining_distri_num[(*it)->tasktype]), std::end(remaining_distri_num[(*it)->tasktype]), 0) == 0)? false : true;
            // if(it == out.begin()){
              // DOP_detection[(*it)->tasktype] += out.size(); 
// #ifdef DEBUG
//               LOCK_ACQUIRE(output_lck);
//               std::cout << "[DEBUG] " << out.size() << " new released tasks from thread " << _nthread << " ===> Current DOP of " << (*it)->kernel_name << " tasks is " << DOP_detection[(*it)->tasktype] << std::endl;
//               LOCK_RELEASE(output_lck);
// #endif
#ifdef INTERVAL_RECOMPUTATION  
              interval_t1 = std::chrono::system_clock::now();
              std::chrono::duration<double> time_interval = interval_t1 - interval_t2;
              if(time_interval.count() >= COMP_INTERVAL){ // if interval between last task distribution computation and now is larger than e.g., 1 second
                (*it)->set_lp_task_distri_state(false);  // set task distribution state to false and recompute the task distribution again
                (*it)->set_hp_task_distri_state(false);
#ifdef DEBUG
                LOCK_ACQUIRE(output_lck);
                std::cout << "[DEBUG] Interval between last task distribution computation and now is larger than " << COMP_INTERVAL << " seconds. " << std::endl;
                LOCK_RELEASE(output_lck); 
#endif
              }
#endif
            // }
          }
          EPTO_for_EDP(_nthread, (*it)); /* Paper 4: EPTO design */     
          LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          worker_ready_q[(*it)->leader].push_back(*it);
          LOCK_RELEASE(worker_lock[(*it)->leader]);
#endif

#ifdef ERASE_target_energy_method1
          /* Method 1: Fix best config to certain cluster */ 
          // ERASE_Target_Energy(_nthread, (*it));
#endif
#ifdef ERASE_target_energy_method2
          /* Method 2: Allow work stealing across clusters */ 
          // if(D_give_A > 0 && std::distance(out.begin(), it) >= out.size() - D_give_A){
          //   (*it)->width = best_width_config[second_best_cluster_config];
          //   // (*it)->leader = A57_best_edp_leader;
          //   if((*it)->width == 4){
          //     (*it)->leader = 2;
          //   }
          //   if((*it)->width <= 2){
          //     (*it)->leader = 2 + 2 * rand()%2;
          //   }
          //   LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          //   worker_ready_q[(*it)->leader].push_back(*it);
          //   LOCK_RELEASE(worker_lock[(*it)->leader]);
          // }else{
          
          JOSS_for_Energy(_nthread, (*it));  /* JOSS design */
          
          // LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          // worker_ready_q[(*it)->leader].push_back(*it);
          // LOCK_RELEASE(worker_lock[(*it)->leader]);
          // } 
#ifdef AcrossCLustersTest
          // D_give_A = 1; /* Simple test: when we allow certail number of work stealings across two clusters */
          if(D_give_A > 0 && (*it)->get_bestconfig_state()==true && std::distance(out.begin(), it) >= out.size() - D_give_A){
            (*it)->leader = 2;   
            // (*it)->width =  2; /* Simply define the <A57, 2 (4)> is the most energy efficient config when A57 steals tasks from Denver cores */
          }
          // else{
          //   (*it)->leader = 0;
          //   (*it)->width = 2;
          // }
#endif          
          LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          worker_ready_q[(*it)->leader].push_back(*it);
          LOCK_RELEASE(worker_lock[(*it)->leader]);
#endif

#ifdef ERASE_target_perf
          ERASE_Target_Perf(_nthread, (*it));
          LOCK_ACQUIRE(worker_lock[(*it)->leader]);
          worker_ready_q[(*it)->leader].push_back(*it);
          LOCK_RELEASE(worker_lock[(*it)->leader]);
#endif

#if (defined ERASE_target_edp_method1) || (defined ERASE_target_edp_method2)
          if(D_give_A > 0 && std::distance(out.begin(), it) >= out.size() - D_give_A){
            (*it)->width = best_width_config[1];
            // (*it)->leader = A57_best_edp_leader;
            if((*it)->width == 4){
              (*it)->leader = 2;
            }
            if((*it)->width <= 2){
              (*it)->leader = 2 + 2 * rand()%2;
            }
            LOCK_ACQUIRE(worker_lock[(*it)->leader]);
            worker_ready_q[(*it)->leader].push_back(*it);
            LOCK_RELEASE(worker_lock[(*it)->leader]);
          }else{
            ERASE_Target_EDP(_nthread, (*it));
            LOCK_ACQUIRE(worker_lock[(*it)->leader]);
            worker_ready_q[(*it)->leader].push_back(*it);
            LOCK_RELEASE(worker_lock[(*it)->leader]);
          }     
#ifdef DEBUG
          LOCK_ACQUIRE(output_lck);
          std::cout <<"[DEBUG] EDP: task "<< (*it)->taskid <<"'s leader id = "<< (*it)->leader << ", width = " << (*it)->width << ", type is " << (*it)->tasktype << std::endl;
          LOCK_RELEASE(output_lck);
#endif
#endif
#ifdef ACCURACY_TEST
          for(int i = (*it)->leader; i < (*it)->leader + (*it)->width; i++){
            LOCK_ACQUIRE(worker_assembly_lock[i]);
            worker_assembly_q[i].push_back((*it));
          }
          for(int i = (*it)->leader; i < (*it)->leader + (*it)->width; i++){
            LOCK_RELEASE(worker_assembly_lock[i]);
          }  
#endif         
        }

      /* Scheduler: Random Work Stealing */
			if(Sched == 3){
        // Case 1: EDP Test for A57 borrow n tasks
        // int ndx = 0;
        // if(std::distance(out.begin(), it) >= out.size() - 2){
        //   ndx = 2 + (std::distance(out.begin(), it) % 2) * 2;
        //   // ndx = 2 + (std::distance(out.begin(), it) % 3);
        //   // ndx = 2+(std::distance(out.begin(), it) % 4);
        //   // ndx = 2 + rand()%4;
        //   (*it)->leader = ndx;
        //   (*it)->width = 2;
        // }else{
        //   ndx = std::distance(out.begin(), it) % 2;
        //   // (*it)->leader = ndx;
        //   (*it)->width = 2;
        // } 
        // Case 2: EDP Test for not borrowing task to A57 
        int ndx = rand() % gotao_nthreads;
        (*it)->leader = ndx;
        (*it)->width = 1;

        // int ndx = 0;
        // if((*it)->tasktype == 0){
        //   ndx = 0;
        //   (*it)->leader = ndx;
        //   (*it)->width = 2;
        // }else{
        //   ndx = 2;
        //   (*it)->leader = ndx;
        //   (*it)->width = 4;
        // }

        LOCK_ACQUIRE(worker_lock[ndx]);
        worker_ready_q[ndx].push_back(*it);
        LOCK_RELEASE(worker_lock[ndx]);
#ifdef DEBUG
        LOCK_ACQUIRE(output_lck);
        std::cout <<"[DEBUG] Task "<< (*it)->taskid <<" is pushed to WSQ of thread "<< ndx << std::endl;
        LOCK_RELEASE(output_lck);
#endif
			}
    }
  }
#if defined SWEEP_Overhead
  t2 = std::chrono::system_clock::now();
  // auto start2_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(t2);
  // auto epoch2 = start2_ms.time_since_epoch();
  std::chrono::duration<double> elapsed = t2 - t1;
  elapsed_overhead += elapsed;
// #ifdef DEBUG
  LOCK_ACQUIRE(output_lck);
  std::cout << "[SWEEP-Overhead] Thread " << _nthread << ", the phase timing overhead: " << elapsed.count() << ", the accumulated SWEEP overhead: " << elapsed_overhead.count() << std::endl;
  LOCK_RELEASE(output_lck);
// #endif      
#endif
#ifdef OVERHEAD_PTT
  end_ptt = std::chrono::system_clock::now();
  elapsed_ptt += end_ptt - start_ptt;
  return elapsed_ptt.count();
#endif
}       