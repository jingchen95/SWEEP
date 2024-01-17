#include "xitao_workspace.h"
namespace xitao {
  std::list<PolyTask *> worker_ready_q[XITAO_MAXTHREADS];
  LFQueue<PolyTask *> worker_assembly_q[XITAO_MAXTHREADS];  
  long int tao_total_steals = 0;  
  long int tao_total_across_steals = 0;
  BARRIER* starting_barrier;
  cxx_barrier* tao_barrier;  
  struct completions task_completions[XITAO_MAXTHREADS];
  struct completions task_pool[XITAO_MAXTHREADS];
  int critical_path;
  int gotao_nthreads;
  int gotao_ncontexts;
  int gotao_thread_base;
  bool gotao_can_exit = false;
  bool gotao_initialized = false;
  bool gotao_started = false;
  bool resources_runtime_conrolled = false;
  bool suppress_init_warnings = false;
  bool ptt_full = false;
  float average[NUMSOCKETS][XITAO_MAXTHREADS] = {0.0}; 
  long avail_freq[NUMSOCKETS][NUM_AVAIL_FREQ];
  long avail_ddr_freq[NUM_DDR_AVAIL_FREQ];
  long cur_ddr_freq;
  long cur_freq[XITAO_MAXTHREADS];
  int cur_ddr_freq_index;
  int cur_freq_index[XITAO_MAXTHREADS];
#if (defined ERASE_target_perf) || (defined ERASE_target_edp)
  float D_A = 0.0f;
#endif
#if (defined ERASE_target_edp_method1) || (defined ERASE_target_edp_method2) || (defined ERASE_target_energy_method2)
  int steal_DtoA = 0;
  int D_give_A = 0;
  int best_cluster_config = -1;
  int second_best_cluster_config = -1;
  int best_width_config[NUMSOCKETS] = {0};
  int best_leader_config[NUMSOCKETS] = {0};
  float best_power_config[NUMSOCKETS] = {0.0};
  float best_perf_config[NUMSOCKETS] = {0.0};
#endif
  std::vector<int> runtime_resource_mapper;   // a logical to physical runtime resource mapper
  std::thread* t[XITAO_MAXTHREADS];
  std::vector<int> static_resource_mapper(XITAO_MAXTHREADS);
  std::vector<int> cluster_mapper(XITAO_MAXTHREADS);
  std::vector<std::vector<int> > ptt_layout(XITAO_MAXTHREADS);
  std::vector<std::vector<std::pair<int, int> > > inclusive_partitions(XITAO_MAXTHREADS);
  GENERIC_LOCK(worker_lock[XITAO_MAXTHREADS]);
  GENERIC_LOCK(worker_assembly_lock[XITAO_MAXTHREADS]);
  std::mutex smpd_region_lock;
  GENERIC_LOCK(output_lck);
}
