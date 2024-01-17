#ifndef SYNTH_STENCIL
#define SYNTH_STENCIL

#include "tao.h"
#include "dtypes.h"
#include <chrono>
#include <iostream>
#include <atomic>
#include <cmath>
#include <vector>
#define PSLACK 8

// Matrix multiplication, tao groupation on written value
class Synth_MatStencil : public AssemblyTask 
{
public: 
  static float time_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
  static float cpu_power_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
  static float ddr_power_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
  static uint64_t cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
  static float mb_table[NUMSOCKETS][XITAO_MAXTHREADS]; /*mb - memory-boundness */
  static bool time_table_state[NUMSOCKETS+1];
  static bool best_config_state;
  static bool enable_cpu_freq_change;
  static bool enable_ddr_freq_change;
  static int best_cpufreqindex;
  static int best_ddrfreqindex;
  static int best_cluster;
  static int best_width;
  static int second_best_cpufreqindex;
  static int second_best_ddrfreqindex;
  static int second_best_cluster;
  static int second_best_width;
  static std::atomic<int> PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static std::atomic<int> PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static bool lp_task_distri_state;
  static bool hp_task_distri_state;
  static int clus0_cpu_freq; 
  static int clus1_cpu_freq;

  Synth_MatStencil(uint32_t _size, int _width, real_t *_A, real_t *_B) : AssemblyTask(_width), A(_A), B(_B) {   
    dim_size = _size;
    block_size = dim_size / (_width * PSLACK);
    if(block_size == 0) block_size = 1;
    block_index = 0;
    uint32_t elem_count = dim_size * dim_size;
/*    A = new real_t[elem_count];
    B = new real_t[elem_count];
    */
    block_count = dim_size / block_size;
  }

  void cleanup() { 
    //delete[] A;
    //delete[] B;

  }

  void execute(int threadid) {
    while(true) {
      int row_block_id = block_index++;
      if(row_block_id > block_count) return;
      int row_block_start =  row_block_id      * block_size;
      int row_block_end   = (row_block_id + 1) * block_size;
      int end = (dim_size < row_block_end) ? dim_size : row_block_end; 
      if (row_block_start == 0) row_block_start = 1;
      if (end == dim_size)      end = dim_size - 1;
      for (int i = row_block_start; i < end; ++i) { 
        for (int j = 1; j < dim_size-1; j++) {
             B[i*dim_size + j] = A[i*dim_size + j] + k * (
             A[(i-1)*dim_size + j] +
             A[(i+1)*dim_size + j] +
             A[i*dim_size + j-1] +
             A[i*dim_size + j+1] +
             (-4)*A[i*dim_size + j] );
        }
      }
    }
  }
/*
  Synth_MatStencil(uint32_t _size, int _width, real_t *_A, real_t *_B) : AssemblyTask(_width), A(_A), B(_B) {   
    dim_size = _size;
    // block_size = dim_size / (_width * PSLACK);
    // if(block_size == 0) block_size = 1;
    block_index = 0;
    // uint32_t elem_count = dim_size * dim_size;
    // block_count = dim_size / block_size;
  }

  void cleanup() { 
  }

  void execute(int threadid) {
    // Add by Jing
    block_size = dim_size / (width * PSLACK);
    if(block_size == 0) block_size = 1;
    block_count = dim_size / block_size;
    int row_block_start =  (threadid-leader) * (block_count/width)* block_size;
    int row_block_end   = ((threadid-leader)+1) * (block_count/width)* block_size;
// #ifdef DEBUG
//     LOCK_ACQUIRE(output_lck);
//     std::cout << "Task " << taskid << ", width: " << width << ". total number of blocks: " << block_count << ". Thread " << threadid << " will do block " << (threadid-leader) * (block_count/width) \
//     << " to block " << ((threadid-leader)+1) * (block_count/width) << "! Currently, thread " << threadid << ",  row_block_start = " << row_block_start<< ", row_block_end = " << row_block_end << std::endl;
//     LOCK_RELEASE(output_lck);
// #endif
    int end = (dim_size < row_block_end) ? dim_size : row_block_end;
    if (row_block_start == 0) row_block_start = 1;
    if (end == dim_size)      end = dim_size - 1;
    for (int i = row_block_start; i < end; ++i) {
      for (int j = 1; j < dim_size-1; j++) {
            B[i*dim_size + j] = A[i*dim_size + j] + k * (
            A[(i-1)*dim_size + j] +
            A[(i+1)*dim_size + j] +
            A[i*dim_size + j-1] +
            A[i*dim_size + j+1] +
            (-4)*A[i*dim_size + j] );
      }
    }
  }
*/
  void set_clus0_cpu_freq(int freq_index){
    clus0_cpu_freq = freq_index;
  }
  void set_clus1_cpu_freq(int freq_index){
    clus1_cpu_freq = freq_index;
  }
  int get_clus0_cpu_freq() {
    return clus0_cpu_freq;
  }
  int get_clus1_cpu_freq() {
    return clus1_cpu_freq;
  }
  void increment_PTT_UpdateFinish(int cpu_freq_index, int clusterid, int index) {
    PTT_UpdateFinish[cpu_freq_index][clusterid][index]++;
  }
  float get_PTT_UpdateFinish(int cpu_freq_index, int clusterid,int index){
    float finish = 0;
    finish = PTT_UpdateFinish[cpu_freq_index][clusterid][index];
    return finish;
  }
  void increment_PTT_UpdateFlag(int cpu_freq_index, int clusterid, int index) {
    PTT_UpdateFlag[cpu_freq_index][clusterid][index]++;
  }
  float get_PTT_UpdateFlag(int cpu_freq_index, int clusterid,int index){
    float finish = 0;
    finish = PTT_UpdateFlag[cpu_freq_index][clusterid][index];
    return finish;
  }
  void set_timetable(int ddr_freq_index, int cpu_freq_index, int clusterid, float ticks, int index) {
    time_table[ddr_freq_index][cpu_freq_index][clusterid][index] = ticks;
  }
  float get_timetable(int ddr_freq_index, int cpu_freq_index, int clusterid, int index) { 
    float time = 0.0;
    time = time_table[ddr_freq_index][cpu_freq_index][clusterid][index];
    return time;
  }
  void set_cpupowertable(int ddr_freq_index, int cpu_freq_index, int clusterid, float power_value, int index) {
    cpu_power_table[ddr_freq_index][cpu_freq_index][clusterid][index] = power_value;
  }
  float get_cpupowertable(int ddr_freq_index, int cpu_freq_index, int clusterid, int index) { 
    float power_value = 0;
    power_value = cpu_power_table[ddr_freq_index][cpu_freq_index][clusterid][index];
    return power_value;
  }
  void set_ddrpowertable(int ddr_freq_index, int cpu_freq_index, int clusterid, float power_value, int index) {
    ddr_power_table[ddr_freq_index][cpu_freq_index][clusterid][index] = power_value;
  }
  float get_ddrpowertable(int ddr_freq_index, int cpu_freq_index, int clusterid, int index) { 
    float power_value = 0;
    power_value = ddr_power_table[ddr_freq_index][cpu_freq_index][clusterid][index];
    return power_value;
  }
  void set_cycletable(int cpu_freq_index, int clusterid, uint64_t cycles, int index) {
    cycle_table[cpu_freq_index][clusterid][index] = cycles;
  }
  uint64_t get_cycletable(int cpu_freq_index, int clusterid, int index) { 
    uint64_t cycles = 0;
    cycles = cycle_table[cpu_freq_index][clusterid][index];
    return cycles;
  }
  void set_mbtable(int clusterid, float mem_b, int index) {
    mb_table[clusterid][index] = mem_b;
  }
  float get_mbtable(int clusterid, int index) { 
    float mem_b = 0;
    mem_b = mb_table[clusterid][index];
    return mem_b;
  }
  bool get_timetable_state(int cluster_index){
    bool state = time_table_state[cluster_index];
    return state;
  }
  void set_timetable_state(int cluster_index, bool new_state){
    time_table_state[cluster_index] = new_state;
  }
  /* Find out the best config for this kernel task */
  bool get_bestconfig_state(){
    bool state = best_config_state;
    return state;
  }
  void set_bestconfig_state(bool new_state){
    best_config_state = new_state;
  }
  /* Enable frequency change or not (fine-grained or coarse-grained) */
  bool get_enable_cpu_freq_change(){
    bool state = enable_cpu_freq_change;
    return state;
  }
  void set_enable_cpu_freq_change(bool new_state){
    enable_cpu_freq_change = new_state;
  }
  bool get_enable_ddr_freq_change(){
    bool state = enable_ddr_freq_change;
    return state;
  }
  void set_enable_ddr_freq_change(bool new_state){
    enable_ddr_freq_change = new_state;
  }
  void set_best_cpu_freq(int cpu_freq_index){
    best_cpufreqindex = cpu_freq_index;
  }
  void set_best_ddr_freq(int ddr_freq_index){
    best_ddrfreqindex = ddr_freq_index;
  }
  void set_best_core_type(int clusterid){
    best_cluster = clusterid;
  }
  void set_best_numcores(int width){
    best_width = width;
  }
  int get_best_cpu_freq(){
    int freq_indx = best_cpufreqindex;
    return freq_indx;
  }
  int get_best_ddr_freq(){
    int ddr_freq_indx = best_ddrfreqindex;
    return ddr_freq_indx;
  }
  int get_best_cluster(){
    int clu_id = best_cluster;
    return clu_id;
  }
  int get_best_numcores(){
    int wid = best_width;
    return wid;
  }
  void set_second_best_cpu_freq(int cpu_freq_index){
    second_best_cpufreqindex = cpu_freq_index;
  }
  void set_second_best_ddr_freq(int ddr_freq_index){
    second_best_ddrfreqindex = ddr_freq_index;
  }
  void set_second_best_cluster(int clusterid){
    second_best_cluster = clusterid;
  }
  void set_second_best_numcores(int width){
    second_best_width = width;
  }
  int get_second_best_cpu_freq(){
    int freq_indx = second_best_cpufreqindex;
    return freq_indx;
  }
  int get_second_best_ddr_freq(){
    int ddr_freq_indx = second_best_ddrfreqindex;
    return ddr_freq_indx;
  }
  int get_second_best_cluster(){
    int clu_id = second_best_cluster;
    return clu_id;
  }
  int get_second_best_numcores(){
    int wid = second_best_width;
    return wid;
  }
  void set_lp_task_distri_state(bool new_state){
    lp_task_distri_state = new_state;
  }
  bool get_lp_task_distri_state(){
    bool state = lp_task_distri_state;
    return state;
  }
  void set_hp_task_distri_state(bool new_state){
    hp_task_distri_state = new_state;
  }
  bool get_hp_task_distri_state(){
    bool state = hp_task_distri_state;
    return state;
  }
private:
  const real_t k = 0.001;
  std::atomic<int> block_index; 
  int dim_size;
  int block_count;
  int block_size;
  real_t* A, *B;
};

#endif
