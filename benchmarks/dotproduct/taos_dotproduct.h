/*! \file 
@brief Contains the TAOs needed for the dot product example
*/
#include "tao.h"
#include <chrono>
#include <iostream>
#include <atomic>

extern "C" {

#include <stdio.h>
#include <stdlib.h> 
#include <unistd.h>

}

// parallel slackness        
#define PSLACK 8  
//#define CRIT_PERF_SCHED
using namespace std;

/*! this TAO will take two vectors and multiply them. 
This TAO implements internal static scheduling.*/
class VecMulSta : public AssemblyTask 
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
  static std::atomic<int> PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static std::atomic<int> PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static bool lp_task_distri_state;
  static bool hp_task_distri_state;
  static int clus0_cpu_freq; 
  static int clus1_cpu_freq;

  //! VecMulSta TAO constructor. 
  /*!
    \param _A is the A vector
    \param _B is the B vector
    \param _C is the Result vector
    \param _len is the length of the vector 
    \param width is the number of resources used by this TAO
    The constructor computes the number of elements per thread.
    In this simple example, we do not instatiate a dynamic scheduler (yet)
  */  
  VecMulSta(double *_A, double  *_B, double *_C, int _len, 
      int width) : A(_A), B(_B), C(_C), len(_len), AssemblyTask(width) 
  {  
     if(len % width) std::cout <<  "Warning: blocklength is not a multiple of TAO width\n";
     block = len / width;
  }

  //! Inherited pure virtual function that is called by the runtime to cleanup any resources (if any), held by a TAO. 
  void cleanup() {  
    
  }
  //! Inherited pure virtual function that is called by the runtime upon executing the TAO
  /*!
    \param threadid logical thread id that executes the TAO
  */
  void execute(int threadid)
  {
    int tid = threadid - leader; 
    for(int i = tid*block; (i < len)  && (i < (tid+1)*block); i++)
          C[i] = A[i] * B[i];
  }
  
  // Add by Jing
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

  int block; /*!< TAO implementation specific integer that holds the number of blocks per TAO */
  int len;   /*!< TAO implementation specific integer that holds the vector length */
  double *A; /*!< TAO implementation specific double array that holds the A vector */
  double *B; /*!< TAO implementation specific double array that holds the B vector */
  double *C; /*!< TAO implementation specific double array that holds the result vector */
};

// #if defined(CRIT_PERF_SCHED)
// float VecMulSta::time_table[XITAO_MAXTHREADS][XITAO_MAXTHREADS]; 
// #endif

/*! this TAO will take two vectors and multiply them. 
This TAO implements internal dynamic scheduling.*/
class VecMulDyn : public AssemblyTask 
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
  static std::atomic<int> PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static std::atomic<int> PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static bool lp_task_distri_state;
  static bool hp_task_distri_state;
  static int clus0_cpu_freq; 
  static int clus1_cpu_freq;
  
  //! VecMulDyn TAO constructor. 
  /*!
    \param _A is the A vector
    \param _B is the B vector
    \param _C is the Result vector
    \param _len is the length of the vector 
    \param width is the number of resources used by this TAO
    The constructor computes the number of elements per thread and overdecomposes the domain using PSLACK parameter
    In this simple example, we do not instatiate a dynamic scheduler (yet)
  */  
  VecMulDyn(double *_A, double  *_B, double *_C, int _len, int width) : A(_A), B(_B), C(_C), len(_len), AssemblyTask(width) 
  {  
    len = _len;
    // if(len % (width)) std::cout <<  "Warning: blocklength is not a multiple of TAO width\n";
    // blocksize = len / (width*PSLACK); 
    // if(!blocksize) std::cout << "Block Length needs to be bigger than " << (width*PSLACK) << std::endl;
    // blocks = len / blocksize;
    // next = 0;
  }

  //! Inherited pure virtual function that is called by the runtime to cleanup any resources (if any), held by a TAO. 
  void cleanup(){ 
  
  }

  //! Inherited pure virtual function that is called by the runtime upon executing the TAO. 
  /*!
    \param threadid logical thread id that executes the TAO
    This assembly can work totally asynchronously
  */
  void execute(int threadid){
   //  int tid = threadid - leader;
    // while(1){
    //   int blockid = next++;
    //   if(blockid > blocks) return;
    //   for(int i = blockid*blocksize; (i < len) && (i < (blockid+1)*blocksize); i++)
    //       C[i] = A[i] * B[i];
    // }
    if(len % (width)){
      std::cout <<  "Warning: blocklength is not a multiple of TAO width\n";
    } 
    blocksize = len / (width*PSLACK); 
    if(!blocksize) std::cout << "Block Length needs to be bigger than " << (width*PSLACK) << std::endl;
    blocks = len / blocksize;
    int start = (threadid-leader) * (blocks/width) * blocksize;
    int end = ((threadid-leader)+1) * (blocks/width) * blocksize;
// #ifdef DEBUG
//     LOCK_ACQUIRE(output_lck);
//     std::cout << "[DEBUG] Task " << taskid << ", leader: " << leader << ". start: " << start <<", end: " << end <<". Total length = " << len << ", task width = " << width << ", blocksize = " << blocksize << ", numblocks = " << blocks << std::endl;
//     LOCK_RELEASE(output_lck);
// #endif
    for(int i = start; (i < len) && (i < end); i++){
// #ifdef DEBUG
//       LOCK_ACQUIRE(output_lck);
//       std::cout << "[DEBUG] Task " << taskid << ": thread id = " << threadid << ", i = " << i << std::endl;
//       LOCK_RELEASE(output_lck);
// #endif
      C[i] = A[i] * B[i];
    }   
  }

  // Add by Jing
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

  int blocks;    /*!< TAO implementation specific integer that holds the number of blocks per TAO */
  int blocksize; /*!< TAO implementation specific integer that holds the number of elements per block */
  int len;       /*!< TAO implementation specific integer that holds the vector length */
  double *A;     /*!< TAO implementation specific double array that holds the A vector */
  double *B;     /*!< TAO implementation specific double array that holds the B vector */
  double *C;     /*!< TAO implementation specific double array that holds the result vector */
  atomic<int> next; /*!< TAO implementation specific atomic variable to provide thread safe tracker of the number of processed blocks */
};

// #if defined(CRIT_PERF_SCHED)
// float VecMulDyn::time_table[XITAO_MAXTHREADS][XITAO_MAXTHREADS]; 
// #endif

/*! this TAO will take a set of doubles and add them all together
*/
class VecAdd : public AssemblyTask 
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
  static std::atomic<int> PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static std::atomic<int> PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
  static bool lp_task_distri_state;
  static bool hp_task_distri_state;
  static int clus0_cpu_freq; 
  static int clus1_cpu_freq;
  
  //! VecAdd TAO constructor. 
  /*!
    \param _in is the input vector for which the elements should be accumulated
    \param _out is the output element holding the summation     
    \param _len is the length of the vector 
    \param width is the number of resources used by this TAO
  */    
  VecAdd(double *_in, double *_out, int _len, int width) :
        in(_in), out(_out), len(_len), AssemblyTask(width) 
  {  

  }
  //! Inherited pure virtual function that is called by the runtime to cleanup any resources (if any), held by a TAO. 
  void cleanup() {     
  }

  //! Inherited pure virtual function that is called by the runtime upon executing the TAO. 
  /*!
    \param threadid logical thread id that executes the TAO. For this TAO, we let logical core 0 only do the addition to avoid reduction
  */
  void execute(int threadid)
  {
    // let the leader do all the additions, 
    // otherwise we need to code a reduction here, which becomes too ugly
    if(threadid != leader) return;
    *out = 0.0;
    for (int i=0; i < len; i++)
       *out += in[i];
  }

  //Add by Jing
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
  
  double *in;  /*!< TAO implementation specific double vector that holds the input to be accumulated */
  double *out; /*!< TAO implementation specific double point to the summation*/
  int len;     /*!< TAO implementation specific integer that holds the number of elements */
};
// #if defined(CRIT_PERF_SCHED)
// float VecAdd::time_table[XITAO_MAXTHREADS][XITAO_MAXTHREADS]; 
// #endif
