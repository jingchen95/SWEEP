/*! \file 
@brief Contains the TAOs needed for sparselu
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
using namespace std;

/*! this TAO will take a set of doubles and add them all together*/
class LU0 : public AssemblyTask 
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

  //! LU0 TAO constructor. 
  /*!
    \param _in is the input vector for which the elements should be accumulated
    \param _out is the output element holding the summation     
    \param _len is the length of the vector 
    \param width is the number of resources used by this TAO
  */    
  LU0(double *_in, int _len, int width) :
        diag(_in), BSIZE(_len), AssemblyTask(width) 
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
    //std::cout << "LU0 tid: " << threadid << std::endl;
    //if(threadid != leader) return;
    int tid = threadid - leader;
    int i, j, k;
    int steps = (BSIZE+width-1 ) / width;     
//    std::cout << "LU0 task needs " << steps << " steps. \n";
    int min = tid * steps;
    int max = std::min(((tid+1)*steps), BSIZE);                                                                    
//    std::cout << "Thread " << threadid <<  " min: "<< min << ", max: " << max << std::endl;
    //for (k=0; k<BSIZE; k++)
    for(k = min; k < max; k++)
       for (i=k+1; i<BSIZE; i++) {
          diag[i*BSIZE+k] = diag[i*BSIZE+k] / diag[k*BSIZE+k];
          for (j=k+1; j<BSIZE; j++)
             diag[i*BSIZE+j] = diag[i*BSIZE+j] - diag[i*BSIZE+k] * diag[k*BSIZE+j];
      }
  }

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
  double *diag;  /*!< TAO implementation specific double vector that holds the input to be accumulated */
  int BSIZE;     /*!< TAO implementation specific integer that holds the number of elements */
};

/*! this TAO will take a set of doubles and add them all together */
class FWD : public AssemblyTask 
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


  //! FWD TAO constructor. 
  /*!
    \param _in is the input vector for which the elements should be accumulated
    \param _out is the output element holding the summation     
    \param _len is the length of the vector 
    \param width is the number of resources used by this TAO
  */    
  FWD(double *_in, double *_out, int _len, int width) :
        diag(_in), col(_out), BSIZE(_len), AssemblyTask(width) 
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
    //std::cout << "FWD tid: " << threadid << std::endl;
    // if(threadid != leader) return;
    int tid = threadid - leader;
    int i, j, k;
    int steps = (BSIZE+width-1 ) / width;     
    int min = tid * steps;
    int max = std::min(((tid+1)*steps), BSIZE);                                                                    
    //for (k=0; k<BSIZE; k++)
    for(k = min; k < max; k++)
      for (i=k+1; i<BSIZE; i++)
        for (j=0; j<BSIZE; j++)
             col[i*BSIZE+j] = col[i*BSIZE+j] - diag[i*BSIZE+k]*col[k*BSIZE+j];
  }
  
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
  double *diag;  /*!< TAO implementation specific double vector that holds the input to be accumulated */
  double *col; /*!< TAO implementation specific double point to the summation*/
  int BSIZE;     /*!< TAO implementation specific integer that holds the number of elements */
};

/*! this TAO will take a set of doubles and add them all together
*/
class BDIV : public AssemblyTask 
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


  //! BDIV TAO constructor. 
  /*!
    \param _in is the input vector for which the elements should be accumulated
    \param _out is the output element holding the summation     
    \param _len is the length of the vector 
    \param width is the number of resources used by this TAO
  */    
  BDIV(double *_in, double *_out, int _len, int width) :
        diag(_in), row(_out), BSIZE(_len), AssemblyTask(width) 
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
    //std::cout << "BDIV tid: " << threadid << std::endl;
    int tid = threadid - leader;
    int i, j, k;
    int steps = (BSIZE+width-1)/width;     
    int min = tid * steps;
    int max = std::min(((tid+1)*steps), BSIZE);                                                                    
    for(i = min; i < max; i++){
       for (k=0; k<BSIZE; k++) {
          row[i*BSIZE+k] = row[i*BSIZE+k] / diag[k*BSIZE+k];
          for (j=k+1; j<BSIZE; j++)
             row[i*BSIZE+j] = row[i*BSIZE+j] - row[i*BSIZE+k]*diag[k*BSIZE+j];
       }
    }
  }

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
  double *diag; /*!< TAO implementation specific double point to the summation*/
  double *row;  /*!< TAO implementation specific double vector that holds the input to be accumulated */
  int BSIZE;     /*!< TAO implementation specific integer that holds the number of elements */
};

/*! this TAO will take a set of doubles and add them all together
*/
class BMOD : public AssemblyTask 
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


  //! BMOD TAO constructor. 
  /*!
    \param _in is the input vector for which the elements should be accumulated
    \param _out is the output element holding the summation     
    \param _len is the length of the vector 
    \param width is the number of resources used by this TAO
  */    
  BMOD(double *_in1, double *_in2,  double *_out, int _len, int width) :
        row(_in1), col(_in2), inner(_out), BSIZE(_len), AssemblyTask(width) 
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
    //std::cout << "BMOD tid: " << threadid << std::endl;
    // if(threadid != leader) return;
    int tid = threadid - leader;
    int i, j, k;
    int steps = (BSIZE+width-1 ) / width;     
    int min = tid * steps;
    int max = std::min(((tid+1)*steps), BSIZE);                                                                    
    for(i = min; i < max; i++)
       for (k=0; k<BSIZE; k++) 
          for (j=0; j<BSIZE; j++)
             inner[i*BSIZE+j] = inner[i*BSIZE+j] - row[i*BSIZE+k]*col[k*BSIZE+j];
  }
  
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
  double *row;  /*!< TAO implementation specific double vector that holds the input to be accumulated */
  double *col;  /*!< TAO implementation specific double vector that holds the input to be accumulated */
  double *inner; /*!< TAO implementation specific double point to the summation*/
  int BSIZE;     /*!< TAO implementation specific integer that holds the number of elements */
};
