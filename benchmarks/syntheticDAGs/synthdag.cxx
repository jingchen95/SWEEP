/*! @example 
 @brief A program that calculates dotproduct of random two vectors in parallel\n
 we run the example as ./dotprod.out <len> <width> <block> \n
 where
 \param len  := length of vector\n
 \param width := width of TAOs\n
 \param block := how many elements to process per TAO
*/
#include "synthmat.h"
#include "synthcopy.h"
#include "synthstencil.h"
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <atomic>
#include <vector>
#include <algorithm>
#include "xitao_api.h"
using namespace xitao;

float Synth_MatMul::time_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatCopy::time_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatStencil::time_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatMul::cpu_power_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatCopy::cpu_power_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatStencil::cpu_power_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatMul::ddr_power_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatCopy::ddr_power_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatStencil::ddr_power_table[NUM_DDR_AVAIL_FREQ][NUM_AVAIL_FREQ][NUMSOCKETS][XITAO_MAXTHREADS];
uint64_t Synth_MatMul::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
uint64_t Synth_MatCopy::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
uint64_t Synth_MatStencil::cycle_table[NUMFREQ][NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatMul::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatCopy::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];
float Synth_MatStencil::mb_table[NUMSOCKETS][XITAO_MAXTHREADS];
bool Synth_MatMul::time_table_state[NUMSOCKETS+1];
bool Synth_MatCopy::time_table_state[NUMSOCKETS+1];
bool Synth_MatStencil::time_table_state[NUMSOCKETS+1];
bool Synth_MatMul::best_config_state;
bool Synth_MatCopy::best_config_state;
bool Synth_MatStencil::best_config_state;
bool Synth_MatMul::enable_cpu_freq_change;
bool Synth_MatCopy::enable_cpu_freq_change;
bool Synth_MatStencil::enable_cpu_freq_change;
bool Synth_MatMul::enable_ddr_freq_change;
bool Synth_MatCopy::enable_ddr_freq_change;
bool Synth_MatStencil::enable_ddr_freq_change;
int Synth_MatMul::best_cpufreqindex;
int Synth_MatCopy::best_cpufreqindex;
int Synth_MatStencil::best_cpufreqindex;
int Synth_MatMul::best_ddrfreqindex;
int Synth_MatCopy::best_ddrfreqindex;
int Synth_MatStencil::best_ddrfreqindex;
int Synth_MatMul::best_cluster;
int Synth_MatCopy::best_cluster;
int Synth_MatStencil::best_cluster;
int Synth_MatMul::best_width;
int Synth_MatCopy::best_width;
int Synth_MatStencil::best_width;
int Synth_MatMul::second_best_cpufreqindex;
int Synth_MatCopy::second_best_cpufreqindex;
int Synth_MatStencil::second_best_cpufreqindex;
int Synth_MatMul::second_best_ddrfreqindex;
int Synth_MatCopy::second_best_ddrfreqindex;
int Synth_MatStencil::second_best_ddrfreqindex;
int Synth_MatMul::second_best_cluster;
int Synth_MatCopy::second_best_cluster;
int Synth_MatStencil::second_best_cluster;
int Synth_MatMul::second_best_width;
int Synth_MatCopy::second_best_width;
int Synth_MatStencil::second_best_width;
std::atomic<int> Synth_MatMul::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> Synth_MatCopy::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> Synth_MatStencil::PTT_UpdateFlag[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> Synth_MatMul::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> Synth_MatCopy::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
std::atomic<int> Synth_MatStencil::PTT_UpdateFinish[NUMSOCKETS][XITAO_MAXTHREADS][XITAO_MAXTHREADS];
bool Synth_MatMul::lp_task_distri_state;
bool Synth_MatCopy::lp_task_distri_state;
bool Synth_MatStencil::lp_task_distri_state;
bool Synth_MatMul::hp_task_distri_state;
bool Synth_MatCopy::hp_task_distri_state;
bool Synth_MatStencil::hp_task_distri_state;
int Synth_MatMul::clus0_cpu_freq;
int Synth_MatCopy::clus0_cpu_freq;
int Synth_MatStencil::clus0_cpu_freq;
int Synth_MatMul::clus1_cpu_freq;
int Synth_MatCopy::clus1_cpu_freq;
int Synth_MatStencil::clus1_cpu_freq;

#if defined(Haswell)
#define MAX_PACKAGES	16
#endif

extern int NUM_WIDTH_TASK[XITAO_MAXTHREADS];

//enum scheduler{PerformanceBased, EnergyAware, EDPAware};
const char *scheduler[] = { "PerformanceBased", "Paper4", "EDPAware", "RWSS"/* etc */ };

int main(int argc, char *argv[])
{
  if(argc != 10) {
    std::cout << "./synbench <Scheduler ID> <MM Block Length> <COPY Block Length> <STENCIL Block Length> <Resource Width> <TAO Mul Count> <TAO Copy Count> <TAO Stencil Count> <Degree of Parallelism>" << std::endl; 
    return 0;
  }
#if defined(Haswell)
  char event_names[MAX_PACKAGES][1][256];
	char filenames[MAX_PACKAGES][1][256];
	char basename[MAX_PACKAGES][256];
	char tempfile[256];
	long long before[MAX_PACKAGES][1];
	long long after[MAX_PACKAGES][1];
	int valid[MAX_PACKAGES][1];
	FILE *fff;
#endif

  std::ofstream timetask;
  timetask.open("data_process.sh", std::ios_base::app);

  const int arr_size = 1 << 27;
	//const int arr_size = 1 << 16;
  real_t *A = new real_t[arr_size];
  real_t *B = new real_t[arr_size];
  real_t *C = new real_t[arr_size];
  memset(A, rand(), sizeof(real_t) * arr_size);
  memset(B, rand(), sizeof(real_t) * arr_size);
  memset(C, rand(), sizeof(real_t) * arr_size);
  int schedulerid = atoi(argv[1]);
  int mm_len = atoi(argv[2]);
  int copy_len = atoi(argv[3]);
  int stencil_len = atoi(argv[4]);
  int resource_width = atoi(argv[5]); 
  int tao_mul = atoi(argv[6]);
  int tao_copy = atoi(argv[7]);
  int tao_stencil = atoi(argv[8]);
  int parallelism = atoi(argv[9]);
  int total_taos = tao_mul + tao_copy + tao_stencil;
  int nthreads = XITAO_MAXTHREADS;
  int tao_types = 0;
  int steal_DtoA = 0;
  int steal_AtoD = 0;

  std::cout << "---------------------- Test Application - Synthetic Benchmarks ---------------------\n";
  std::cout << "---------------------- You choose " << scheduler[schedulerid] << " scheduler ---------------------\n";

  if(tao_mul > 0){
    tao_types++;
  }
  if(tao_copy > 0){
    tao_types++;
  }
  if(tao_stencil > 0){
    tao_types++;
  }
  gotao_init_hw(-1,-1,-1);
  gotao_init(schedulerid, tao_types, steal_DtoA, steal_AtoD);

  int indx = 0;
  std::ofstream graphfile;
  graphfile.open ("graph.txt");
  graphfile << "digraph DAG{\n";
  
  int current_type = 0;
  int previous_tao_id = 0;
  int current_tao_id = 0;
  AssemblyTask* previous_tao;
  AssemblyTask* startTAO;

  // create first TAO
  if(tao_mul > 0) {
    previous_tao = new Synth_MatMul(mm_len, resource_width,  A + indx * mm_len * mm_len, B + indx * mm_len * mm_len, C + indx * mm_len * mm_len);
    previous_tao->tasktype = 0;
    previous_tao->kernel_name = "MM";
    graphfile << previous_tao_id << "  [fillcolor = lightpink, style = filled];\n";
    tao_mul--;
    indx++;
    if((indx + 1) * mm_len * mm_len > arr_size) indx = 0;
  } else if(tao_copy > 0){
    previous_tao = new Synth_MatCopy(copy_len, resource_width,  A + indx * copy_len * copy_len, B + indx * copy_len * copy_len);
    previous_tao->tasktype = 0;
    previous_tao->kernel_name = "CP";
    graphfile << previous_tao_id << "  [fillcolor = skyblue, style = filled];\n";
    tao_copy--;
    indx++;
    if((indx + 1) * copy_len * copy_len > arr_size) indx = 0;
  } else if(tao_stencil > 0) {
    previous_tao = new Synth_MatStencil(stencil_len, resource_width, A+ indx * stencil_len * stencil_len, B+ indx * stencil_len * stencil_len);
    previous_tao->tasktype = 0;
    previous_tao->kernel_name = "ST";
    graphfile << previous_tao_id << "  [fillcolor = palegreen, style = filled];\n";
    tao_stencil--;
    indx++;
    if((indx + 1) * stencil_len * stencil_len > arr_size) indx = 0;
  }
  startTAO = previous_tao;
  previous_tao->criticality = 1;
  total_taos--;
#ifdef random_dop
  int increment = rand()%10 * 2 + 4; // random number between 2 and 20
  for(int i = 0; i < total_taos; i+=increment)
#else
  for(int i = 0; i < total_taos; i+=parallelism) 
#endif
  {
    AssemblyTask* new_previous_tao;
    int new_previous_id;
#ifdef DVFS
    //for(int j = 0; j < tao_types; j++) {
      AssemblyTask* currentTAO;
      // switch(current_type) {
      //   case 0:
          if(tao_mul > 0) { 
            for(int k = 0; k < parallelism/tao_types; k++){
              currentTAO = new Synth_MatMul(mm_len, resource_width, A + indx * mm_len * mm_len, B + indx * mm_len * mm_len, C + indx * mm_len * mm_len);
              previous_tao->make_edge(currentTAO);  
              currentTAO->tasktype = 0;                               
              graphfile << "  " << previous_tao_id << " -> " << ++current_tao_id << " ;\n";
              graphfile << current_tao_id << "  [fillcolor = lightpink, style = filled];\n";
              tao_mul--;
              indx++;
              if((indx + 1) * mm_len * mm_len > arr_size) indx = 0;          

              if(k == 0) {
                new_previous_tao = currentTAO;
                new_previous_tao->criticality = 1;
                new_previous_id = current_tao_id;
              }
            }
            // break;
          }
        // case 1: 
          if(tao_copy > 0) {
            //for(int k = parallelism/tao_types; k < parallelism; k++){
            for(int k = 0; k < parallelism/tao_types; k++){
              currentTAO = new Synth_MatCopy(copy_len, resource_width, A + indx * copy_len * copy_len, B + indx * copy_len * copy_len);
              previous_tao->make_edge(currentTAO); 
              currentTAO->tasktype = 1;  
              graphfile << "  " << previous_tao_id << " -> " << ++current_tao_id << " ;\n";
              graphfile << current_tao_id << "  [fillcolor = skyblue, style = filled];\n";
              tao_copy--;
              indx++;
              if((indx + 1) * copy_len * copy_len > arr_size) indx = 0;     
              if(k == 0) {
                new_previous_tao = currentTAO;
                new_previous_tao->criticality = 1;
                new_previous_id = current_tao_id;
              }         
            }
            // break;
          }
        // case 2: 
          if(tao_stencil > 0) {
            //currentTAO = new Synth_MatStencil(len, resource_width);
            currentTAO = new Synth_MatStencil(stencil_len, resource_width, A+ indx * stencil_len * stencil_len, B+ indx * stencil_len * stencil_len);
            currentTAO->tasktype = 2;  
/*
            if(indx % 2 == 0){
              //currentTAO = new Synth_MatStencil(stencil_len, resource_width, A+(indx-1)*stencil_len*stencil_len, B+(indx-1)*stencil_len*stencil_len);
              currentTAO = new Synth_MatStencil(stencil_len, resource_width, A, B);
            }else{
              //currentTAO = new Synth_MatStencil(stencil_len, resource_width, B+(indx-1)*stencil_len*stencil_len, A+(indx-1)*stencil_len*stencil_len);
              currentTAO = new Synth_MatStencil(stencil_len, resource_width, B, A);
            }
*/        
            previous_tao->make_edge(currentTAO); 
            graphfile << "  " << previous_tao_id << " -> " << ++current_tao_id << " ;\n";
            graphfile << current_tao_id << "  [fillcolor = palegreen, style = filled];\n";
            tao_stencil--;
            indx++;
            if((indx + 1) * stencil_len * stencil_len > arr_size) indx = 0;
            break;
          }
        // default:
        //   if(tao_mul > 0) { 
        //     //currentTAO = new Synth_MatMul(len, resource_width);
        //     currentTAO = new Synth_MatMul(mm_len, resource_width, A + indx * mm_len * mm_len, B + indx * mm_len * mm_len, C + indx * mm_len * mm_len);
        //     previous_tao->make_edge(currentTAO); 
        //     graphfile << "  " << previous_tao_id << " -> " << ++current_tao_id << " ;\n";
        //     graphfile << current_tao_id << "  [fillcolor = lightpink, style = filled];\n";
        //     tao_mul--;
        //     break;
        //   }
      // }
      // if(j == 0) {
      //   new_previous_tao = currentTAO;
      //   new_previous_tao->criticality = 1;
      //   new_previous_id = current_tao_id;
      // }
      current_type++;
      if(current_type >= tao_types) current_type = 0;
    // }
#else
#ifdef random_dop
    for(int j = 0; j < increment; ++j)
#else    
    for(int j = 0; j < parallelism; ++j) 
#endif
    {
      AssemblyTask* currentTAO;
      switch(current_type) {
        case 0:
          if(tao_mul > 0) { 
            currentTAO = new Synth_MatMul(mm_len, resource_width, A + indx * mm_len * mm_len, B + indx * mm_len * mm_len, C + indx * mm_len * mm_len);
            currentTAO->tasktype = 0; /* If the DAG only include MM tasks, then the task type number should be 0. With multiple kernels, task type can be other numbers > 0 */
            currentTAO->kernel_name = "MM";
            previous_tao->make_edge(currentTAO);                                 
            graphfile << "  " << previous_tao_id << " -> " << ++current_tao_id << " ;\n";
            graphfile << current_tao_id << "  [fillcolor = lightpink, style = filled];\n";
            tao_mul--;
            indx++;
            if((indx + 1) * mm_len * mm_len > arr_size) indx = 0;
            break;
          }
        case 1: 
          if(tao_copy > 0) {
            currentTAO = new Synth_MatCopy(copy_len, resource_width, A + indx * copy_len * copy_len, B + indx * copy_len * copy_len);
            currentTAO->tasktype = 0; /* If the DAG only include CP tasks, then the task type number should be 0. With multiple kernels, task type can be other numbers > 0 */
            currentTAO->kernel_name = "CP";
            previous_tao->make_edge(currentTAO); 
            graphfile << "  " << previous_tao_id << " -> " << ++current_tao_id << " ;\n";
            graphfile << current_tao_id << "  [fillcolor = skyblue, style = filled];\n";
            tao_copy--;
            indx++;
            if((indx + 1) * copy_len * copy_len > arr_size) indx = 0;
            break;
          }
        case 2: 
          if(tao_stencil > 0) {
            //currentTAO = new Synth_MatStencil(len, resource_width);
            currentTAO = new Synth_MatStencil(stencil_len, resource_width, A+ indx * stencil_len * stencil_len, B+ indx * stencil_len * stencil_len);
/*              if(indx % 2 == 1){
              //currentTAO = new Synth_MatStencil(stencil_len, resource_width, A+(indx-1)*stencil_len*stencil_len, B+(indx-1)*stencil_len*stencil_len);
              currentTAO = new Synth_MatStencil(stencil_len, resource_width, A, B);
            }else{
              //currentTAO = new Synth_MatStencil(stencil_len, resource_width, B+(indx-1)*stencil_len*stencil_len, A+(indx-1)*stencil_len*stencil_len);
              currentTAO = new Synth_MatStencil(stencil_len, resource_width, B, A);
            } */
            currentTAO->tasktype = 0; /* If the DAG only include ST tasks, then the task type number should be 0. With multiple kernels, task type can be other numbers > 0 */
            currentTAO->kernel_name = "ST";     
            previous_tao->make_edge(currentTAO); 
            graphfile << "  " << previous_tao_id << " -> " << ++current_tao_id << " ;\n";
            graphfile << current_tao_id << "  [fillcolor = palegreen, style = filled];\n";
            tao_stencil--;
            indx++;
            if((indx + 1) * stencil_len * stencil_len > arr_size) indx = 0;
            break;
          }
          
        default:
          if(tao_mul > 0) { 
            //currentTAO = new Synth_MatMul(len, resource_width);
            currentTAO = new Synth_MatMul(mm_len, resource_width, A + indx * mm_len * mm_len, B + indx * mm_len * mm_len, C + indx * mm_len * mm_len);
            currentTAO->tasktype = 0;
            currentTAO->kernel_name = "MM";   
            previous_tao->make_edge(currentTAO); 
            graphfile << "  " << previous_tao_id << " -> " << ++current_tao_id << " ;\n";
            graphfile << current_tao_id << "  [fillcolor = lightpink, style = filled];\n";
            tao_mul--;
            break;
          }
      }
#ifdef random_dop
      if(j == increment - 1) 
#else
      if(j == parallelism - 1) 
#endif
      {      
        new_previous_tao = currentTAO;
        new_previous_tao->criticality = 1;
        new_previous_id = current_tao_id;
      }
      current_type++;
      if(current_type >= tao_types) current_type = 0;
    }
#endif
    previous_tao = new_previous_tao;
    previous_tao_id = new_previous_id;
#ifdef random_dop
    increment = rand() % 10 * 2 + 4;
#endif 
  }
  //close the output
  graphfile << "}";
  graphfile.close();
  gotao_push(startTAO, 0);
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  auto start1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(start);
  auto epoch1 = start1_ms.time_since_epoch();
  goTAO_start();
  goTAO_fini();
  end = std::chrono::system_clock::now();
  auto end1_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(end);
  auto epoch1_end = end1_ms.time_since_epoch();
  std::chrono::duration<double> elapsed_seconds = end-start;
  timetask << "python Energy.py " << epoch1.count() << "\t" <<  epoch1_end.count() << "\n";
  timetask.close();
  std::cout << total_taos + 1 << "," << parallelism << "," << epoch1.count() << "\t" <<  epoch1_end.count() << "," << elapsed_seconds.count() << "," << (total_taos+1) / elapsed_seconds.count() << "\n";
#if (defined Target_EPTO)
  std::cout << "Total number of steals across clusters: " << tao_total_across_steals << "\n";
#endif
  std::cout << "Total number of steals: " << tao_total_steals << "\n";
  std::cout << "\n\n";
  return 0;
}
