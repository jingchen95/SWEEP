#ifndef _CONFIG_H
#define _CONFIG_H

/* Set different target (only one is allowed) */
//#define ERASE_target_perf
//#define ERASE_target_energy_method1
// #define ERASE_target_energy_method2 // JOSS design
#define Target_EPTO // Paper 4 design

/* Platforms */
#define TX2
//#define ALDERLAKE

#define XITAO_MAXTHREADS 6  //==> Move to Make CXXflags in command line
#define START_CLUSTER_A 0   //==> Move to Make CXXflags in command line
#define START_CLUSTER_B 2   //==> Move to Make CXXflags in command line
#define NUMSOCKETS 2        //==> Move to Make CXXflags in command line
#define NUMFREQ 2           /*Train tasks with 2 different frequencies: currently 2.04GHz and 1.11GHz */
#define TRAIN_MAX_A_FREQ 2035200 // Sample and model phase
#define TRAIN_MED_A_FREQ 1113600 // Sample and model phase
#define TRAIN_MED_A_FREQ_idx 6 
#define TRAIN_MAX_B_FREQ 2035200 // Sample and model phase
#define TRAIN_MED_B_FREQ 1113600 // Sample and model phase
#define TRAIN_MED_B_FREQ_idx 6
#define NUM_AVAIL_FREQ 12
#define NUM_DDR_AVAIL_FREQ 5 // No DDR frequency tuning om ALder lake
#define DDR_FREQ_TUNING
#define INTERVAL_RECOMPUTATION // Paper 4 design: recompute the task distribution every interval 
#define COMP_INTERVAL 4.0 // seconds
#define TRAIN_METHOD_1
#define Performance_Model_Time    // Only use sampled time to calculate memory-boundness
//#define Performance_Model_Cycle // Use Cycles to calculate memory-boundness
#define FINE_GRAIN_THRESHOLD 0.001
#define NUM_TRAIN_TASKS 5
#define HP_LP_Threshold 2
#define TASK_NUM_INTERVAL 100

/* SWEEP adaption to various EPTO metrics: EDP = Power * Time^2, ED2P = Power * Time^3, E2DP = Power^2 * Time^3 */
#define DELAY_METRIC 2.0 
#define POWER_METRIC 1.0 

#define SWEEP_Overhead // Measure the timing overhead that consumed by SWEEP scheduler

//#define random_dop // Benchmarks: synthetic benchmark using random parallelism between 2 - 20

//#define DEBUG // Open debug option

//#define Pro_fix_D 
//#define Loose_Parallelism
// #define AAWS_CASE // try AAWS case when alpha > beta (Denver - 0.65GHz, A57 - 1.11GHz with memory copy)

#define DOP_TRACE // task parallelism tracing - count the total number of ready tasks (inc. tasks in queue + released tasks)
#define ADD_CPU_FREQ_TUNING // Paper 4 add CPU frequency tuning
#define ADD_DDR_FREQ_TUNING // Paper 4 add DDR frequency tuning

#define WORK_STEALING // Enable or disable work stealing
#define STEAL_ATTEMPTS 1
//#define WITHIN_CLUSTER // [NO USE] Enable or disable work stealing within cluster
//#define ACROSS_CLUSTER_AFTER_IDLE // [NO USE] Enable or disable work stealing across cluster

/* Schedulers with sleep for idle work stealing loop */
#define SLEEP
#define IDLE_SLEEP 1000              // Idle sleep try to reduce the power consumption
#define SLEEP_LOWERBOUND 1000000    //Sleep time bound settings (nanoseconds)
#define SLEEP_UPPERBOUND 8000000
#define EAS_SLEEP /* Energy aware scheduler */
#define EAS_NoCriticality /* Energy aware scheduler without criticality */
#define RWSS_SLEEP /* Random working stealing with sleep */

/* Accumulte the total exec time this thread complete */
#define EXECTIME

/* Accumulte the number of task this thread complete */
#define NUMTASKS_MIX

/* 2021 Sep 28: paper II takes DVFS into account*/
// #define dvfs

//#define Model_Computation_Overhead // measure the time that consumed for populating the look-up tables, inc execution time and power tables
//#define Search_Overhead // Measure the time that consumed by searching for the best config in poly_task.cpp
//#define Optimized_Search // Optimized searching method to reduce the searching space 
//#define Exhastive_Search // Search for all possible configurations

// Scenario 1: We allow some % of performance slowdown for each task
//#define perf_contraints  // Explore JOSS with performance contraints  
//#define PERF_SLOWDOWN 0.1 // COnstraint: 0.1 means allowing 10% of performance slowdown, basedline is when running with highest frequency and all fastest core 

// Scenario 2: Based on JOSS (minimizing energy) version, we specify how much performance improvement we want to achieve
//#define perf_improve
//#define PERF_SPEEDUP 1 // it means 1.2 times speedup based on JOSS default execution time

//#define ALLOWSTEALING

//#define AcrossCLustersTest // Allow stealing across clusters - starter, not the final version

//#define JOSS_NoMemDVFS // Reducing the total energy consumption (inc. CPU+DDR) without mem DVFS 

//#define EDP_TEST_  // Test minimize EDP per task

//#define FineStrategyTest
//#define FINE_GRAIN_THRESHOLD 0.0001

/* Performance Counters - calculate Memory-boundness */
// #define PERF_COUNTERS

/* Power Model for Hebbe in Synthetic DAG Benchmarks */
// #define Haswell
// #define NUMSOCKETS 2
// #define COREPERSOCKET 10

// Power Profiling Kernel tasks, especially for applications with multiple kernels
//#define PowerProfiling

/* Average same configuration */
#define AveCluster

/* Blancing EDP between Denver and A57 clusters */
// #define ERASE_target_edp_method1

/* Test if energy increase (cost) is lower than execution time reduction (benefit) */
//#define ERASE_target_edp_method2

/* Test if energy optimization can use the same method as EDP */
//#define ERASE_target_test

/* Check the PTT Accuracy between prediction and real time*/
//#define PTTaccuracy

/* Check the accuracy between the energy prediction and real energy */
//#define Energyaccuracy
/* If application has multiple kernels, can not use single ptt_full (sparseLU test)*/
//#define MultipleKernels


/* ERASE uses second energy efficient config - to compare the energy with most energy efficent one*/
//#define second_efficient

//#define CATA
//#define Hermes

// #define FREQLEVELS 2

// #define EAS_PTT_TRAIN 2
/* Frequency setting: 0=>MAX, 1=>MIN */
#define A57 0
#define DENVER 0





/* Performance-oriented with sleep */
// #define FCAS_SLEEP
// #define CRI_COST
//#define CRI_PERF

// CATS Scheduler
//#define CATS



// #define ACCURACY_TEST




/* Accumulte the PTT visiting time */
// #define OVERHEAD_PTT

// #define ONLYCRITICAL

// #define PARA_TEST

//#define NEED_BARRIER

//#define DynaDVFS

#define GOTAO_THREAD_BASE 0
#define GOTAO_NO_AFFINITY (1.0)
#define TASK_POOL 100
#define TAO_STA 1

#define L1_W   1
#define L2_W   2
#define L3_W   6
#define TOPOLOGY { L1_W, L2_W}
#define GOTAO_HW_CONTEXTS 1
#endif
