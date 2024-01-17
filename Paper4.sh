#!/bin/bash

# --- Synthetic DAGs - very high parallelism ---
# --- Goal: test if paper 4 design achieves the same performance (load balancing) as GRWS --- "
target="EDP_4s"
parallelism="16 8 4"

for metric in $target
do
for dop in $parallelism
do
    for((k=0;k<3;k++))
    do
        # ./benchmarks/syntheticDAGs/synbench 3 256 0 0 1 10000 0 0 $dop >> ./Results/MM_GRWS_p${dop}.txt
        # sleep 30 
        echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
        echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
        echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
        echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        sleep 5
        echo "/*---------------------------------------------------------------*/"
        echo "Start running Paper 4 - MM - 256 --- with parallelism = $dop, $k th execution"
        export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench_${metric} 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_${metric}_p${dop}.txt
        echo "/*---------------------------------------------------------------*/"
        sleep 15
    done
done

for dop in $parallelism
do
    for((k=0;k<5;k++))
    do
        # ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_GRWS_p${dop}.txt
        # sleep 30 
        echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
        echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
        echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
        echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        sleep 5
        echo "/*---------------------------------------------------------------*/"
        echo "Start running Paper 4 - MC - 4096 --- with parallelism = $dop, $k th execution"
        export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench_${metric} 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_${metric}_p${dop}.txt
        echo "/*---------------------------------------------------------------*/"
        sleep 15
    done
done


for dop in $parallelism
do
    for((k=0;k<5;k++))
    do
        echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
        echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
        echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
        echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        sleep 5
        echo "/*---------------------------------------------------------------*/"
        echo "Start running Paper 4 - ST - 2048 --- with parallelism = $dop, $k th execution"
        export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench_${metric} 1 0 0 2048 1 0 0 50000 $dop >> ./Results/ST_${metric}_p${dop}.txt
        echo "/*---------------------------------------------------------------*/"
        sleep 15
    done
done

# --- Sparse LU ---
for((k=0;k<15;k++))
do
    echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
    echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
    echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
    echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    sleep 3
    echo "/*---------------------------------------------------------------*/"
    echo "PAPER 4: sparse LU 32, 256 Begin the $k execution! "
    export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/sparselu/sparselu_${metric} 1 32 256 >> ./Results/slu_32_256_${metric}.txt
    echo "PAPER 4 End the $k execution! "
    echo "/*---------------------------------------------------------------*/"
    sleep 10
done

for((k=0;k<15;k++))
do
    echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
    echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
    echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
    echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    sleep 3
    echo "/*---------------------------------------------------------------*/"
    echo "PAPER 4: sparse LU 32, 512 Begin the $k execution! "
    export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/sparselu/sparselu_${metric} 1 32 512 >> ./Results/slu_32_512_${metric}.txt
    echo "PAPER 4 End the $k execution! "
    echo "/*---------------------------------------------------------------*/"
    sleep 10
done

for((k=0;k<15;k++))
do
    echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
    echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
    echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
    echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    sleep 3
    echo "/*---------------------------------------------------------------*/"
    echo "PAPER 4: sparse LU 64, 256 Begin the $k execution! "
    export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/sparselu/sparselu_${metric} 1 64 256 >> ./Results/slu_64_256_${metric}.txt
    echo "PAPER 4 End the $k execution! "
    echo "/*---------------------------------------------------------------*/"
    sleep 10
done


for((k=0;k<15;k++))
do
    echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
    echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
    echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
    echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    sleep 3
    echo "/*---------------------------------------------------------------*/"
    echo "PAPER 4: sparse LU 64, 512 Begin the $k execution! "
    export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/sparselu/sparselu_${metric} 1 64 512 >> ./Results/slu_64_512_${metric}.txt
    echo "PAPER 4 End the $k execution! "
    echo "/*---------------------------------------------------------------*/"
    sleep 10
done
done


# --- Fibnacci ---
# for((k=0;k<1;k++))
# do
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 800000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 1881600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 1881600 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 5
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 - Fib - Begin the $k execution!"
#         # 57236 tasks with term 55 grain_size 34
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/fibonacci/fibonacci 3 55 34 >> ./Results/fib_ED2P.txt
#         sleep 5
#         echo "Paper 4 End the $k execution!"
#         echo "/*---------------------------------------------------------------*/"
# done

# --- Dot Product ---
# for((k=0;k<15;k++))
# do
#     echo "/*---------------------------------------------------------------*/"
#     echo "Paper 4 - Dot Product - Begin the $k execution!"
#     echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#     echo 1331200000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#     echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#     echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#     echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#     sleep 5
#     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/dotproduct/dotprod 1 1000 40000000 1 1000000 >> ./Results/dotprod.txt
#     sleep 5
#     echo "Paper 4 End the $k execution!"
#     echo "/*---------------------------------------------------------------*/"
# done

# --- 2D Heat ---
# 2023/08/18, Step 1: run Heat with Paper 4 final version;
# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - small - with CPU + MEM frequency tuning - ED2P - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao 1 ./benchmarks/heat/small.dat >> ./Results/heat_small_ED2P.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - big - with CPU + MEM frequency tuning - ED2P- $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao 1 ./benchmarks/heat/big.dat >> ./Results/heat_big_ED2P.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - huge - with CPU + MEM frequency tuning - ED2P - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao 1 ./benchmarks/heat/huge.dat >> ./Results/heat_huge_ED2P.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - small - with CPU + MEM frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao-cpu-mem 1 ./benchmarks/heat/small.dat >> ./Results/heat_small_CPU_MEM.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - big - with CPU + MEM frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao-cpu-mem 1 ./benchmarks/heat/big.dat >> ./Results/heat_big_CPU_MEM.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - huge - with CPU + MEM frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao-cpu-mem 1 ./benchmarks/heat/huge.dat >> ./Results/heat_huge_CPU_MEM.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# 2023/08/18, Step 3: run Heat with Paper 4 with only CPU frequency tuning;
# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - small - with CPU frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao-cpu 1 ./benchmarks/heat/small.dat >> ./Results/heat_small_CPU.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - big - with CPU frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao-cpu 1 ./benchmarks/heat/big.dat >> ./Results/heat_big_CPU.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - huge - with CPU frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao-cpu 1 ./benchmarks/heat/huge.dat >> ./Results/heat_huge_CPU.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# 2023/08/18, Step 2: run Heat with Paper 4 without any frequency tuning;
# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - small - NO frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao 1 ./benchmarks/heat/small.dat >> ./Results/heat_small.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - big- NO frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao 1 ./benchmarks/heat/big.dat >> ./Results/heat_big.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done

# for((k=0;k<10;k++))
# do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Paper 4 Begin the Heat - huge - NO frequency tuning - $k execution! "
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 3 
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/heat/heat-tao 1 ./benchmarks/heat/huge.dat >> ./Results/heat_huge.txt
#         sleep 3
#         echo "Paper 4 End the $k execution! "
#         echo "/*---------------------------------------------------------------*/"        
# done


# for dop in $parallelism
# do
#     for((k=0;k<5;k++))
#     do
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 5
#         echo "/*---------------------------------------------------------------*/"
#         echo "Start running Paper 4 - ST (CPU DVFS) - 2048 --- with parallelism = $dop, $k th execution"
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 1 0 0 2048 1 0 0 50000 $dop >> ./Results/ST_2048_CPU_p${dop}.txt
#         echo "/*---------------------------------------------------------------*/"
#         sleep 15
#     done
# done





# for dop in $parallelism
# do
#     for((k=0;k<3;k++))
#     do
#         echo "/*---------------------------------------------------------------*/"
#         echo "Start running Paper 4 - ST - 2048 --- with parallelism = $dop, $k th execution"
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 1 0 0 2048 1 0 0 50000 $dop >> ./Results/ST_2048_p${dop}.txt
#         echo "/*---------------------------------------------------------------*/"
#         # jetson_clocks --show >> ./Results/MC_Paper4_p${dop}.txt
#         sleep 15
#     done
# done

# for dop in $parallelism
# do
#     for((k=0;k<10;k++))
#     do
#         echo 1 >/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
#         echo 1866000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
#         echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state
#         echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#         echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#         sleep 5
#         echo "/*---------------------------------------------------------------*/"
#         echo "Start running Paper 4 - ST - 512 --- with parallelism = $dop, $k th execution"
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 1 0 0 512 1 0 0 50000 $dop >> ./Results/ST_ED2P_p${dop}.txt
#         echo "/*---------------------------------------------------------------*/"
#         # jetson_clocks --show >> ./Results/MC_Paper4_p${dop}.txt
#         sleep 5
#     done
# done


# --- Synthetic DAGs - Relatively HIGH parallelism - Monitor AAWS case - config.h: #define AAWS_CASE ---

# for dop in $parallelism
# do
#     echo 1113600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#     echo 499200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    # for((k=0;k<2;k++))
    # do
    #     echo "/*---------------------------------------------------------------*/"
    #     echo "Start running fixed assignment - MC d2_w1 --- with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_d2_w1; ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 1 0 5000 0 $dop >> ./Results/MC_D2_W1_p${dop}.txt
    #     echo "/*---------------------------------------------------------------*/"
    #     jetson_clocks --show
    #     sleep 10
    # done
    # for((k=0;k<2;k++))
    # do
    #     echo "/*---------------------------------------------------------------*/"
    #     echo "Start running fixed assignment - MC A4_w1 --- with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_A4_w1; ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 1 0 5000 0 $dop >> ./Results/MC_A4_W1_p${dop}.txt
    #     echo "/*---------------------------------------------------------------*/"
    #     jetson_clocks --show
    #     sleep 10
    # done
    # for((k=1;k<3;k++))
    # do
    #     echo "/*---------------------------------------------------------------*/"
    #     echo "Start running fixed assignment - MC A4_w2 --- with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_A4_w2; ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 2 0 5000 0 $dop >> ./Results/MC_A4_W2_p${dop}.txt
    #     echo "/*---------------------------------------------------------------*/"
    #     jetson_clocks --show
    #     sleep 10
    # done
    # for((k=1;k<3;k++))
    # do
    #     echo "/*---------------------------------------------------------------*/"
    #     echo "Start running fixed assignment - MC d2_w2 --- with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_d2_w2; ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 2 0 5000 0 $dop >> ./Results/MC_D2_W2_p${dop}.txt
    #     echo "/*---------------------------------------------------------------*/"
    #     jetson_clocks --show
    #     sleep 10
    # done
    # for((k=0;k<5;k++))
    # do
    #     echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    #     echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    #     sleep 5
    #     echo "/*---------------------------------------------------------------*/"
    #     echo "Start running Paper 4 - MM - 256 --- with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 1 256 0 0 1 5000 0 0 $dop >> ./Results/MM_256_AAWS_p${dop}.txt
    #     echo "/*---------------------------------------------------------------*/"
    #     jetson_clocks --show
    #     sleep 10 
    # done
    # echo 1113600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    # echo 499200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    # sleep 5
    # for((k=1;k<3;k++))
    # do
    #     echo "/*---------------------------------------------------------------*/"
    #     echo "Start running GRWS - MM - 256 --- with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 3 256 0 0 1 5000 0 0 $dop >> ./Results/MM_256_GRWS_p${dop}.txt
    #     echo "/*---------------------------------------------------------------*/"
    #     jetson_clocks --show
    #     sleep 10
    # done
    # for((k=0;k<5;k++))
    # do
    #     echo 2035200 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    #     echo 2035200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    #     sleep 5
    #     echo "/*---------------------------------------------------------------*/"
    #     echo "Start running Paper 4 - MC - 4096 --- with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 1 0 4096 0 1 0 5000 0 $dop >> ./Results/MC_4096_AAWS_p${dop}.txt
    #     echo "/*---------------------------------------------------------------*/"
    #     jetson_clocks --show
    #     sleep 10 
    # done
    # echo 1113600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    # echo 499200 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
    # sleep 5
    # for((k=0;k<3;k++))
    # do
    #     echo "/*---------------------------------------------------------------*/"
    #     echo "Start running GRWS - MC - 4096 --- with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 1 0 5000 0 $dop >> ./Results/MC_4096_GRWS_p${dop}.txt
    #     echo "/*---------------------------------------------------------------*/"
    #     jetson_clocks --show
    #     sleep 10
    # done
# done


# for((k=0;k<2;k++))
# do
#     echo "/*---------------------------------------------------------------*/"
#     echo "GRWS: sparse LU 32, 256 Begin the $k execution! "
#     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/sparselu/sparselu 3 32 256 >> ./Results/slu_32_256_GRWS.txt
#     echo "GRWS End the $k execution! "
#     echo "/*---------------------------------------------------------------*/"
#     sleep 20
# done



# for((k=0;k<2;k++))
# do
#     echo "/*---------------------------------------------------------------*/"
#     echo "GRWS: sparse LU 32, 512 Begin the $k execution! "
#     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/sparselu/sparselu 3 32 512 >> ./Results/slu_32_512_GRWS.txt
#     echo "GRWS End the $k execution! "
#     echo "/*---------------------------------------------------------------*/"
#     sleep 20
# done



# for((k=0;k<2;k++))
# do
#     echo "/*---------------------------------------------------------------*/"
#     echo "GRWS: sparse LU 64, 256 Begin the $k execution! "
#     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/sparselu/sparselu 3 64 256 >> ./Results/slu_64_256_GRWS.txt
#     echo "GRWS End the $k execution! "
#     echo "/*---------------------------------------------------------------*/"
#     sleep 20
# done



# for((k=0;k<2;k++))
# do
#     echo "/*---------------------------------------------------------------*/"
#     echo "GRWS: sparse LU 64, 512 Begin the $k execution! "
#     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/sparselu/sparselu 3 64 512 >> ./Results/slu_64_512_GRWS.txt
#     echo "GRWS End the $k execution! "
#     echo "/*---------------------------------------------------------------*/"
#     sleep 20
# done


# --- Synthetic DAGs - Relatively LOW parallelism [] ---
# parallelism="2"
# for dop in $parallelism
# do
#     for((k=0;k<3;k++))
#     do
#         echo "Start running MM - Paper4 with parallelism = $dop, $k th execution"
#         export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_LP_p${dop}.txt
#         sleep 30
#     done
        # export XITAO_LAYOUT_PATH=./ptt_layout_d2_w2; ./benchmarks/syntheticDAGs/synbench 3 256 0 0 2 10000 0 0 $dop >> ./Results/MM_2D_p${dop}.txt
        # sleep 30 
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MC - Paper4 with parallelism = $dop, $k th execution"
    #     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_LP_p${dop}.txt
    #     sleep 30
    #     # export XITAO_LAYOUT_PATH=./ptt_layout_d2_w2; ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 2 0 20000 0 $dop >> ./Results/MC_2D_p${dop}.txt
    #     # sleep 30 
    # done
# done

# --- Synthetic DAGs - how to detect Low or high task parallelism? - using GRWS as test ---
# parallelism="8 9 10 11"
# for dop in $parallelism
# do
#     echo "Start running GRWS with parallelism = $dop"
#     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 3 256 0 0 1 1000 0 0 $dop >> MM_GRWS_DOPTest.txt
# done

# parallelism="7 8 9 10 11"
# for dop in $parallelism
# do
#     echo "Start running GRWS with parallelism = $dop"
#     export XITAO_LAYOUT_PATH=./ptt_layout_tx2; ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 1 0 2000 0 $dop >> MC_GRWS_DOPTest.txt
# done

# --- Synthetic DAGs - Relatively LOW parallelism [WS + interval = 4s] ---
# parallelism="8 6 4 2"
# for dop in $parallelism
# do
#     for((k=0;k<3;k++))
#     do
#         echo "Start running Paper4 with parallelism = $dop, $k th execution"
#         # ./benchmarks/syntheticDAGs/synbench 3 256 0 0 1 10000 0 0 $dop >> ./Results/MM_GRWS_p${dop}.txt
#         ./benchmarks/syntheticDAGs/synbench 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_WS_4s_p${dop}.txt
#         # sleep 30 
#         # ./benchmarks/syntheticDAGs/synbench 3 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_GRWS_p${dop}.txt
#         # ./benchmarks/syntheticDAGs/synbench 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_WS_4s_p${dop}.txt
#         sleep 30 
#     done
# done


    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MM --- Paper4 - No work stealing - with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_NoWS 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_NoWS_p${dop}.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MM --- Paper4 - with work stealing - with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_WS 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_WS_p${dop}.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MM --- Paper4 - with work stealing - with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_WS_Appr3 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_WS_p${dop}_Appr3.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MM --- Paper4 - No work stealing - 4s interval with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_NoWS_4s 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_NoWS_4s_p${dop}.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MM --- Paper4 - with work stealing - 4s interval with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_WS_4s 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_WS_4s_p${dop}.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MM --- Paper4 - with work stealing - 4s interval with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_WS_4s_Appr3 1 256 0 0 1 10000 0 0 $dop >> ./Results/MM_WS_4s_p${dop}_Appr3.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MC --- Paper4 - No work stealing - with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_NoWS 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_NoWS_p${dop}.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MC --- Paper4 - with work stealing - with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_WS 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_WS_p${dop}.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MC --- Paper4 - with work stealing - with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_WS_Appr3 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_WS_p${dop}_Appr3.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MC --- Paper4 - No work stealing - 4s interval with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_NoWS_4s 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_NoWS_4s_p${dop}.txt
    #     sleep 30 
    # done
    # for((k=0;k<3;k++))
    # do
    #     echo "Start running MC --- Paper4 - with work stealing - 4s interval with parallelism = $dop, $k th execution"
    #     ./benchmarks/syntheticDAGs/synbench_WS_4s 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_WS_4s_p${dop}.txt
    #     sleep 30 
    # done
#     for((k=0;k<3;k++))
#     do
#         echo "Start running MC --- Paper4 - with work stealing - 4s interval with parallelism = $dop, $k th execution"
#         ./benchmarks/syntheticDAGs/synbench_WS_4s_Appr3 1 0 4096 0 1 0 20000 0 $dop >> ./Results/MC_WS_4s_p${dop}_Appr3.txt
#         sleep 30 
#     done


