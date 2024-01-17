// solver.h -- jacobi solver as a TAO_PAR_FOR_2D_BASE class
#include "heat.h"
#include "tao.h"
#include "tao_parfor2D.h"

#ifdef DO_LOI
#include "loi.h"
#endif

// chunk sizes for KRD
#define KRDBLOCKX 128  // 128KB
#define KRDBLOCKY 128

#define JACOBI2D 0
#define COPY2D 1

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

class jacobi2D : public TAO_PAR_FOR_2D_BASE
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

  jacobi2D(void *a, void*c, int rows, int cols, int offx, int offy, int chunkx, int chunky, 
                         gotao_schedule_2D sched, int ichunkx, int ichunky, int width, float sta=GOTAO_NO_AFFINITY,
                         int nthread=0) 
                         : TAO_PAR_FOR_2D_BASE(a,c,rows,cols,offx,offy,chunkx,chunky,
                                         sched,ichunkx,ichunky,width,sta) {}

                int ndx(int a, int b){ return a*gotao_parfor2D_cols + b; }

                int compute_for2D(int offx, int offy, int chunkx, int chunky)
                {

                    double *in = (double *) gotao_parfor2D_in;
                    double *out = (double *) gotao_parfor2D_out;
                    double diff; double sum=0.0;

                    // global rows and cols
                    int grows = gotao_parfor2D_rows;
                    int gcols = gotao_parfor2D_cols;


                    int xstart = (offx == 0)? 1 : offx;
                    int ystart = (offy == 0)? 1 : offy;
                    int xstop = ((offx + chunkx) >= grows)? grows - 1: offx + chunkx;
                    int ystop = ((offy + chunky) >= gcols)? gcols - 1: offy + chunky;

#if DO_LOI
    kernel_profile_start();
#endif
                    for (int i=xstart; i<xstop; i++) 
                        for (int j=ystart; j<ystop; j++) {
                        out[ndx(i,j)]= 0.25 * (in[ndx(i,j-1)]+  // left
                                               in[ndx(i,j+1)]+  // right
                                               in[ndx(i-1,j)]+  // top
                                               in[ndx(i+1,j)]); // bottom
                               
// for similarity with the OmpSs version, we do not check the residual
//                        diff = out[ndx(i,j)] - in[ndx(i,j)];
//                        sum += diff * diff; 
                        }
#if DO_LOI
    kernel_profile_stop(JACOBI2D);
#if DO_KRD
    for(int x = xstart; x < xstop; x += KRDBLOCKX)
      for(int y = ystart; y < ystop; y += KRDBLOCKY)
      {
      int krdblockx = ((x + KRDBLOCKX - 1) < xstop)? KRDBLOCKX : xstop - x;
      int krdblocky = ((y + KRDBLOCKY - 1) < ystop)? KRDBLOCKY : ystop - y;
      kernel_trace1(JACOBI2D, &in[ndx(x,y)], KREAD(krdblockx*krdblocky)*sizeof(double));
      kernel_trace1(JACOBI2D, &out[ndx(x,y)], KWRITE(krdblockx*krdblocky)*sizeof(double));
      }
#endif
#endif

//                     std::cout << "compute offx " << offx << " offy " << offy 
//                          << " chunkx " << chunkx << " chunky " << chunky 
//                          << " xstart " << xstart << " xstop " << xstop 
//                          << " ystart " << ystart << " ystop " << ystop
//                          << " affinity "  << get_affinity()
//                          << " local residual is " << sum << std::endl;
		return 0;
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
};

class copy2D : public TAO_PAR_FOR_2D_BASE
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
  
  copy2D(void *a, void*c, int rows, int cols, int offx, int offy, int chunkx, int chunky, 
                         gotao_schedule_2D sched, int ichunkx, int ichunky, int width, float sta=GOTAO_NO_AFFINITY,
                         int nthread=0) 
                         : TAO_PAR_FOR_2D_BASE(a,c,rows,cols,offx,offy,chunkx,chunky,
                                         sched,ichunkx,ichunky,width,sta) {}

                  int ndx(int a, int b){ return a*gotao_parfor2D_cols + b; }

                  int compute_for2D(int offx, int offy, int chunkx, int chunky)
                  {
                    double *in = (double *) gotao_parfor2D_in;
                    double *out = (double *) gotao_parfor2D_out;
                    double diff; double sum=0.0;

                    // global rows and cols
                    int grows = gotao_parfor2D_rows;
                    int gcols = gotao_parfor2D_cols;

                    int xstart = offx;
                    int ystart = offy;

                    int xstop = ((offx + chunkx) >= grows)? grows: offx + chunkx;
                    int ystop = ((offy + chunky) >= gcols)? gcols: offy + chunky;
#if DO_LOI
    kernel_profile_start();
#endif

                    for (int i=xstart; i<xstop; i++) 
                        for (int j=ystart; j<ystop; j++) 
                           out[ndx(i,j)]= in[ndx(i,j)];

//                    std::cout << "copy: offx " << offx << " offy " << offy 
//                          << " chunkx " << chunkx << " chunky " << chunky 
//                          << " xstart " << xstart << " xstop " << xstop 
//                          << " ystart " << ystart << " ystop " << ystop
//                          << " affinity "  << get_affinity()  
//                          << std::endl;
#if DO_LOI
    kernel_profile_stop(COPY2D);
#if DO_KRD
    for(int x = xstart; x < xstop; x += KRDBLOCKX)
      for(int y = ystart; y < ystop; y += KRDBLOCKY)
      {
      int krdblockx = ((x + KRDBLOCKX - 1) < xstop)? KRDBLOCKX : xstop - x;
      int krdblocky = ((y + KRDBLOCKY - 1) < ystop)? KRDBLOCKY : ystop - y;
      kernel_trace1(JACOBI2D, &in[ndx(x,y)], KREAD(krdblockx*krdblocky)*sizeof(double));
      kernel_trace1(JACOBI2D, &out[ndx(x,y)], KWRITE(krdblockx*krdblocky)*sizeof(double));
      }
#endif
#endif
    return 0;
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
};

/*
 * Blocked Jacobi solver: one iteration step
 */
/*double relax_jacobi (unsigned sizey, double (*u)[sizey], double (*utmp)[sizey], unsigned sizex)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = 4;
    bx = sizex/nbx;
    nby = 4;
    by = sizey/nby;
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                utmp[i][j]= 0.25 * (u[ i][j-1 ]+  // left
                         u[ i][(j+1) ]+  // right
                             u[(i-1)][ j]+  // top
                             u[ (i+1)][ j ]); // bottom
                diff = utmp[i][j] - u[i][j];
                sum += diff * diff; 
            }

    return sum;
}
*/
/*
 * Blocked Red-Black solver: one iteration step
 */
/*
double relax_redblack (unsigned sizey, double (*u)[sizey], unsigned sizex )
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = 4;
    bx = sizex/nbx;
    nby = 4;
    by = sizey/nby;
    // Computing "Red" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                unew= 0.25 * (    u[ i][ (j-1) ]+  // left
                      u[ i][(j+1) ]+  // right
                      u[ (i-1)][ j ]+  // top
                      u[ (i+1)][ j ]); // bottom
                diff = unew - u[i][j];
                sum += diff * diff; 
                u[i][j]=unew;
            }
    }

    // Computing "Black" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                unew= 0.25 * (    u[ i][ (j-1) ]+  // left
                      u[ i][ (j+1) ]+  // right
                      u[ (i-1)][ j     ]+  // top
                      u[ (i+1)][ j     ]); // bottom
                diff = unew - u[i][ j];
                sum += diff * diff; 
                u[i][j]=unew;
            }
    }

    return sum;
}
*/
/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
/*
double relax_gauss (unsigned padding, unsigned sizey, double (*u)[sizey], unsigned sizex )
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = 8;
    bx = sizex/nbx;
    nby = 8;
    by = sizey/nby;
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++){
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++)
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                unew= 0.25 * (    u[ i][ (j-1) ]+  // left
                      u[ i][(j+1) ]+  // right
                      u[ (i-1)][ j     ]+  // top
                      u[ (i+1)][ j     ]); // bottom
                diff = unew - u[i][ j];
                sum += diff * diff; 
                u[i][j]=unew;
                }
        }

    return sum;
}
*/
