/* cuda_fs_bench.cu
   Noah Thornton
   This is a benchmark to compare the performance of different file
   access methods within CUDA on a gpu.
  */

/* Generic includes */
#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Cuda includes */
#include <curand.h>
#include <curand_kernel.h>

#include "gpu_fs_bench.ch"

struct global {
  long PG_SIZE;
  unsigned NUM_THREADS;
  unsigned NUM_BLOCKS;
  unsigned TOTAL_THREADS;
  unsigned NUM_PAGES;
  unsigned PADDED_NUM_PAGES;
  unsigned PAGES_PER_THREAD;
  char * file_name;
  size_t file_size;
  unsigned allocation_type;
  bool readers;
  bool writers;
  bool random;
  void * file_mem;
  char * shared_mem;
  cudaMemoryAdvise advise;
  int device;
  int mmap_advise;
} global;

long PG_SIZE = 0;

__global__
void print_char_parallel(void * file_mem) {
  char * c = (char *) file_mem;
  c += blockIdx.x * blockDim.x + threadIdx.x;
  printf("thead:%d:block:%d:char:%c\n", threadIdx.x, blockIdx.x, *c); 
}

__host__
void * file_mmap(char * file_name, size_t allocation_size) {
  assert(allocation_size % global.PG_SIZE == 0);

  int fd = open(file_name, O_RDWR | O_CREAT, 0666);
  void * file_mem = mmap(0, allocation_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, fd, 0);
                         //MAP_SHARED | MAP_LOCKED, fd, 0);
  if (file_mem == MAP_FAILED) {
    perror("mmap error\n");
  }

  // TODO add madvise
  // madvise(file_mem, allocation_size, global.mmap_advice);
  // MADV_RANDOM
  // MADV_SEQUENTIAL

  return file_mem;
}

__host__
void * gpu_file_mmap(char * file_name, size_t allocation_size) {
  assert(allocation_size % global.PG_SIZE == 0);

  void * file_mem = file_mmap(file_name, allocation_size);

  CUDA_CALL(cudaHostRegister(file_mem, allocation_size, cudaHostRegisterMapped));
  
  return file_mem;
}

__host__
void * gpu_file_malloc(char * file_name, size_t allocation_size) {
  assert(allocation_size % global.PG_SIZE == 0);

  void * file_mem = file_mmap(file_name, allocation_size);
  global.file_mem = file_mem;
  void * dev_ptr = NULL;

  CUDA_CALL(cudaMalloc(&dev_ptr, allocation_size));
  CUDA_CALL(cudaMemcpy(dev_ptr, file_mem, allocation_size, cudaMemcpyHostToDevice));

  return dev_ptr;
}

__host__
void * gpu_file_malloc_managed(char * file_name, size_t allocation_size) {
  assert(allocation_size % global.PG_SIZE == 0);

  void * file_mem = file_mmap(file_name, allocation_size);
  void * unified_ptr = NULL;

  CUDA_CALL(cudaMallocManaged(&unified_ptr, allocation_size, cudaMemAttachGlobal));
  memcpy(unified_ptr, file_mem, allocation_size);

  return unified_ptr;
}

__host__
char * map_file_to_gpu() {
  char * shared_mem = NULL;
  switch (global.allocation_type) {
    case _MMAP:
      //fprintf(stderr, "MMAP\n");
      shared_mem = (char *) gpu_file_mmap(global.file_name, global.file_size);
      break;
    case _MALLOC:
      //fprintf(stderr, "MALLOC\n");
      shared_mem = (char *) gpu_file_malloc(global.file_name, global.file_size);
      break;
    case _MALLOC_MANAGED:
      //fprintf(stderr, "MALLOC_MANAGED\n");
      shared_mem = (char *) gpu_file_malloc_managed(global.file_name, global.file_size);
      break;
  }

  if (global.advise) {
    assert(global.allocation_type != _MALLOC);
    // #define cudaCpuDeviceId ((int)-1)

    // cudaMemAdviseSetReadMostly
    // cudaMemAdviseSetPreferredLocation
    // cudaMemAdviseSetAccessedBy
    int dev_id = 0;
    CUDA_CALL(cudaGetDevice(&dev_id));
    //printf("Mem %p:size:%d\n", shared_mem, global.file_size);
    //printf("Advise %d:id:%d\n", global.advise, dev_id);
    CUDA_CALL(cudaMemAdvise(shared_mem, global.file_size,
                            global.advise, dev_id));
  }

  return shared_mem;
}

/* Run a memory access benchmark on a memory mapped file
 * buffer           - memory backed file to access
 * page_seq         - array of ints representing page number sequence to access
 * rand_state_arr   - random number generators
 * total_threads    - total number of threads running kernel
 * pages_pre_thread - number of pages for each thread
 * PG_SIZE          - size of a page
 * granularity      - how much each thread will access
 * random_access    - if readers access randomly 
 * readers          - designates threads read data from buffer
 * writers          - designates threads write data to buffer
 */
__global__
void launch_bench(char * buffer,
                  int * page_seq,
                  curandState_t * rand_state,
                  size_t total_threads,
                  size_t pages_per_thread,
                  size_t PG_SIZE,
                  size_t granularity, 
                  bool random_access,
                  bool readers, 
                  bool writers) {
  unsigned global_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned bytes_per_thread = pages_per_thread * PG_SIZE;
  page_seq = page_seq + (pages_per_thread * global_index); // TODO verify correct
  

  int cur_page_num = 0;
  char * cur_page = NULL;



  for (int i = 0; i < pages_per_thread; i++) {
    cur_page_num = page_seq[i];
    if (cur_page_num != -1) {
      cur_page = buffer + (cur_page_num * PG_SIZE);

      for (int j = 0; j < PG_SIZE; j += granularity) { 
        volatile char tmp = 0;
        if (readers) {
          tmp = cur_page[j];
        } 
        
        if (writers) {
          cur_page[j] = (curand(rand_state) % 92) + 33; // valid ascii range
        }
      }
    }
  }
}

__host__
void update_file_from_gpu() {
  switch (global.allocation_type) {
    case _MALLOC:
      CUDA_CALL(cudaMemcpy(global.file_mem, 
                           global.shared_mem,
                           global.file_size,
                           cudaMemcpyDeviceToHost));
    case _MMAP:
    case _MALLOC_MANAGED:
      break;
  }
}

/* Initialize random number generators for each thread */
__global__
void gpu_rand_init(curandState_t * rand_state_arr) {
      unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
          curand_init(index, 0, 0, (rand_state_arr + index));
}

/* Setup sequence of pages for each thread to access*/
__host__
int * page_seq_init() {
  size_t rand_seq_size = sizeof(int) * global.PADDED_NUM_PAGES;
  int * rand_seq_arr = (int *) malloc(rand_seq_size);

  printf("total thread: %d\n", global.TOTAL_THREADS);
  printf("num pages: %d\n", global.NUM_PAGES);
  printf("padded num pages: %d\n", global.PADDED_NUM_PAGES);

  /* Populate sequence */
  for (int i = 0; i < global.PADDED_NUM_PAGES; i++) {
    if (i < global.NUM_PAGES) {
      rand_seq_arr[i] = i;
    } else {
      rand_seq_arr[i] = -1;
    }
  }

  /* Randomize if necessary */
  if (global.random) {
    for (int i = 0; i < global.PADDED_NUM_PAGES - 1; i++) 
    {
      int j = i + rand() / (RAND_MAX / (global.PADDED_NUM_PAGES - i) + 1);
      int tmp = rand_seq_arr[j];
      rand_seq_arr[j] = rand_seq_arr[i];
      rand_seq_arr[i] = tmp;
    }
  }

  int * dev_rand_seq_arr = NULL;
  CUDA_CALL(cudaMalloc(&dev_rand_seq_arr, rand_seq_size));
  CUDA_CALL(cudaMemcpy(dev_rand_seq_arr, rand_seq_arr, 
                       rand_seq_size, cudaMemcpyHostToDevice));

  return dev_rand_seq_arr;
}

/* Usage ./gpu_fs.o UNUSED UNUSED <file_name> <file_size> <allocation_type>
                    <read> <write> <random> <cudaMemoryAdvise> <dev_id> <madvise> */
int main(int argc, char ** argv) {
  assert(argc == 12);


  /* Parse args */
  global.file_name        = (char *) argv[3];
  global.file_size        = (size_t) atoi(argv[4]);
  global.allocation_type  = (unsigned) atoi(argv[5]);
  global.readers          = (bool) atoi(argv[6]);
  global.writers          = (bool) atoi(argv[7]);
  global.random           = (bool) atoi(argv[8]);
  global.file_mem         = NULL;
  global.shared_mem       = NULL;
  global.advise           = (cudaMemoryAdvise) atoi(argv[9]); //XXX 0 means no advise
  global.device           = (int) atoi(argv[10]); //XXX only matters if advise is non 0
  global.mmap_advise      = (int) atoi(argv[11]); //XXX -1 no advice

  /* Compute other handy values */
  global.PG_SIZE = (long) sysconf(_SC_PAGE_SIZE);
  //global.NUM_THREADS = 1024;
  //global.NUM_BLOCKS = 32;
  global.NUM_THREADS = 1;
  global.NUM_BLOCKS = 2;
  global.TOTAL_THREADS = global.NUM_THREADS * global.NUM_BLOCKS;
  global.NUM_PAGES = global.file_size / global.PG_SIZE;
  if (global.NUM_PAGES % global.TOTAL_THREADS == 0) {
    global.PADDED_NUM_PAGES = global.NUM_PAGES;
  } else {
    global.PADDED_NUM_PAGES = global.NUM_PAGES +
                              ((global.TOTAL_THREADS) - (global.NUM_PAGES % global.TOTAL_THREADS));
  }
  global.PAGES_PER_THREAD = global.PADDED_NUM_PAGES / global.TOTAL_THREADS;
  printf("pages_pre_thread %d\n", global.PAGES_PER_THREAD);

  assert(global.file_size % global.PG_SIZE == 0);
  assert(global.PADDED_NUM_PAGES % global.TOTAL_THREADS == 0);


  /* Setup file memory */
  cudaEvent_t start, end;
  float mili = 0;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&end));

  CUDA_CALL(cudaEventRecord(start));
  global.shared_mem = map_file_to_gpu();
  CUDA_CALL(cudaEventRecord(end));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&mili, start, end));
  printf("Memory init time: %f\n", mili);
  
  /* Setup random number generator */
  curandState_t * rand_state_arr = NULL;
  size_t rand_state_size = sizeof(curandState_t) * (global.NUM_BLOCKS * global.NUM_THREADS);
  CUDA_CALL(cudaMalloc(&rand_state_arr, rand_state_size));
  gpu_rand_init<<<global.NUM_BLOCKS, global.NUM_THREADS>>>((curandState_t *) rand_state_arr);

  /* Setup sequence of pages to access */
  int * dev_rand_seq_arr = page_seq_init();

  /* Run benchmark */
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&end));
  CUDA_CALL(cudaEventRecord(start));
  launch_bench<<<global.NUM_BLOCKS, global.NUM_THREADS>>>(global.shared_mem, 
                                            dev_rand_seq_arr,
                                            rand_state_arr,
                                            global.TOTAL_THREADS,
                                            global.PAGES_PER_THREAD,
                                            global.PG_SIZE,
                                            1 /*granularity*/,
                                            global.random,
                                            global.readers,
                                            global.writers);
  CUDA_CALL(cudaEventRecord(end));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&mili, start, end));

  printf("Kernel time: %f\n", mili);

  /* Copy data back to file from gpu if malloc */
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&end));
  CUDA_CALL(cudaEventRecord(start));
  update_file_from_gpu();
  CUDA_CALL(cudaEventRecord(end));
  CUDA_CALL(cudaEventSynchronize(end));
  CUDA_CALL(cudaEventElapsedTime(&mili, start, end));
  printf("Copyback time: %f\n", mili);

  CUDA_CALL(cudaDeviceSynchronize());
}
