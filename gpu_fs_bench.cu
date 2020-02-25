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


long PG_SIZE = 0;

__global__
void print_char_parallel(void * file_mem) {
  char * c = (char *) file_mem;
  c += blockIdx.x * blockDim.x + threadIdx.x;
  printf("thead:%d:block:%d:char:%c\n", threadIdx.x, blockIdx.x, *c); 
}

__host__
void * file_mmap(char * file_name, size_t allocation_size) {
  assert(allocation_size % PG_SIZE == 0);

  int fd = open(file_name, O_RDWR | O_CREAT, 0666);
  void * file_mem = mmap(0, allocation_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
  if (file_mem == MAP_FAILED) {
    perror("mmap error\n");
  }

  return file_mem;
}

__host__
void * gpu_file_mmap(char * file_name, size_t allocation_size) {
  assert(allocation_size % PG_SIZE == 0);

  void * file_mem = file_mmap(file_name, allocation_size);

  CUDA_CALL(cudaHostRegister(file_mem, PG_SIZE, cudaHostRegisterMapped));
  
  return file_mem;
}

__host__
void * gpu_file_malloc(char * file_name, size_t allocation_size) {
  assert(allocation_size % PG_SIZE == 0);

  void * file_mem = file_mmap(file_name, allocation_size);
  void * unified_ptr = NULL;

  CUDA_CALL(cudaMalloc(&unified_ptr, allocation_size));
  memcpy(unified_ptr, file_mem, allocation_size);

  return unified_ptr;
}

__host__
void * gpu_file_malloc_managed(char * file_name, size_t allocation_size) {
  assert(allocation_size % PG_SIZE == 0);

  void * file_mem = file_mmap(file_name, allocation_size);
  void * dev_ptr = NULL;

  CUDA_CALL(cudaMallocManaged(&dev_ptr, allocation_size));
  CUDA_CALL(cudaMemcpy(dev_ptr, file_mem, allocation_size, cudaMemcpyHostToDevice));

  return dev_ptr;
}

/* Run a memory access benchmark on a memory mapped file
 * bufffer        - memory to access
 * rand_state_arr - random number generators
 * total_threads  - total number of threads running kernel
 * size           - size of buffer
 * granularity    - how much each thread will access
 * random_access  - if readers access randomly 
 * readers        - designates threads read data from buffer
 * writers        - designates threads write data to buffer
 *
 * XXX For simiplicity make size an even multiple of total_threads
 */
__global__
void launch_bench(char * buffer,
                  curandState_t * rand_state_arr,
                  size_t total_threads,
                  size_t size,
                  size_t granularity, 
                  bool random_access,
                  bool readers, 
                  bool writers) {
    unsigned global_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned bytes_per_thread = size / total_threads;
    unsigned min_index = bytes_per_thread * global_index;
    //unsigned max_index = min_index + bytes_per_thread - 1;

    curandState_t * rand_state = rand_state_arr + global_index;

    //printf("id %d:%d; Index range %d:%d\n", blockIdx.x, threadIdx.x, 
           //min_index, max_index);

    unsigned index_offset = 0;
    for (int i = 0; i < bytes_per_thread; bytes_per_thread += granularity) { 
      if (random_access) {
        index_offset = curand(rand_state) % bytes_per_thread;
      } else { /* Sequential */
        index_offset = i;
      }

      volatile char tmp = 0;
      if (readers) {
        tmp = buffer[min_index + index_offset];
      } 
      
      if (writers) {
        buffer[min_index + index_offset] = (curand(rand_state) % 92) + 33; // valid ascii range
      }
    }
}

/* Initialize random number generators for each thread */
__global__
void random_init(curandState_t * rand_state_arr) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(0, 0, 0, (rand_state_arr + index));
}

enum allocation_type {
  _mmap = 0,
  _malloc = 1,
  _malloc_managed = 2
};

/* Usage ./gpu_fs.o file_size blocks threads allocation_type read write random 
   blocks - blocks to run
   threads - threads per block
   file_name - name of file to map
   file_size - size of file mapping, should be multiple of PG_SIZE
   allocation_type - 0 = mmap, 1 = cudaMalloc, 2 = cudaMallocManaged
   read = read data?
   write = write data?
   random = random access?
   */
int main(int argc, char ** argv) {
  assert(argc == 9);
  DEBUG_PRINT("Hello cuda\n");

  /* Parse args */
  //unsigned NUM_BLOCKS = atoi(argv[1]);
  //unsigned NUM_THREADS = atoi(argv[2]);
  unsigned NUM_THREADS = 1;
  unsigned NUM_BLOCKS = 1;
  char * file_name = argv[3];
  size_t file_size = atoi(argv[4]);
  unsigned allocation_type = atoi(argv[5]);
  bool read = atoi(argv[6]);
  bool write = atoi(argv[7]);
  bool random = atoi(argv[8]);

  printf("filename: %s\n", file_name);

  PG_SIZE = sysconf(_SC_PAGE_SIZE);
  assert(file_size % PG_SIZE == 0);

  /* Setup file memory */
  char * shared_mem = NULL;
  //switch (allocation_type) {
  //  case _mmap:
  //    shared_mem = (char *) gpu_file_mmap(file_name, file_size);
  //    break;
  //  case _malloc:
  //    shared_mem = (char *) gpu_file_malloc(file_name, file_size);
  //    break;
  //  case _malloc_managed:
  //    shared_mem = (char *) gpu_file_malloc_managed(file_name, file_size);
  //    break;
  //}
  shared_mem = (char *) gpu_file_mmap(file_name, file_size);
  DEBUG_PRINT("Ptr: %p\n", shared_mem);

  /* Setup random number generator */
  curandState_t * rand_state_arr = NULL;
  CUDA_CALL(cudaMalloc(&rand_state_arr, 
                       sizeof(curandState_t) * (NUM_BLOCKS * NUM_THREADS)));
  random_init<<<NUM_BLOCKS, NUM_THREADS>>>((curandState_t *) rand_state_arr);

  cudaEvent_t start, end;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&end));

  CUDA_CALL(cudaEventRecord(start));
  launch_bench<<<NUM_BLOCKS, NUM_THREADS>>>(shared_mem, 
                                            rand_state_arr,
                                            NUM_BLOCKS * NUM_THREADS,
                                            file_size, 
                                            1 /*granularity*/,
                                            random,
                                            read,
                                            write);
  //CUDA_CALL(cudaEventRecord(end));
  //cudaError_t err = cudaEventSynchronize(end);
  //printf("cudaerr %x\n", err);
  float mili = 0;
  //CUDA_CALL(cudaEventElapsedTime(&mili, start, end));

  printf("Kernel time: %f\n", mili);

  //__sync_bool_compare_and_swap(sharedMem, (int) 0, (int) 1);
  CUDA_CALL(cudaDeviceSynchronize());
}
