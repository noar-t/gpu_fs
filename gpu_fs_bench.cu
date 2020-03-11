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
                         MAP_SHARED, fd, 0);
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

  if (global.advise) { //TODO add a flag for memadvise
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
    //void * tmp = NULL;
    //CUDA_CALL(cudaMalloc(&tmp, 4));
    //CUDA_CALL(cudaMallocManaged(&tmp, 4096, cudaMemAttachGlobal));
    //CUDA_CALL(cudaMemAdvise(shared_mem, 4096, cudaMemAdviseSetReadMostly, dev_id));
  }

  return shared_mem;
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
    unsigned max_index = min_index + bytes_per_thread - 1;

    curandState_t * rand_state = rand_state_arr + global_index;

    //printf("id %d:%d; Index range %d:%d\n", blockIdx.x, threadIdx.x, 
           //min_index, max_index);


    unsigned index_offset = 0;
    for (int i = 0; i < bytes_per_thread; i += granularity) { 
      if (random_access) {
        index_offset = curand(rand_state) % bytes_per_thread;
      } else { /* Sequential */
        index_offset = i;
      }
      assert(index_offset + min_index <= max_index);

      volatile char tmp = 0;
      if (readers) {
        tmp = buffer[min_index + index_offset];
      } 
      
      if (writers) {
        buffer[min_index + index_offset] = (curand(rand_state) % 92) + 33; // valid ascii range
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
void random_init(curandState_t * rand_state_arr) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(index, 0, 0, (rand_state_arr + index));
}

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
  assert(argc == 12);

  /* Parse args */
  global = {
    .PG_SIZE =         (long) sysconf(_SC_PAGE_SIZE),
    .NUM_THREADS =     1024,//(unsigned) atoi(argv[1]),
    .NUM_BLOCKS =      32,//(unsigned) atoi(argv[2]),
    .file_name =       (char *) argv[3],
    .file_size =       (size_t) atoi(argv[4]),
    .allocation_type = (unsigned) atoi(argv[5]),
    .readers =         (bool) atoi(argv[6]),
    .writers =         (bool) atoi(argv[7]),
    .random =          (bool) atoi(argv[8]),
    .file_mem =        NULL,
    .shared_mem =      NULL,
    .advise =          (cudaMemoryAdvise) atoi(argv[9]), //XXX 0 means no advise
    .device =          (int) atoi(argv[10]), //XXX only matters if advise is non 0
    .mmap_advise =     (int) atoi(argv[11]), //XXX -1 no advice
  };
  assert(global.file_size % global.PG_SIZE == 0);
  assert(global.file_size % global.NUM_THREADS * global.NUM_BLOCKS == 0);

  //int blockSize;   // The launch configurator returned block size
  //int minGridSize; // The minimum grid size needed to achieve the
  //                 // maximum occupancy for a full device launch
  //int gridSize;    // The actual grid size needed, based on input size

  //cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                      //launch_bench, 0, 0);


  /* Setup file memory */
  cudaEvent_t start, end;
  float mili = 0;
  char * shared_mem = NULL;
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
  random_init<<<global.NUM_BLOCKS, global.NUM_THREADS>>>((curandState_t *) rand_state_arr);

  /* Run benchmark */
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&end));
  CUDA_CALL(cudaEventRecord(start));
  launch_bench<<<global.NUM_BLOCKS, global.NUM_THREADS>>>(global.shared_mem, 
                                            rand_state_arr,
                                            global.NUM_BLOCKS * global.NUM_THREADS,
                                            global.file_size, 
                                            1 /*granularity*/,
                                            random,
                                            read,
                                            write);
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
