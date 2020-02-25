#ifdef DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__);
#else
#define DEBUG_PRINT(...) do {} while(false)
#endif

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d:Error:%d\n",__FILE__,__LINE__,x); \
      exit(EXIT_FAILURE);}} while(0)

__global__
void print_char_parallel(void * file_mem);

__host__
void * gpu_file_mmap(char * file_name, size_t allocation_size);
