all:
	nvcc gpu_fs_bench.cu -g -o gpu_fs.o -DDEBUG=1
