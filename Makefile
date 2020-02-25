all:
	nvcc gpu_fs_bench.cu -g -o gpu_fs.o

debug:
	nvcc gpu_fs_bench.cu -g -o gpu_fs_debug.o -DDEBUG=1
