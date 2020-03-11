import subprocess
import statistics
import re
import sys

#blocks = [1, 2, 4, 8, 16] #XXX add more
#threads = [1024, 2048, 4096] #XXX add more
#mapping_size = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576] #4k to 1gb

executable = './gpu_fs.o'
input_file = 'testFile.txt'
mapping_size = [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912]#, 1073741824] # 1mb to 1gb
#mapping_size = [1073741824]
allocation_type = [0, 1, 2] #mmap, cudaMalloc, cudaMallocManaged
#allocation_type = [1, 2] #mmap, cudaMalloc, cudaMallocManaged
read = [True] #[True, False]
write = [False]#[True, False]
random_access = [True, False]
#XXX none, read mostly, set prefered location(need to pair with id), set accesed by
gpu_memadvise = [0, 1, 3, 5]
cpu_memadvise = [-1]
#device_id = [-1, 0] #use CUDA_VISIBLE_DEVICES to set device so we can use 0
device_id = [0] #use CUDA_VISIBLE_DEVICES to set device so we can use 0
regex = re.compile(r'[+-]?([0-9]*[.])?[0-9]+')




def main():
    num_blocks = 32
    num_threads = 1024
    print('blocks, threads, mapping size, allocation type, read, write, random, gpu_advise, dev_id, cpu_advise\n')
    for size in mapping_size:
        for allocation in allocation_type:
                for reading in read:
                    for writing in write:
                        for random in random_access:
                            for gpu_advise in gpu_memadvise:
                                if not allocation == 2 and not gpu_advise == 0:
                                    #skip gpu memadvise for non mallocManaged
                                    continue

                                for dev_id in device_id:
                                    for cpu_advise in cpu_memadvise:
                                        load_results = list()
                                        exec_results = list()
                                        cleanup_results = list()
                                        command = [executable, str(num_blocks), str(num_threads),
                                                   input_file, str(size), str(allocation),
                                                   str(int(reading == True)), str(int(writing == True)),
                                                   str(int(random == True)), str(gpu_advise),
                                                   str(dev_id), str(cpu_advise)]

                                        #print(command)
                                        sys.stderr.write(str(command))
                                        sys.stderr.write('\n')
                                        #continue
                                        for i in range(3): #XXX run 10 iterations
                                            p = subprocess.Popen(command,
                                                                 stdout=subprocess.PIPE)
                                            (std, _) = p.communicate()

                                            output = std.decode('utf-8')
                                            sys.stderr.write(output)
                                            output = output.splitlines()
                                            #print(regex.search(output[0]))#.group())
                                            load_results.append(float(regex.search(output[0]).group()))
                                            exec_results.append(float(regex.search(output[1]).group()))
                                            cleanup_results.append(float(regex.search(output[2]).group()))
                                        print('{0}, {1}, {2}, {3}, {4}, {5}, {6}: {7}, {8}, {9}, {10}, {11}, {12}'.format(
                                               num_blocks, num_threads, size, allocation,
                                               reading, writing, random, statistics.mean(load_results),
                                               statistics.mean(exec_results), statistics.mean(cleanup_results),
                                               gpu_advise, dev_id, cpu_advise))



if __name__ == '__main__':
    main()




