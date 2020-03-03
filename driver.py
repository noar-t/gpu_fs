import subprocess
import statistics
import re


executable = './gpu_fs.o'
blocks = [1, 2, 4, 8, 16] #XXX add more
threads = [1024, 2048, 4096] #XXX add more
input_file = 'testFile.txt'
mapping_size = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576] #4k to 1gb
allocation_type = [0, 1, 2] #mmap, cudaMalloc, cudaMallocManaged
read = [True, False]
write = [True, False]
random_access = [True, False]

regex = re.compile(r'[+-]?([0-9]*[.])?[0-9]+')




def main():
    print('blocks, threads, mapping size, allocation type, read, write, random\n')
    for num_blocks in blocks:
        for num_threads in threads:
            for size in mapping_size:
                for allocation in allocation_type:
                    for reading in read:
                        for writing in write:
                            for random in random_access:
                                load_results = list()
                                exec_results = list()
                                cleanup_results = list()
                                command = [executable, str(num_blocks), str(num_threads),
                                           input_file, str(size), str(allocation),
                                           str(int(reading == True)), str(int(writing == True)),
                                           str(int(random == True))]

                                #print(command)
                                for i in range(10): #XXX run 10 iterations
                                    p = subprocess.Popen(command,
                                                         stdout=subprocess.PIPE)
                                    (std, _) = p.communicate()

                                    output = std.decode('utf-8')
                                    output = output.splitlines()
                                    #print(regex.search(output[0]))#.group())
                                    load_results.append(float(regex.search(output[0]).group()))
                                    exec_results.append(float(regex.search(output[1]).group()))
                                    cleanup_results.append(float(regex.search(output[2]).group()))
                                    #print(regex.search(output[1]))#.group())
                                    #print(regex.search(output[2]))#.group())
                                #sub_results.append(int(outstd.split('\n')[0]))
                                print('{0}, {1}, {2}, {3}, {4}, {5}, {6}: {7}, {8}, {9}'.format(
                                      num_blocks, num_threads, size, allocation,
                                       reading, writing, random, statistics.mean(load_results),
                                       statistics.mean(exec_results), statistics.mean(cleanup_results)))



if __name__ == '__main__':
    main()




