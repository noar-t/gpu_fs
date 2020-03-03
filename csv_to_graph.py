from enum import IntEnum
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
import matplotlib.pyplot as plt

import csv

columns = ['blocks', ' threads', ' mapping size', ' allocation type',
           ' read', ' write', ' random']

#three graphs transfer time, execution time, total tim
# XXX for now we will lock it at 2 blocks, 4096 threads, read, write, randomm

class Indexes(IntEnum):
    BLOCK   = 0
    THREAD  = 1
    SIZE    = 2
    TYPE    = 3
    READ    = 4
    WRITE   = 5
    RANDOM  = 6
    LOAD_T  = 7
    EXEC_T  = 8
    CLEAN_T = 9

class Type(IntEnum):
    MMAP     = 0
    MALLOC   = 1
    MALLOC_M = 2



#x: transfer size
#y: blocks + threads
#z: total time
#x = [list(), list(), list()]
#y = [list(), list(), list()]
mmap     = list()
malloc_m = list()
malloc   = list()

with open('output.csv', mode='r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if len(row) > 0 and i > 1:
            blocks = int(row[Indexes.BLOCK])
            threads = int(row[Indexes.THREAD])
            if blocks == 16 and threads == 4096: #XXX limiting the axis
                allocation_size = int(row[Indexes.SIZE])
                allocation_type = int(row[Indexes.TYPE])

                read = 'True' == row[Indexes.READ][1:]
                write = 'True' == row[Indexes.WRITE][1:]
                random = 'True' == row[Indexes.RANDOM][1:]
                print(read)

                if random and write and not read:
                    transfer_time = float(row[Indexes.LOAD_T])
                    exec_time = float(row[Indexes.EXEC_T])
                    clean_time = float(row[Indexes.CLEAN_T])

                    total_time = transfer_time + exec_time# + clean_time
                    array_elem = (allocation_size, total_time)

                    if allocation_type == Type.MMAP:
                        mmap.append(array_elem)
                    elif allocation_type == Type.MALLOC_M:
                        malloc_m.append(array_elem)
                    elif allocation_type == Type.MALLOC:
                        malloc.append(array_elem)



x_vals = [x for x,y in mmap]
x_vals.sort()

sort_func = lambda a: a[1]
mmap.sort(key=sort_func)
malloc_m.sort(key=sort_func)
malloc.sort(key=sort_func)

mmap     = [y for x,y in mmap]
malloc_m = [y for x,y in malloc_m]
malloc   = [y for x,y in malloc]
print(x_vals)
print(mmap)
print(malloc_m)
print(malloc)

plt.xscale('log', basex=2)
plt.plot(x_vals, mmap, label = 'mmap Unified')
plt.plot(x_vals, malloc, label = 'CUDA malloc')
plt.plot(x_vals, malloc_m, label = 'malloc Managed')

plt.xlabel('Mapping Size (b)')
plt.ylabel('Time (msec)')

plt.title('1 Block 1024 Thread Sequential Read')
plt.legend()
plt.savefig('1_1024_sequential_read.png')

