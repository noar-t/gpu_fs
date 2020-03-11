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
    GPU_A   = 10
    DEV_ID  = 11
    CPU_A   = 12

class AllocType(IntEnum):
    MMAP     = 0
    MALLOC   = 1
    MALLOC_M = 2

class MemAdvise(IntEnum):
    UNSET       = 0
    READMOSTLY  = 1
    PREFEREDLOC = 3
    ACCESSEDBY  = 5

class DevID(IntEnum):
    CPU = -1
    GPU = 0


#x: transfer size
#y: blocks + threads
#z: total time
#x = [list(), list(), list()]
#y = [list(), list(), list()]
mmap              = list()
malloc_m          = list()
malloc_m_read     = list()
malloc_m_prefered = list()
malloc_m_accessed = list()
malloc            = list()

with open('advise_read_results.csv', mode='r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        print(row)
        blocks = int(row[Indexes.BLOCK])
        threads = int(row[Indexes.THREAD])
        allocation_size = int(row[Indexes.SIZE])
        allocation_type = int(row[Indexes.TYPE])

        read = 'True' == row[Indexes.READ][1:]
        write = 'True' == row[Indexes.WRITE][1:]
        random = 'True' == row[Indexes.RANDOM][1:]

        if random:
            transfer_time = float(row[Indexes.LOAD_T])
            exec_time = float(row[Indexes.EXEC_T])
            clean_time = float(row[Indexes.CLEAN_T])

            total_time = transfer_time + exec_time# + clean_time
            array_elem = (allocation_size, total_time)

            if allocation_type == AllocType.MMAP:
                mmap.append(array_elem)
            elif allocation_type == AllocType.MALLOC_M:
                advise = int(row[Indexes.GPU_A])
                if advise == MemAdvise.UNSET:
                    malloc_m.append(array_elem)
                elif advise == MemAdvise.READMOSTLY:
                    malloc_m_read.append(array_elem)
                elif advise == MemAdvise.PREFEREDLOC:
                    malloc_m_prefered.append(array_elem)
                elif advise == MemAdvise.ACCESSEDBY:
                    malloc_m_accessed.append(array_elem)
            elif allocation_type == AllocType.MALLOC:
                malloc.append(array_elem)



x_vals = [x for x,y in mmap]
x_vals.sort()

sort_func = lambda a: a[1]
mmap.sort(key=sort_func)
malloc_m.sort(key=sort_func)
malloc_m_read.sort(key=sort_func)
malloc_m_prefered.sort(key=sort_func)
malloc_m_accessed.sort(key=sort_func)
malloc.sort(key=sort_func)

mmap     = [y for x,y in mmap]
malloc_m = [y for x,y in malloc_m]
malloc_m_read = [y for x,y in malloc_m_read]
malloc_m_prefered = [y for x,y in malloc_m_prefered]
malloc_m_accessed = [y for x,y in malloc_m_accessed]
malloc   = [y for x,y in malloc]
print(x_vals)
print(mmap)
print(malloc_m)
print(malloc_m_read)
print(malloc_m_prefered)
print(malloc_m_accessed)
print(malloc)

plt.xscale('log', basex=2)
#plt.yscale('log', basey=10)
plt.plot(x_vals, mmap, label = 'mmap')
plt.plot(x_vals, malloc, label = 'CUDA malloc')
plt.plot(x_vals, malloc_m, label = 'malloc Managed - No Hint')
plt.plot(x_vals, malloc_m_read, label = 'malloc Managed - ReadMostly')
plt.plot(x_vals, malloc_m_prefered, label = 'malloc Managed - PreferedLocation (GPU)')
plt.plot(x_vals, malloc_m_accessed, label = 'malloc Managed - AccessedBy (GPU)')

plt.xlabel('Mapping Size (b)')
plt.ylabel('Time (msec)')

plt.title('32x1024 Random Read Advise Bench')
plt.legend()
plt.savefig('32_1024_read_only_advise.png', dpi=1200)

