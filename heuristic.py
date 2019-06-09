from sys import argv

# <n machines> <input file> <output file>

nm = int(argv[1])

with open(argv[2]) as fp:
    while True:
        if fp.readline().startswith('[Tl]'):
            break
    x = fp.readline()
            
x = x.strip(' \n').split(' ')
x = [int(i) for i in x]

T = float(sum(x))
t = T / float(nm)

s = 0
k = 0
tt = t
part = [[] for _ in range(nm)]
for j in x:
    s += j
    part[k].append(j)
    if s >= tt:
        k += 1
        tt += t

with open(argv[3], 'w') as fp:
    i = 0
    for p in part:
        fp.write('{} {}\n'.format(str(i), str(i+len(p))))
        i += len(p)
    
