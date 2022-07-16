
import sys
print (sys.version)

import statistics
print('NormalDist' in dir(statistics))
from statistics import NormalDist

m1 = 5
std1 = 1
m2 = 4
std2 = 2
fff = NormalDist(mu=m1, sigma=std1)
mm=NormalDist(mu=m1, sigma=std1).overlap(NormalDist(mu=m2, sigma=std2))
mm2=(0.5*NormalDist(mu=m1, sigma=std1)).overlap(NormalDist(mu=m2, sigma=std2))
print(mm,mm2)



