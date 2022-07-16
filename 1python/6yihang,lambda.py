#[表达式 for 变量 in 列表 if 条件1 if condiction 2]
#       for varible2 xxxx


import json
import glob, os
import shutil
import copy


#lambda
fun= lambda x,y:x+y
#x=int(input('x='))    #这里要定义int整数，否则会默认为字符串
#y=int(input('y='))
#print(fun(x,y))
print(fun(3,6))

def fun(x,y):
	return (x+y)

print(list(map(fun,[1],[2])))
list(map(fun,[1,2],[3,4]))


# 一行完成 and set tuple
li = [1,2,3,4,5,6,7,8,9]

# list format
print ([x**2 for x in li if x>5])
even = [x for x in li if x % 2 == 0]

#dict format
kk = (dict([(x,x*10) for x in li if x > 5 if x<8]))
print (kk)

print ([ (x, y) for x in range(10) if x % 2 >0 if x > 3 for y in range(10) if y > 7 if y != 8 ])

# use 下标记
vec=[2,4,6]
vec2=[4,3,-9]
sq = [vec[i]+vec2[i] for i in range(len(vec))]
print (sq)

# with function
testList = [1,2,3,4]
def mul2(x):
    return x*2
print ([mul2(i) for i in testList])

# simple function
worked = True
result = 'done' if worked else 'not yet'
print(result)

# one by one
print ([x*y for x in [1,2,3] for y in [1,10,100]])

# 2d
list_2d = [ [0 for i in range(5)] for i in range(5)] 
print(list_2d)
