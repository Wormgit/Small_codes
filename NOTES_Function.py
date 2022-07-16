'''
下划线占位符：容易辨认大数的位数
枚举函数 enumerate()：不需要显性创建索引
打包函数 zip()：能同时遍历多个迭代器
解包：将值赋给正确的变量
动态属性 setattr()：用尽可能少的代码快速创建对象
密码函数 getpass()：让输入的密码不可见

包含，保留小数，补0，zip，解包，查找重复，假定
扩展名, 判断数字
查看内存消耗
'''

#是否包含特定
#string
name = 'Swaroop'
if name.startswith('Swa'):   #endwith
    print('Yes, the string starts with "Swa"')
if name.find('war') != -1:
    print('Yes, it contains the string "war"')

in   
not in

#保留两位小数
round(a,2)
print('{0:.3f}'.format(1.0/3))   对于浮点数 '0.333' 保留小数点(.)后三位

#print 补0, 格式化输出对齐补充填充(%,format,函数三种方式)
print('name:%20s,age:%06d'%(name,age)) #对齐输出：name左对齐，站位20个字符，不足的补充空格。age 使用左对齐占位6个字符，不足的补充空格

##zip() 同时遍历多个迭代器
names = ['小罗伯特唐尼', '托比·马奎尔', '克里斯蒂安·贝尔', '杰森·莫玛']
actors = ['钢铁侠', '蜘蛛侠', '蝙蝠侠', '水行侠']
universes = ['漫威', '漫威', 'DC', 'DC']

for name, actor, universe in zip(names, actors, universes):
    print(f'{name}是来自{universe}的{actor}')

a = zip(names, actors, universes)
print(*a) #只能用一次 print(list(a)) 替换也可
print(*a)


#解包
a, b, *c = 1, 2, 3, 4, 5
a, b, *_, d = 1, 2, 3, 4, 5 #从头和尾开始一一解包，再把多余的全部赋给 c

text = '5678'
#print 格式
print(text.ljust(20))
print(text.rjust(20))
print(text.center(20))
print(text.ljust(20,'/'))
print(text.rjust(20,'~'))
print(text.center(20,"*"))
print('{0:_^11}'.format('hello'))# 使用 (^) 定义 '___hello___'字符串总长度为 11
print(r"Newlines are indicated by \n") #r指定一个 原始（Raw） 字符串

# Print 换行or 不换
print('a', end=' ') #指定以空格结
s = '''This is a multi-line string. 
This is the second line.'''
print(s)  #分行符号 '''

# 查找重复 find duplicate from list
from iteration_utilities import unique_everseen
list(unique_everseen(duplicates([1,1,2,1,2,3,4,2])))
[1, 2]
# 每次遇到重复就出来一次
from iteration_utilities import duplicates
m = list(duplicates([1,1,2,1,2,3,4,2]))
[1, 1, 2, 2]


#为了满足多个（或）条件：
select_indices = np.where( np.logical_and( x > 1, x < 5) )[0] #   1 < x <5
select_indices = np.where( np.logical_or( x < 1, x > 5 ) )[0] # x <1 or x >5
array[i<4]	布尔索引
x = array([5, 2, 3, 1, 4, 5])
y = array(['f','o','o','b','a','r'])
>>> output = y[np.logical_and(x > 1, x < 5)] # desired output is ['o','o','a']
y[(1 < x) & (x < 5)]


#假定
assert

#print skill
print ('What\'s up?')
print ("What's your name?")
print('\\')
print ('first line\nthe second line')
print("first. \
second sentence.")

#后缀
m=xx.split('.')[-1]

#是否是字符or数字
isinstance(item,str)

#查看内存消耗
data.memory_usage(deep=True)

