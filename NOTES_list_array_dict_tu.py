列表、元组与字符串,同样拥有一种切片(Slicing)运算符
下标[]，slice [:],call(argumnets),属性引用.attribute
(绑定or 元组)，[list], {dic:value}, {set}
列表是可变的（Mutable）而字符串是不可变的（Immutable）
*param 组成 Tuple  **param 组成 Dictionary

不可变数据（3 个）：Number（数字）、String（字符串）、Tuple（元组）； 
可变数据（3 个）：List（列表）、Dictionary（字典）、Set（集合）。 
###########################  list 操作     ##########################
#a.append()

# string to a rough int
a='2.1'
print(int(float(a)))

#切片
name[1:3]
name[:-1]

#查 find, 索引
num[0]
l.index('you') # the first find index based on value
print(a.index(3)) # 显示列表a中第一次出现的值为3的项的索引
# all
get_index1(firstforder, m[3])

#升降序排列
l.sort(reverse = True) #参数默认为 False,升序。 直接返回为 None，它直接在原列表上进行排序，原列表改变了，sorted 会开辟一个新的内存空间来存放排序好的列表。
L=[('b',2),('a',1),('c',3),('d',4)]
sorted(L, cmp=lambda x,y:cmp(x[1],y[1]))   # 利用cmp函数
sorted()新列表

#某个元素次数 
list.count(obj) 

#在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表） 
list.extend(seq) 

#颠倒
num.reverse() #更好用
num[::-1]
lyst[:2:-1] #[9, 8, 7, 6, 5, 4, 3]，步长为负数时，起始下标缺省，默认为-1，等价于lyst[-1:2:-1]. 就是步长为正数时，首末缺省下标分别是0和n；步长为负时，首末缺省下标分别是-1和-n-1


#新建creat 
list(range(5))      结果 [0, 1, 2, 3, 4] 
list_2d = [ [0 for i in range(5)] for i in range(5)] 
list_empty = [0]*10 


#list合并	
l = [1, 2, 3]
j = [4, 5, 6]
l.extend(j)
l + j

#insert
num.insert(2, 'three') #position 2
num[2:2] = ['three']

#删
del num[0]
num.pop()  删除最后一个元素。
num.pop(1)
num.remove(4) 删除列表中第一个出现的值为4的项,如果删除的元素有多个的话，只会删除先出现的那一个，其他的不会删除。

#清空
l.clear() 全部清空列表元素，用 clear 方法

#改,赋值
num[1] = 'two'

#copy
如果想要实现a、b从此毫无瓜葛，那么简单的拷贝实现有两种：
a = [1,2,3,4,5]
b = a[:] #真正实现拷贝
b = list(a) #也可实现拷贝
an assignment statement for lists does not create a copy. You have to use slicing operation to make a copy of the sequence.

shoplist = ['apple', 'mango', 'carrot', 'banana']
mylist = shoplist
del shoplist[0]
print('mylist is', mylist)
mylist = shoplist[:]

# 二维demention
multi_dim_a = [[1,2,3],
	       [2,3,4],
	       [3,4,5]] # 三行三列
print(multi_dim_a[0][2])

# #list 存dict形式
dict_1 = {'color': 'yellow', 'points': 10,  'pints': 10,}
dict_2 = {'color': 'red', 'points': 15}
lists = [dict_1, dict_2]
# for dict in lists:
#     print(dict)

###########################  set       操作     ##########################
# #set 创建
a = set('abracadabra') #每个独立
b = {'abracadabra'} #都在一起 or  set({'abracadabra'})
print (type(a), a)
print (type(b), b)

a = {x for x in 'abracadabra' if x not in 'abc'}
print (a)

# 两个set
bri = set(['brazil', 'russia', 'india'])
print(('india' in bri))
bric = bri.copy()
bric.add('china')
print(bric.issuperset(bri)) #one includes all in anohter one
bri.remove('russia')
print('bri & bric', bri & bric)  #print(set1.intersection(set2))
# print(a - b)     # a 和 b 的差集
# print(a | b)     # a 和 b 的并集
# print(a & b)     # a 和 b 的交集
# print(a ^ b)     # a 和 b 中不同时存在的元素
#unique_char2 = set(char_list)
#print(unique_char2.difference({'a', 'e', 'i'}))
print(bric,bri)

char_list = ['a', 'b', 'c', 'c', 'd', 'd', 'd']
sentence = 'Welcome Back to This Tutorial'
unique_char = set(char_list)
print(unique_char)  # delete multiple things directely  乱序
print(type(unique_char))
print(set(sentence))
m = set(char_list+ list(sentence)) #通过list可以加一堆
unique_char.add('x') #只能一个一个加，乱序
#unique_char.remove('x')   #delete   unique_char.discard('d')
unique_char.clear()
print(m)



###########################  array(np) 操作     ##########################
array 设定大小后不可改变

#切片和索引
array[i<4]	布尔索引
array[:1]	选择第0行数据项 (与[0:1, :]相同)

#数据转换
top_2.astype(np.float128) 

#创建
axis 0 指行  1 指列
np.zeros((1,2)) 创建全0数组
np.ones((1,2))
np.empty((2,2)) 空数组

#size & type
array.shape        维度(行,列)
len(array)
array.size
array.dtype 	   数据类型
array.astype(type) 转换数组类型
type(array)        数组类型

# max min 
np.min()
import heapq
f = heapq.nlargest(20, top_2)
indmax = np.argpartition(log_prob_norm, -50)[-50:] # 50个最大的
indmin = np.argpartition(log_prob_norm, 14)[0:14]  # 14个最小的
index_bottom = np.where(log_prob_norm >= np.median(log_prob_norm))[0]# 一半取

#最小值索引
min_index2 = np.argsort(buffer2)[1:5]
max_index2 = np.argsort(buffer2)[-1] #最大

# sort
array.sort(axis=0) 按照指定轴排序一个数组
np.copy(array)
other = array.copy()

# 插入
np.insert(array, 1, 2, axis) 沿着数组0轴或者1轴插入数据项

# merge
np.append()
np.concatenate((a,b),axis=1) #水平组合  没有内存占用大的问题
np.hstack((a,b))  	     #水平组合
np.concatenate((a,b),axis=0) #vertical组合
np.vstack((a,b))             #vertical组合
np.stack()
np.dstack()		    #深度组合：沿着纵轴方向组合
column_stack()              #列组合
row_stack()                 #行组合
a==b                        #用来比较两个数组

# 删除
np.delete(array, 1, axis) 

# 分离
numpy.split()
np.array_split(array, 3) 将数组拆分为大小（几乎）相同的子数组
numpy.hsplit(array, 3) 在第3个索引处水平拆分数组

# 数组形状变化
np.resize((2,4)) 将数组调整为形状(2,4)
other = ndarray.flatten()
numpy.flip() 翻转一维数组中元素的顺序
np.ndarray[::-1]
reshape 改变数组的维数
squeeze 从数组的形状中删除单维度条目
expand_dims 扩展数组维度
array = np.transpose(other)数组转置
array.T 数组转置
a = array(［0, 1, 2],
       [3, 4, 5],
       [6, 7, 8］)
b = array(［ 0, 2, 4],
       [ 6, 8, 10],
       [12, 14, 16］)

# 数学计算
array.sum()	数组求和
np.sqrt(x)
np.sqrt(x)	平方根	
np.sin(x)	元素正弦	
np.log(x)	元素自然对数
np.dot(x,y)	点积
np.roots([1,0,-4])	给定多项式系数的根
array.max(axis=0)	数组求最大值（沿着0轴）	
array.cumsum(axis=0)	指定轴求累计和	

# 运算
** （乘方）
// （整除）   向下取整至最接近的整数
% （取模
<< （左移）二进制
& （按位与） | （按位或）^ （按位异或）~ （按位取反）
!= （不等于）not （布尔“非”）  and  or

# 比较
np.array_equal(x,y)	数组比较	
math.isinf() 最大

# 统计
np.mean(array)	
np.median(array)	Median	
array.corrcoef()	Correlation Coefficient	
np.std(array)	        Standard Deviation


###########################  tuple 操作     ##########################
zoo = ('python', 'elephant', 'penguin')   #tuple type
new_zoo = 'monkey', 'camel', zoo ,'dog'   #tuple type

print('Number of animals in the zoo is', len(zoo))
print('Number', len(new_zoo))

print('All animals in new zoo are', new_zoo)
print('Animals brought from old zoo are', new_zoo[2])
print('Last animal brought from old zoo is', new_zoo[2][2])
print('Number of animals in the new zoo is', len(new_zoo)-1+len(new_zoo[2]))
print('Number of animals in the new zoo is', len(new_zoo)-1+len(zoo))



###########################  dict 操作     ##########################

#type, add delete
def func():
    return 0

d4 = {'apple':[1,4,3], 'pear':2, 'orange':{1:3, 3:'a'}, 'banana':3, 'tree':func}
d2 = {1:'a', 'b':2, 'io':'c'}
print(d4['pear'][3])    # 3 is key
print(d4['apple'][0])   # 0 is index

ab = {
	'Swaroop': 'swaroop@swaroopch.com',
	'Larry': 'larry@wall.org',
	'Matsumoto': 'matz@ruby-lang.org',
	'Spammer': 'spammer@hotmail.com'
}
print("Swaroop's address is", ab['Swaroop'])

del ab['Spammer']               #delete itm
print('\nThere are {} contacts in the address-book\n'.format(len(ab)))

# 属性，
#ab.items()
#ab.keys()
#ab.values() 
for name, address in ab.items():#there are items
	print('Contact {} at {}'.format(name, address))

ab['Guido'] = 'guido@python.org'#add
if 'Guido' in ab:
	print("\nGuido's address is", ab['Guido'])
print('\nThere are {} contacts in the address-book\n'.format(len(ab)))

dict.clear()  # 清空字典
del dict         # 删除字典

#最灵活的内置数据结构类型。列表是有序的对象集合，字典是无序的对象集合。字典当中的元素是通过键来存取的，而不是通过偏移存取。 
#key不一定是 string类型 
#元素唯一性：集合是无重复元素的序列，会自动去除重复元素；字典因为其key唯一性，所以也不会出现相同元素 
#键必须不可变，所以可以用数字，字符串或元组充当，而用列表就不行 
#不允许同一个键出现两次。创建时如果同一个键被赋值两次，后一个值会被记住， 

dict = {} 
dict['one'] = "This is one"  
dict[2] = "This is two"  
tinydict = {'name': 'john','code':6734, 'dept': 'sales'} 
print dict[2] # 输出键为 2 的值  
print tinydict # 输出完整的字典  
print tinydict.keys() # 输出所有键 ,结果是list形式 
print tinydict.values() # 输出所有值, 结果是list形式 

for i in b.values(): 
    print(i) 
for c in b.keys(): 
    print(c) 
 
key in dict 
如果键在字典dict里返回true，否则返回false 

 

for c in dict: 
    print(c,':',dict[c]) 

 
dict1 = {'abc':1,"cde":2,"d":4,"c":567,"d":"key1"} 
 
for k,v in dict1.items(): 
   print(k,":",v) 

dict_1 = dict([('a',1),('b',2),('c',3)])#元素为元组的列表 
dict_2 = dict({('a',1),('b',2),('c',2)})#元素为元组的集合 
dict_3 = dict([['a',1],['b',2],['c',3]])#元素为列表的列表 
dict_4 = dict((('a',1),('b',2),('c',3)))#元素为元组的元组 




# 嵌套
cities={
    '北京':{
        '朝阳':['国贸','CBD','天阶','我爱我家','链接地产'],
        '海淀':['圆明园','苏州街','中关村','北京大学'],
        '昌平':['沙河','南口','小汤山',],
        '怀柔':['桃花','梅花','大山'],
        '密云':['密云A','密云B','密云C']
    },
    '河北':{
        '石家庄':['石家庄A','石家庄B','石家庄C','石家庄D','石家庄E'],
        '张家口':['张家口A','张家口B','张家口C'],
        '承德':['承德A','承德B','承德C','承德D']
    }
}

for i in cities['北京']['海淀']:
    print(i)
for i in cities['北京']:
    print(i)


# 调换key and value  当然要注意原始 value 的类型,必须是不可变类型：
dic = {
    'a': 1,
    'b': 2,
    'c': 3,
}
reverse = {v: k for k, v in dic.items()}
print(dic)
print(reverse)

#获取字典中最大的值及其键：
prices = {
    'A':123,
    'B':450.1,
    'C':12,
    'E':444,
}

max_prices = max(zip(prices.values(), prices.keys()))
print(max_prices) # (450.1, 'B')


