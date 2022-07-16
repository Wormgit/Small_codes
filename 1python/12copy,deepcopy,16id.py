import copy
a=[1,2,3]
b=a ##绑定索引
print(id(a)) #address
b[0] =11
print(a,b)
print(id(a)==id(b))



#only change the second layer's value
a=[1,2,[3,4]]  #第三个值为列表[3,4],即内部元素 
d=copy.copy(a) #浅拷贝a中的[3，4]内部元素的引用，非内部元素对象的本身  # wont change a  shallow copy
print(id(a)==id(d))
print(id(a[1])==id(d[1]))

a[2][0]=3333  #改变a中内部原属列表中的第一个值
d[2][1]=100
d[0]=100
a[0]=200
print(a,d)     #这时d中的列表元素也会被改变

a=[1,2,[3,4]]  
e=copy.deepcopy(a) #e为深拷贝了a
a[2][0]=333 #改变a中内部元素列表第一个的值
print(a,e)