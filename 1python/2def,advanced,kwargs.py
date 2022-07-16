# for and else
for i in range(1,10,2):
	print(i)
else:
	print('over\n')




#Documentation_Strings #文档字符串
def print_max(x, y):
	'''print the biggest number of the two.
	Inputs are integer'''
	x = int(x)
	y = int(y)
	if x > y:
		print(x, 'is maximum')
	else:
		print(y, 'is maximum')
		
print_max(3, 5)
print(print_max.__doc__)



print('\n')
# globel 
x = 50 
def global__():
    global x
    print('x is', x)
    x = 2
    print('Changed global x to', x)

global__()
print('Value of x is', x)
print('\n')






# function_keyword function_default 
def func(a, b=5, c=10):
    print('a is', a, 'and b is', b, 'and c is', c)

func(3, 7)
func(25, c=24)
func(c=50, a=100)
print('\n')





# function_variable 可变参数 function_varargs.py
#*args 和 **kwargs 数组参数，**kwargs 称作为字典参数
#当我们声明一个*param的星号参数时,从此处开始直到结束的所有位置参数(Positional Arguments)都将被收集并汇集成一个称为“param”的元组(Tuple)。
#当我们声明一个**param参数都将被收集并汇集成一个名为的双星号参数时,从此处开始直至结束的所有关键字param的字典(Dictionary)。
def total(a=5, *numbers, **phonebook):
    print('a', a)
    for single_item in numbers:
        print('single_item', single_item)
    for first_part, second_part in phonebook.items():
        print('dict:',first_part, second_part)

                                                 #*grades veribal length of paprmeters 可变参数在函数定义不能出现在特定参数和默认参数前面
def portrait(name, color='red', *grades, **kw):  # 关键字参数 自动封装成一个字典(dict).
    print('name is', name)
    print('color', color)

    total_grade = 0
    for grade in grades:
        total_grade += grade
    print(name, 'total grade is ', total_grade)

    for k, v in kw.items():
        print(k, v)


print(total(10, 1, 2, 3,9, Jack=1123, John=2231, m=1))
portrait('Tim', 5, 6, age=18, country='China', education='bachelor')
portrait('Mike', age=24, country='China', education='bachelor')
# 通过可变参数和关键字参数，任何函数都可以用 universal_func(*args, **kw) 表达

ab = {
	'Swaroop': 'swaroop@swaroopch.com',
	'Larry': 'larry@wall.org',
	'Matsumoto': 'matz@ruby-lang.org',
	'Spammer': 'spammer@hotmail.com'
}
actors = ['钢铁侠', '蜘蛛侠', '蝙蝠侠', '水行侠']
print(total(10, 1, 2, actors, *actors, ab, *ab, **ab))
#在一个dict对象的前面，添加**，表示字典的解包，它会把dict对象中的每个键值对元素，依次转换为一个一个的关键字参数传入到函数中
#二者同时存在，一定需要将*args放在**kwargs之前



#function 高级
#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test


def hi(name="yasoob"):
    print("now you are inside the hi() function")

    def greet():
        return "now you are in the greet() function"

    def welcome():
        return "now you are in the welcome() function"

    if name == "yasoob":
        return greet  # 有()这个函数就会执行, 不带, 那它可以被到处传递
    else:
        return welcome

a = hi # 与hi用法一模一样
print('\n')
print('*****')
print(a('k'))
print(hi('k'))
print('*****')
b = hi()
print(b)
print(b())
print('*****')

hi()() # 两个都执行
hi()
print('*****')





del hi
def hi():
    return "hi yasoob!"
def doSomethingBeforeHi(func):
    print("I am doing some boring work before executing hi()")
    print(func())

doSomethingBeforeHi(hi)
print()

#只有那些位于参数列表末尾的参数才能被赋予默认参数值，意即在函数的参数列表中拥有默认参数值的参数不能位于没有默认参数值的参数之前。
#这是因为值是按参数所处的位置依次分配的。举例来说， def func(a, b=5) 是有效的，但 def func(a=5, b) 是无效的。





if __name__ == '__main__':  # 如果外部调用该脚本,不会执行， for testing
    pass

