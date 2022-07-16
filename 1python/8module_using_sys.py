import sys,os

print('The command line arguments are:')
for i in sys.argv:
	print(i)
print('\n\nThe PYTHONPATH is', sys.path, '\n')
###sys包含了与 Python 解释器及其环境相关的功能
###运行的脚本名称在 sys.argv 的列表中总会位列第一

print(os.getcwd()) #程序目前所处在的目录


###返回由对象所定义的名称列表, if 这一对象是一个模块,则该列
###表会包括函数内所定义的函数、类与变量
print(dir(sys))
print(dir())
a = 5
tie = [1,2,4]
print(dir())
del a
print(dir())

m = sys.version_info
print(m)