# myobject.method(arg1, arg2)  自动将其转
# MyClass.method(myobject, arg1, arg2)


##example 1
class Calculator:       #首字母要大写， 首次调用可没参数
    '''
    jkjk
    '''
    name='Good Calculator'  #属性, cal = Calculator() 不进入 Calculator
    price=18

    def add(self,x,y):
        print(self.name)
        print(x + y)
    def minus(self,x,y):
        print(x-y)
    def say_hi(self):
        print('Hello, how are you?', self.name)

cal = Calculator()
print(cal)   # address
cal.add(1,2)
Calculator().say_hi()
m = cal.price   # 外部调用
#m = dir(cal)




##example 2
class Calculator2:
    name = '123'  # 属性     # 首次调用进入init. 首次call需要参数。 给出参数默认值如weight=9也可 ignore 初次调用参数。

    def __init__(self, name, price, height, width, weight=9):  # 注意，这里的下划线是双下划线
        self.name = name  # self 内部调用
        self.price = price
        self.h = height
        self.wi = width
        self.we = weight

    def add(self, x, y):
        print(self.name)
        result = x + y
        print(result)

    def minus(self, x, y):
        result = x - y
        print(result)

    def times(self, x, y):
        print(x * y)

    def divide(self, x, y):
        print(x / y)


c = Calculator2('another calculator', 18, 17, width=16)
c.add(1, 2)
m = c.name
m = 1





###example 2
class SchoolMember:  # base class  共有特征     Polymorphism
    # 无需被主程序调用，只用在继承即可
    '''Represents any school member.
    如果我们增加或修改了    SchoolMember     
    的任何功能,它将自动反映在子    类型中'''   #来为所有老师与学生添加一条新的 ID 卡字段
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print('(Initialized SchoolMember: {})'.format(self.name))

    def tell(self):
        '''Tell my details.'''
        print('Name:"{}" Age:"{}"'.format(self.name, self.age), end=" ")


class Teacher(SchoolMember): #Derived Classes  独有的
    '''Represents a teacher.'''
    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)   #继承 int
        self.salary = salary                     # 新的数据
        print('(Initialized Teacher: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Salary: "{:d}"'.format(self.salary))


class Student(SchoolMember):
    '''Represents a student.'''
    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print('(Initialized Student: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Marks: "{:d}"'.format(self.marks))


t = Teacher('Mrs. Shrividya', 40, 30000)
s = Student('Swaroop', 25, 75)
print()
members = [t, s]
for member in members:
    # Works for both Teachers and Students
    member.tell()







#字段(Field)有两种类型——类变量与对象变量
# 类变量(Class Variable)是共享的(Shared)
# 对象变量(Object variable)由类的每一个独立的对象或实例所拥有
###example 3
# population belongs to robot, 类变量
# name:  对象变量
# 当一个对象变量与一个类变量名称
# 相同时,类变量将会被隐藏。

class Robot:
    """Represents a robot, with a name."""

    # A class variable, counting the number of robots, shared in droid1 nad droid2
    population = 0   #we refer to the population class variable as Robot.population and not as self.population

    def __init__(self, name):
        """Initializes the data."""
        self.name = name # this is an object variable #对象中所具有的方法
        print("(Initializing {})".format(self.name))

        # When this person is created, the robot
        # adds to the population
        Robot.population += 1

    def die(self):  #对象
        """I am dying."""
        print("{} is being destroyed!".format(self.name))

        Robot.population -= 1

        if Robot.population == 0:
            print("{} was the last one.".format(self.name))
        else:
            print("There are still {:d} robots working.".format(
                Robot.population))

    def say_hi(self):
        """Greeting by the robot."""
        print("Greetings, my masters call me {}.".format(self.name))

    @classmethod   #classmethod(类方法)或是一个staticmethod(静态方法)
    def how_many(cls):
        """Prints the current population."""
        print("We have {:d} robots.".format(cls.population))

print('\n')
droid1 = Robot("R2-D2")
droid1.say_hi()
Robot.how_many()

droid2 = Robot("C-3PO")
droid2.say_hi()
Robot.how_many()
m = Robot.__doc__ # Robot.say_hi.__doc__

print("\nRobots can do some work here.\n")

print("Robots have finished their work. So let's destroy them.")
droid1.die()
droid2.die()

Robot.how_many()

#how_many = classmethod(how_many)?