

# define a Fib class
class Fib(object):
    def __init__(self, max):
        self.max = max
        self.n, self.a, self.b = 0, 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < self.max:
            r = self.b
            self.a, self.b = self.b, self.a + self.b
            self.n = self.n + 1
            return r
        raise StopIteration()

# using Fib object
for i in Fib(5):
    print(i)


# Python 使用 yield 关键字也能实现类似迭代的效果，yield 语句每次 执行时，立即返回结果给上层调用者，
# 而当前的状态仍然保留，以便迭代器下一次循环调用。这样做的 好处是在于节约硬件资源，在需要的时候才会执行，并且每次只执行一次
def fib(max):
    a, b = 0, 1
    while max:
        r = b
        a, b = b, a+b
        max -= 1
        yield r

# using generator
for i in fib(5):
    print(i)
