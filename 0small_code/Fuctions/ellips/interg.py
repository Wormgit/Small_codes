
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate
from scipy.stats import norm
from math import sqrt
import pandas as pd

from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon


def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

fig,ax = plt.subplots()

##these next few lines are pretty important because
##otherwise your ellipses might only be displayed partly
##or may be distorted

ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_aspect('equal')
##first ellipse in blue
ellipse1 = create_ellipse((0,0),(4,2),20)  #((0,0),(2,4),10)  (0,0),(5,10),0
verts1 = np.array(ellipse1.exterior.coords.xy)
patch1 = Polygon(verts1.T, color = 'blue', alpha = 0.5)
ax.add_patch(patch1)
##second ellipse in red
ellipse2 = create_ellipse((1,-1),(3,2),80) #(1,-1),(3,2),50  (3.0,1.0),(5,5),0
verts2 = np.array(ellipse2.exterior.coords.xy)
patch2 = Polygon(verts2.T,color = 'red', alpha = 0.5)
ax.add_patch(patch2)
##the intersect will be outlined in black
intersect = ellipse1.intersection(ellipse2)
verts3 = np.array(intersect.exterior.coords.xy)
patch3 = Polygon(verts3.T, facecolor = 'none', edgecolor = 'black')
ax.add_patch(patch3)
##compute areas and ratios
print('area of ellipse 1:',ellipse1.area)
print('area of ellipse 2:',ellipse2.area)
print('area of intersect:',intersect.area)
print('intersect/ellipse1:', intersect.area/ellipse1.area)
print('intersect/ellipse2:', intersect.area/ellipse2.area)
plt.show()


df = pd.DataFrame({
    'Time' : [1,2,3,4,5],
    'X' : [3.0,4.5,5.0,10.0,3.0],
    'Y' : [1.0,0.0,-1.0,-2.0,-3.0],
    })
ellipse1 = create_ellipse((0,0),(5,10),0)
df['ellipse2']  = df[['X', 'Y']].apply(
        lambda xy:create_ellipse([*xy],(5,5),0),
        axis=1
        )
df['Area'] = df['ellipse2'].apply(lambda x: ellipse1.intersection(x).area)
df.drop('ellipse2', axis=1, inplace=True)
print(df)





def gaussian(x, mu, delta):
    exp = np.exp(- np.power(x - mu, 2) / (2 * np.power(delta, 2)))
    c = 1 / (delta * np.sqrt(2 * np.pi))
    return c * exp


x = np.arange(-3, 3, 0.01)
for delta in [0.2, 0.5, 1]:
    y = gaussian(x, 0, delta)
    plt.plot(x, y, label='u=0, delta={}'.format(delta))

plt.legend()
plt.show()



f = lambda x, y : 16*x*y
g = lambda x : 0
h = lambda y : sqrt(1-4*y**2)
i,_ = integrate.dblquad(f, 0, 0.5, g, h)
print (i)





m1 = 1
std1 = 0.2
m2 = 1.2
std2 = 0.1

number = 3000
# Generate random data uniformly distributed.
a = np.random.normal(m1, std1, number)
b = np.random.normal(m2, std2, number)

# Obtain KDE estimates foe each set of data.
xmin, xmax = -1., 2.
x_pts = np.mgrid[xmin:xmax:1000j]
# Kernels.
ker_a = stats.gaussian_kde(a)
ker_b = stats.gaussian_kde(b)
# KDEs for plotting.
kde_a = np.reshape(ker_a(x_pts).T, x_pts.shape)
kde_b = np.reshape(ker_b(x_pts).T, x_pts.shape)


# Random sample from a KDE distribution.
sample = ker_a.resample(size=number)

# Compute the points below which to integrate.
iso = ker_b(sample)

# Filter the sample.
insample = ker_a(sample) < iso

# As per Monte Carlo, the integral is equivalent to the
# probability of drawing a point that gets through the filter.
integral = insample.sum() / float(insample.shape[0])

print(f'?? {integral}')

plt.xlim(0.4,1.9)
plt.plot(x_pts, kde_a)
plt.plot(x_pts, kde_b)

plt.show()

# Calculate overlap between the two KDEs.
def y_pts(pt):
    y_pt = min(ker_a(pt), ker_b(pt))
    return y_pt
# Store overlap value.
overlap = integrate.quad(y_pts, -1., 2.)

print(f'??{overlap}')







def half_circle(x):
    return (1-x**2)**0.5
pi_half, err = integrate.quad(half_circle, -1, 1)

mmmm=half_circle(0)
print(pi_half*2)
#经典的分小矩形计算面积总和的方式
N = 10000
x = np.linspace(-1, 1, N)
dx = 2.0/N
y = half_circle(x)
print(dx * np.sum(y[:-1] + y[1:])) # 面积的两倍
print(np.trapz(y, x)*2)



def half_sphere(x, y):
    return (1-x**2-y**2)**0.5
integrate.dblquad(half_sphere, -1, 1, lambda x:-half_circle(x), lambda x:half_circle(x))
#于func2d(x,y)函数进行二重积分，其中a,b为变量x的积分区间，而gfun(x)到hfun(x)为变量y
print(f'the int is {np.pi*4/3/2}')




result1 = integrate.quad(lambda x: np.exp(-x), a=0, b=np.inf)  # quad(func,a,b),对函数func从a到b积分。np.inf表示无穷大
print(result1, type(result1))  # 返回结果为：元组（积分值，误差值）。 因为电脑算的数值积分不同于定积分不定积分，会有一个误差值

result2 = integrate.dblquad(lambda y, x: x * y ** 2, a=0, b=5, gfun=lambda x: 0,
                            hfun=lambda x: 6)  # ∫_0^6【∫_0^5〖xy^3 ⅆx〗】 ⅆy
# integral of func(y,x) from x=a..b ,and y=gfun(x)..hfun(x)
print('积分值，误差值：', result2)


def f(x, y):  # 积分函数
    return x * y
def bound_x(y):  # 积分内容边界
    return [0, 1 - 2 * y]
def bound_y():  # 积分内容边界
    return [0, 0.5]
result3 = integrate.nquad(f, [bound_x, bound_y])  # nquad(积分函数，积分边界)。 ∫_0^0.5【∫_0^(1-2y)【xyⅆx】】ⅆy
# 这其中被调用的函数f(x,y)、bound_x(y)均不写括号（），因为我们没有要传入的实参，写了反而会因为没有传入参数而出现错误。bound_y()可写可不写，因为定义的函数中就不需要接收参数
print(result3)

"""
可写成result3 = integrate.nquad(lambda x,y: x*y, [bound_x,[0,0.5]])
"""




# 尽管可以定义和scipy.quad采用高斯符号形式（或其他方式），但cdf用于获得高斯积分。

norm.cdf(1.96)

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])



# 3std
lower = min(m1 - 3 * std1,m2 - 3 * std2)
upper = max(m2 + 3 * std2, m1 + 3 * std1)

#Get point of intersect
tmp_result = solve(m1, m2, std1, std2)
result = []
for item in range(len(tmp_result)):
    if upper > tmp_result[item] > lower :
        result.append(tmp_result[item])

# Get point on surface
x = np.linspace(lower, upper, 10000)
plt.plot(x, norm.pdf(x, m1, std1))
plt.plot(x, norm.pdf(x, m2, std2))
plt.plot(result, norm.pdf(result, m1, std1), 'o')

# 'lower' and 'upper' represent the lower and upper bounds of the space within which we are computing the overlap
if(len(result)==0): # Completely non-overlapping
    overlap = norm.cdf(r, m2, std2)
    print('Totally overlap')

elif(len(result)==1): # One point of contact
    r = result[0]
    if(m1>m2):
        tm,ts=m2,std2
        m2,std2=m1,std1
        m1,std1=tm,ts

    if(r<lower): # point of contact is less than the lower boundary. order: r-l-u
        overlap = (norm.cdf(upper,m1,std1)-norm.cdf(lower,m1,std1))
        print("ATTENTION")
    elif(r<upper): # point of contact is more than the upper boundary. order: l-u-r
        overlap = (norm.cdf(r,m2,std2)-norm.cdf(lower,m2,std2))+(norm.cdf(upper,m1,std1)-norm.cdf(r,m1,std1))
    else: # point of contact is within the upper and lower boundaries. order: l-r-u
        overlap = (norm.cdf(upper,m2,std2)-norm.cdf(lower,m2,std2))
        print("ATTENTION")

    plt.plot(lower, norm.pdf(lower, m1, std1), 'o')
    plt.plot(upper, norm.pdf(upper, m2, std2), 'o')
    # Plots integrated area
    olap = plt.fill_between(x[x > r], 0, norm.pdf(x[x > r], m1, std1), alpha=0.3)
    olap = plt.fill_between(x[x < r], 0, norm.pdf(x[x < r], m2, std2), alpha=0.3)

    # integrate
    area = norm.cdf(r, m2, std2) + (1. - norm.cdf(r, m1, std1))
    print("Area under curves no limit", area)



elif(len(result)==2): # Two points of contact
    r1 = result[0]
    r2 = result[1]
    if(r1>r2):
        temp=r2
        r2=r1
        r1=temp
    if(std1>std2):
        tm,ts=m2,std2
        m2,std2=m1,std1
        m1,std1=tm,ts

    lower = m1 - 3 * std1
    upper = m2 + 3 * std2
    plt.plot(lower, norm.pdf(lower, m1, std1), 'o')
    plt.plot(upper, norm.pdf(upper, m2, std2), 'o')
    if(r1<lower):
        print("ATTENTION")
        if(r2<lower):           # order: r1-r2-l-u
            overlap = (norm.cdf(upper,m1,std1)-norm.cdf(lower,m1,std1))
        elif(r2<upper):         # order: r1-l-r2-u
            overlap = (norm.cdf(r2,m2,std2)-norm.cdf(lower,m2,std2))+(norm.cdf(upper,m1,std1)-norm.cdf(r2,m1,std1))
        else:                   # order: r1-l-u-r2
            overlap = (norm.cdf(upper,m2,std2)-norm.cdf(lower,m2,std2))
    elif(r1<upper):
        if(r2<upper):         # order: l-r1-r2-u
            overlap = (norm.cdf(r1,m1,std1)-norm.cdf(lower,m1,std1))+(norm.cdf(r2,m2,std2)-norm.cdf(r1,m2,std2))+(norm.cdf(upper,m1,std1)-norm.cdf(r2,m1,std1))
            # Plots integrated area
            plt.fill_between(x[(x < r1)], 0, norm.pdf(x[(r1 > x)], m1, std1), alpha=0.3)
            plt.fill_between(x[(r1 < x) &(x < r2)], 0, norm.pdf(x[(r1 < x) &(x < r2)], m2, std2), alpha=0.3)
            plt.fill_between(x[ (x > r2)], 0, norm.pdf(x[ (x > r2)], m1, std1), alpha=0.3)
        else:
            print("ATTENTION")# order: l-r1-u-r2
            overlap = (norm.cdf(r1,m1,std1)-norm.cdf(lower,m1,std1))+(norm.cdf(upper,m2,std2)-norm.cdf(r1,m2,std2))
    else:                       # l-u-r1-r2
        overlap = (norm.cdf(upper,m1,std1)-norm.cdf(lower,m1,std1))
        print("ATTENTION")

print("Area under curves UPPER AND LOWER 3", overlap)
plt.show()
