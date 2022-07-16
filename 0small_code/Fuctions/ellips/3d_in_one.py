from pylab import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
len = 5
step = 0.1


# fig = plt.figure()
# ax = Axes3D(fig)
# def build_gaussian_layer(mean, sd, height):
#     x = np.arange(-len, len, step)
#     y = np.arange(-len, len, step)
#     x, y = np.meshgrid(x, y)
#     z = np.exp(-((y-mean[1])**2 + (x - mean[0])**2)/(2*(sd**2)))
#     z = z/(np.sqrt(2*np.pi)*sd)
#     return (x, y, height*z)
#
# #具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
# # x6, y6, z6 = build_layer(-0.22)
# # ax.plot_surface(x6, y6, z6, rstride=1, cstride=1, color='pink')
# x3, y3, z3 = build_gaussian_layer(mean=[0, 0], sd = 2, height = 1)
# ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='rainbow') # rainbow
# x3, y3, z3 = build_gaussian_layer(mean=[0.5, 0.5], sd = 2.2, height = 1)
# ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='rainbow') # rainbow
# ax.view_init(elev=45,azim=0)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.datasets.samples_generator import make_blobs
n_components = 3
X, truth = make_blobs(n_samples=300, centers=n_components,
                      cluster_std = [2, 1.5, 1],
                      random_state=42)
plt.scatter(X[:, 0], X[:, 1], s=50, c = truth)
plt.title(f"Example of a mixture of {n_components} distributions")
plt.xlabel("x")
plt.ylabel("y")
# Extract x and y
x = X[:, 0]
y = X[:, 1]
# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
plt.show()

class Distribution():
    def __init__(self, mu, Sigma, height):
        self.mu = mu
        self.sigma = Sigma
        self.h = height

    def tow_d_gaussian(self,x):
        mu = self.mu
        Sigma =self.sigma
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)

        N = np.sqrt((2*np.pi)**n*Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized way across all the input variables.
        fac = np.einsum('...k,kl,...l->...',x-mu,Sigma_inv,x-mu)
        U, s, Vt = np.linalg.svd(self.sigma)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
        print(f'angle: {angle},{width},{height}')
        return self.h * np.exp(-fac/2)/N

    def build_layer(self, z_value):
        x = np.arange(-len, len, step)
        y = np.arange(-len, len, step)
        z1 = np.full(x.size, z_value / 2)
        z2 = np.full(x.size, z_value / 2)
        z1, z2 = np.meshgrid(z1, z2)
        z = z1 + z2
        x, y = np.meshgrid(x, y)
        return (x, y, z)


def con(X, Y, Z, *args, zdir='z', offset=None, **kwargs):
    from mpl_toolkits.mplot3d import art3d
    had_data = True
    jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
    cset = super().contour(jX, jY, jZ, *args, **kwargs)
    add_contour_set(cset, extend3d, stride, zdir, offset)
    add_contourf_set(cset, zdir, offset)
    auto_scale_xyz(X, Y, Z, had_data)
    return cset

if __name__=='__main__':

    N = 500
    X = np.linspace(-len,len,N)
    Y = np.linspace(-len,len,N)
    X, Y = np.meshgrid(X,Y)

    mu = np.array([0.,0.])
    Sigma = np.array([[1.,-0.5],[-0.5,1.5]])
    height = np.array([0.5])
    pos = np.empty(X.shape+(2,))
    pos[:,:,0] = X
    pos[:,:,1] = Y

    p2 = Distribution(mu,Sigma,height)
    Z = p2.tow_d_gaussian(pos)


    mu = np.array([0.,3.])
    Sigma = np.array([[2.,0],[0,2]])
    height = np.array([1])
    pos = np.empty(X.shape+(2,))
    pos[:,:,0] = X
    pos[:,:,1] = Y

    p1 = Distribution(mu,Sigma,height)
    Z1 = p1.tow_d_gaussian(pos)

    fig = plt.figure(figsize=(7,7))
    ax = Axes3D(fig)
    ax.contourf(X, Y, Z, levels=5, cmap='coolwarm')
    cset = ax.contour(X, Y, Z, levels=5, colors='k', linewidth=.5)  # the flat version
    ax.clabel(cset, inline=1, fontsize='xx-small')
    ffff = cset.cvalues
    layers = cset.layers
    zmax = cset.zmax
    #con(X, Y, Z1, levels=5, cmap='coolwarm')
    allsegs = cset.allsegs


    cset2 = ax.contour(X, Y, Z1, colors='k')  # the flat version
    ax.clabel(cset2, inline=1, fontsize=10)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zticks([])
    ax.view_init(90, 0)
    plt.show()

    fig = plt.figure(figsize=(7,7))
    ax2 = Axes3D(fig)
    ax2.contour(X, Y, Z, zdir='z', offset=0)
    ax2.contour(X, Y, Z1, zdir='z',offset=0)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.set_zticks([])
    ax2.view_init(90, 0)
    plt.show()

    fig =plt.figure(figsize=(7,7))
    ax = fig.add_subplot(2,2,1,projection='3d')
    ax.plot_surface(X,Y,Z,rstride=3,cstride=3,linewidth=1,cmap='rainbow', antialiased =True)  #cm.viridis
    ax.contour(X, Y, Z, zdir='z', offset=-np.max(Z), cmap = plt.get_cmap('rainbow'))  # the flat version
    ax.plot_surface(X,Y,Z1,rstride=3,cstride=3,linewidth=1,cmap=cm.viridis, antialiased =True)  #cm.viridis
    ax.contour(X,Y,Z1,zdir='z',offset=-np.max(Z1), cmap = plt.get_cmap('rainbow'))
    ax.set_zlim(min(-np.max(Z),-np.max(Z1)), np.max(Z))
    #ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27,-21)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    ax1.plot_surface(X, Y, Z,rstride=3,cstride=3,linewidth=1,cmap='rainbow', antialiased =True)  #cm.viridis
    ax1.plot_surface(X, Y, Z1, rstride=3, cstride=3, linewidth=1, cmap=cm.viridis, antialiased=True)
    ax1.view_init(30, -70)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')

    ax4 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4.plot_surface(X,Y,Z,rstride=3,cstride=3,linewidth=1,cmap='rainbow', antialiased =True)  #cm.viridis
    ax4.plot_surface(X, Y, Z1, rstride=3, cstride=3, linewidth=1, cmap=cm.viridis, antialiased=True)
    ax4.view_init(0, 0)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_zticks([])
    ax4.set_xlabel(r'$x$')
    ax4.set_ylabel(r'$y$')

    ax3 = fig.add_subplot(2, 2, 4, projection='3d')
    ax3.plot_surface(X,Y,Z,rstride=3,cstride=3,linewidth=1,cmap='rainbow', antialiased =True) #.cm.coolwarm
    ax3.plot_surface(X, Y, Z1, rstride=3, cstride=3, linewidth=1, cmap=cm.viridis, antialiased=True)
    ax3.view_init(0, -90)
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$y$')
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                        left=.01, right=.99)

    plt.show()


