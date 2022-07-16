import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

cmaps['Miscellaneous'] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar']


cm = plt.cm.get_cmap('Greys')
xy = range(20)
z = (5,4,3,3,5,4,3,3,5,4,3,3,5,4,3,3,5,4,3,3)
sc = plt.scatter(xy, xy, c=z, s=35, vmin=6, vmax=10, cmap=cm)  #vmin=0, vmax=20
plt.colorbar()
plt.show()

# cm = plt.cm.get_cmap('summer_r')  #plt.cm.rainbow  summer  rainbow  RdYlBu  plt.cm.get_cmap('Greys')
# colors = ["orangered", "yellow", "greenyellow", "springgreen","aqua", "royalblue", "blueviolet"]

class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)


fig, (ax, ax2, ax3) = plt.subplots(nrows=3,
                                   gridspec_kw={"height_ratios":[3,2,1], "hspace":0.25})

x = np.linspace(-13,4, 110)
norm=SqueezedNorm(vmin=-13, vmax=4, mid=0, s1=1.7, s2=4)

line, = ax.plot(x, norm(x))
ax.margins(0)
ax.set_ylim(0,1)

im = ax2.imshow(np.atleast_2d(x).T, cmap="Spectral_r", norm=norm, aspect="auto")
cbar = fig.colorbar(im ,cax=ax3,ax=ax2, orientation="horizontal")
plt.show()


import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np; np.random.seed(0)
import pandas as pd

x = np.arange(12)
y = np.random.rand(len(x))*51
c = np.random.rand(len(x))*3+1.5
df = pd.DataFrame({"x":x,"y":y,"c":c})

cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)

fig, ax = plt.subplots()
mm= cmap(norm(df.c.values))
ax.bar(df.x, df.y, color=cmap(norm(df.c.values)))
ax.set_xticks(df.x)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])  # only needed for matplotlib < 3.1
fig.colorbar(sm)

plt.show()


import numpy as np
import matplotlib.pyplot as plt


# Have colormaps separated into categories:
# http://matplotlib.org/examples/color/colormaps_reference.html
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list, nrows)

#十分类散点图绘制
randlabel = np.random.randint(0,1,10)
randdata = np.reshape(np.random.rand(10*2),(10,2))


cm = plt.cm.get_cmap('RdYlBu')
z = randlabel
sc = plt.scatter(randdata[:,0], randdata[:,1], c=z, vmin=0, vmax=10, s=35,edgecolors='k', cmap=cm)
plt.colorbar(sc)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors


# 画出显著变化和不显著变化趋势的站点，同时标注显著变化者senslope的大小
def plot_scatter():
	fig = plt.figure(figsize=(6,4))

	# customized colormap
	colors = ['lightskyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'navy']
	cmap = mpl.colors.ListedColormap(colors[::-1]) # 让颜色倒过来，原来的

	# even bounds gives a contour-like effect
	bounds = [ round(elem, 3) for elem in np.linspace(-0.15, -0.05, 5)] #
	# get one more color than bounds from colormap 在bounds之外再多一个颜色
	colors = plt.get_cmap(cmap)(np.linspace(0,1,len(bounds)+1))
	# create colormap without the outmost colors 不用最外面的颜色来创造colormap
	cmap = mpl.colors.ListedColormap(colors[1:-1]) # 1:-1是从第二个到倒数第二个，左闭右开
	# set upper/lower color
	cmap.set_over(colors[-1])
	cmap.set_under(colors[0])

	m = Basemap(llcrnrlon=113.7, llcrnrlat=22.35, urcrnrlon=114.7, urcrnrlat=22.9,\
			rsphere=(6378137.00,6356752.3142),\
			resolution='l', area_thresh=1000., projection='lcc', lat_1=22.5, lat_0=22.5, lon_0=114) #- shape文件，画出区域轮廓

	lons = [114.1267222, 114.111825, 114.04908059999998, 113.8175, 114.47, 114.1122222, 114.1658333, 114.0036111, 113.8936271, 113.8855675, 114.104705, 114.1508333, 114.5002778, 113.7672222, 114.1011111, 113.9536389, 113.8911111, 113.8979694, 114.5340278, 114.1436111, 114.2238889, 114.3366799, 114.3044444, 113.9308333, 114.3922222, 113.9147222, 113.8361111, 114.02916670000002, 114.2236111, 113.8686111, 114.1488889, 114.1230556, 114.1933333, 114.02916670000002, 113.9727778, 114.48444440000002, 114.2752778, 114.2422222, 114.01416670000002, 113.9288889, 114.5264328, 113.94485559999998, 114.19001670000002, 114.4861111, 114.1794444, 114.2430556, 114.1508333, 114.09555559999998, 114.2622222, 114.0163889, 114.3002778, 114.3552778, 113.8602778, 113.8797222, 114.2338889, 114.0708333, 114.43194440000002, 113.89305559999998]
	lats = [22.55097222, 22.55905833, 22.52501111, 22.67416667, 22.60055556, 22.60555556, 22.56472222, 22.54166667, 22.65377067, 22.47006253, 22.54966819, 22.55166667, 22.56944444, 22.72333333, 22.57194444, 22.76244444, 22.78166667, 22.70483, 22.59697222, 22.57166667, 22.555, 22.59546331, 22.7775, 22.48861111, 22.615, 22.5475, 22.77916667, 22.62305556, 22.64638889, 22.495, 22.61638889, 22.69611111, 22.64638889, 22.5725, 22.59638889, 22.53444444, 22.56666667, 22.72416667, 22.54888889, 22.67888889, 22.48137411, 22.50378611, 22.57101111, 22.48277778, 22.66222222, 22.58388889, 22.65611111, 22.5375, 22.76694444, 22.53027778, 22.67888889, 22.7125, 22.80166667, 22.55277778, 22.68527778, 22.56861111, 22.61888889, 22.58861111]
	values = [-0.03101638020010544, -0.059602408594572324, -0.08074725429957398, -0.052523178817571285, -0.06097954730989012, -0.048576189387818935, -0.05321303447484441, -0.08990485880945824, -0.07035292627778422, -0.08904485101486563, -0.050582862579880206, -0.08667040863306046, -0.2381958334486063, -0.09043589588854052, -0.015791558268944444, -0.04750735872842685, -0.0827816789190754, -0.0495746307369816, -0.09748174185420576, -0.05328468550193526, -0.0385492789056475, -0.12603457458801204, -0.03785426812465535, -0.10934222695289253, -0.06257603144203898, -0.05322345906174508, -0.0437720889282609, -0.029440526555296737, -0.0446847919726688, -0.05879880517527025, -0.042610207214769935, -0.07981280621664595, -0.056233334347744406, -0.07486982006522186, -0.064798640683293, -0.060273137801299975, -0.08829009917157292, -0.02696058143578384, -0.047529515842470234, -0.051839812353009496, -0.05970518244018841, -0.08892429975993077, -0.2594588992844997, -0.0853022809983619, -0.11143855759589913, -0.05064439000559071, -0.02366547879452241, -0.1743014882414493, -0.05539755391763087, -0.04722722346208863, -0.04903534243112728, -0.0613871242134162, -0.05259194456906426, -0.0933057542530412, -0.05063961430132472, -0.05351251987776765, -0.03849744311833736, -0.04819026109972248]

	sgt_scatter = m.scatter(lons, lats, s=60, c=values, marker="o", vmin=-0.15, vmax=-0.05, latlon=True, cmap=cmap)

	m.colorbar(extend='both', ticks=bounds)

	m.drawmeridians(np.arange(10, 125, 0.5), labels=[1,0,0,1])
	m.drawparallels(np.arange(15, 30, 0.3),labels=[1,0,0,0])
	plt.show()

if __name__ == '__main__':
	plot_scatter()
