# matplotlib可视化
[matplotlib.pyplot.bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar)
>matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)

- x： float or array-like
- height： float or array-like，the height of bars
- width：float or array-like, default: 0.8，the width of bars
- bottom：float or array-like, default: 0
- align：{'center', 'edge'}, default: 'center'
	- 'center': Center the base on the x positions.
	- 'edge': Align the left edges of the bars with the x positions.
	- To align the bars on the right edge pass a negative width and **align='edge'**.

其他的参数：
- color：color or list of color, optional；The colors of the bar faces.
- edgecoloor：optional；The colors of the bar edges.
- tick_label ： str or list of str, optional；The tick labels of the bars. Default: None
- [xerr, yerr](https://matplotlib.org/stable/gallery/statistics/errorbar_features.html)：float or array-like of shape(N,) or shape(2, N), optional
	- If not None, add horizontal / vertical **errorbars** to the bar tips. The values are +/- sizes relative to the data:
	- scalar: symmetric +/- values for all bars
	- shape(N,): symmetric +/- values for each bar
- ecolor:color or list of color, **default: 'black'** ;errorcolor
- capsize：float,（浮点型数据）
- error_kw：dict, optional
- log：bool, default: False  # ax.set_yscale('log')
- **kwargs：详情查看上方链接

## 1.  bar_stacked
```python
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html#sphx-glr-gallery-lines-bars-and-markers-bar-stacked-py
import matplotlib.pyplot as plt

labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# scores
men_means = [25, 35, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
# std
men_std = [2, 3, 4, 1, 2]
women_std = [3, 5, 2, 3, 3]
width = 0.37 # the width of the bars

fig, ax = plt.subplots() # 绘制白板
ax.bar(labels, men_means, width, yerr = men_std, label = 'Men', align = 'edge') # yerr
ax.bar(labels, women_means, width, yerr = women_std,
       bottom = men_means, label = 'Women', align = 'edge')

ax.set_ylabel('Scores') # ylabel
ax.set_title('Scores by Group and Gender') # title
ax.legend() # legend

plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224174037169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
## 2. barchart
```python
###  https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))
width = 0.37
fig, ax = plt.subplots()

rects1 = ax.bar(x-width/2, men_means, width, label = 'men_means')
rects2 = ax.bar(x+width/2, women_means, width, label ='women_means')

# set of ax
ax.set_ylabel('Scores')
ax.set_title('Scores by Group and Gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add some text for labels
def autolabel(rects):
    """Add some text for labels above each bars."""
     for rect in rects:
         height = rect.get_height()
         ax.annotate('{}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()
```
**结果**：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224173902340.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
## 3. Plotting the coherence of two signals
```python
## Plotting the coherence of two signals
import numpy as np
import matplotlib.pyplot as plt

# Fixing the random state
np.random.seed(19680801)

dt = 0.01 # 步长
t = np.arange(0, 30, dt) # 生成3000*1的序列
# np.random.rand()不会产生负值，生成在(0,1)服从正态分布之间的随机数
nse_1 = np.random.randn(len(t)) # 生成3000*1的序列，服从正态分布，会出现负值，
nse_2 = np.random.randn(len(t)) # 生成3000*1的序列，服从正态分布，会出现负值，

# tow signals with a coherent at 10 hz and a noise part
s_1 = np.sin(2 * np.pi * 10 * t) + nse_1 # 信号1
s_2 = np.sin(2 * np.pi * 10 * t) + nse_2 # 信号2

# 绘制 2 * 1 的 图像，
fig, axs = plt.subplots(2,1)
# 绘制子图1
axs[0].plot(t,s_1, t,s_2)
axs[0].set_xlim(0,2)
axs[0].set_xlabel('time')
axs[0].set_ylabel('$ s_1 \ \ and \ \ s_2 $')
axs[0].grid(True)

# 绘制子图2，两个信号之间的相关性
# cohere function https://www.dsprelated.com/freebooks/mdft/Coherence_Function.html
cxy, f = axs[1].cohere(s_1, s_2, 256, 1. / dt)
axs[1].set_ylabel('coherence')

# 使图像看起来更紧凑
fig.tight_layout()
plt.savefig('E:\Python_Code\爬虫\可视化\pictures\cohere')
plt.show()

```
**结果：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210225105217773.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
## 4. Errorbar limit selection
```python 
# Errorbar limit selection

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure() # 绘制画布
x = np.arange(10) # 0开始，步长1, 9结束
y = 2.5 * np.sin(x / 20 * np.pi)
yerr = np.linspace(0.05, 0.2, 10) # 0.05-0.2 生成10个数

plt.errorbar(x, y+3, yerr = yerr, label = 'both limits (default)')
plt.errorbar(x, y+2, yerr = yerr, uplims =True, label = 'uplims = True')
plt.errorbar(x, y+1, yerr = yerr, uplims =True, lolims=True,
             label = 'uplims = True,lolims=True')

upperlimits = [True, False] * 5 # 列表复制5份
lowelimits = [False, True] * 5 # 列表复制5份
plt.errorbar(x, y, yerr = yerr, uplims =upperlimits, lolims=lowelimits,
             label = 'subsets of uplims and lolims')

plt.legend(loc = 'lower right')
fig.tight_layout()
fig.savefig(r'E:\Python_Code\爬虫\可视化\pictures\Errorbar limit selection') #保存图片
```
**结果：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210225113904511.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

## 5. Pyplot tutorial

```python
import  matplotlib.pyplot as plt
fig = plt.figure() # 生成画布
plt.plot([1, 2, 3, 4]) # 绘制在y轴上
plt.ylabel('some numbers') 
plt.show() # 查看图片
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\some numbers') # 图像保存
```
结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210226094354259.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
为什么横坐标是0-3，而纵坐标是1.0 - 4.0? Python 绘制单个 list 或者 array ，matplotlib假定他是关于y轴的序列，自动生成横坐标。由于Python默认是从0开始，生成与y轴同等长度序列，因此是[0, 1, 2, 3]

```python
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210226110052909.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

**Formatting the style of your plot**

对于每个配对的参数 x 和 y ，存在可选的第三个参数（字符串格式），可以选择绘制的颜色和线的类型。字符串格式的命令与matlab中的格式一样！

```python
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro') # 红色，圆圈
plt.axis([0,6 ,0,20]) # 坐标轴范围限制， [xmin, xmax, ymin, ymax]
plt.show() 
```
matplotlib想要绘制数组数据时，需要导入numpy模块，这时用matplotlib绘制列表数据的方法将变得低效，应把所有的序列转化为数组类型！
```python
# # red dashes, blue squares and green triangles
import numpy as np
t = np.arange(0., 5., 0.2) # 0-5间隔0.5 ，25元素
fig = plt.figure()
plt.plot(t,t,'r--', t,t**2,'bs', t,t**3,'g^') # 绘制 red dashes, blue squares and green triangles
```
**结果：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210226112606759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
**Plotting with keyword strings**

```python
# 生成字典型数据
data = { 'a' : np.arange(50) , # 自然数0-50
         'c' : np.random.randint(0, 50, 50), # 正态分布，正整数，50个元素
         'd' : np.random.randn(50) } # 正态分布，50个元素，可以产生负数

data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig = plt.figure()
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter
plt.scatter('a', 'b', c = 'c', s = 'd',data = data) # 绘制散点图
plt.xlabel('entry a')
plt.ylabel('entry b')

plt.tight_layout()
plt.show()
```
**结果**：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210226114202654.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
**Plotting with categorical variables**
```python
# Plotting with categorical variables
names = ['group_a', 'group_b', 'group_c']
values = [0, 10, 100]
plt.figure( figsize=(9,3) ) # 生成画布

plt.subplot(131) #bar
plt.bar(names, values)
plt.subplot(132) #scatter
plt.scatter(names, values)
plt.subplot(133) # plot
plt.plot(names, values)
plt.suptitle('Categorical plotting')

plt.tight_layout()
plt.show()
```
结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210226123454745.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
**Controlling line properties**
```python
plt.plt(x, y, linewidth = 2.0) # linewidth
line, = plt.plot(x, y, '-')
line.set_antialiased(False)
# using setp
line = plt.plot(x1,y1, x2,y2)
#  use keyword args
plt.setp(lines, color = 'r', linewidth = 2.0)
# or MATLAB style string value pairs
plt.setp(lines, 'color', 'r', 'linewidth', '2.0')
```
## 6.  Horizontal bar chart
```python
## Horizontal bar chart
import  matplotlib.pyplot as plt
import numpy as np

# fix random state
np.random.seed(19680801)
plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ['tom', 'dick', 'harry', 'slim', 'jim']
y_pos = np.arange(len(people))
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html#numpy.random.rand
performance = 3 + 10 * np.random.rand(len(people)) # np.random.rand 从[0,1)均匀分布中抽样
error  = np.random.rand(len(people))

ax.barh(y_pos, performance, xerr = error, align = 'center') # Horizontal bar
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

fig.tight_layout()
plt.show()
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Horizontal bar chart.pdf')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210227215912246.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

## 7. Broken Barh
```python
# Broken Barh
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue') # x轴：110:140; 150:160 ；y轴： 10：19.填充蓝色
ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),
               facecolors = ('tab:orange', 'tab:green', 'tab:red'))
ax.set_ylim(5, 35)
ax.set_xlim(0, 200)

ax.set_xlabel('seconds since start')
ax.set_yticks([15, 25])
ax.set_yticklabels(['tom', 'jim'])
ax.grid()
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html#matplotlib.axes.Axes.annotate
# 添加文字
ax.annotate('race interrupt', (61, 25),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')
fig.tight_layout()
plt.show()
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Broken Barh.pdf')
```
结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210227215611691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
## 8. Errorbar subsampling
```python
## Errorbar subsampling
import matplotlib.pyplot as plt
import numpy as np

# example data
x = np.arange(0.1, 4, 0.1)
y_1 = np.exp(-1.0 * x)
y_2 = np.exp(-0.5 * x)

# example variable error bar values
y_1_error = 0.1 + 0.1 * np.sqrt(x)
y_2_error = 0.1 + 0.1 * np.sqrt(x/2)

fig, (ax0, ax1, ax2) = plt.subplots(nrows = 1, ncols = 3, sharex=True,
                                    figsize = (12, 6))
ax0.set_title('all errorbars')
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html#matplotlib.axes.Axes.errorbar
ax0.errorbar(x, y_1, y_1_error)
ax0.errorbar(x, y_2, y_2_error)

ax1.set_title('only every 6th errorbar')
ax1.errorbar(x, y_1, yerr = y_1_error, errorevery = 6) # 间隔6
ax1.errorbar(x, y_2, yerr = y_2_error, errorevery = 6) # 间隔6

ax2.set_title('second series shifted by 3')
ax2.errorbar(x, y_1, yerr = y_1_error, errorevery = (0,6))  # 起点：0，间隔6
ax2.errorbar(x, y_2, yerr = y_2_error, errorevery = (3,6))  # 起点：3，间隔6

fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Errorbar subsampling.pdf')
```

结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228104100691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
##
```python
# Stem Plot
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.1, 2 * np.pi, 41)
y = np.exp(np.sin(x))

fig = plt.figure()
plt.stem(x, y)
plt.show()
```
结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022811201596.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

```python
# 对图像进行优化
markerline, stemlines, baseline = plt.stem(
    x, y, linefmt= 'grey', markerfmt='D', bottom=1.1)
markerline.set_markerfacecolor('none')
plt.show()
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Stem Plot')
```
结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228111951370.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
## 9. Filling the area between lines
```python
## Filling the area between lines

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 2, 0.01)
y_1 = np.sin(2 * np.pi * x)
y_2 = 0.8 * np.sin(4 * np.pi * x)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex= True, figsize=(6,6)) #  sharex= True 仅保留一个横坐标

ax1.fill_between(x, y_1)
ax1.set_title('fill between $y_1$ and 0')

ax2.fill_between(x, y_1, 1)
ax2.set_title('fill between $y_1$ and 1')

ax3.fill_between(x, y_1, y_2)
ax3.set_title('fill between $y_1$ and $y_2$')
ax3.set_xlabel('x')

fig.tight_layout()
plt.show()
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Filling the area between lines.jpg')
```

![Filling the area between lines](https://img-blog.csdnimg.cn/20210304142938197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
## 10. Confidence bands
```python
# Confidence bands
N = 21
x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1, 9.9, 13.9, 15.1, 12.5]
# filter a linear curve an estimate its y-values and errors
a, b = np.polyfit(x, y,deg = 1) # 一阶线性拟合
y_est = a * x + b # estimation
y_err = x.std() * np.sqrt(1/len(x) + 
                          (x - x.mean())**2 / np.sum((x - x.mean())**2)) # error

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
ax.fill_between(x, y_est-y_err, y_est+y_err, alpha = 0.2) # 透明度设为0.2
ax.plot(x, y, 'ro', color = 'tab:brown')
fig.tight_layout()
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Confidence bands.jpg')
```
![Confidence bands](https://img-blog.csdnimg.cn/20210304150221968.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

```python
# Selectively filling horizontal regions
x = np.array([0, 1, 2, 3])
y_1 = np.array([0.8, 0.8, 0.2, 0.2])
y_2 = np.array([0, 0, 1, 1])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x, y_1, 'o--')
ax1.plot(x, y_2, 'o--')
ax1.fill_between(x, y_1, y_2, where=(y_1 > y_2), color = 'C0', alpha = 0.2) #  填充 y_1 > y_2 的区域
ax1.fill_between(x, y_1, y_2, where=(y_1 < y_2), color = 'C1', alpha = 0.2) #  填充 y_1 > y_2 的区域
ax1.set_title('interpolation=False')

ax2.plot(x, y_1, 'o--')
ax2.plot(x, y_2, 'o--')
ax2.fill_between(x, y_1, y_2, where=(y_1 > y_2), color='C0', alpha = 0.2,
                 interpolate=True) #  填充 y_1 > y_2 的区域, 并差值空白区域
ax2.fill_between(x, y_1, y_2, where=(y_1 < y_2), color='C1', alpha = 0.3,
                 interpolate=True) #  填充 y_1 < y_2 的区域, 并差值空白区域
ax2.set_title('interpolation=True')

fig.tight_layout()
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Selectively filling horizontal regions.jpg')
```
![Selectively filling horizontal regions](https://img-blog.csdnimg.cn/20210304153528112.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
## 11 Selectively marking horizontal regions
```python
# Selectively marking horizontal regions across the whole Axes
fig, ax = plt.subplots()
x = np.arange(0, 4 * np.pi, 0.01)
y = np.sin(x)
ax.plot(x, y, 'b') # plot blue

threshold = 0.75 # 门槛值
ax.axhline(threshold, color = 'green', lw=2, alpha = 0.7)
ax.fill_between(x, 0, 1, where=(y > threshold), # 0,1之间 高于门槛值
                color = 'green', alpha = 0.5, transform = ax.get_xaxis_transform()) # transform = ax.get_xaxis_transform()，覆盖整个yaxis
ax.set_title('Selectively marking horizontal regions')
fig.tight_layout()
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Selectively marking horizontal regions.jpg')
```
![Selectively marking horizontal regions](https://img-blog.csdnimg.cn/20210304154934589.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)
## 12. Eventplot Demo
```python 
# Eventplot Demo
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 8.0
# fixed random state reproducibility
np.random.seed(19680801)

# random data
data1 = np.random.random([6,50])
# set different colors for each set of positions
colors1 = ['C{}'.format(i) for i in range(6)]

# for different line properties for each set of positions
# note that some overlap
lineoffsets1 = [-15, -3, 1, 1.5, 6, 10]
linelengths1 = [5, 2, 1, 1, 3, 1.5]

fig, axs = plt.subplots(2, 2)

# create a horizontal plot
axs[0, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                    linelengths=linelengths1)
# create a vertical plot
axs[1,0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                    linelengths=linelengths1, orientation='vertical')

# create another set of random data
# the gamma distribution is used for aesthetic purposes
data2 = np.random.gamma(4, size=[60, 50])

# use individual values for the parameters this time
# these values will be used for all data sets (except lineoffsets2,
# which sets the increment between each data set in this usuage)
colors2 = 'black'
lineoffsets2 = 1
linelengths2 = 1
# create a horizontal plot
axs[0,1].eventplot(data2, colors = colors2, lineoffsets=lineoffsets2,
                   linelengths=linelengths2)
# create a vertical plot
axs[1,1].eventplot(data2, colors = colors2, lineoffsets=lineoffsets2,
                   linelengths=linelengths2, orientation='vertical')
axs[1,1].set_title('vertical')

fig.tight_layout()
fig.savefig('E:\Python_Code\爬虫\可视化\pictures\Eventplot Demo.jpg')
```



结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210307172520531.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

```python
# Discrete distribution as horizontal bar chart
import numpy as np
import matplotlib.pyplot as plt

category_names = ['Strongly disagree', 'disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']

results = {
    'Question1': [10, 15, 17, 32, 26],
    'Question2': [26, 22, 29, 10, 13],
    'Question3': [35, 37, 7, 2, 19],
    'Question4': [32, 11, 9, 15, 33],
    'Question5': [21, 29, 5, 5, 40],
    'Question6': [8, 19, 5, 30, 38]
}

# 定义函数
def survey(results, category_names):
    """
    Parameters
    -------------------------------
    :param results:dict
        A mapping from questions labels to a list of answers per category.
        It's assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    :param category_names: list of string
        the category labels
    :return:fig, ax
    """
    labels = list(results.keys()) # 获取key
    data = np.array(list(results.values())) # 获取value; ndarray
    data_cum = data.cumsum(axis=1) # 按行累加
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1])
    )

    fig, ax = plt.subplots(figsize=(9.2, 5)) # 生成画布
    ax.invert_yaxis() # 顺序Q1 -> Q6
    ax.xaxis.set_visible(False) # xaxis不可见
    ax.set_xlim(0, np.sum(data,axis=1).max()) # 设置x范围

   # 循环 barh 绘制横向bar；
    for i, (colname, color) in enumerate(zip(category_names,category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label = colname, color = color)
        xcenters = starts + widths / 2

        r, g ,b ,_ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey' # 定义text_color
        # 添加数字
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha = 'center', va = 'center',
                   color = text_color)
            ax.legend(ncol=len(category_names), bbox_to_anchor = (0, 1),
                      loc = 'lower left', fontsize = 'small')

    fig.tight_layout()
    fig.savefig('E:\Python_Code\爬虫\可视化\pictures\ Discrete distribution.jpg')
    return fig, ax

```
**结果：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021030719191254.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

##  13  Scatter plot
[Python 星号（*）操作符的用法](https://blog.csdn.net/yzj99848873/article/details/48025593?utm_source=blogxgwz0)

```python
## 导入库
import matplotlib.pyplot as plt
import numpy as np

# fix random state for reproducibility
np.random.seed(10)
x = np.arange(0, 50, 2)
#  x.shape = 25 是 x 的维度 ， * 可起到自动解包的作用
# https://blog.csdn.net/yzj99848873/article/details/48025593?utm_source=blogxgwz0
y = x ** 1.3 + np.random.rand(*x.shape) * 30.0
s = np.random.rand(*x.shape) * 800 + 500
fig = plt.subplot()
plt.scatter(x, y, s, c = 'green', alpha = 0.5, marker=r'$\clubsuit$',
            label = "Luck") # \clubsuit 是Tex 命令

## 设置坐标轴参数
plt.xlabel('Leprechauns')
plt.ylabel('Gold')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()
plt.savefig("D:\PycharmProjects\pythonProject\PythonMatPlot\Scatter Symbol.png")
```
结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210320192918932.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

 ##  14  Spectrum Representation


```python
## Spectrum Representation
import matplotlib.pyplot as plt
import  numpy as np

np.random.seed(10)

dt = 0.01 # sampling interval
Fs = 1 / dt # Frequency of sampling
t = np.arange(0, 10, dt)
# generate noise
nse = np.random.rand(len(t))
r = np.exp( -t / 0.05)
# https://blog.csdn.net/u011599639/article/details/76254442
#  np.convolve 为 numpy 的卷积函数
cnse = np.convolve(nse, r) * dt
cnse = cnse[:len(t)] # 取元组前 len(t)个

s = 0.1 * np.sin(4 * np.pi * t) + cnse # the signal
figs , axs = plt.subplots( nrows = 3, ncols = 2, figsize=(7,7))

# plot time signal
axs[0, 0].set_title('time signal')
axs[0, 0].plot(t, s, color = 'C0')
axs[0, 0].set_xlabel('time')
axs[0, 0].set_ylabel('Amplitude')

# plot different spectrum
axs[1, 0].set_title('Magnitude Spectrum')
axs[1, 0].magnitude_spectrum(s, Fs = Fs, color = 'C1')

axs[1, 1].set_title('Log of Magnitude Spectrum')
axs[1, 1].magnitude_spectrum(s, Fs = Fs, scale='dB', color = 'C1')

axs[2, 0].set_title('phase spectrum')
axs[2, 0].phase_spectrum(s, Fs = Fs, color = 'C2')

axs[2, 1].set_title('angle spectrum')
axs[2, 1].angle_spectrum(s, Fs = Fs, color = 'C2')

axs[0, 1].remove() # 不展示空图，位置[0,1]

figs.tight_layout()
plt.show()
plt.savefig("D:\PycharmProjects\pythonProject\PythonMatPlot\Spectrum Representation.png")
```

![结果](https://img-blog.csdnimg.cn/20210320202907426.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

## 15  Scatter Masked

```
import matplotlib.pyplot as plt
import  numpy as np

## fix random state for reproducility
np.random.seed(10)

N = 100
r0 = 0.6
fig = plt.subplot()
x = 0.9 * np.random.rand(N)
y = 0.9 * np.random.rand(N)
area = (20 * np.random.rand(N))**2 # 0 to 10 point radii
c = np.sqrt(area)
r = np.sqrt( x ** 2 + y ** 2)

area1 = np.ma.masked_where( r < r0 , area)
area2 = np.ma.masked_where(r >= r0 , area)
plt.scatter(x, y, area1, marker='^', c = c)
plt.scatter(x, y, area2, marker='o', c = c)

# show the boundary
theta = np.arange(0, np.pi/2, 0.01)
plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))
```

![image-20210403201815890](C:%5CUsers%5CRankFan%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210403201815890.png)