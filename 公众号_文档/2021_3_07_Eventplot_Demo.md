hello, 大家好，最近看了一下matplotlib的图，还是挺多的，最近一些向大家分享的是`Lines, bars and markers`,画图的`demo code`很具有学习的价值。

-----

##  Eventplot Demo
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
**结果**：
![Eventplot Demo](https://img-blog.csdnimg.cn/20210307172520531.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

## Discrete distribution as horizontal bar chart
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
![Discrete distribution](https://img-blog.csdnimg.cn/2021030719191254.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JhbmtfZnhs,size_16,color_FFFFFF,t_70)

------
-------
觉得不从的朋友可以点赞，评论，转发呦！