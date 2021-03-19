## 理解SVM(Support Vector Machines)



[TOC]

### 导语

这个假期在看《机器学习实战》这本书，它主要介绍了实现机器学习的主流算法并用Python编写代码。我看的这本是2013年出版的，相对来说书可能有点老，但是打打基础对算法有一个初步认识还是很有帮助。放假之前我还计划寒假要把《深度学习》那本花书看完，结果是只有开头，没有然后。SVM（支持向量机）这一章花了我很多时间，书上讲得不是很详细，理论知识很多很复杂，我就去搜集了一些资料，网上很多朋友都写得特别好。因为SVM这个东西本身就比较难懂，所以我只大概介绍一下基本概念以及其SMO（Sequential Minimal Optimization，序列最小化）算法的实现。

### 什么是SVM

SVM是一种处理二类分类问题的模型，其基本思想就是：通过找到离分隔超平面（separating hyperplane）最近的点，确保这些点离分隔面的距离尽可能远，最终可转化为一个凸二次规划问题的求解。

什么是分隔超平面，考虑下图二维平面中的数据点分布，我们是不是可以画出一条直线将两类数据点分开，这条将数据集分隔开的直线就称为分隔超平面。但是什么样的超平面是最好的呢？通过找到离这条直线最近的点，确保他们离直线的距离尽可能大，以此构建的线性分类器，在对测试集进行预测时结果也就越可信。所谓支持向量就是离分隔超平面最近的那些点，接下来要做的就是最大化支持向量到分隔超平面的距离。（下图来自《Sequential Minimal Optimization:A Fast Algorithm for Training Support Vector Machines》）[^2]



<img src="C:\Users\王颜溶\AppData\Roaming\Typora\typora-user-images\image-20210310145445445.png" alt="image-20210310145445445" style="zoom:25%;" />

### SMO高效优化算法

首先我们讨论一种简单的情况，即假设数据都是线性可分的，后面再讨论线性不可分的情况。用$\mathbf{x}$表示数据点，用y表示类别，其取值为1或-1（这样取值可以大大简化计算过程，实际取两个不同的值也完全可以），代表两个不同的类。这个1和-1的分类标准起源于**Logistic回归**，[具体内容可参考这篇文章的解释。](https://blog.csdn.net/v_JULY_v/article/details/7624837)

分隔超平面的方程可以表示为
$$
\mathbf{w}^{\text{T}}\mathbf{x}+b=0. \notag
$$

如上图1.1-(b)所示，该二维平面上的超平面用$f(\mathbf{x})=\mathbf{w}^{\text{T}}\mathbf{x}+b$表示，$f(\mathbf{x})=0$即$(x_{1},x_{2})$是位于超平面上的点，$f(\mathbf{x})>0$即$(x_{1},x_{2})$是属于$y=1$类的数据点，$f(\mathbf{x})<0$即$(x_{1},x_{2})$是属于$y=-1$类的数据点。数据点分类正确就有$yf(\textbf{x})>0$，那接下来如何确定这个超平面呢？

要计算一个点到分隔超平面的距离，要给出点到超平面的垂线距离，该值为$y (\mathbf{w}^{\text{T}}\mathbf{x}+b)/\|\textbf{w}\|=|f(\mathbf{x})|/\|\textbf{w}\|$，也被称为点到超平面的几何间隔，$y (\mathbf{w}^{\text{T}}\mathbf{x}+b)$被称为点到超平面的函数间隔。现在目标是找出定义的$\text{w}$和$b$，为此，我们先要找到具有最小间隔的数据点，然后对该间隔最大化。于是最大间隔分类器的目标函数定义为
$$
\arg\max_{\text{w},b}\left \{ \min_{m}\left ( y\left ( \mathbf{w}^{\text{T}}\mathbf{x}+b \right ) \right )\cdot \frac{1}{\|\textbf{w}\|}  
\right \}
\quad \quad\text{subject to}\quad y_{i}(\textbf{w}^{\text{T}}\textbf{x}_{i}+b)\ge y (\mathbf{w}^{\text{T}}\mathbf{x}+b),i=1,\dots,n.
$$

直接求解上述最优化问题比较困难，所以我们令所有支持向量的$y (\mathbf{w}^{\text{T}}\mathbf{x}+b)$均为1，接着就可以通过求$\|\textbf{w}\|^{-1}$的最大值来得到最终解。注意，但并不是所有数据点的$y (\mathbf{w}^{\text{T}}\mathbf{x}+b)$均为1，只有离超平面最近的点值才为1，离超平面越远的点，其$y (\mathbf{w}^{\text{T}}\mathbf{x}+b)$的值大于1。这时上述优化问题转化为
$$
\max \frac{1}{\|\textbf{w}\|}  \quad \text{subject to} \quad y_{i}(\textbf{w}^{\text{T}}\textbf{x}_{i}+b)\ge1,i=1,\dots,n.
$$
求$1/\|\textbf{w}\|$的最大值即为求$\frac{1}{2} \|\textbf{w}\|^{2}$的最小值，所以上述优化问题等价于
$$
\min \frac{1}{2} \|\textbf{w}\|^{2}  \quad \text{subject to} \quad y_{i}(\textbf{w}^{\text{T}}\textbf{x}_{i}+b)\ge1,i=1,\dots,n.
$$
这是一个凸二次规划问题，可以用著名的拉格朗日乘子法求解，通过给每一个约束条件加上一个拉格朗日乘子$\alpha$，定义拉格朗日函数
$$
\mathcal{L}\left (\textbf{w},b,\boldsymbol{\alpha}\right)= \frac{1}{2} \|\textbf{w}\|^{2}-\sum_{i=1}^{n}\alpha_{i}
\left \{ y_{i}(\textbf{w}^{\text{T}}\textbf{x}_{i}+b)-1 \right \}
$$
然后令
$$
\theta \left ( \textbf{w} \right ) =\max_{\alpha_{i}\ge0} \mathcal{L}\left (\textbf{w},b,\boldsymbol{\alpha}\right)
$$
当所有约束条件都满足时，有$\theta \left ( \textbf{w} \right )=\frac{1}{2} \|\textbf{w}\|^{2}$，也就是前面要最小化的量。所以目标函数又变为
$$
\min_{\textbf{w},b} \theta \left ( \textbf{w} \right )=
\min_{\textbf{w},b}\max_{\alpha_{i}\ge0} \mathcal{L}\left (\textbf{w},b,\boldsymbol{\alpha}\right)
$$
直接求解不太容易，所以我们交换一下求解顺序，将目标函数变为
$$
\max_{\alpha_{i}\ge0}\min_{\textbf{w},b}\mathcal{L}\left (\textbf{w},b,\boldsymbol{\alpha}\right)
$$
交换以后的新问题是原始问题的对偶问题，这两个问题的最优值在满足KKT条件时相等，对偶问题和原始问题的转化以及KKT条件请查阅其他相关资料。

接下来就是对偶问题的具体求解过程：

(1) 固定$\boldsymbol{\alpha}$，求$\mathcal{L}$关于$\textbf{w}$和$b$的最小化问题。
$$
\begin{aligned}
\frac{\partial  \mathcal{L}}{\partial \textbf{w}}&=0 \Rightarrow \textbf{w}=\sum_{i=1}^{n} \alpha_{i} y_{i} \textbf{x}_{i}  \\
\frac{\partial  \mathcal{L}}{\partial b}&=0 \Rightarrow \sum_{i=1}^{n} \alpha_{i} y_{i}=0  
\end{aligned}
$$

把(8)和(9)带入(4)，得到
$$
\begin{equation}
	\begin{aligned}
		\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) &=\frac{1}{2}\|\mathbf{w}\|^{2}-\sum_{i=1}^{n} \alpha_{i}\left\{y_{i}\left(\mathbf{w}^{\text{\text{T}}} \mathbf{x}_{i}+b\right)-1\right\} \\
		&=\frac{1}{2} \mathbf{w}^{\text{T}} \mathbf{w}-\sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{w}^{\text{T}} \mathbf{x}_{i}-\sum_{i=1}^{n} \alpha_{i} y_{i} b+\sum_{i=1}^{n} \alpha_{i} \\
		&=\frac{1}{2} \mathbf{w}^{\text{T}} \sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}-\sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{w}^{\text{T}} \mathbf{x}_{i}-\sum_{i=1}^{n} \alpha_{i} y_{i} b+\sum_{i=1}^{n} \alpha_{i} \\
		&=\frac{1}{2} \mathbf{w}^{\text{T}} \sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}-\mathbf{w}^{\text{T}} \sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}-\sum_{i=1}^{n} \alpha_{i} y_{i} b+\sum_{i=1}^{n} \alpha_{i} \\
		&=-\frac{1}{2} \mathbf{w}^{\text{T}} \sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}-\sum_{i=1}^{n} \alpha_{i} y_{i} b+\sum_{i=1}^{n} \alpha_{i} \\
		&=-\frac{1}{2} \mathbf{w}^{\text{T}} \sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}-b \sum_{i=1}^{n} \alpha_{i} y_{i}+\sum_{i=1}^{n} \alpha_{i} \\
		&=-\frac{1}{2}\sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}^{\text{T}} \sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}-b \sum_{i=1}^{n} \alpha_{i} y_{i}+\sum_{i=1}^{n} \alpha_{i} \\
		&=-\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i} y_{i}\mathbf{x}_{i}^{\text{T}} \alpha_{j} y_{j} \mathbf{x}_{j}-b \sum_{i=1}^{n} \alpha_{i} y_{i}+\sum_{i=1}^{n} \alpha_{i} \\
		&=\sum_{i=1}^{n} \alpha_{i} -\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i}\alpha_{j} y_{i}y_{j} \mathbf{x}_{i}^{\text{T}}\mathbf{x}_{j}.
	\end{aligned}	
\end{equation}
$$
此时拉格朗日函数只包含变量$\alpha_{i}$，接下来求关于的$\alpha_{i}$最大化问题。

(2) 求关于$\boldsymbol{\alpha}$的最大。通过上一步，已经求出$\textbf{w}$和$b$，从而目标函数变为
$$
\begin{align}
&\max_{\alpha_{1},\dots,\alpha_{n}} \sum_{i=1}^{n} \alpha_{i} -\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i}\alpha_{j} y_{i}y_{j} \mathbf{x}_{i}^{\text{T}}\mathbf{x}_{j}\\
&\quad \text { s.t. } \ \ \alpha_{i} \geq 0, i=1, \ldots, n  \notag \\
&\quad \quad \quad\sum_{i=1}^{n} \alpha_{i} y_{i}=0. \notag 
\end{align}
$$
针对上述优化问题，1996年[John C.Platt发布了一个称为SMO（Sequential Minimal Optimization）的算法](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)，用于训练SVM。SMO是一个将非常大的二次规划问题分解为多个小优化问题来求解的算法，SMO算法的求解时间短很多，它对于线性SVM和稀疏数据集最快。我们一旦求出每一个训练数据点所对应的拉格朗日乘子$\alpha_{i}$，也就可以通过
$$
\textbf{w}=\sum_{i=1}^{n} \alpha_{i} y_{i} \textbf{x}_{i},\quad b=\textbf{w}^{\text{T}}\textbf{x}_{k}-y_{k} \ \ \text{for some}\ \ \alpha_{k}>0
$$
求出$\textbf{w}$和$b$的值。<u>**注意：John C.Platt的SMO算法一文中定义线性SVM的输出公式为$u=\mathbf{w}^{\text{T}}\mathbf{x}-b$，那么分隔超平面就为$\mathbf{w}^{\text{T}}\mathbf{x}-b=0$，我们之前的叙述中是定义$f(\mathbf{x})=\mathbf{w}^{\text{T}}\mathbf{x}+b$，从此处开始，后文均采用定义$u=\mathbf{w}^{\text{T}}\mathbf{x}-b$，这两种定义实质是一样的，请读者自行分辨。**</u>

但是往往并不是所有数据集都是线性可分，这时需要引入松弛变量（slack variable）至原始优化问题(3)中，目标函数变为
$$
\begin{align}
&\min \frac{1}{2} \|\textbf{w}\|^{2} +C\sum_{i=1}^{n}\xi_{i}  \\
& \ \ \text{s.t.} \ \  y_{i}(\textbf{w}^{\text{T}}\textbf{x}_{i}-b)\ge1-\xi_{i},i=1,\dots,n. \notag \\

\end{align}
$$
其中$\xi_{i}$是松弛变量，$C$是一个权衡参数。这时就可以允许有些数据点处于超平面的错误一侧。把这个新的最优化问题(13)转变为对偶形式后最终的目标函数变为
$$
\begin{align}
& \min_{\alpha_{1},\dots,\alpha_{n}}\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i}\alpha_{j} y_{i}y_{j} K\left ( \mathbf{x}_{i},\mathbf{x}_{j} \right ) 
- \sum_{i=1}^{n} \alpha_{i} \\
& \quad\text{s.t.} \ \ \ \ 0\le \alpha_{i}\le C,i=1,\dots,n,  \\
&\quad \quad \quad\sum_{i=1}^{n} \alpha_{i} y_{i}=0, 
\end{align}
$$
此处$K$是一个核函数，引入核函数是为了处理非线性数据和统一线性问题和非线性问题的形式，省去分开讨论的麻烦。常用的核函数有多项式核、高斯核、线性核，线性核$ K\left ( \mathbf{x}_{i},\mathbf{x}_{j} \right ) = \left \langle  \mathbf{x}_{i},\mathbf{x}_{j} \right \rangle$就是原始空间中的内积，此时目标函数就从(14)变为(11)。有关引入核函数的具体内容我们下期再写，这期主要讲计算过程。KKT条件是解决一个正定二次规划问题的充分必要条件，上述规划问题需要满足的条件是
$$
\begin{align}
\alpha_{i}=0\Leftrightarrow y_{i}\mu_{i}\ge1 \\
0< \alpha_{i}<C\Leftrightarrow y_{i}\mu_{i}=1 \\
\alpha_{i}=C\Leftrightarrow y_{i}\mu_{i}\le1
\end{align}
$$
(17)表明$\alpha_{i}$位于边界内部；(17)表明$\alpha_{i}$是支持向量，位于边界；(19)表明$\alpha_{i}$在两条边界之间。下面我们要解决的问题是关于$\mathbf{\alpha}$求目标函数(14)的最小值。为了求解拉格朗日乘子$\alpha_{1},\dots,\alpha_{n}$，每次选取一对$\alpha_{i}和$$\alpha_{j}$，然后固定除$\alpha_{i}和$$\alpha_{j}$外的其他乘子，使目标函数只是关于$\alpha_{i}$和$\alpha_{j}$的函数，$\alpha_{j}$可由$\alpha_{i}$和其他乘子表示，在得到$\alpha_{i}$和$\alpha_{j}$的解之后，用$\alpha_{i}$和$\alpha_{j}$改进其他乘子。这样，不断的从$n$个乘子中任意抽取两个求解，迭代求解子问题，最终达到求解原问题的目的。**为什么每次选取两个拉格朗日乘子进行优化？这是由约束条件$\sum_{i=1}^{n} \alpha_{i} y_{i}=0$决定的。**现在假设选取$\alpha_{1}和$$\alpha_{2}$，子问题的目标函数为
$$
\frac{1}{2}\alpha_{1}^{2}K_{11}+\frac{1}{2}\alpha_{2}^{2}K_{22}+sK_{12}\alpha_{1}\alpha_{2}+y_{1}\alpha_{1}v_{1}
+y_{2}\alpha_{2}v_{2}-\alpha_{1}-\alpha_{2}+\psi，
$$
其中
$$
s=y_{1}y_{2},\quad
K_{ij}=K\left ( \mathbf{x}_{i},\mathbf{x}_{j} \right ) \\
v_{i}=\sum_{j=3}^{n} y_{j}\alpha_{j}^{*}K_{ij}=u_{i}+b^{*}-y_{1}\alpha_{1}^{*}K_{1i}-
y_{2}\alpha_{2}^{*}K_{2i},\quad u_{i}=\sum_{j=1}^{n} y_{j}\alpha_{j}K_{ij}-b \notag
$$

在乘子更新之前记作$\alpha_{1}^{old}$、$\alpha_{2}^{old}$，更新之后记为$\alpha_{1}^{new}$、$\alpha_{2}^{new}$，它们彼此之间需要满足的约束条件是
$$
\alpha_{1}^{old}y_{1}+\alpha_{2}^{old}y_{2}=\alpha_{1}^{new}y_{1}+\alpha_{2}^{new}y_{2}=\zeta,
$$
$\zeta$是一个常数。两个乘子不好同时求解，所以可以先求$\alpha_{2}^{new}$，再用$\alpha_{2}^{new}$表示$\alpha_{1}^{new}$。接下来我们确定$\alpha_{2}^{new}$的取值范围$L\le\alpha_{2}^{new}\le H$：

(1)当$y_{1}\ne y_{2}$时，由$\alpha_{1}^{old}y_{1}+\alpha_{2}^{old}y_{2}=\alpha_{1}^{new}y_{1}+\alpha_{2}^{new}y_{2}=\zeta$知，$\alpha_{1}^{old}-\alpha_{2}^{old}=\zeta$或$-\alpha_{1}^{old}+\alpha_{2}^{old}=\zeta$，所以有$L=\max \left( 0,-\zeta  \right)$，$H=\min\left( C,C-\zeta \right )$。

<img src="C:\Users\王颜溶\AppData\Roaming\Typora\typora-user-images\image-20210311171040966.png" alt="image-20210311171040966" style="zoom:50%;" />

(2)当$y_{1}= y_{2}$时，由$\alpha_{1}^{old}y_{1}+\alpha_{2}^{old}y_{2}=\alpha_{1}^{new}y_{1}+\alpha_{2}^{new}y_{2}=\zeta$知，$\alpha_{1}^{old}+\alpha_{2}^{old}=\zeta$或$-\alpha_{1}^{old}-\alpha_{2}^{old}=\zeta$，所以有$L=\max \left( 0,\zeta-C  \right)$，$H=\min\left( C,\zeta \right )$。

<img src="C:\Users\王颜溶\AppData\Roaming\Typora\typora-user-images\image-20210311172315391.png" alt="image-20210311172315391" style="zoom:50%;" />

综上我们有
$$
\begin{cases}
   L=\max \left( 0,\alpha_{2}^{old}-\alpha_{1}^{old} \right),H=\min(C,C+\alpha_{2}^{old}-\alpha_{1}^{old}) \quad \text{if} \ \ y_{1}\ne y_{2}\\
   \\
  L=\max \left( 0,\alpha_{2}^{old}+\alpha_{1}^{old}-C \right),H=\min(C,\alpha_{2}^{old}+\alpha_{1}^{old}) \quad \text{if} \ \ y_{1}= y_{2}
\end{cases}
$$
式(21)两边同时乘以$y_{1}$可得
$$
\alpha_{1}+s\alpha_{2}=\alpha_{1}^{*}+s\alpha_{2}^{*}=w,\quad w=-y_{1}\sum_{i=3}^{n}y_{i} \alpha_{i}^{*}.
$$
从而$\alpha_{1}=w-s\alpha_{2}$,代入(22)有
$$
\frac{1}{2}\left ( w-s\alpha_{2} \right )^{2} K_{11}+\frac{1}{2}\alpha_{2}^{2}K_{22}+sK_{12}\left ( w-s\alpha_{2} \right )\alpha_{2}+y_{1}\left ( w-s\alpha_{2} \right )v_{1}
+y_{2}\alpha_{2}v_{2}-\left ( w-s\alpha_{2} \right )-\alpha_{2}+\psi，
$$
式(25)两边对$\alpha_{2}$求导并令其为0，结合$s=y_{1}y_{2}$、$K_{ij}=K\left ( \mathbf{x}_{i},\mathbf{x}_{j} \right )$、$v_{i}=\sum_{j=3}^{n} y_{j}\alpha_{j}^{*}K_{ij}=u_{i}+b^{*}-y_{1}\alpha_{1}^{*}K_{1i}-
y_{2}\alpha_{2}^{*}K_{2i}$有
$$
\begin{align}
-s\left ( w-s\alpha_{2} \right )K_{11}+\alpha_{2}K_{22}-K_{12}\alpha_{2}+sK_{12}\left ( w-s\alpha_{2} \right )-y_{2}v_{1}
+y_{2}v_{2}+s-1=0 \\
\Rightarrow \alpha_{2}^{new,unc}\left ( K_{11}+K_{22}-2K_{12} \right ) =\alpha_{2}^{old}\left ( K_{11}+K_{22}-2K_{12} \right )+y_{2}\left ( u_{1}-u_{2}+y_{2}-y_{1} \right )
\end{align}
$$
unc指unconstrained，令$E_{i}=u_{i}-y_{i}$，$\eta=K_{11}+K_{22}-2K_{12}$，式(26)两边同除$\eta$有
$$
\alpha_{2}^{new,unc}=\alpha_{2}^{old}+\frac{y_{2}
\left ( E_{1}-E_{2} \right )}{\eta} 
$$
考虑上文$L\le\alpha_{2}^{new}\le H$的约束条件，最终得到
$$
\alpha_{2}^{new}=\begin{cases}
    H,\quad &\text{ if } \ \alpha_{2}^{new,unc} >H\\
    \\
    \alpha_{2}^{new,unc},\quad &\text{ if } \  L\le \alpha_{2}^{new,unc} \le H \\
    \\
    L,\quad &\text{ if } \ \alpha_{2}^{new,unc} <L
\end{cases}
$$
至此我们已经求出了$\alpha_{2}^{new}$，那么$\alpha_{1}^{new}=\alpha_{1}^{old}+y_{1}y_{2}\left (\alpha_{2}^{old}-\alpha_{2}^{new}  \right )$。而$b$在满足条件
$$
b=\begin{cases}
    b_{1},\quad &\text{ if } \ 0<\alpha_{1}^{new}<C\\
    \\
    b_{2},\quad &\text{ if } \  0<\alpha_{2}^{new}<C \\
    \\
    (b_{1}+b_{2})/2 ,\quad &\text{ otherwise.} 
\end{cases}
$$
时更新为
$$
b_{1}^{new}=b^{old}-E_{1}-y_{1}\left ( \alpha_{1}^{new}-\alpha_{1}^{old} \right ) K\left ( \mathbf{x}_{1},\mathbf{x}_{1}
 \right ) -y_{2}\left ( \alpha_{2}^{new}-\alpha_{2}^{old} \right ) K\left ( \mathbf{x}_{1},\mathbf{x}_{2} \right ) \\
 b_{2}^{new}=b^{old}-E_{2}-y_{1}\left ( \alpha_{1}^{new}-\alpha_{1}^{old} \right ) K\left ( \mathbf{x}_{1},\mathbf{x}_{2}
 \right ) -y_{2}\left ( \alpha_{2}^{new}-\alpha_{2}^{old} \right ) K\left ( \mathbf{x}_{2},\mathbf{x}_{2} \right )
$$
注意：在每次更新两个乘子的值后，$b$和$E_{i}$也都要进行更新，直到不再发生任何乘子的修改。这时我们就得到
$$
u=\sum_{i=1}^{n} \alpha_{i}y_{i}\left \langle \mathbf{x}_{i},\mathbf{x} \right \rangle -b
$$

那么有一个问题是我们该如何选取$\alpha_{1}$和$\alpha_{2}$呢？这时我们

(1)先遍历所有乘子，把第一个违反KKT条件的乘子作为更新对象，令为$\alpha_{2}$，

(2)在所有不违反KKT条件的乘子中，选择使$\mid E_{1}-E_{2}\mid$最大的$\alpha_{1}$进行更新，使得能最大限度增大目标函数的值。

这种方法被称为启发式法。

### 一个简单的数据集应用

数据集和代码（Python）均来自《机器学习实战》这本书，代码一定要自己敲一遍才可以，有助于加深理解，原始数据集分布如左图所示，在使用完整SMO算法找到分隔超平面后的结果如右图所示，包括画圈的支持向量机。

<center class="half">
<img src="C:\Users\王颜溶\Desktop\SVM\Figure_1.png" style="zoom:50%;" alt="Figure_1"  />
<img src="C:\Users\王颜溶\Desktop\SVM\Figure_2.png" style="zoom:43%;" alt="Figure_2"  />
<center>





我们现在已经成功训练出分类器了，但是这个数据集中的数据点恰好分布在一条直线的两边，如果两类数据点分别分布在一个圆的外部和内部呢？如下图所示，我们要怎么找到分隔超平面呢？这时就要引入核函数，来解决数据非线性可分的情况。这部分下期再写，哈哈哈最近开学有点小忙，我会尽量每天写一点的。祝好！

<img src="C:\Users\王颜溶\Desktop\SVM\Figure_3.png" style="zoom:50%;" alt="Figure_3"  />



```python

from numpy import *
from time import sleep

#打开文件并对其进行逐行解析
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
#原始数据可视化
def showDataSet(dataMat, labelMat):
    data_plus = []
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = array(data_plus)              #转换为numpy矩阵
    data_minus_np = array(data_minus)            #转换为numpy矩阵
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1]) #负样本散点图
    plt.title('original data')
    plt.ylim(-8, 6)
    plt.xlim(-2, 12)
    plt.show()
    return data_plus, shape(data_plus), type(data_plus)
if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')                        #加载训练集
    showDataSet(dataArr, labelArr)
#选择不等于i的j
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j
#用于调整大于H或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj
#简化版SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m): #multiply(a,b)是乘法，如果a,b是两个数组，那么对应元素相乘
            fXi = float(multiply(alphas, labelMat).T * \
                        (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
            ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * \
                            (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])# 对原始的alphas的复制，分配新的内存
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i]) #将alphas[j]调整到0 C之间
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print('L == H'); continue
            #continue意味着本次循环结束直接运行下一次for循环
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0: print('eta >= 0'); continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001): #abs()函数返回数字的绝对值
                    print('j not moving enough'); continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) *\
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T -\
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]) : b = b1
                elif (0 < alphas[j]) and (C > alphas[j]) : b = b2
                else: b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' %(iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print('iteration number: %d' % iter)
    return b, alphas
#完整版SMO算法
class optStruct: # 构建一个类，以实现其成员变量的填充
    def __init__(self, dataMatIn, classLabels, C, toler):#C 松弛变量
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))#根据矩阵行数初始化alpha参数为0 
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * \
                (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0 #初始化
    oS.eCache[i] = [1, Ei] #根据Ei更新误差缓存
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]#返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:  #有不为0的误差
        for k in validEcacheList:  #遍历,找到最大的Ek
            if k == i: continue
            Ek = calcEk(oS, k) #计算Ek
            deltaE = abs(Ei - Ek) #计算|Ei-Ek|
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else: #没有不为0的误差
        j = selectJrand(i, oS.m) #随机选择alpha_j的索引值
        Ej = calcEk(oS, j)
    return j, Ej
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
    ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print('L = H'); return 0
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0: print('eta >= 0'); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)#更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j not moving enough'); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b -Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold) * \
        oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j]-alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        return 1
    else: return 0
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)#初始化数据结构
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    #遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:#遍历整个数据集       
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS) #使用优化的SMO算法
            print('fullSet, iter %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else: #遍历不在边界0和C的alpha
            nonBounds = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]#元素所在的行
            for i in nonBounds:#.A  array
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False#遍历一次后改为非边界遍历
        elif (alphaPairsChanged == 0): entireSet = True #如果alpha没有更新,计算全样本遍历
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas
dataArr, labelArr = loadDataSet('testSet.txt')
b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)       
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w
ws = calcWs(alphas, dataArr, labelArr)
#分类结果可视化
def showClassifer(dataMat, classLabels, w, b):
    data_plus = []             
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = array(data_plus)              #转换为numpy矩阵
    data_minus_np = array(data_minus)            #转换为numpy矩阵   
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s = 30, alpha = 0.7)   #正样本散点图
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1], s = 30, alpha = 0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1 * x1) / a2, (-b - a1 * x2) / a2
    #画离散点，通过参数'o'实现
    plt.plot([x1, x2], [y1, y2]) #通过2个离散的点画连线
	plt.title('Classification result of SMO algorithm')
    plt.ylim(-8, 6)
    plt.xlim(-2, 12)
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s = 150, c = 'none', alpha = 0.7, linewidth = 1.5, edgecolor = 'red')
    plt.show()
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。    
showClassifer(dataArr, labelArr, ws, b)    
```



以上就是我对SVM算法的一些梳理和总结，其间也参照了其他一些小伙伴的文章。如果有什么问题请大家多多包涵，多多指正，我还只是一个懒惰的算法小萌新，欢迎你们和我交流! Mua~

> 参考文章（可通过链接跳转至原文）

\[1] [*支持向量机通俗导论（理解SVM的三层境界）*](https://blog.csdn.net/v_JULY_v/article/details/7624837)

\[2] [*《Sequential Minimal Optimization:A Fast Algorithm for Training Support Vector Machines》*](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)

\[3] [*《机器学习实战》*](https://www.manning.com/books/machine-learning-in-action)

### 后记

我很喜欢王小波的文字，尤其是《黄金时代》，有两段话我一直印象深刻，第一段是：

> 那一天我二十一岁，在我一生的黄金时代，我有好多奢望。我想爱，想吃，还想在一瞬间变成天上半明半暗的云，后来我才知道，生活就是个缓慢受锤的过程，人一天天老下去，奢望也一天天消逝，最后变得像挨了锤的牛一样。可是我过二十一岁生日时没有预见到这一点。我觉得自己会永远生猛下去，什么也锤不了我。

第二段是：

> 人活在世上，就是为了忍受摧残，一直到死。想明了这一点，一切都能泰然处之。

要好好珍惜生猛的自己！