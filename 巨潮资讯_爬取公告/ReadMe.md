**参考链接**：https://www.lianxh.cn/news/94192bcec139e.html

### 导入封装文件

如果`import JuChao_stock_notice`失败，可能是由于没有将`JuChao_stock_notice.py`放在正确的位置，可以通过以下命令查看可以存放的路径，建议存放在Python文件夹下的Scripts中。

```
import sys
sys.path()
```

当然也可以添加当前路径到`sys.path`中：

```
sys.path.append["C:\Users\RankFan\Desktop\巨潮资讯_爬取公告"]
```

### 路径设置

需要自行更改`os.chdir("C:/Users/RankFan/Desktop/贴吧/notice")` 