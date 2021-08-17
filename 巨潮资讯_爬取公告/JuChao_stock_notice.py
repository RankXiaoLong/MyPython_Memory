import xlrd # #读取excel
import xlwt  # #写入excel
import json  # #解析
import requests # #获取网页内容
import math    # #数学函数
import os
from urllib.request import urlretrieve  # #下载网络文件到本地


class Stock_Notice():
    def __init__(self,):
        url_stock_json = "http://www.cninfo.com.cn/new/data/szse_stock.json"
        self.url_stock_json = url_stock_json
        stock_list_A = []
        self.stock_list_A = stock_list_A
        dir_stock_list = []
        self.dir_stock_list = dir_stock_list

    def get_stock_list(self):
        """
        输入必须是json文件：http://www.cninfo.com.cn/new/data/szse_stock.json
        :return:
        """
        # url_stock_json = "http://www.cninfo.com.cn/new/data/szse_stock.json"
        # self.url_stock_json = url_stock_json
        ret_stock_json = requests.get(url=self.url_stock_json)

        ret_stock_json = ret_stock_json.content
        stock_list = json.loads(ret_stock_json)["stockList"]
        self.stock_list = stock_list
        return self.stock_list

    def get_stock_A(self):
        for stock in self.stock_list:
            if stock["category"] == "A股":
                self.stock_list_A.append(stock)
            else:
                break
        i = 0
        for stock in self.stock_list_A:
            if stock["code"][0] == "0" or stock["code"][0] == "3":
                self.stock_list_A[i]["column"] = "szse"
                self.stock_list_A[i]["plate"] = "sz"
            else:
                self.stock_list_A[i]["column"] = "sse"
                self.stock_list_A[i]["plate"] = "sh"
            i = i + 1
        return self.stock_list_A

    # 获取股票公告页数
    def get_stock_page(self, stock_list_A, url_query):
        # stock_list_new = stock_list_A[:10] #以前10个为例，直接删除这行
        # url_query = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
        i = 0
        for stock in stock_list_A:
            data = {
                "stock": stock["code"] + "," + stock["orgId"],
                "tabName": "fulltext",
                "pageSize": 30,
                "pageNum": 1,
                "column": stock["column"],
                "plate": stock["plate"],
                "isHLtitle": "true",
            }

            ret = requests.post(url=url_query, data=data)
            if ret.status_code == 200:
                ret = ret.content
                ret = str(ret, encoding="utf-8")
                total_ann = json.loads(ret)["totalAnnouncement"]
                stock_list_A[i]["pages"] = math.ceil(total_ann / 30)
                print(f"成功获取第{i}个股票页数！%d" % stock_list_A[i]["pages"])
                i = i + 1
            else:
                break

    # 将stock_list信息写入excel
    def save_to_excel(self, stock_list_A, excel_name):
        """
        输入数据类型是 stock_list_new
        :param stock_list_new:
        :return:
        """
        w = xlwt.Workbook()
        ws = w.add_sheet("股票信息")
        title_list = ["orgId", "category", "code", "pinyin",
                      "zwjc", "pages", "column", "plate"]

        j = 0
        for title in title_list:
            ws.write(0, j, title)
            j = j + 1

        i = 1
        for stock in stock_list_A:
            content_list = [stock["orgId"], stock["category"],
                            stock["code"], stock["pinyin"],
                            stock["zwjc"], stock["pages"],
                            stock["column"], stock["plate"]]
            j = 0
            for content in content_list:
                ws.write(i, j, content)
                j = j + 1
            i = i + 1
        w.save(excel_name + ".xls")

    # 上步骤的基础上，我们就可以对股票代码和页数进行循环，以获所有公告地址。
    def get_notice_url(self, stock_list, url_query):
        for stock in stock_list:
            # url_query = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
            name = stock["code"]

            w = xlwt.Workbook()
            ws = w.add_sheet(name)
            title_list = ["secCode", "secName", "announcementTitle",
                          "adjunctUrl", "columnId"]
            j = 0
            for title in title_list:
                ws.write(0, j, title)
                j = j + 1

            i = 1
            for page in range(1, int(stock["pages"]) + 1):
                data = {
                    "stock": stock["code"] + "," + stock["orgId"],
                    "tabName": "fulltext",
                    "pageSize": 30,
                    "pageNum": page,
                    "column": stock["column"],
                    "plate": stock["plate"],
                    "isHLtitle": "true",
                }

                ret = requests.post(url_query, data=data)
                ret = ret.content
                ret = str(ret, encoding="utf-8")
                ann_list = json.loads(ret)["announcements"]
                for ann in ann_list:
                    # print(ann)
                    content_list = [ann["secCode"], ann["secName"],
                                    ann["announcementTitle"], ann["adjunctUrl"], ann["columnId"]]
                    j = 0
                    for content in content_list:
                        ws.write(i, j, content)
                        j = j + 1
                    i = i + 1
            print(f"成功写入{name}！")
            w.save(f"{name}.xls")

    # 读入excel
    def read_to_excel(self, dir_name):
        w = xlrd.open_workbook(dir_name)  # Stock_Information.xls
        ws = w.sheet_by_name("股票信息")
        n_row = ws.nrows
        n_col = ws.ncols

        for i in range(1, n_row):
            dict = {}
            for j in range(n_col):
                title = ws.cell_value(0, j)
                value = ws.cell_value(i, j)
                dict[title] = value
            self.dir_stock_list.append(dict)
        return self.dir_stock_list

    ## 判断页数
    def is_le_N(self, stock_list):
        i = 0
        for stock in stock_list:
            if stock["pages"] >= N:
                print(i, stock["code"], stock["pages"])
            i = i + 1

    def mk_dir(self, files):
        for file in files:
            if not os.path.exists("PDF"):  # 不存在，创建pdf子文件夹
                os.mkdir("PDF")
            if not os.path.exists(f"./PDF/{file[0:6]}/"):  # 不存在，创建pdf子文件夹
                os.mkdir(f"./PDF/{file[0:6]}/")

    def load_notice_pdf(self, files, N):
        for file in files:
            w = xlrd.open_workbook(file)
            ws = w.sheet_by_name(file.replace(".xls", ""))
            n_row_load = ws.nrows
            n_col_load = ws.ncols
            for i in range(1, n_row_load):
                if i <= N:  # 下载个数 N
                    if N < n_row_load:
                        url = "http://static.cninfo.com.cn/" + ws.cell_value(i, 3)
                        print(url)
                        name = ws.cell_value(i, 2) + ".pdf"
                        urlretrieve(url, filename="./PDF/" + f'{file[0:6]}/' + name)  # + /PDF/ +
                        print(f"{file} 成功下载 第{i}个 文件名为：{name}！")
                    else:
                        break
                        # print(f"{N}大于{file[0:6]}股票公告文件个数")
                else:
                    break
                     # print(f"{file[0:6]}股票已经超过{N}个")