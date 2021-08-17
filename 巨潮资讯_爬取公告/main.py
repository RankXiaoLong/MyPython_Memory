
import JuChao_stock_notice as snn
import os
import re
 
os.chdir("C:/Users/RankFan/Desktop/贴吧/notice") # Change by yourself according to your computer!

url_stock_json = "http://www.cninfo.com.cn/new/data/szse_stock.json"
url_query = "http://www.cninfo.com.cn/new/hisAnnouncement/query"


stock_notice = snn.Stock_Notice() # 实例化

stock_list = stock_notice.get_stock_list() # all stocks
stock_list_A_stock = stock_notice.get_stock_A() # A 股
# stock_list_A_stock[:1]

need_stock = []
for stock in stock_list_A_stock:
    if stock['code'] == '000001': # 平安保险
        need_stock.append(stock)
        stock_notice.get_stock_page(need_stock, url_query)
        stock_notice.save_to_excel(need_stock, "Stock Information")
        stock_notice.get_notice_url(need_stock, url_query)
    else:
        break

files = [f for f in os.listdir() if re.match("\d{6}", f)]
stock_notice.mk_dir(files)
stock_notice.load_notice_pdf(files, 51) # 获得前51个公告