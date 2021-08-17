## ref
# https://www.cnblogs.com/wenqiangit/p/11252741.html
# https://blog.csdn.net/weixin_38664232/article/details/97259159
# https://blog.csdn.net/weixin_41100555/article/details/88555658

import pandas as pd

series_city = pd.Series(["厦门", "福州", "杭州", "合肥"])
series_province = pd.Series(["福建", "福建", "浙江", "安徽"])

# dict to Dataframe
dict_city = {"city": series_city}
df_city = pd.DataFrame(dict_city)

# Create Dataframe Directly
data = [
    ["厦门", "福建"],
    ["福州", "福建"],
    ["杭州", "浙江"],
    ["合肥", "安徽"],
]
df = pd.DataFrame(data, columns=["城市", "省份"])

# Series to DataframeSig_series_df_city = series_city.to_frame(name="city") # single series

Se_to_df_city = pd.DataFrame({"city": series_city, "province": series_province}) # muti series
Se_to_df_city.rename(columns={"city": "城市"})
