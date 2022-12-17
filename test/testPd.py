#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : testPd.py
@Author  : jay.zhu
@Time    : 2022/12/16 15:33
"""
import pandas as pd

# 三个字段 name, site, age
name = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 98]
namelist = {"name", "site", "age"}
list0 = [name, st, ag]


# 字典
# dict0 = {nameItem : valItem for nameItem in namelist for valItem in list0}
dict0 = {
    "one":[1,2,3,4],
    "two":[0.1, 0.2,"",""]
}


df = pd.DataFrame(dict0)

# 保存 dataframe
df.to_csv('site.csv')
