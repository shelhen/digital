# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: works.py
@Project: digital 
@Time: 2025/02/10  10:57
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: 插入一段描述。
"""
import requests
import pandas as pd
from time import time, sleep
from random import random

total_result = []

def parse(data):
    """
    companyName;companyType;jobType;location;releaseTime;endTime
    announcement;relatedLink
    recruitmentTarget;recruitmentType;
    :return:
    """
    CTM = {
        1: "互联网",
        2: "制造业",
        3: "银行",
        4: "国企",
        5: "外企",
        6: "民企",
        7: "事业单位"
    }
    STM = {
        1: "未投递",
        2: "已投递",
        3: "已笔试",
        4: "已面试",
        5: "面试通过",
        6: "已挂",
        7: "暂不投递"
    }
    JTM = {
        '1': "暑假实习",
        '2': "日常实习",
        '3': "秋招提前批",
        '4': "秋招正式批",
        '5': "秋招补录",
        '6': "春招"
    }
    RTM = {
        '1': "2025届本科及以上",
        '2': "2025届硕士及以上",
        '3': "2024/2025届本科及以上"
    }
    result = []
    for rec in data["data"]['recruitmentInfoList']:
        location = rec.get('location', None)
        item = dict(
            公司名称=rec["companyName"],
            公司类型=CTM.get(rec["companyType"], "无"),
            岗位类型=rec['jobType'],
            工作地点=location if location else "无",
            招聘类型=JTM.get(rec['recruitmentType'], "无"),
            招聘公告=rec['announcement'],
            相关链接=rec['relatedLink'],
            学历要求=RTM.get(rec['recruitmentTarget'], "无"),
            发布日期=rec['releaseTime'],
            截止日期=rec['endTime']
        )
        print(item)
        # if
        result.append(item)
    return result


def crawl(i):
    headers = {
        "Origin": "https://givemeoc.com",
        "Referer": "https://givemeoc.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    }
    url = "https://www.givemeoc.com/job/recruitments/list?pageNum={0}".format(i)
    retry_time = 0
    while retry_time<3:
        try:
            response = requests.get(url, headers=headers)
            result = response.json()
            return result
        except Exception as e:
            print("Error have happened, For The Reason {0}".format(e))


if __name__ == '__main__':
    for num in range(1, 217):
        result = crawl(num)
        if result is None:
            print("第{0}页爬取失败。".format(num))
            continue
        items = parse(result)
        total_result.extend(items)
        sum = len(total_result)
        pd.DataFrame(total_result).to_excel('./result.xlsx', index=False)
        sleep(0.2*random())
        if len(total_result)%100==0:
            sleep(random()+1)
            print("已爬取{0}个条目，完成率{1:.2%}".format(sum, sum/4329))
