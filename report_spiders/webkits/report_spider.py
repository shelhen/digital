# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: report_spider.py
@Project: digital_evalutate 
@Time: 2025/02/07  22:55
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: 插入一段描述。
"""
import os
import re
import json
import time
import random
import requests
import multiprocessing
from uuid import NAMESPACE_URL, uuid3
from utils import generate_random_ua, proxy


class ReportSpider(object):
    def __init__(self):
        # "年报链接(未处理).txt"
        # self.save_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.paths=(
            '../results/年报链接(未处理).txt'
        )
        self.exclude_keywords = ['英文', '摘要', '已取消', '公告', '核查意见', 'h股', '财务报表']
        self.has_storages = set()
        self.session = requests.Session()
        self.session.headers = {
            'Accept': 'application/json,text/plain,*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Host': 'www.cninfo.com.cn',
            'Origin': 'http://www.cninfo.com.cn',
        }
        proxy_meta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % proxy
        self.session.proxies = {"http": proxy_meta, "https": proxy_meta}
        if os.path.exists('../results/orgs.json'):
            with open('../results/orgs.json', 'r', encoding="utf8") as f:
                self.org_map = json.load(f)
        else:
            self.get_orgs()

    def search_report(self, scode:str, year:int):
        url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
        plate, column = ('sh', 'sse') if int(scode)> 600000 else ('sz', 'szse')
        plate, column = ('', 'szse') if year > 2020 else (plate, column)
        data = {
            'pageNum': '1',
            "pageSize": "30",
            "tabName": "fulltext",
            "stock": f'{scode},{self.org_map[scode]}',
            "seDate": f"{year + 1}-01-01~{year + 1}-12-30",
            "column": column,
            "category": "category_ndbg_szsh",
            "isHLtitle": "true",
            'sortName': '',
            'sortType': '',
            'plate': plate,
            'searchkey': '',
            'secid': '',
        }
        retry_time = 0
        while retry_time<3:
            try:
                self.session.headers['User-Agent'] = generate_random_ua()
                response = self.session.post(url, data=data, timeout=10)
                reports = response.json()
                if reports["announcements"]:
                    for report in reports["announcements"]:
                        title = re.sub(r'\W', "", report['announcementTitle'].lower())
                        # 检查该文档是否需要下载
                        flag = True
                        for kw in self.exclude_keywords:
                            if kw in title:
                                flag = False
                                break
                        if flag:
                            self.parse(report, scode, year, title)
                else:
                    print(f'文件获取失败:{scode}-{year}')
            except Exception as e:
                print("遇到异常{0}，尝试重试，当前重试第{1}次".format(e, retry_time))
                time.sleep(3 * random.random() + 2)
                retry_time += 1

    def parse(self, report, scode, year, title):
        """
        仅仅用于格式化标题和保存下载链接，不下载pdf文件。
        :param report:
        :param stock:
        :param title:
        :return:
        """
        down_url = report['adjunctUrl']
        announcements = report['announcementTime']
        shortname = report['secName'].replace('*', '').replace(' ', '')
        bulletinId = down_url.split('/')[2][:-4]
        pdfurl = "http://static.cninfo.com.cn/" + down_url
        filename = f"{scode},{year},{announcements},{shortname},{bulletinId},{title},{pdfurl}"
        uuid = uuid3(NAMESPACE_URL, filename)
        if uuid in self.has_storages:
            print(f"{filename}已经爬取过了，跳过.")
        else:
            with open(self.paths[0], 'a', encoding='utf8') as f:
                f.write(f"{uuid},{filename}\n")
            print(f"{filename}下载成功！")

    def get_orgs(self):
        szse_url = 'http://www.cninfo.com.cn/new/data/szse_stock.json'
        bj_url = 'http://www.cninfo.com.cn/new/data/bj_stock.json'
        szse_orgs = self.session.get(szse_url, timeout=10).json()["stockList"]
        bj_orgs = self.session.get(bj_url, timeout=10).json()["stockList"]
        szse_orgs.extend(bj_orgs)
        self.org_map = {org["code"]:org['orgId'] for org in szse_orgs}
        with open('../results/orgs.json', 'w', encoding="utf8") as f:
            json.dump(self.org_map, f, ensure_ascii=False)

    def main(self):
        # 读取所有需要爬取的股票代码
        with open('../results/中国上市制造业股票代码.txt', 'r', encoding='utf8') as f:
            codes = [code.strip() for code in f.readlines() if code.strip()]
        # 读取已经爬取的数据信息
        with open(self.paths[0], 'r', encoding='utf8') as f:
            for item in f.readlines():
                if item.strip():
                    uuid, scode, year, *_ =item.strip().split(',')
                    self.has_storages.add(uuid)
                    self.has_storages.add(f"{scode}-{year}")
        for code in codes:
            with multiprocessing.Pool() as pool:
                for year in range(2014, 2025):
                    if f"{code}-{year}" in self.has_storages:
                        print(f"{code}-{year}已经下载，跳过.")
                    else:
                        pool.apply_async(self.search_report, args=(code, int(year)))
                pool.close()
                pool.join()
        self.check()

    def check(self):
        with open(self.paths[0],'r',encoding='utf8') as f:
            result = {cont.strip() for cont in f.readlines() if len(cont.strip().split(',')) == 8}
        with open(self.paths[0], 'w', encoding='utf8') as f:
            f.write('\n'.join(list(result)))


if __name__ == '__main__':
    report_spider = ReportSpider()