# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: infospider.py
@Project: digital_evalutate 
@Time: 2025/02/07  11:20
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: 获取企业基本信息的接口，可能不太全：1.给定企业编码，搜索返回企业信息；2.直接爬取所有企业信息。
"""
from time import time
import requests
from utils import generate_random_ua, proxy, get_enckey


class InfoSpider(object):

    def __init__(self):
        self.base_url = 'http://webapi.cninfo.com.cn{0}'
        self.session = requests.Session()
        self.session.headers = {
            "Host": "webapi.cninfo.com.cn",
            "Origin": "http://webapi.cninfo.com.cn",
            "Referer": "http://webapi.cninfo.com.cn/"
        }
        proxy_meta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % proxy
        self.session.proxies = {"http": proxy_meta, "https": proxy_meta}

    def get_total_enterprises(self):
        """
        首先以此获取所有证监会行业信息，其次根据行业信息以此查询全部的企业编码信息。
        """
        params = {
            "indtype": "008001",
            "indcode": "0",  # 二级三级通过查询后取得，可以为空，为空默认为取全部行业列表
            "format": "json",
            "@column": "SORTCODE,SORTNAME,PARENTCODE"
        }
        self.session.headers['User-Agent'] = generate_random_ua()
        self.session.headers["Cookie"] = f"Hm_lvt_489bd07e99fbfc5f12cbb4145adb0a9b={int(time())}; Hm_lpvt_489bd07e99fbfc5f12cbb4145adb0a9b={int(time())}"
        self.session.headers['Accept-Enckey'] = get_enckey()[1]
        res = self.session.get(self.base_url.format('/api/stock/p_public0002'), params=params, timeout=10).json()
        total_codes = []
        for ind in res['records']:
            # 137002表示通过证监会行业分类查询；platecode=C,具体行业包括C11-C99表示查询制造业
            # 137006,表示通过地区分类查询；platecode=320000;表示查询江苏省的
            self.session.headers['Accept-Enckey'] = get_enckey()[1]
            self.session.headers['User-Agent'] = generate_random_ua()
            self.session.headers["Cookie"] = f"Hm_lvt_489bd07e99fbfc5f12cbb4145adb0a9b={int(time())}; Hm_lpvt_489bd07e99fbfc5f12cbb4145adb0a9b={int(time())}"
            params = {
                'platetype': '137002',
                'platecode': ind["SORTCODE"],
                '@column': 'SECCODE,SECNAME',
                '@orderby': 'SECCODE:asc'
            }
            res = self.session.get(self.base_url.format('/api/stock/p_public0004'), params=params, timeout=10).json()
            for record in res['records']:
                total_codes.append(record['SECCODE'])

        with open("../results/enterprise.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(total_codes))
        return total_codes

    def get_enterprise_info(self, scode:str):
        """
        根据给定的企业编码查询企业详情，其中企业编码可以为一个或多个企业编码由逗号连接的字符串
        :param scode:
        :return:
        """
        self.session.headers['User-Agent'] = generate_random_ua()
        self.session.headers["Cookie"] = f"Hm_lvt_489bd07e99fbfc5f12cbb4145adb0a9b={int(time())}; Hm_lpvt_489bd07e99fbfc5f12cbb4145adb0a9b={int(time())}"
        self.session.headers["mcode"], self.session.headers['Accept-Enckey'] = get_enckey()
        # api = self.base_url.format("/api/stock/p_stock2100")
        # 两个api都可以查，但是特征列不同，暂时选择老版，新版对应关系：https://webapi.cninfo.com.cn/#/apiDoc
        api = self.base_url.format("/api/sysapi/p_sysapi1018")
        params = {
            'scode': scode,
            "@column": "SECCODE,SECNAME,F001V,F002D,F003V,F004V,F009D,F015V,F016V,F024V",
            # "@column": "SECCODE,SECNAME,ORGNAME,F001V,F002D,F003V,F004V,F009D,F015V,F016V,F024V,F025V,F026V,F027V,F028V,F029V,F030V,F031V"
        }
        res = self.session.get(api, params=params).json()
        for record in res['records']:
            result = f"{record['SECCODE']} {record['F001V']} {record['SECNAME'].replace(' ', '')} {record['F024V']} {record['F002D']} {record['F009D']} {record['F004V']} {record['F003V']} {record['F015V']} {record['F016V']}"
            print(result)

    def split_codes(self, codes=None):
        """
        组织批量股票代码的查询结构。
        :return:
        """
        split_lists = []
        count = len(codes)
        if len(codes) < 20:
            return [','.join(codes)]
        index = 0
        while index < count:
            chunk_size = 20 if 50<count-index<70 else min(50, len(codes) - index)
            split_lists.append(','.join(codes[index:index + chunk_size]))
            index += chunk_size
        return split_lists

    def main(self):
        with open('../results/enterprise.txt', 'r', encoding='utf8') as f:
            codes = [code.strip() for code in f.readlines()]
        # codes = infospider.get_total_enterprises()
        for code_str in self.split_codes(codes[:120]):
            infospider.get_enterprise_info(code_str)


if __name__ == '__main__':
    infospider = InfoSpider()
    infospider.main()


