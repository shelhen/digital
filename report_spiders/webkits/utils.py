# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: utils.py
@Project: digital_evalutate 
@Time: 2025/02/07  11:15
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: 可能用到的工具模块。
"""
import os
import re
import time
import js2py
from base64 import b64encode
import requests
from fake_useragent import UserAgent


def generate_random_ua():
    ua = UserAgent(
        browsers=['Edge', 'Chrome', "Opera", " Safari", "Android", ],
        os=["Windows", "Mac OS X"],
    )
    return ua.random


def textextract(src, target):
    # 实现多线程的文字提取工作，同时，对无法正确处理的文本进行ocr提取再次返回进行文字提取
    exceptions=[]
    pathrows = [entry.name for entry in os.scandir(src) if entry.name.endswith('.txt')]
    for entryName in pathrows:
        src_ = os.path.join(src, entryName)
        with open(src_, 'r', encoding='utf8') as f:
            text =f.read().replace(' ', '')
        res1 = re.finditer(r'\s+[第三四五六七八九十、章节§\d]{2,}\s*董事会[工作]*报告\s?[^”…⋯.\-„]', text)
        res2 = re.finditer(r'\s+[第三四五六七八九十章节、§\d]{2,}\s*重[要大]事\s?项\s?[^”…⋯.\-„\w]', text)
        res3 = re.finditer(r'\s+[第三四五六七八九十章节、§\d]{2,}\s*[公司经营情况管理层]{3,7}讨论与分\s?析\s?[^”…⋯.\-„]', text)
        target1 = [i for i in res1]
        target2 = [i for i in res2]
        target3 = [i for i in res3]
        if len(target1) in [1, 2] and len(target2) in [1,2]:
            text_ = text[target1[-1].start():target2[-1].end()]
        elif len(target3) in [1, 2] and len(target2) in [1,2]:
            text_ = text[target3[-1].start():target2[-1].end()]
        else:
            exceptions.append(entryName)
            print(f'无法提取文件: {entryName}')
            continue
        with open(os.path.join(target, entryName),'w',encoding='utf8') as f:
            f.write(text_)


def get_enckey():
    with open("./juchaozixun.js", 'r', encoding='utf8') as f:
        result = f.read()
    ts = str(int(time.time()))
    context = js2py.EvalJs(enable_require=True)
    context.execute(result)
    return b64encode(ts.encode('utf-8')).decode(), context.getResCode_(ts)


proxy = {"host": "u3991.10.tn.16yun.cn", "port": "6442", "user": "16KPIRES", "pass": "552380", "name": "yiniuyun"}



if __name__ == '__main__':
    session = requests.Session()
    # session["GET"]()

