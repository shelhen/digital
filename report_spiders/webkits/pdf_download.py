# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: pdf_download.py
@Project: digital_evalutate 
@Time: 2025/02/08  00:16
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: 根据整理好的链接下载全部pdf。
"""
import os
import time
import random
from pathlib import Path
import multiprocessing
import requests
import fitz
from utils import generate_random_ua, proxy


class PdfDownloader(object):
    def __init__(self):
        self.base_path = Path("E:/年报文本")
        self.paths = (
            '../results/下载链接(清洗后).txt',
            self.base_path / "reports",
            self.base_path / "txts"
        )
        self.headers = {
            'User-Agent': generate_random_ua,
            'Host': 'www.cninfo.com.cn',
            'Accept-Encoding': 'gzip,deflate',
            'Connection': 'keep-alive',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        }
        proxy_meta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % proxy
        self.proxies = {"http": proxy_meta, "https": proxy_meta}

    def download_pdf(self):
        print("程序开始运行，用时大约4小时左右，请耐心等待……")
        with open(self.paths[0], 'r', encoding='utf8') as f:
            url_dicts = {url.strip().split(',')[0]: url.strip().split(',')[1] for url in f.readlines() if url.strip() != ''}
        with multiprocessing.Pool() as pool:
            for name, url in url_dicts.items():
                save_path = self.paths[1] / name
                if os.path.exists(save_path):
                    print(f"{name}已经下载,跳过")
                else:
                    pool.apply_async(self.crawler, args=(url, save_path))
            pool.close()
            pool.join()
        print('pdf年报全部下载完成')

    def crawler(self, url, save_path):
        retry_time = 0
        while retry_time < 3:
            try:
                with requests.get(url, stream=True, headers=self.headers, timeout=10, proxies=self.proxies, verify=False) as r:
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f'保存“{save_path}”文件成功！')
                break
            except requests.exceptions.RequestException as e:
                print("遇到异常{0}，尝试重试，当前重试第{1}次".format(e, retry_time))
                time.sleep(3 * random.random() + 2)
                retry_time += 1

    def get_files(self, path, type):
        return [entry.name for entry in os.scandir(path) if entry.name.endswith(type)]

    def pdf_extract(self, filepath, savepath):
        text = ''
        with fitz.open(filepath) as pdf:
            for page in pdf:
                text += page.get_text("block", sort=True).lower()
        with open(savepath, 'w', encoding='utf8') as f:
            f.write(text)
        print(f"{savepath}转化成功。")

    def tranlate_by_fitz(self):
        pdfpath = self.paths[1]
        textpath = self.paths[2]
        pdfpaths = self.get_files(pdfpath, '.pdf')
        with multiprocessing.Pool() as pool:
            for file in pdfpaths:
                save_path = os.path.join(textpath, file[:-3] + 'txt')
                pdf_path = os.path.join(pdfpath, file)
                if os.path.exists(save_path):
                    print(f"{file}已经转化, 跳过！")
                else:
                    pool.apply_async(self.pdf_extract, args=(pdf_path, save_path))
            pool.close()
            pool.join()
        print('txt年报全部转化完成')


