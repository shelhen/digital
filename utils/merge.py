# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: main.py
@Project: common
@Time: 2024/11/19   01:58
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
@Brief:
"""
import os
import re
import jieba
import warnings
import pandas as pd
from pathlib import Path
from config import _EvaluationIndicatorSystem
from config import base_path, KeywordBy
warnings.filterwarnings('ignore')

"""
更新说明，咋
"""
def extract_corp_performance(info_path, start_year=2020):
    save_path = Path(__file__).parent.parent / 'data'
    corp_unfin_performance = pd.read_csv(save_path / "数字化单位及数字化程度(未筛选).csv", dtype={'股票代码': 'object'})
    stocks = set(corp_unfin_performance["股票代码"].tolist())
    # 创建一个绩效指标数据表
    corp_performance = pd.DataFrame(
        [{"股票代码": stock, "截止日期": year} for stock in stocks for year in range(start_year, 2024)])
    corp_performance.sort_values(["股票代码", "截止日期"], inplace=True)
    corp_performance.reset_index(drop=True, inplace=True)
    # 财务绩效指标合并：其中托宾q值/销售期间费用率为备用指标
    for path in os.scandir(info_path / '财务'):
        file_name = path.name
        if '.xlsx' in file_name:
            corp_fin_perf_indicators = _EvaluationIndicatorSystem[file_name.split('.')[0]]
            dataset = pd.read_excel(info_path / '财务' / file_name, skiprows=[0, 2], dtype={'股票代码': 'object'},
                                    parse_dates=['统计截止日期'])
            dataset = dataset[(dataset['公告来源'] == 0) & (dataset['报表类型编码'] == 'A') & (
                        dataset['统计截止日期'].dt.month == 12)]
            dataset['统计截止日期'] = dataset['统计截止日期'].dt.year.astype('int')
            dataset = dataset[(dataset['统计截止日期'] > start_year) & (dataset['股票代码'].isin(stocks))]
            dataset.rename(columns={"统计截止日期": '截止日期'}, inplace=True)
            dataset = dataset[['股票代码', '截止日期', *corp_fin_perf_indicators]]
            corp_performance = pd.merge(corp_performance, dataset, how='outer', on=['股票代码', '截止日期'])
    # 非财务绩效数据
    for path in os.scandir(info_path / '非财务'):
        file_name = path.name
        if '.xlsx' in file_name:
            corp_unfin_perf_indicators = _EvaluationIndicatorSystem[file_name.split('.')[0]]
            dataset = pd.read_excel(info_path / '非财务' / file_name, skiprows=[0, 2],
                                    dtype={'股票代码': 'object', '证券代码': 'object', '统计截止日期': 'datetime64[ns]',
                                           '截止日期': 'datetime64[ns]'})
            if "证券代码" in dataset.columns:
                dataset.rename(columns={"证券代码": "股票代码"}, inplace=True)
            if "统计截止日期" in dataset.columns:
                dataset.rename(columns={"统计截止日期": "截止日期"}, inplace=True)
            dataset = dataset[(dataset["截止日期"].dt.month == 12) & (dataset["截止日期"].dt.year > start_year) & (
                dataset["股票代码"].isin(stocks))]
            if "报表类型" in dataset.columns:
                if "应付职工薪酬" in file_name:
                    dataset = dataset[dataset["报表类型"] == "A"]
                else:
                    dataset = dataset[dataset["报表类型"] == 1]
            if "数据来源" in dataset.columns:
                dataset = dataset[dataset["数据来源"] == 0]
            if "统计口径" in dataset.columns:
                dataset = dataset[dataset["统计口径"] == 1]
            if "申请类型编码" in dataset.columns:
                dataset = dataset[(dataset["地区"] == 1) & (dataset["申请类型编码"] == 'S5201')]  # 6是累计授权、
            dataset["截止日期"] = dataset["截止日期"].dt.year.astype('int')
            dataset = dataset[['股票代码', '截止日期', *corp_unfin_perf_indicators]]
            corp_performance = pd.merge(corp_performance, dataset, how='outer', on=['股票代码', '截止日期'])
    corp_performance.rename(columns={
        "董事长与总经理兼任情况": '两职合一',
        "前十大股东持股比例(%)": "股权集中度(%)",
        "其中：独立董事人数": "独董人数",
        "息税折旧摊销前收入（EBITDA）": "EBITDA率",
        "资产报酬率B": "资产报酬率",
        "净资产收益率（ROE）B": "净资产收益率(ROE)",
        "投入资本回报率（ROIC）": "投入资本回报率(ROIC)",
        '托宾Q值B': '托宾Q值',
        "总资产周转率B": "总资产周转率",
        "应收账款周转率B": "应收账款周转率",
        "流动资产周转率B": "流动资产周转率",
        "存货周转率B": "存货周转率",
        "资本保值增值率B": "资本保值增值率",
        "总资产增长率B": "总资产增长率",
        "资本积累率B": "资本积累率",
        "营业利润增长率B": "营业利润增长率"
    }, inplace=True)
    corp_performance.sort_values(["股票代码", "截止日期"], inplace=True)
    corp_performance.reset_index(drop=True, inplace=True)
    corp_performance.to_csv(save_path / f'{start_year}-2023目标企业绩效指标数据(未处理).csv', index=False,
                            encoding='utf-8')


def extract_corp_mda():
    """
    提取mda文本
    这段代码来源于自己的windows主机，需要挨个提取txt中的管理层讨论与分析内容，运行时间大约需要31min，如果是远程pull代码的话，不要运行。
    如果是本地跑的话，将base_path路径处理好。
    :return:
    """
    # 将txt 文本载入到csv中
    pattern1 = re.compile(r'\W')
    pattern2 = re.compile(r'[^\u4E00-\u9FA5A-Za-z0-9]')

    keywords = KeywordBy['keywords']
    # 读取停用词
    with open(base_path / '政策停留词.txt', 'r', encoding='utf8') as f:
        eco_stops = f.read()
    with open(base_path / '中文停用词合并.txt', 'r', encoding='utf8') as f:
        chi_stops = f.read()
    content = eco_stops + chi_stops
    stopwords = {con for con in content.split('\n') if con != ''}

    dataset = []
    for dir_name in os.scandir(base_path / 'MDA管理层讨论与分析'):
        for file_name in os.scandir(dir_name.path):
            scid, date = file_name.path.split('\\')[-1].split('.')[0].split('_')
            year, month, day = date.split('-')
            if month == '06':
                continue
            with open(file_name.path, 'r', encoding='utf-8') as f:
                text = pattern1.sub(' ', f.read().lower())
            # 计算总词数
            clear_text = pattern2.sub(" ", text)
            text_lists = jieba.cut(clear_text, HMM=True)
            # 去除停留词
            text_lists = [item for item in text_lists if item not in stopwords]
            counts = {keyword: text.count(keyword.lower()) for keyword in keywords}
            counts["mda总词数"] = len(text_lists)
            dataset.append({'股票代码': scid, '截止日期': year, '月份': month, '讨论与分析内容': text, **counts})

    dataset = pd.DataFrame(dataset)
    dataset.drop(['月份'], axis=1, inplace=True)
    # 该文件大约3G左右，暂存在window系统上了。
    dataset.to_csv(base_path / 'MDA管理层讨论与分析.csv', index=False)


def extract_indicator():

    root_path = Path(__file__).parent.parent / 'data' / '所有指标'
    files  = [root_path/ file.name for file in os.scandir(root_path)]
    return files




def process_content(files):
    result = []
    filters = [
        "股票代码", "统计截止日期", "公告日期", "所属省份", "行业代码", "行业名称", "行业代码1", "行业名称1",
        '证券代码', '股票简称', '证券简称', '数据来源', '公告来源', '行业代码C', '行业名称C', '股票简称', '上市状态',
        '报表类型', '办公地址' '说明', '币种', '申请类型', '截止日期', '注册地址', "办公地址"
    ]
    for file in files:
        temp = list()
        try:
            with open(file, 'r', encoding='utf-8') as f:
                cons = f.readlines()
            for cont in cons:
                cont = cont.strip()
                if not cont:
                    continue
                inds = cont.split(' ')
                if len(inds) == 1:
                    continue
                ind = inds[1].strip(']').strip('[')
                if ind in filters:
                    continue
                temp.append(ind)
        except Exception as e:
            print(f"wrong happened {e},{file}")
        result.extend(temp)
    result = sorted(list(set(result)))
    print('、'.join(result))


def search_ind(files, keywords):
    # 给定指标，分词，将分词结果再所有文件中搜索，如果找到，返回文件名和相关指标，否则输出找不到
    result = []
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            cons = [cont.strip() for cont in f.readlines() if cont.strip()]
        for con in cons:
            for keyword in keywords:
                if keyword in con:
                    result.append({"source": filepath, "feature":con})
    if len(result) == 0:
        print("找不到相关指标")
    else:
        print("共搜索到{}个结果".format(len(result)))
        for item in result:
            print(f"来源:{item['source']}，name{item['feature']}")
        # print("搜索结果：{}".format(result))
    return result




if __name__ == '__main__':
    files = extract_indicator()
    info_path = Path(r'C://datas//企业绩效数据')
    # extract_corp_mda()
    # 将数字化数据提取存储在 base_path / '2010-2023企业数字化转型水平数据.csv'
    # extract_corp_performance(info_path, start_year=2014)
    keywords = ["物流效率", "物流","订单","订单交付周期","交付","周期","供应链"]
    search_ind(files, keywords)
    # 数字化投资比重-软件及信息技术投资-生产流程自动化比例-数字化管理系统覆盖率-高管中IT背景占比
    # 数字化战略表述
    # 专利




