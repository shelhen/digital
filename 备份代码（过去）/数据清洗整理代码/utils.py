import os
import numpy as np
import pandas as pd
pd.options.io.excel.xlsx.writer = 'xlsxwriter'


def make_ftranslate(tranlate_=r'.\datas\financial.txt'):
    # 返回财务指标集合，和转换字典
    finas = set()
    with open(tranlate_, "r", encoding='utf-8') as f:
        lines = f.readlines()
    translate = {line.split(':')[0]: line.split(':')[1].replace('\n', '') for line in lines}
    for k, v in translate.items():
        finas |= {k, v}
    return finas, translate


def get_findicators(path=r'C:\datas\企业绩效数据\制造企业绩效评价指标体系.xlsx'):
    # 返回财务指标的中文指标体系
    dataset = pd.read_excel(path)
    indicators = dataset['具体指标']
    indicators.dropna(inplace=True)
    indicators = list(indicators.to_list())
    indicators.extend(['股票代码', '股票简称', '统计截止日期', '报表类型编码', '公告来源', '行业代码', '行业名称'])
    return indicators


def get_files(path, type='txt'):
    type_1 = '.' + type
    type_2 = '[DES]'
    files = []
    files_ = []
    for dir_ in os.listdir(path):
        path_ = os.path.join(path, dir_)
        for entry in os.scandir(path_):
            if type_2 in str(entry.name):
                files_.append(os.path.join(path_, entry.name))
            elif (entry.name.endswith(type_1)) and '00.00.readme.txt' not in str(entry.name):
                files.append(os.path.join(path_, entry.name))
            else:
                continue
    return files, files_


def save_ftranslate(path=r'C:\datas\企业绩效数据\财务绩效数据', tranlate_=r'.\datas\financial.txt'):
    # 生成了一个财务绩效中英文指标体系的转化txt
    indicators = get_findicators()
    if os.path.exists(tranlate_):
        os.remove(tranlate_)
    files = get_files(path)[1]
    for file_ in files:
        with open(file_, "r", encoding='utf-8') as f:
            content = f.readlines()
        for line in content:
            res = line.split('-')[0].rstrip(' ').split(' ')
            if len(res) > 1:
                if res[1].replace('[', '').replace(']', '') in indicators:
                    with open(tranlate_, 'a', encoding='utf8') as f:
                        f.write(f"{res[0]}:{res[1].replace('[', '').replace(']', '')}\n")
    with open(tranlate_, 'a', encoding='utf8') as f:
        f.write(f"上市日期:上市日期\n上市省份:上市省份")


def get_unfindicators():
    unfi_tranlate = {
        '股票代码': 'Symbol',
        '统计截止日期': 'EndDate',
        '主营业务收入': 'Fn04804',
        '纳税总额A': 'TotalTax',
        '捐赠总额': 'DonationAmount',
        '研发人员数量': 'RDPerson',
        '研发人员数量占比(%)': 'RDPersonRatio',
        '研发投入金额': 'RDSpendSum',
        '研发营收比(%)': 'RDSpendSumRatio',
        '董事人数': 'DirectorNumber',
        '独立董事人数': 'IndependentDirectorNumber',
        '业务类型': 'FN_Fn01004',
        '销售(采购)额': 'FN_Fn01005',
        '销售(采购)额占年度业务总额比例(%)': 'FN_Fn01006',
        '销售额': '销售额',
        '采购额': '采购额',
        '本期销售额占年度销售总额比例(%)': '本期销售额占年度销售总额比例(%)',
        '本期采购额占年度采购总额比例(%)': '本期采购额占年度采购总额比例(%)',
        '财务波动': '财务波动',
        '长期绩效增长': '长期绩效增长',
        '组织韧性': '组织韧性',
        '内部控制指数': '内部控制指数',
        '股权性质': 'S0702b',
        '董事会规模': 'Y0401b',
        '员工人数': 'Y0601b',
        '两职合一': 'Y1001b',
        'stkcd': 'stkcd',
        'accper': 'accper',
        'Accper':'Accper',
        '两权分离度(%)': 'Seperation',
        "股权集中度4(%)": 'Shrcr4',
        "职工薪酬":'C001020000',
        '授权专利数':'Patents',
        '授权专利数2':'Patents2',
        '上市日期':'上市日期',
        '上市省份':'上市省份'
    }
    _unfi_tranlate = dict()
    indicators = set()
    for k, v in unfi_tranlate.items():
        _unfi_tranlate[v] = k
        indicators |= {k, v}
    return indicators, _unfi_tranlate







