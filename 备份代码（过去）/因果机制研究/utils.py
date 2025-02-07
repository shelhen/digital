import pandas as pd
import numpy as np


financial_dict={
    '总资产净利润率(ROA)B':'F050202B',
    '净资产收益率（ROE）B':'F050502B',
    '投入资本回报率（ROIC）':'F051201B',
    '研发费用率':'F053401B',
    '成本费用利润率':'F052101B',
    '销售期间费用率':'F052001B',
    '托宾Q值B':'F100902A',
    '资本积累率B':'F080302A',
    '资产负债率':'F011201A',
    '应收账款周转率B':'F040202B',
    '应付账款周转率B':'F040802B',
    '存货周转率B':'F040502B',
    '固定资产周转率B':'F041402B',
    '总资产周转率B':'F041702B',
    '资本密集度':'F041601B',
    '股票代码':'Stkcd',
    '截止日期':'Accper',
}

trans_dict={
    'id_str':'股票代码', 'year':'截止日期', 'LISTINGSTATE_11':'上市状态', 'firm_age':'企业年限', 'Sicda_str':'行业代码', 'EquityNature_p':'股权性质', 'S0702b_p':'实际控制人性质 ', 'prov_off':'上市省份', 'city_off':'上市城市', 'B001100000':'营业总收入', 'lnSale':"ln(营业收入)", 'size':'ln(总资产)', 'roa':'总资产净利润率', 'roe':'净资产收益率', 'IsSTx':'是否*ST', 'IsSTPT':'是否ST或PT', 'tobin':'托宾Q值', 'RDPerson':	'研发人员数量', 'RDPersonRatio':'研发人员数量占比(%)', 'RDSpendSum':'研发投入金额', 'RDSpendSumRatio':'研发投入占营业收入比例(%)','Ispatent': '是否申请专利', 'IsInvention': '是否申请发明专利', 'IsUtilityModel': '是否申请实用新型专利', 'IsDesign': '是否申请外观设计专利', 'Patents_sum': '年度申请专利总数', 'Invention_sum': '申请发明专利总数', 'UtilityModel_sum': '申请实用新型专利总数', 'Design_sum': '申请外观设计专利总数', 'Patents1': '专利数量-国内已申请', 'Invention1': '发明专利数量-国内已申请', 'UtilityModel1': '实用新型数量-国内已申请', 'Design1': '外观设计数量-国内已申请', 'Patents2': '专利数量-国内已获得', 'Invention2': '发明专利数量-国内已获得', 'UtilityModel2': '实用新型数量-国内已获得', 'Design2': '外观设计数量-国内已获得', 'Patents3': '专利数量-国内已授权', 'Invention3': '发明专利数量-国内已授权', 'UtilityModel3': '实用新型数量-国内已授权', 'Design3': '外观设计数量-国内已授权', 'Patents4': '专利数量-国内截至报告期末累计获得', 'Invention4': '发明专利数量-国内截至报告期末累计获得', 'UtilityModel4': '实用新型数量-国内截至报告期末累计获得', 'Design4': '外观设计数量-国内截至报告期末累计获得', 'Patents5': '专利数量-国内截止报告期末累计已被受理', 'Invention5': '发明专利数量-国内截止报告期末累计已被受理', 'UtilityModel5': '实用新型数量-国内截止报告期末累计已被受理', 'Design5': '外观设计数量-国内截止报告期末累计已被受理', 'Patents6': '专利数量-国内截止报告期末累计已授权', 'Invention6': '发明专利数量-国内截止报告期末累计已授权', 'UtilityModel6': '实用新型数量-国内截止报告期末累计已授权', 'Design6': '外观设计数量-国内截止报告期末累计已授权', 'Patents7': '专利数量-国外已申请', 'Invention7': '发明专利数量-国外已申请', 'UtilityModel7': '实用新型数量-国外已申请', 'Design7': '外观设计数量-国外已申请', 'Patents8': '专利数量-国外已获得', 'Invention8': '发明专利数量-国外已获得', 'UtilityModel8': '实用新型数量-国外已获得', 'Design8': '外观设计数量-国外已获得', 'Patents9': '专利数量-国外已授权', 'Invention9': '发明专利数量-国外已授权', 'UtilityModel9': '实用新型数量-国外已授权', 'Design9': '外观设计数量-国外已授权', 'Patents10': '专利数量-国外截至报告期末累计获得', 'Invention10': '发明专利数量-国外截至报告期末累计获得', 'UtilityModel10': '实用新型数量-国外截至报告期末累计获得', 'Design10': '外观设计数量-国外截至报告期末累计获得', 'Patents11': '专利数量-国外截止报告期末累计已被受理', 'Invention11': '发明专利数量-国外截止报告期末累计已被受理', 'UtilityModel11': '实用新型数量-国外截止报告期末累计已被受理', 'Design11': '外观设计数量-国外截止报告期末累计已被受理', 'Patents12': '专利数量-国外截止报告期末累计已授权', 'Invention12': '发明专利数量-国外截止报告期末累计已授权', 'UtilityModel12': '实用新型数量-国外截止报告期末累计已授权', 'Design12': '外观设计数量-国外截止报告期末累计已授权', 'green_patent1': '绿色专利独立申请', 'green_patent2': '绿色发明专利独立申请', 'green_patent3': '绿色实用新型专利独立申请', 'green_patent4': '绿色专利联合申请', 'green_patent5': '绿色发明专利联合申请', 'green_patent6': '绿色实用新型专利联合申请'
}


def fillna_(dataset, rate):
    # 缺失值填充
    amount = dataset.shape[0]
    names = dataset.columns.tolist()
    for name in names:
        if (dataset[name].dtype=='float64') and (dataset[name].isnull().sum()>0) and (dataset[name].isnull().sum()<amount*rate):
            dataset[name].fillna(dataset[name].mean(), inplace=True)


def fillna__(dataset, code, rate=0.5):
    # 按照code指标将dataset分为n类，分别将各类拆分为n个dataset
    codes = list(set(dataset[code].values.tolist()))
    dataset_ = dataset[dataset[code]==codes[0]]
    fillna_(dataset_, rate)
    for i in range(1, len(codes)):
        new_dataset = dataset[dataset[code]==codes[i]]
        fillna_(new_dataset, rate)
        dataset_ = pd.concat([dataset_, new_dataset])
    return dataset_


def normalize(matrix:pd.DataFrame, neg_vec, pass_vec, method='pass'):
    """
    :param matrix:
    :param method: z/m/other
    :param neg_vec:负向指标
    :param pass_vec:无需标准化的指标
    :return:
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if method=='z':
        # z_score标准化
        zscore = StandardScaler()
        return zscore.fit_transform(matrix)
    elif method=='m':
        # 不考虑指标方向的极差标准化
        minmax = MinMaxScaler()
        return minmax.fit_transform(matrix)
    else:
        metrix = matrix.copy()
        for name, item in matrix.items():
            max_ = item.max()
            min_ = item.min()
            if name in neg_vec:
                for index, row in item.items():
                    metrix.loc[index, name] = (max_ - matrix.loc[index, name]) / (max_ - min_)
            elif name in pass_vec:
                metrix[name] = matrix[name]
            else:
                for index, row in item.items():
                    metrix.loc[index, name] = (matrix.loc[index, name] - min_) / (max_ - min_)
        return metrix


def entropy(matrix):
    """熵权法求取权重"""
    names = matrix.columns
    project, indicator = matrix.shape
    p = pd.DataFrame(np.zeros([project, indicator], dtype='float64'), columns=names)
    for i in range(indicator):
        p.iloc[:, i] = matrix.iloc[:, i] / matrix.iloc[:, i].sum()
    E = -1 / np.log(project) * (p * np.log(p)).sum()
    W = (1 - E) / sum((1 - E))
    return W


if __name__ == '__main__':
    import pandas as pd
    # 此代码运行时间较长，已经暂存结果,需要的数据直接放在 trans_dict 中即可
    supply = pd.read_excel(r'C:\datas\企业信息数据\上市公司数据.xlsx',dtype={'id_str':'object'})
    supply_infomation = supply[trans_dict.keys()]
    supply_infomation.dropna(subset=['id_str','Sicda_str'],inplace=True)
    supply_infomation['year'] = supply_infomation['year'].apply(lambda x:int(x))
    supply_infomation['Sicda_str'] = supply_infomation['Sicda_str'].apply(lambda x: int(x[1:]))
    supply_infomation['EquityNature_p'] = supply_infomation['EquityNature_p'].apply(lambda x:1 if x=='国企' else 0)
    supply_infomation=supply_infomation[(supply_infomation['year']>2010)&(supply_infomation['firm_age']>=0)&(supply_infomation['Sicda_str']>12)&(supply_infomation['Sicda_str']<44)]
    supply_infomation['LISTINGSTATE_11'] =supply_infomation['LISTINGSTATE_11'].fillna('*ST')
    supply_infomation['firm_age'] = supply_infomation['firm_age'].apply(lambda x:int(x))
    supply_infomation.loc[supply['id_str']=='002532','prov_off']='浙江省'
    supply_infomation.loc[supply['id_str']=='002532','city_off']='台州市'
    supply_infomation.loc[supply['id_str']=='600197','city_off']='伊犁哈萨克自治州'
    supply_infomation.loc[supply['id_str']=='600075','city_off']='伊犁哈萨克自治州'
    supply_infomation.loc[supply['id_str']=='600531','city_off']='焦作市'
    supply_infomation.rename(columns=trans_dict,inplace=True)
    supply_infomation.sort_values(by=['股票代码','截止日期'],inplace=True)
    supply_infomation.to_csv('./data/supply_infomation.csv', index=False)

