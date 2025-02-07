import pandas as pd
import numpy as np


def normalize(matrix:pd.DataFrame, method='z'):
    """
    :param matrix:
    :param method: z/m/other
    :return:
    """
    directions=[
        ['销售期间费用率','总营业成本率','应付账款周转率B','存货周转率B','资产负债率','负债与权益市价比率',
         '销售费用增长率','管理费用增长率','营业总成本增长率'],
        ['企业年限', '现金比率', '速动比率','产权比率', '权益乘数', '经营活动产生的现金流量净额／负债合计', '流动比率', '总资产增长率B', '资本保值增值率B', '营业总收入增长率', '固定资产增长率B', '资本积累率B',
         '现金资产比率', '营业收入现金净含量', '资本支出与折旧摊销比', '营运指数', '全部现金回收率', '净利润现金净含量', '投入资本回报率（ROIC）', '成本费用利润率', '营业利润率', '总资产净利润率(ROA)C',
         '营业净利率', '应收账款周转率B', '财务杠杆', '经营杠杆', '董事会规模', '两职合一', '股权性质', '两权分离度(%)', '股权集中度4(%)', '创新投入', '内部控制指数', '财务波动', '长期绩效增长',
         '组织韧性', '独董比例', '供应链集中能力', '提供岗位增长率', '员工收入增长率', '研发强度', '创新产出', '行业类别'],
        ['股票代码', '截止日期', '股票简称', '上市省份', 'digital']
    ]
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
        for name, item in metrix.items():
            max_ = item.max()
            min_ = item.min()
            if name in directions[0]:
                for index, row in item.items():
                    metrix.loc[index, name] = (max_ - metrix.loc[index, name]) / (max_ - min_)
            elif name in directions[1]:
                for index, row in item.items():
                    metrix.loc[index, name] = (metrix.loc[index, name] - min_) / (max_ - min_)
        return metrix


def calculate(table):
    # 计算准确率与灵敏度
    table = np.array(table)
    Accuracy = (table[0, 0] + table[1, 0]) / np.sum(table)
    Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
    print(f"准确率Accuracy:{Accuracy}，灵敏度Sensitivity:{Sensitivity}。")
    return Accuracy,Sensitivity