import pymysql
import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
# pip install pymysql
# pip install gensim

LocationWeight =  {
    '上海':0.035,
    '澳门':0.000,
    '香港':0.000,
    '云南':0.0122,
    '内蒙古':0.0061,
    '北京':0.0425,
    '四川':0.0108,
    '安徽':0.1033,
    '山东':0.1215,
    '山西': 0.0061,
    '广东':0.0243,
    '新疆':0.004,
    '江苏':0.3376,
    '江西':0.0101,
    '河北':0.0423,
    '河南':0.1123,
    '浙江':0.0305,
    '海南':0.0040,
    '湖北':0.002,
    '湖南':0.0040,
    '福建':0.0140,
    '贵州':0.0061,
    '辽宁':0.0203,
    '重庆':0.011,
    '陕西':0.023,
    '黑龙江':0.006,
    '天津':0.002,
    '青海':0.001,
    '吉林':0.001,
    '广西':0.002,
    '甘肃':0.002,
    '宁夏':0.001,
    '台湾':0.000,
    '西藏':0.002
}
Translates={'上海': '上海市',
            '云南': '云南省',
            '内蒙古': '内蒙古自治区',
            '北京': '北京市',
            '吉林': '吉林省',
            '四川': '四川省',
            '天津': '天津市',
            '宁夏': '宁夏回族自治区',
            '安徽': '安徽省',
            '山东': '山东省',
            '山西': '山西省',
            '广东': '广东省',
            '广西': '广西壮族自治区',
            '新疆': '新疆维吾尔自治区',
            '江苏': '江苏省',
            '江西': '江西省',
            '河北': '河北省',
            '河南': '河南省',
            '浙江': '浙江省',
            '海南': '海南省',
            '湖北': '湖北省',
            '湖南': '湖南省',
            '甘肃': '甘肃省',
            '福建': '福建省',
            '贵州': '贵州省',
            '辽宁': '辽宁省',
            '重庆': '重庆市',
            '陕西': '陕西省',
            '黑龙江': '黑龙江省',
            '青海': '青海省',
            '西藏': '西藏自治区',
            '台湾': '台湾省',
            '香港': '香港特别行政区',
            '澳门': '澳门特别行政区'
            }


# 省份处理
def location_generate(location):
    keys = np.array(list(LocationWeight.keys()))
    values = np.array(list(LocationWeight.values()))
    return np.random.choice(a= keys,p =values)  if location not in keys else location


def condence(conment):
    """压缩去重函数"""
    for i in [1,2]:
        j=0
        while j<len(conment)-2*i:
            # 至少重复了两次
            if conment[j:j+i]==conment[j+i:j+2*i] and conment[j:j+i]==conment[j+2*i:j+3*i]:
                k=j+2*i
                while k+i<len(conment) and conment[j:j+i]==conment[k+i:k+2*i]:
                    k+=i
                conment=conment[:j+i]+conment[k+i:]
            j+=1
        i+=1
    for i in [3,4,5]:
        j=0
        while j<len(conment)-2*i:
            # 至少重复了一次
            if conment[j:j+i]==conment[j+i:j+2*i]:
                k=j+i
                while k+i<len(conment) and conment[j:j+i]==conment[k+i:k+2*i]:
                    k+=i
                conment=conment[:j+i]+conment[k+i:]
            j+=1
        i+=1
    return conment


def format_data(data, set_key_list):
    '''格式化需要计算的数据，将原始数据格式转换成二维数组'''
    formated_data = []
    # ech_line = ''
    for ech_line in data.tolist():
        temp = []  # 筛选出format_data中属于关键词集合的词
        for e in ech_line:
            if e in set_key_list:
                temp.append(e)
        ech_line = temp
        ech_line = list(set(filter(lambda x: x != '', ech_line)))  # set去掉重复数据
        formated_data.append(ech_line)
    return formated_data


def count_matrix(matrix, formated_data):
    for row in range(1, len(matrix)):
        # 遍历矩阵第一行，跳过下标为0的元素
        for col in range(1, len(matrix)):
            # 遍历矩阵第一列，跳过下标为0的元素
            # 实际上就是为了跳过matrix中下标为[0][0]的元素，因为[0][0]为空，不为关键词
            if matrix[0][row] == matrix[col][0]:
                # 如果取出的行关键词和取出的列关键词相同，则其对应的共现次数为0，即矩阵对角线为0
                matrix[col][row] = str(0)
            else:
                counter = 0  # 初始化计数器
                for ech in formated_data:
                    # 遍历格式化后的原始数据，让取出的行关键词和取出的列关键词进行组合，
                    # 再放到每条原始数据中查询
                    if matrix[0][row] in ech and matrix[col][0] in ech:
                        counter += 1
                    else:
                        continue
                matrix[col][row] = str(counter)
    return matrix


def prepare_data():
    pwd = 'Password123.'
    database='xuzhou_remarks'
    table_namse = ['xiecheng', 'mafeng', 'qiongyou', 'qunaer', 'tongcheng', 'dazhong']
    mysql = pymysql.connect(host='127.0.0.1', user='root', password=pwd, port=3306, charset='utf8mb4', database=database)
    dataset = pd.DataFrame()
    # 导入数据
    for table in table_namse:
        mysql_str = f'select * from {table}'
        data = pd.read_sql(mysql_str, con=mysql, index_col='id', parse_dates=['datetime'])
        dataset = pd.concat([dataset, data])
    # 去除整行完全为空的数据
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.drop(dataset[(dataset.score>5) | (dataset.score<0) ].index)
    dataset['datetime'] = dataset['datetime'].astype('datetime64[s]')
    dataset = dataset[dataset['datetime'].dt.year>2013]
    dataset = dataset[~((dataset['datetime'].dt.year==2014) & (dataset['datetime'].dt.month < 2))]
    dataset[dataset['score']==0] = 1
    dataset['location'] = dataset["location"].apply(lambda x: location_generate(x))
    dataset['location'] = dataset['location'].apply(lambda x: Translates[x])
    dataset.dropna(subset=['location'], inplace=True)
    dataset = dataset[~(dataset['datetime']==1)]
    dataset.to_csv('./datas/datas.csv', index=None)
    # return dataset


def analysis(content: pd.Series, t=50):
    dictionary = corpora.Dictionary(content)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(i) for i in content]
    perplexity_scores = []
    coherence_scores = []
    for topic in range(1, t+1):
        ldamodel = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            chunksize=2000,  # 控制每次处理的文档数，增大数值将提高速度，但是过大将影响模型的质量
            alpha='auto',
            eta='auto',
            iterations=400,
            num_topics=topic,
            passes=20,  # 控制在整个语料库上训练模型的频率，
            eval_every=None# 不建议评估模型的困惑，太过浪费时间。
        )
        perplexity_scores.append(ldamodel.log_perplexity(corpus))
        coherence_scores.append(CoherenceModel(model=ldamodel, corpus=corpus, dictionary=dictionary, coherence='u_mass').get_coherence())
    return pd.Series(perplexity_scores), pd.Series(coherence_scores)


def boson_predict(comment:pd.Series):
    # 基于情感词典
    # BosonNLP情感词分类积极消极文本，进行评论文本情感倾向分析
    # 读取情感字典文件
    feelingscore = pd.DataFrame()
    feelingscore['comment'] = comment
    feelingscore['length'] = feelingscore['comment'].apply(lambda x: len(x))
    wordtemp = [word for corpus in feelingscore['comment'] for word in corpus]
    BosonNLP = pd.read_csv("./datas/BosonNLP.txt",sep = " ",encoding = "utf-8",header =None)
    BosonNLP.columns = ["word", "score"]  # 重新命名列名
    # 导入否定词词库，并新增1列为'value'，取值全部为-1
    notdata = pd.read_csv("./datas/not.csv", encoding='utf-8')
    notdata['value'] = -1
    # 将wordtemp，否定词表，词语情感分值表合并
    notdata = pd.merge(pd.DataFrame(wordtemp, columns=['content']), notdata, how='left', left_on='content',
                       right_on='term')
    BosonNLP = pd.merge(notdata, BosonNLP, how= 'left', left_on = 'content', right_on='word')
    BosonNLP.drop(['term','word'], axis=1, inplace=True)
    # 情感得分会受词汇长度一影响
    # 给各词语指定归属标号，即词语所属记录的序号
    index = []
    for i in range(feelingscore['length'].shape[0]):
        index.extend([i]*feelingscore['length'][i])
    #将ID作为df的index
    BosonNLP.index = index
    #获取不含有否定词的句子的ID
    noneNotId = list(BosonNLP.loc[BosonNLP['value'].isnull(),:].index)
    # 将没有否定词的句子的所有词的情感分值直接相加
    for i in noneNotId:
        feelingscore.loc[i,'Score'] = BosonNLP.loc[i,'score'].sum()
    # 取出含有否定词的句子ID
    NotID = list(set(BosonNLP.loc[BosonNLP['value'].notnull(),:].index))
    # 以value为基础合并，若df['value']的数据缺失,则用df['score']的数据值填充
    BosonNLP = BosonNLP.reset_index()
    BosonNLP['score'] = BosonNLP['value'].combine_first(BosonNLP['score'])
    BosonNLP['score'] = BosonNLP['score'].fillna(0)
    BosonNLP.set_index('index', drop=True)
    # 本算法不考虑多重否定
    for i in NotID:
        score = 0
        Ser = BosonNLP.loc[i,'score']  # 取出其中一个有否定词句子的index和score
        try:
            lenNot = Ser.shape[0]     # 获取句子包含的总共词语数目
        except:
            lenNot = 1
            continue
        cirlist = [k for k in range(lenNot)]
        cirlist.reverse()
        # 从后往前计算一个句子的情感分值，防止最前面出现否定词无效
        for j in cirlist:
            # 若句子的最后一个词为否定词，Score初始化为-1
            if  Ser.iloc[j] == -1:
                if j == (lenNot-1):
                    score = -1
                elif (j ==0) & (Ser.iloc[j+1]!=-1):
                    # 在遇见-1时，前项已经加过，因而应将前项减去再加上(-1*得分)
                    score = score- Ser.iloc[j] + (-1* Ser.iloc[j])
                else:
                    if (Ser.iloc[j+1]!=-1) & (Ser.iloc[j-1]!=-1):
                        score = score- Ser.iloc[j] + (-1* Ser.iloc[j])
                        # 在双重否定的情况下，不需要做操作
            else:
                score += Ser.iloc[j]
        feelingscore.loc[i,'Score'] = score
    feelingscore.fillna(0, inplace=True)
    return feelingscore['Score']/feelingscore['length']


def Df_to_sql(df, name, db, psw, user='root', ip='localhost'):
    '''
    :param df: The data to pass
    :param name: The name of the table in db
    :param psw: Your password of your database
    :param ip: Your IP
    :param db: The name of your database
    :param user: root
    :return: None
    '''
    pass
    # from sqlalchemy import create_engine
    # con = create_engine('mysql+pymysql://{}:{}@{}/{}'.format(user, psw, ip, db))  # mysql+pymysql的意思为：指定引擎为pymysql
    # df.to_sql(name, con,if_exists='replace')


def word_count(sentences, word):
    sum = 0
    for sentence in sentences:
        sum += sentence.count(word)
    return sum


if __name__ == '__main__':
    prepare_data()
    # dataset = pd.read_csv('./result/feelingscore.csv', sep='/')
    # dataset['content'] = dataset['content'].apply(lambda x: eval(x))
    # posegment = dataset.loc[dataset['feelingscore'] > 0, 'content']
    # negegment = dataset.loc[dataset['feelingscore'] < 0, 'content']
    # segment = {'pos':posegment,'neg': negegment,'all':dataset['content']}
    # data = pd.DataFrame()
    # for key,value in segment.items():
    #     data[f"{key}perplexity_score"], data[f"{key}coherence_score"] = analysis(value)
    # data.to_csv('./result/perplexity_coherence_scores.csv', index=False)


