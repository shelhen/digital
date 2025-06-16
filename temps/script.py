# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: script.py
@Project: digital 
@Time: 2025/03/14  01:59
@Author: xieheng
@Email: xieheng@163.com
@Software: PyCharm
--------------------------------------------------------
@Brief: 插入一段描述。
"""
from temps.utils import pdf2text, text_clean


def extra_text_from_paths(filepaths: list):
    """从所给路径提取文本txt"""
    # for filepath in filepaths:
    #     extrac_text_from_pdf(filepath)


    pass



if __name__ == '__main__':

    pdf2text("杨和高 - 2024 - 企业数字化转型：概念内涵、统计测度技术路线和改进思路.pdf")
    text = """

    """.strip().replace('￼', '')
    result = text_clean(text).replace('*', '').replace('	', '').replace(' ', '')
    # result = clean_by_list(text)
    print(result)






