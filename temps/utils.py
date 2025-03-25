# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: utils.py
@Project: digital 
@Time: 2025/03/14  13:14
@Author: xieheng
@Email: xieheng@163.com
@Software: PyCharm
--------------------------------------------------------
@Brief: 插入一段描述。
"""
import os
import fitz  # PyMuPDF


def text_clean(text, subs:list=None):
    if subs is None:
        subs = ['\n', "", ""]
    else:
        subs.extend(['\n', "", ""])
    for cl in subs:
        text = text.replace(cl, "")
    return text.replace(',', "，").replace(";", "；")


def clean_by_list(_text: str):
    texts = _text.split("\n")
    return '\n'.join([text_clean(_te) for _te in texts if text_clean(_te)])

def get_all_font(font_name=None):
    import matplotlib.font_manager as fm

    # 获取所有字体名称
    font_list = sorted([f.name for f in fm.fontManager.ttflist])
    print(font_list)
    # 获取所有的字体路径
    font_paths = {f.name: f.fname for f in fm.fontManager.ttflist}
    if font_name:
        print(font_paths[font_name])



def pdf2text(pdf_path: str, page_range=None, save_file=None):

    # 参数检查
    if not os.path.exists(pdf_path):
        print("所给pdf路径不存在文件，请检查")
        return
    # if save_file is None:
    #     save_file = pdf_path.split('/')
    result = []
    if page_range is None:
        with fitz.open(pdf_path) as pdf_document:
            for page in pdf_document:
                result.append(text_clean(page.get_text()))
    else:
        start, end = page_range
        with fitz.open(pdf_path) as pdf_document:
            for i in range(start, end+1):
                try:
                    page = pdf_document[i]
                    result.append(text_clean(page.get_text()))
                except IndexError:
                    print("超出提取索引")
                    break
                except Exception as e:
                    print("未知错误：{}".format(e))
                    continue
    if save_file is None:
        save_file = f"{pdf_path.split('.', -1)[0]}"+'.txt'
    dirname = os.path.dirname(os.path.abspath(save_file))
    saved = {file.name.strip() for file in os.scandir(dirname) if file}
    if save_file in saved:
        name, _t = save_file.split('.',-1)
        new_path = f"{name}[新].{_t}"
    else:
        new_path = save_file
    with open(new_path, 'a', encoding='utf8') as f:
        f.write('\n'.join(result))


if __name__ == '__main__':
    get_all_font(font_name='Heiti TC')
