# -*- encoding: utf-8 -*-



def text_clean(text, subs:list=None):
    if subs is None:
        subs = ['\n', "", "", " ", " "]
    else:
        subs.extend(['\n', "", "", " ", " "])
    for cl in subs:
        text = text.replace(cl, "")
    return text.replace(',', "，").replace(";", "；")


def clean_by_list(_text: str):
    texts = _text.split("\n")
    return '\n'.join([text_clean(_te) for _te in texts if text_clean(_te)])

if __name__ == '__main__':
    text = """
根据第一章对制造业数字化转型的背景以及第二章中已有的集中数字化转型评价体系的构建维度以及国内外学者的研究的总结归纳，本研究将从战略规划、技术基础、数字化创新、组织变革、行业影响力这五个维度构建制造业数字化转型评价体系
    """.strip()
    result = text_clean(text)
    # result = clean_by_list(text)
    print(result)