# -*- encoding: utf-8 -*-
"""
--------------------------------------------------------
@File: open_ai接口.py
@Project: digital_evalutate 
@Time: 2024/09/10  05:06
@Author: shelhen
@Email: shelhen@163.com
@Software: PyCharm
--------------------------------------------------------
# @Brief: openai python接口类。
api开发文档：https://platform.openai.com/docs/guides/speech-to-text/quickstart

"""
from openai import OpenAI


def init_prompt():
    # prompt
    prompt = """
        Task: Please try to answer my question as concisely as possible.
        Input: {text}
        Output: All answers are in simplified Chinese,
        """
    return prompt


def text_generate(sentence):
    client = OpenAI(api_key=api_key)
    prompt = init_prompt()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a university professor in management science and engineering. Please help me with my graduate thesis writing. "},
            {
                "role": "user",
                "content": prompt.format(text=sentence)
            }
        ]
    )
    return completion.choices[0].message

if __name__ == '__main__':
    # 获取的API密钥
    api_key = "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"
    background = "With the continuous popularization and widespread application of digital technology, digital transformation has become an important way for manufacturing companies to achieve the conversion of old and new driving forces.\nDigital transformation has changed the entire picture of enterprise operations, from internal management to external interactions, and has had a profound impact on every link in the enterprise value chain. This has also raised new requirements and challenges for the evaluation of digital enterprise performance.\nUnder the wave of rapidly developing digital economy, use performance evaluation tools to accurately evaluate the performance of digital enterprises from a micro level, grasp the development status of digital enterprises, identify problems and obstacles in the process of digital transformation, and clarify the goals and paths of transformation. It is of great significance for digital technology to empower the high-quality development of the manufacturing industry.\n"
    curreent="The development of enterprise performance evaluation research can be roughly divided into three stages. Traditional corporate performance evaluation is mostly based on the requirements of accounting standards, using ratio methods such as DuPont analysis to calculate different financial performance indicators, and determines the strength of the company's financial capabilities through comparison. With the continuous development of evaluation practice activities, some statistical operations models have been incorporated into the enterprise performance evaluation system. These methods have significantly improved the persuasiveness of evaluation and are a useful supplement to the traditional ratio evaluation method. However, some single evaluation methods are too subjective, and some completely ignore the actual significance of the indicators. Therefore, some scholars improve the evaluation effect through the integration or improvement of methods. These integrated methods can complement each other's advantages to a certain extent and improve the accuracy of evaluation, but they still have their own shortcomings.\n"
    sulution = "In view of this, I plan to use China’s A-share digital manufacturing companies from 2013 to 2022 as the research object, from the aspects of profitability, debt solvency, operating capabilities, development capabilities, risk capabilities, innovation capabilities, social contribution, corporate strategy, supply chain capabilities and A comprehensive evaluation index system for digital enterprise performance is constructed at ten levels of the ownership structure. The expected performance of the enterprise is calculated based on the factor analysis method. The enterprise performance evaluation model is constructed using Adam's optimized neural network algorithm to evaluate and predict the performance of the digital enterprise. Finally, based on the importance of features The results are explained from a sexual perspective in order to provide theoretical guidance for the formulation of digital transformation strategies for manufacturing companies.\n"

    decs ="Based on the above research background, research status and research ideas, help me plan the thesis title (possible) and research framework of a small paper. The planning results will clarify the framework in the form of first- and second-level paragraph titles, and in the form of an abstract. Briefly describe what each paragraph should contain, and highlight the areas that need attention."
    sentence = background + curreent+ sulution+decs
    result = text_generate(sentence)
    print(result)









