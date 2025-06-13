import json
import logging
import spacy
from word2number import w2n
import re
import numpy as np
from openai import OpenAI
from tqdm import tqdm 
from dotenv import load_dotenv
import os

# 配置日志记录
logging.basicConfig(level=logging.INFO,  # 设置日志级别
                    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
                    handlers=[logging.StreamHandler()])  # 输出到控制台

load_dotenv(".env")
# 加载 spaCy 模型
nlp = spacy.load("./model/en_core_web_trf")
API_KEY = os.getenv("NUM_API_KEY")
BASE_URL= os.getenv("NUM_BASE_URL")
MODEL_NAME= os.getenv("NUM_MODEL_NAME")
def parse_json_safe(result):
    """
    尝试将字符串解析为 JSON，如果失败则抛出异常。
    :param result: 待解析的字符串
    :return: 解析后的 JSON 对象
    :raises ValueError: 如果解析失败，抛出解析失败异常
    """
    try:
        result = json.loads(result)
        return result
    except json.JSONDecodeError as e:
        # 提供详细的错误信息，包括错误位置和上下文
        error_position = e.pos
        error_linenum = e.lineno
        error_colnum = e.colno
        context_start = max(0, error_position - 20)
        context_end = min(len(result), error_position + 20)
        error_context = result[context_start:context_end]
        error_msg = (
            f"解析结果失败，返回内容不是有效的 JSON 格式：\n"
            f"错误位置：行 {error_linenum}, 列 {error_colnum}\n"
            f"错误上下文：{error_context}\n"
            f"错误信息：{e}"
        )
        raise ValueError(error_msg)

# 定义主要数字词和连接词
main_number_words = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six',
    'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
    'eighteen', 'nineteen', 'twenty', 'thirty', 'forty',
    'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
    'hundred', 'thousand', 'million', 'billion', 'trillion'
]

connector_words = ['point', 'negative']

def extract_numbers(text):
    """
    从英文文本中提取所有数字，包括：
    - 阿拉伯数字（整数和小数）
    - 带有特殊符号的数字（如 "120/70"）
    - 英文文字表达的数字（如 "seventy-two"）
    - 负数和百分比
    - 英文文字表达的小数（如 "thirty point five"）
    
    为每个数字分配一个唯一的标签（如 [NUM0], [NUM1], ...）并按出现顺序排序。
    
    参数:
        text (str): 输入的文本。
    
    返回:
        list: 包含元组的列表，每个元组格式为 (标签, 起始位置, 结束位置, 数值)。
    """
    potential_matches = []

    # 1. 处理带有特殊符号的数字，如 "120/70" 或 "1,200/70"
    special_pattern = re.compile(
        r'(-?\d{1,}(?:,\d{3})*(?:\.\d+)?)[\s*/\-]+(-?\d{1,}(?:,\d{3})*(?:\.\d+)?)'
    )
    for match in special_pattern.finditer(text):
        num1, num2 = match.groups()
        start1, end1 = match.span(1)
        start2, end2 = match.span(2)

        # 第一个数字
        num_clean1 = num1.replace(',', '')
        value1 = float(num_clean1) if '.' in num_clean1 else int(num_clean1)
        potential_matches.append((start1, end1, value1))

        # 第二个数字
        num_clean2 = num2.replace(',', '')
        value2 = float(num_clean2) if '.' in num_clean2 else int(num_clean2)
        potential_matches.append((start2, end2, value2))

    # 2. 处理紧跟单位的阿拉伯数字（整数和小数）
    number_pattern = re.compile(
        r'(?<![\w.])(-?\d{1,}(?:,\d{3})*(?:\.\d+)?)(?=\s*[a-zA-Z%µμ]*)'
    )
    for match in number_pattern.finditer(text):
        start, end = match.span(1)
        num_str = match.group(1)
        num_clean = num_str.replace(',', '')
        value = float(num_clean) if '.' in num_clean else int(num_clean)
        potential_matches.append((start, end, value))

    # 3. 处理独立的阿拉伯数字（整数和小数）
    standalone_number_pattern = re.compile(
        r'(?<![\w.])(-?\d{1,}(?:,\d{3})*(?:\.\d+)?)(?![a-zA-Z%µμ])'
    )
    for match in standalone_number_pattern.finditer(text):
        start, end = match.span(1)
        num_str = match.group(1)
        num_clean = num_str.replace(',', '')
        value = float(num_clean) if '.' in num_clean else int(num_clean)
        potential_matches.append((start, end, value))

    # 4. 处理英文文字表达的数字，包括小数
    number_words_pattern = re.compile(
        rf'\b(?:{"|".join(main_number_words + connector_words)})'
        rf'(?:[-\s](?:{"|".join(main_number_words + connector_words)}))*\b',
        re.IGNORECASE
    )
    for match in number_words_pattern.finditer(text):
        start, end = match.span()
        word_num = match.group(0).lower()

        # 过滤不相关的匹配
        if not any(word in word_num for word in main_number_words):
            continue
        if re.search(r'[~]', word_num):
            continue

        value = text2num_decimal(word_num)
        if value is None:
            continue  # 转换失败则跳过
        # 处理负数
        if word_num.startswith("negative") or word_num.startswith("-"):
            value = -abs(value)
        potential_matches.append((start, end, value))

    # 5. 按起始位置排序
    potential_matches.sort(key=lambda x: x[0])

    # 6. 移除重叠的匹配项
    final_matches = []
    seen_positions = set()
    for start, end, value in potential_matches:
        if any(pos in seen_positions for pos in range(start, end)):
            continue  # 跳过重叠部分
        final_matches.append((start, end, value))
        seen_positions.update(range(start, end))

    # 7. 分配标签
    extracted_numbers = []
    for count, (start, end, value) in enumerate(final_matches):
        label = f"[NUM{count}]"
        extracted_numbers.append((label, start, end, value))

    return extracted_numbers

def text2num_decimal(word_num):
    """
    将英文文字表达的数字转换为数字类型。
    使用 `word2number` 库进行转换。
    """
    try:
        return w2n.word_to_num(word_num)
    except:
        return None


def mask_num(text, extract_num_tup):
    """
    :param text: 原始文本
    :param extract_num_tup: extract_numbers 返回值，tup[0]:掩码、tup[1]:token 起始位置、tup[2]:token 结束位置、tup[3]:token 值
    :return:
    """
    new_text = text
    # 反转元组列表从后向前替换
    reverse_extract_num_tup = list(reversed(extract_num_tup))
    for tup in reverse_extract_num_tup:
        new_text = new_text[:tup[1]] + tup[0] + new_text[tup[2]:]  # 使用 new_text 而不是 text
    return new_text


def restore_numbers(masked_text, extract_num_tup):
    """
    恢复掩码为扰动后的数值
    :param masked_text: 替换后的文本
    :param extract_num_tup: extract_numbers 返回值，tup[0]:掩码、tup[1]:token 起始位置、tup[2]:token 结束位置、tup[3]:token 值
    :return: 恢复后的文本
    """
    restored_text = masked_text

    for tup in extract_num_tup:
        restored_text = restored_text.replace(tup[0], str(tup[3]))

    return restored_text

def check_list_lengths(ori_numbers, json_data):
    if len(ori_numbers) != len(json_data):
        raise ValueError(
            f"列表json_data和列表ori_numbers长度不一致,json_data的长度是：{len(json_data)}，ori_numbers的列表长度是：{len(ori_numbers)}, json_data = {json_data}, ori_numbers = {ori_numbers}")


def get_noise_tup(num_tup, noise_list):
    """
    :param num_tup: 原始元组列表
    :param noise_list: 差分结果列表
    :return: 差分扰动后的元组列表
    """
    # 确保两个列表长度一致
    if len(num_tup) != len(noise_list):
        raise ValueError("元组列表和替换列表的长度必须相同")

    # 创建一个新的列表，包含修改后的元组
    modified_tuples = [
        (*tup[:-1], new_item)  # 解包元组并替换最后一项，同时将新值转换为整数
        for tup, new_item in zip(num_tup, noise_list)
    ]

    return modified_tuples


def add_gaussian_noise(extract_num_tup, json_data, epsilon=1, delta=0.5):
    """
    向数值添加满足差分隐私的高斯扰动
    :param extract_num_tup: extract_numbers 返回值，tup[0]:掩码、tup[1]:token 起始位置、tup[2]:token 结束位置、tup[3]:token 值
    :param epsilon: 隐私预算
    :param delta: 失败概率
    :param json_data: 由LLM输出的存放灵敏度（数值变化对结果的影响）的json格式数据
    :return: 加扰动后的值列表
    """
    
    noise_tup_list = []
    ori_numbers_list = [tup[3] for tup in extract_num_tup]
    # try:
    #     check_list_lengths(extract_num_tup, json_data)
    # except ValueError as e:
    #     print(f"发生错误: {e}")

    for i in range(len(extract_num_tup)):
        if not json_data[i][4]:
            noise_tup_list = get_noise_tup(extract_num_tup, ori_numbers_list)
            continue

        sensitivity = json_data[i][2]

        # 计算噪声标准差
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon

        # 生成高斯噪声
        noisy = np.random.normal(0, sigma)

        if '.' in str(ori_numbers_list[i]):

            original_value = ori_numbers_list[i]
            disturbance = round(noisy + original_value, get_decimal_places(original_value))
            if json_data[i][3] == "false":
                while disturbance * original_value <= 0:
                    noisy = np.random.normal(0, sigma)
                    disturbance = round(noisy + original_value, get_decimal_places(original_value))
        else:
            original_value = int(ori_numbers_list[i])


            disturbance = round(original_value + noisy)

            if str(json_data[i][3]).lower() == "false" and original_value != 0:
                while disturbance * original_value <= 0:
                    noisy = np.random.normal(0, sigma)
                    disturbance = round(original_value + noisy)


        ori_numbers_list[i] = str(disturbance)
        noise_tup_list = get_noise_tup(extract_num_tup, ori_numbers_list)
        # 返回加扰动后的值
    return noise_tup_list


def get_decimal_places(value):
    """
    获取浮点数的小数位数
    :param value: 浮点数值
    :return: 小数位数
    """
    if '.' in str(value):
        return len(str(value).split('.')[1])
    return 0  # 如果是整数，返回0


def sensitivity_acquisition(masked_text):
    """
    利用LLM的语言理解能力获取合理的差分隐私扰动范围

    :param masked_text: 已经完成掩码保护的文本
    :return: json形式的回复，格式为[[占位符, 数值修饰的实体, 参考扰动范围], ...]
             注意：这个函数现在会在达到最大重试次数时抛出异常而不是终止程序。
    """

    # 定义正则表达式模式
    pattern = r'\[NUM\d+\]'

    # 使用 re.findall 找到所有匹配的标签
    matches = re.findall(pattern, masked_text)

    # Store the last exception to re-raise it
    last_exception = None

    max_retries = 10
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key= API_KEY,
                base_url= BASE_URL,
            )

            systemContent = f"""
           You are an expert in medicine, linguistics and differential privacy. I will provide you with a natural language text containing the masking tag "[NUM]". Note that only masks with the shape [NUM] are considered masks. That is to say, you only need to handle the part shaped like [NUM]. First of all, please read through the text to determine how many mask tags are present in the text to ensure that your output matches the number of tags present in the text.
           Markers may be mixed with other content through symbols such as "-", please identify all [NUM] completely. Follow these steps to complete your task


           1: Read Read all the content of the input and output from beginning to end first. Match all masks in a regular matching manner, determine how many masks are in the text in the format [NUMi], where i is an integer placeholder. If the mask tags are connected by symbols such as "-", "/", etc., please regard them as multiple different masks.(For example, [NUM0]/[NUM1] are two masks)
           2: Read the context of each mask tag carefully, Determine what the thing modified by the number values masked by [NUM] is
           3: Based on your extensive prior knowledge, you should provide a specific and reasonable sensitivity value based on the range of differential privacy perturbations acceptable to that entity. Note that you need to provide the sensitivity, not the variance, WHICH MUST BE AN INTEGER OR A FLOATING POINT NUMBER BUT NOT 0. This value will be used to calculate the Gaussian noise parameter for differential privacy using the formula:
            [sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / 0.5))) / 1]. Please study the example I gave to carefully calculate each sensitivity
            NOTE: This sensitivity value should be a certain NUMBER, DON'T BE ANY WORDS. For you do not know or are not sure of the sensitivity do not answer 0, but to use 0.849, so that the variance according to the formula is 1. Only if you think that this value should not be added to the perturbation to answer 0!

            4: In addition, ‘true’ or ‘false’ is used to indicate whether the content modified by the numerical content masked by the mask is allowed to change from positive to negative or from negative to positive after perturbation.
            5: Use "true" or "false" to indicate whether the values used to describe the entity are allowed to be perturbed (e.g., the name of a defined drug as a proper noun is not suitable for perturbation).

            For you do not know the sensitivity do not use 0, with 0.849 so that the variance according to the formula is 1

            Example:"I met him [NUM0] days ago", there is only one mask here, which is [NUM0], and [NUM0] is modifying the number of days since the meeting, you need to give the sensitivity of this "number of days since the meeting", and at the same time, since it is modifying the number of days, it can't become negative after perturbation.
            I'm going to go with my experience here (you should have more experience as an expert) and assume that the number of days met can be perturbed within 3 days, so the sigma is 3, and the inverse gives the sensitivity here as roughly 2.548
            You need to give the sensitivity of this "number of days since meeting" at this point, and since the modifier is the number of days, it can't be made negative after perturbation, and since it's not a proper noun, it's allowed to be perturbed. So your output should be JSON ['NUM0', number of days since the meeting, 2.548, false, true].

            Another example: I saw him on the [NUM0] of [NUM1], 2023, where [NUM] modifies the month, so please decide for yourself what to modify in relation to the text. Here [NUM0] is presumed to modify the specific date of a month, [NUM1] is presumed to modify a certain month, other cases, please combine with the text to determine the specific content of the modification. So [NUM0] for the date of the perturbation range of assumptions for the 5 words, according to the formula projected sensitivity of 4.246

            Please output the content in the following format:
            [["NUM0", the thing modified by the number values masked by [NUM0] is, perturbation range 0, true/false, true/false], ["NUM1", ...]]
             where the first item is the label mask you are currently processing, and the second item is the entity modified by this numerical mask. The third item is the value that modifies this entity variable in sign, the forth item is this number suitable for change. After processing each mask, place the result into a list, and finally output the entire two-dimensional list.


            Please do not have a situation where the maximum number of mask tags in the text is only [NUM30], and your result tags contain [NUM31], which is more than the number of text tags that need to be processed in the text

            Please review your answer as you answer to make sure you haven't missed any of the mask marks.
            Because the text can be lengthy, please ensure you read all the content carefully when responding, and don't miss any of the [NUM] mask tags, especially at the end of the text!

            Only handle the [NUM] type masks present in the text. Do not introduce new masks, and leave all other masks as they are.

            Please make sure that the number of masks in the format [NUM] matches the number of results you output.
            Be sure to keep your results in the above form and don't add the rest of the content, I'll follow up by loading it directly with json.load(), so make sure your output loads correctly!
            """

            userContent = f"""{masked_text}, The number of mask markers is:{len(matches)}"""

            completion = client.chat.completions.create(
                model= MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': systemContent},
                    {'role': 'user', 'content': userContent}
                ],
                temperature=0.8,
            )
            sensitivity_tup = parse_json_safe(completion.choices[0].message.content)

            if len(matches) != len(sensitivity_tup):
                 logging.info(f"masked_text : {masked_text}")
                 logging.info(f"sensitivity_tup: {sensitivity_tup}")
                 last_exception = ValueError("模型给出的替换元组长度与需要替换的标签数量不符，重新请求")
                 raise last_exception 

            float_sensitivity_tup = []
            for item in sensitivity_tup:
                if not isinstance(item, list) or len(item) < 5:
                     last_exception = ValueError(f"模型返回的元组格式不正确: {item}. 需要至少5项.")
                     raise last_exception 

                try:
                    float_item = float(item[2])
                    each_tup = (item[0], item[1], float_item, str(item[3]).lower(), str(item[4]).lower()) # Ensure true/false are lower case strings
                except ValueError as e:
                    last_exception = ValueError(f"无法将文本 '{item[2]}' 转换为浮点数 (标签 {item[0]}). 重新请求") 
                    raise last_exception 
                except Exception as e:
                     last_exception = ValueError(f"处理模型返回的元组时发生未知错误: {item}. 错误: {e}. 重新请求")
                     raise last_exception

                float_sensitivity_tup.append(each_tup)

            return float_sensitivity_tup

        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} failed for masked text '{masked_text[:50]}...': {e}")
            # 如果是最后一次尝试，则向外抛出异常
            if attempt == max_retries - 1:
                print("Max retries reached. Re-raising the last exception.")
                raise last_exception 

def contexts_num_replace(contexts):
    """
    处理一个批次的文本，对其中的数值进行掩码、LLM敏感度获取和高斯扰动。

    :param contexts: 一个列表的列表，外层列表的每个元素是一个包含字符串的列表
    :return: 一个元组 (processed_contexts_list, failed_indices_list)
             processed_contexts_list: 与输入结构相同，但其中的数值已进行扰动。
                                      如果某个外层列表处理失败，则该位置的结果不会被添加到此列表中。
             failed_indices_list: 包含在处理过程中发生错误的外层列表的索引列表。
    """
    all_context_num_replace_list = []
    failed_indices = [] # 用于存储处理失败的外层列表索引的列表

    # 遍历contexts列表，并使用enumerate获取索引和值
    for outer_index, context in enumerate(tqdm(contexts, desc="generate contexts num replace")):
        try:
            context_num_replace_list = [] # 用于存储当前'context'列表处理后的字符串结果的列表
            # 遍历当前context列表中的每个字符串
            for i in range(len(context)):
                context_k = context[i]

                # 1、获取提取到的数值元组列表
                extract_num_tup = extract_numbers(context_k)

                # 2、根据是否提取到数字来决定是否进行掩码和扰动
                if not extract_num_tup:
                    # 如果没有提取到数字，处理后的文本就是原始文本
                    noised_text = context_k
                else:
                    # 如果提取到数字，进行掩码、LLM调用、扰动
                    masked_text = mask_num(context_k, extract_num_tup)

                    # 3、通过llm模型获取参考敏感度 (仅在生成了NUM标签时调用)
                    # 检查mask_num是否实际生成了任何[NUM]标签
                    if re.search(r'\[NUM\d+\]', masked_text):
                        # 调用sensitivity_acquisition，此函数在重试耗尽后可能会抛出异常
                        sensitivity_tup = sensitivity_acquisition(masked_text)
                        # 4、加入高斯扰动
                        noise_tup_list = add_gaussian_noise(extract_num_tup, sensitivity_tup)
                        # 5、获取扰动后的文本
                        noised_text = restore_numbers(masked_text, noise_tup_list)
                    else:
                        # 如果extract_num_tup不为空但mask_num未能创建标签，这是一种异常情况
                        # 回退到使用原始文本
                        logging.warning(f"提取到数字，但在掩码文本中未找到[NUM]标签 (context {outer_index} 中的索引 {i})：{context_k[:100]}...")
                        noised_text = context_k # 回退到原始文本

                # 将当前字符串context_k的处理结果添加到内部列表
                context_num_replace_list.append(noised_text)

            # 如果内部循环成功处理完当前'context'列表中的所有字符串，
            # 将结果列表添加到总列表中。
            all_context_num_replace_list.append(context_num_replace_list)

        except Exception as e:
            # 如果try块内发生任何异常（例如，sensitivity_acquisition在重试耗尽后抛出异常），在此处捕获
            print(f"处理外部索引为 {outer_index} 的context时捕获到异常：{e}")
            # 将失败的外层context列表的索引添加到failed_indices列表中
            failed_indices.append(outer_index)
            all_context_num_replace_list.append('')
            # 对于这个失败的context，不向all_context_num_replace_list添加任何内容。
            # 外层循环将继续处理下一个context列表。

    # 返回成功处理的contexts列表和失败的索引列表
    return all_context_num_replace_list, failed_indices