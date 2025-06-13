from typing import List
from tqdm import tqdm
import json
import random
import spacy
from openai import OpenAI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from torch.nn.functional import cosine_similarity
import logging
# # 配置日志记录
# logging.basicConfig(level=logging.INFO,  # 设置日志级别
#                     format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
#                     handlers=[logging.StreamHandler()])  # 输出到控制台
from dotenv import load_dotenv
import os
load_dotenv(".env")
model_path = './model/'
API_KEY = os.getenv("ENTITY_API_KEY")
BASE_URL= os.getenv("ENTITY_BASE_URL")
MODEL_NAME= os.getenv("ENTITY_MODEL_NAME")

# 1、加载spacy模型
def load_spacy(model_name):
    if model_name == "en_core_web_trf":
        # 创建空的 spaCy 模型
        nlp_web = spacy.load(f"{model_path}en_core_web_trf")  # 或加载预训练模型，如 nlp = spacy.load("en_core_web_sm")

        # 创建 EntityRuler 并添加到管道中
        ruler = nlp_web.add_pipe("entity_ruler")  # 用组件名称注册 EntityRuler
        patterns = [
            {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"}}]}
        ]
        ruler.add_patterns(patterns)
        return nlp_web
    elif model_name == "en_core_sci_scibert":
        nlp_sci = spacy.load(f"{model_path}en_core_sci_scibert")
        return nlp_sci

# 2、加载bert模型
def load_bert_model(model_name):
    # 初始化BERT模型和tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if model_name == "sentence-transformers/all-MiniLM-L6-v2":
    #     tokenizer = BertTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    #     model = BertModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    #     model.eval()
    if model_name == "all-MiniLM-L6-v2":
        tokenizer = BertTokenizer.from_pretrained(f"{model_path}all-MiniLM-L6-v2")
        model = BertModel.from_pretrained(f"{model_path}all-MiniLM-L6-v2").to(device)
        model.eval()
        return tokenizer, model, device

# 3、实体过滤与提取
def filter_entities(entities):
    filtered_entities = []
    for ent in entities:
        # 判断实体是否合理
        if '(' in ent.text or ')' in ent.text:  # 排除包含括号的实体
            continue
        # if any(char.isdigit() for char in ent.text):  # 排除包含数字的实体（如错误识别的数值实体）
        #     continue
        filtered_entities.append(ent)
    return filtered_entities


def extract_entities_without_sec(context :str , primary_nlp, secondary_nlp, probability):
    """
    提取文本中的实体，并去除实体文本中的换行符，同时合并部分重叠的实体。
    :param context: 输入文本
    :param primary_nlp: 主模型
    :param secondary_nlp: 次模型 (可以为 None)
    :param probability: 非主模型得到的实体及名词替换的概率 (0-100)
    :return: ori_entities：[(实体起始位置索引，实体)]
    """
    # 主模型提取实体
    primary_doc = primary_nlp(context)
    # 主模型实体过滤
    primary_filter_entities = filter_entities(primary_doc.ents)
    # 去除实体文本中的换行符
    primary_entities = [
        (ent.start_char, ent.end_char, ent.text.replace('\n', ''))
        for ent in primary_filter_entities
    ]

    combined_entities = []

    # 添加主模型的实体
    for pri_start, pri_end, pri_text in primary_entities:
        combined_entities.append((pri_start, pri_end, pri_text))

    # 如果次模型不为 None，则进行次模型的实体提取和名词识别
    if secondary_nlp is not None:
        # 次模型提取实体
        secondary_doc = secondary_nlp(context)
        # 次模型实体过滤
        secondary_filter_entities = filter_entities(secondary_doc.ents)
        # 去除实体文本中的换行符
        secondary_entities = [
            (ent.start_char, ent.end_char, ent.text.replace('\n', ''))
            for ent in secondary_filter_entities
        ]

        # 处理次模型的实体，合并部分重叠的实体
        for sec_start, sec_end, sec_text in secondary_entities:
            overlap = False
            for i, (pri_start, pri_end, pri_text) in enumerate(combined_entities):
                # 判断是否存在部分重叠
                if sec_start < pri_end and sec_end > pri_start:
                    overlap = True
                    # 合并实体
                    new_start = min(pri_start, sec_start)
                    new_end = max(pri_end, sec_end)
                    new_text = context[new_start:new_end].replace('\n', '')
                    combined_entities[i] = (new_start, new_end, new_text)
                    break
            if not overlap:
                random_number = random.randint(0, 100)
                # 如果随机数小于 probability，记录实体
                if random_number < float(probability):
                    combined_entities.append((sec_start, sec_end, sec_text))

        # 去重
        unique_entities = list(set(combined_entities))

        noun_list = []

        for token in primary_doc:
            if token.pos_ == "NOUN":
                should_add = True
                # 去除名词中的换行符
                noun_text = token.text.replace('\n', '')
                if not noun_text:
                    continue
                for uni_start, uni_end, uni_text in unique_entities:
                    if noun_text in uni_text and uni_start <= token.idx <= uni_end:
                        should_add = False
                        break
                if should_add:
                    random_number = random.randint(0, 100)
                    if random_number < float(probability):
                        noun_list.append((token.idx, token.idx + len(noun_text), noun_text))

        # 合并实体和名词列表，并去重
        ori_entities = [(entity_tuple[0], entity_tuple[2]) for entity_tuple in unique_entities + noun_list]
    else:
        unique_entities = list(set(combined_entities))
        noun_list = []
        for token in primary_doc:
            if token.pos_ == "NOUN":
                should_add = True
                # 去除名词中的换行符
                noun_text = token.text.replace('\n', '')
                if not noun_text:
                    continue
                for uni_start, uni_end, uni_text in unique_entities:
                    if noun_text in uni_text and uni_start <= token.idx <= uni_end:
                        should_add = False
                        break
                if should_add:
                    random_number = random.randint(0, 100)
                    if random_number < float(probability):
                        noun_list.append((token.idx, token.idx + len(noun_text), noun_text))
        # 如果次模型为 None，则只处理主模型的实体
        ori_entities = [(entity_tuple[0], entity_tuple[2]) for entity_tuple in unique_entities + noun_list]

    ori_entities = list(set(ori_entities))
    # 按起始位置降序排序
    reverse_ori_entities = sorted(ori_entities, key=lambda x: x[0], reverse=True)
    return reverse_ori_entities


# 4、构造对话模型
def get_client():
    client = OpenAI(
        api_key = API_KEY,
        base_url = BASE_URL
    )
    return client


import json


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
        error_lin_enum = e.lineno
        error_col_num = e.colno
        context_start = max(0, error_position - 20)
        context_end = min(len(result), error_position + 20)
        error_context = result[context_start:context_end]
        error_msg = (
            f"解析结果失败，返回内容不是有效的 JSON 格式：\n"
            f"错误位置：行 {error_lin_enum}, 列 {error_col_num}\n"
            f"错误上下文：{error_context}\n"
            f"错误信息：{e}"
        )
        raise ValueError(error_msg)


def get_llm_replace_entities(client, entity_list, is_batch=False):
    if len(entity_list) == 0:
        return entity_list
    """
    与模型建立单段对话，并验证返回结果能否解析为 JSON 格式
    :param client: 用于给出参考近义词的模型
    :param entity_list: 从上下文中提取出的实体列表
    :return: JSON格式的近义词替换列表
    """

    if is_batch:
        request_entity_list = [entity_tup[1] for entity in entity_list for entity_tup in entity]
    else:
        request_entity_list = [entity_tup[1] for entity_tup in entity_list]

    # 用户输入内容
    user_content = f"""{request_entity_list}"""

    # 系统提示内容，强调多样性和隐私保护
    system_content = f"""
        As an expert specialising in medicine, linguistics and privacy protection. I will provide a list containing a large number of entities. Your task is to generate SEVEN alternative entities for each entity in the list. These alternative entities should use different privacy-preserving strategies, and generate the corresponding alternative entities by randomly selecting SEVEN strategies for each entity from the following list, giving priority to entities with multiple possible meanings to give substitutions from a variety of different possible meanings:
        1:Randomization: Replace with a random but related entity.
        2:Obfuscation: Replace with a more general or vague description.
        3:Differential Privacy: Add noise or perturbation to the entity.
        4:Semantic Abstraction: Replace with a conceptually related but less specific entity.
        5:Fictional Replacement: Replace with a completely fictional or fabricated entity.
        6:Synonym Replacement: Use synonyms or closely related terms.
        7:Domain-Specific Variation: Replace with commonly used variations within the same field.
        For entities that require high accuracy, such as specialized medical expressions and legal texts, minimize the degree of variation while providing expressions from different perspectives to maximize contextual adaptability.
        Return the result in JSON format, structured as. The original entity being replaced is labeled ori. Place the original entity in the first position in the list. NOT output as markdown format: 
        [
        ["ori1", "replacement_entity1_1",...,"replacement_entity1_5",...], 
        ["ori2", "replacement_entity2_1",...,"replacement_entity2_5",...],
        ....
        ]
        If there is only one item in the list of entities please give a replacement for that item instead of outputting seven paragraphs of the same answer.
        Ensure that the length of the input list of entities is equal to the length of the output 2D list.
    """

    completion = client.chat.completions.create(
        model = MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ],
        temperature=2.0,  # 增加随机性，鼓励生成更多样化的替代
        top_p=0.9,  # 扩大生成的覆盖范围
    )

    # 获取模型生成的内容
    result = completion.choices[0].message.content
    return result


def embed_texts(texts, tokenizer, model, device):
    """对一批文本进行BERT嵌入"""
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 获取 [CLS] token 的嵌入
    return embeddings


def compute_similarities(original_embedding, alternative_embeddings):
    """
    计算原始文本嵌入与替换后文本嵌入的余弦相似度。
    original_embedding: 原始文本的单个嵌入 (1, hidden_size)
    alternative_embeddings: 替换后文本的嵌入矩阵 (num_alternatives, hidden_size)
    """
    # 计算余弦相似度
    similarities = cosine_similarity(original_embedding, alternative_embeddings)
    return similarities.squeeze(0).cpu().numpy()  # 返回为 NumPy 数组


def select_replacement_entity(context, reverse_ori_entities, parse_llm_replace_entities, tokenizer, model, device):
    """
    处理整个上下文，计算所有实体替换后的第二相似结果。
    """

    selected_replacements = []
    # 原始上下文嵌入

    original_embedding = embed_texts([context], tokenizer, model, device)
    # count = 0

    for (start_pos, entity), replacements in zip(reverse_ori_entities, parse_llm_replace_entities):
        # 对于每个实体生成替换后的上下文
        replaced_texts = []
        for replacement in replacements:
            replaced_text = context[:start_pos] + replacement + context[start_pos + len(entity):]
            replaced_texts.append(replaced_text)
        # 一次性嵌入所有替换后的文本
        alternative_embeddings = embed_texts(replaced_texts, tokenizer, model, device)
        # 计算相似度并选出第二相似
        similarities = compute_similarities(original_embedding, alternative_embeddings)
        # logging.info(f"Similarity score: {similarities}")
        second_similar_idx = np.argsort(-similarities)[1]  # 选出第二相似的索引
        selected_replacements.append(replacements[second_similar_idx])  # 记录第二相似的替换实体

    return selected_replacements


# 在字典中追加数据
def add_new_entry(dic, key, value):
    if isinstance(dic, dict):
        dic[key] = value
    else:
        print("Unexpected data format. Expected a dictionary.")


def check_dic(entity, dic):
    """
    用于在dic中检查是否已经存放有指定实体对应的列表
    :param entity: 目标实体
    :param dic: 实体替换列表字典
    :return: [实体的替换列表]
    """
    if entity in dic:
        return dic[entity]


def prepare_replacements(entities, replace_entities_dic):
    """
    :param entities: extract_entities返回的实体元组列表
    :return: request_entities：字典中不存在的，需要向模型发起请求的实体
             replacement_entities：[(索引的次序，具体的替换列表)]，最后根据索引次序将对应的替换列表插入到最后的替换列表中
    """
    # replacement_entities：[(实体次序，[替换实体列表])]用于记录已经在存储库中存放的实体的对应替换
    replacement_entities = []
    request_entities = entities[:]

    # del_list用于存放需要删除的实体次序
    del_list = []
    for n, (idx, entity) in enumerate(entities):
        if entity in replace_entities_dic:
            replacement_entities.append((n, check_dic(entity, replace_entities_dic)))
            del_list.append(n)
    del_list.reverse()
    if del_list:
        for n in del_list:
            del request_entities[n]
    return request_entities, replacement_entities


def get_replacement_entities_list(context: str, replace_entities_dic_path, primary_nlp, secondary_nlp, probability):
    """
    得到替换实体列表
    :param context: 上下文文本
    :param replace_entities_dic_path: 存储在本地的替换列表路径
    :param primary_nlp: 主分词模型
    :param secondary_nlp: 副分词模型
    :param probability: 替换概率（百分制）
    :return: reverse_ori_entities = [(22, 'apple'), (11, 'banana')]， parse_llm_replace_entities：[[模型给出的替换列表]...[]]
    """
    try:
        with open(replace_entities_dic_path, "r", encoding="utf-8") as f:
            replace_entities_dic = json.load(f)
    
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或格式不对，初始化为空字典
        replace_entities_dic = {}
    # reverse_ori_entities:[(起始索引，实体)]
    client = get_client()
    reverse_ori_entities = extract_entities_without_sec(context, primary_nlp, secondary_nlp, probability)
    # logging.info(f"Reverse ori entities: {reverse_ori_entities}")
    request_entities, replacement_entities = prepare_replacements(reverse_ori_entities, replace_entities_dic)
    # logging.info(f"Request entities: {request_entities}")
    parse_llm_replace_entities = []
    # llm_replace_entities = get_llm_replace_entities(client, request_entities)
    max_retries = 10
    if not is_list_of_empty_lists(request_entities):
        for attempt in range(max_retries):
            try:
                with open(replace_entities_dic_path, "r", encoding="utf-8") as f:
                    replace_entities_dic = json.load(f)
                request_entities, replacement_entities = prepare_replacements(reverse_ori_entities, replace_entities_dic)
                # 获得llm提供的替换列表并解析
                llm_replace_entities = get_llm_replace_entities(client, request_entities)
                parse_llm_replace_entities = parse_json_safe(llm_replace_entities)

                one_check_length(request_entities, parse_llm_replace_entities, replace_entities_dic_path)
                # print(f"parse_llm_replace_entities:{parse_llm_replace_entities}")
                one_save_replace_entities(replace_entities_dic_path, request_entities, parse_llm_replace_entities)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("All attempts failed for this context.")
                    parse_llm_replace_entities = None  # 或者根据需要处理失败情况
        # 将从字典中找到的replacement_entities按照其中元组中的次序插入parse_llm_entities
    for replacement_entity in replacement_entities:
        parse_llm_replace_entities.insert(replacement_entity[0], replacement_entity[1])
    return reverse_ori_entities, parse_llm_replace_entities


def get_final_context(ori_context, reverse_ori_entities, parse_llm_replace_entities, tokenizer, model, device):

    # 基于bert选择相似度第二高的结果
    # logging.info("开始选择替换实体")
    replace_entity = select_replacement_entity(
        ori_context, reverse_ori_entities, parse_llm_replace_entities, tokenizer, model, device)
    
    for n, _ in enumerate(replace_entity):
        length = len(reverse_ori_entities[n][1])
        start = reverse_ori_entities[n][0]
        # logging.info(f"start = {start}")
        ori_context = ori_context[:start] + replace_entity[n] + ori_context[start + length:]
    return ori_context

    
def one_check_length(request_entities, llm_replace_entities, replace_entities_dic_path):
    if len(request_entities) != len(llm_replace_entities):
        if len(request_entities) > len(llm_replace_entities):
            one_part_save_replace_entities(replace_entities_dic_path, llm_replace_entities)
        raise ValueError(
            f"实体长度错误，需要被替换的实体个数为{len(request_entities)},需要被替换的实体为：{request_entities} 模型给出的替换实体数量为：{len(llm_replace_entities)}")


def is_list_of_empty_lists(lst):
    return all(isinstance(item, list) and len(item) == 0 for item in lst)



def get_ori_final_context(contexts :list[list[str]] , primary_nlp, secondary_nlp, probability, tokenizer, model, device,  replace_entities_dic_path = None):
    all_ori_final_context = []
    if replace_entities_dic_path is None:
        with open('./result/replace_entities_dic.json', 'w', encoding='utf-8') as f:
            json.dump({}, f)
            replace_entities_dic_path = './result/replace_entities_dic.json'
    for context in tqdm(contexts, desc="generate final contexts"):
        ori_final_context = []
        for i in range(len(context)):
            # logging.info("执行替换实体列表获取")
            context_k = context[i]
            # logging.info(f"context_k = {context_k[:100]}")
            reverse_ori_entities, parse_llm_replace_entities = get_replacement_entities_list(context_k, replace_entities_dic_path, primary_nlp, secondary_nlp, probability)
            # logging.info(f"需要被替换的实体：{reverse_ori_entities}")
            # logging.info("确定最终替换")
            new_context = get_final_context(
            context_k,
            reverse_ori_entities,
            parse_llm_replace_entities,
            tokenizer,
            model,
            device
            )
            # logging.info(f"替换结果：{new_context}")
            ori_final_context.append(new_context)
        all_ori_final_context.append(ori_final_context)
    return all_ori_final_context


def save_replace_entities(replace_entities_dic_path, batch_request_entities, request_entities_replacement_list):
    """
    批量存储替换实体及其对应的替换列表
    :param replace_entities_dic_path: 替换字典路径
    :param batch_request_entities: 批量请求实体列表
    :param request_entities_replacement_list: 请求实体的替换列表
    :return:
    """
    try:
        with open(replace_entities_dic_path, "r", encoding="utf-8") as f:
            replace_entities_dic = json.load(f)
    
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或格式不对，初始化为空字典
        replace_entities_dic = {}

    # 将批量实体展平
    request_entities = [entity_tup[1] for entity in batch_request_entities for entity_tup in entity]

    for n, entity in enumerate(request_entities):
        add_new_entry(replace_entities_dic, entity,
                      request_entities_replacement_list[n])
    open(replace_entities_dic_path, "w").write(json.dumps(replace_entities_dic))
    
def one_save_replace_entities(replace_entities_dic_path, request_entities, request_entities_replacement_list):
    """
    批量存储替换实体及其对应的替换列表
    :param replace_entities_dic_path: 替换字典路径
    :param request_entities: 请求实体列表
    :param request_entities_replacement_list: 请求实体的替换列表
    :return:
    """
    try:
        with open(replace_entities_dic_path, "r", encoding="utf-8") as f:
            replace_entities_dic = json.load(f)
    
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或格式不对，初始化为空字典
        replace_entities_dic = {}

    # 将批量实体展平
    for n, (_, entity) in enumerate(request_entities):
        add_new_entry(replace_entities_dic, entity,
                      request_entities_replacement_list[n])
    open(replace_entities_dic_path, "w").write(json.dumps(replace_entities_dic))
    
def one_part_save_replace_entities(replace_entities_dic_path,  request_entities_replacement_list):
    """
    批量存储替换实体及其对应的替换列表
    :param replace_entities_dic_path: 替换字典路径
    :param request_entities_replacement_list: 请求实体的替换列表
    :return:
    """
    try:
        with open(replace_entities_dic_path, "r", encoding="utf-8") as f:
            replace_entities_dic = json.load(f)
    
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或格式不对，初始化为空字典
        replace_entities_dic = {}
    
    logging.info("存入部分实体")
    # 将批量实体展平
    for entities_list in request_entities_replacement_list:
        add_new_entry(replace_entities_dic, entities_list[0],
                      entities_list[1:])
    open(replace_entities_dic_path, "w").write(json.dumps(replace_entities_dic))

