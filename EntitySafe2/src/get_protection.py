import os
import argparse
import pandas as pd
from numerical_pertubation import contexts_num_replace
from entities_replace import get_client, load_bert_model, get_ori_final_context, load_spacy
from retrieval_database import construct_retrieval_database
parser = argparse.ArgumentParser(description='开始合成数据流程')
parser.add_argument('--p', type=str, required=True, help='替换率，用以控制实体替换比率以适应不同的有用性与隐私性的平衡要求，0~100，表示需要进行替换的实体比率')
parser.add_argument('--n', type=bool, required=True, help='用户决定是否需要为命名体识别阶段启用补充模型')
args = parser.parse_args()
"""
You can run this file by running following codes:
export CUDA_VISIBLE_DEVICES=1
python get_origin_context.py --dataset_name="chat" --attack_method="per"
You can also run:
python get_origin_context.py --dataset_name="wiki" --attack_method="per"
python get_origin_context.py --dataset_name="chatdoctor" --attack_method="attack"
python get_origin_context.py --dataset_name="chatdoctor" --attack_method="untarget"
python get_origin_context.py --dataset_name="wiki_pii100" --attack_method="attack"
python get_origin_context.py --dataset_name="wiki_pii100" --attack_method="untarget"
"""

if not os.path.exists('./result'):
    os.makedirs('./result')



# data_df = pd.read_csv('./data/data.csv')[:5]
# data = [[row] for row in data_df['document']]
# num_replace_result_list, failed_indices = contexts_num_replace(data)
# num_replace_result_df = pd.DataFrame(num_replace_result_list, columns=['num_replace_result'])
# num_replace_result_df.to_csv('./result/num_replace_result_df.csv')
# failed_indices_df = pd.DataFrame(failed_indices, columns=['failed_indices'])
# failed_indices_df.to_csv('./result/failed_indices_df.csv')


client = get_client()
nlp_web = load_spacy("en_core_web_trf")
nlp_sci = load_spacy("en_core_sci_scibert")
# 加载bert模型参数
tokenizer, model, device = load_bert_model('all-MiniLM-L6-v2')

num_replace_result_df = pd.read_csv('./result/num_replace_result_df.csv')
num_replace_result_list = data = [[row] for row in num_replace_result_df['num_replace_result']]
if args.n == True:
    final_result = get_ori_final_context(num_replace_result_list, nlp_web, nlp_sci, args.p, tokenizer, model, device)
else:
    final_result = get_ori_final_context(num_replace_result_list, nlp_web, args.p, tokenizer, model, device)
failed_indices_df = pd.DataFrame(final_result, columns=['final_result'])
failed_indices_df.to_csv('./result/final_result.csv')

construct_retrieval_database()