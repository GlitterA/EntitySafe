import langchain
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def construct_retrieval_database() -> 'langchain.vectorstores.chroma.Chroma':

    documents_df = pd.read_csv("./result/final_result.csv")  
    vector_store_path = './retrieval_database'
    documents = documents_df['final_result'].tolist()
    embed_model = HuggingFaceEmbeddings(model_name="./model/all-MiniLM-L6-v2")
    docs = [Document(page_content=text) for text in documents]
    # 4. 转换为LangChain的Document格式（可添加元数据）

    retrieval_database = Chroma.from_documents(documents=docs,
                                               embedding=embed_model,
                                               persist_directory=vector_store_path)
    return retrieval_database



def load_retrieval_database_from_address() -> 'langchain.vectorstores.chroma.Chroma':

    # 初始化嵌入模型
    embed_model = HuggingFaceEmbeddings(model_name="./model/all-MiniLM-L6-v2")

    # 新建空数据库（如果 ./chroma_db 不存在）
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory='./retrieval_database'  # 新路径
    )
    return retrieval_database

