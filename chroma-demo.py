import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import warnings
warnings.filterwarnings("ignore")

# 1. 读取 CSV 数据
data = pd.read_csv("/Users/mac/Downloads/jobs_extended_prod_20241008.csv")  # 替换为你的 CSV 文件路径

# 2. 使用 SentenceTransformer 模型加载
model = SentenceTransformer('all-MiniLM-L6-v2')  # 可以根据需要选择不同的模型

# 3. 初始化 Chroma 持久化客户端，并指定存储路径
chroma_client = chromadb.PersistentClient(path="/Users/mac/PycharmProjects/job_extended_20241008")  # 指定你的持久化存储路径
collection = chroma_client.create_collection(name="job_collection")  # 创建集合

cnt = 0
total = len(data)
for index, row in data.iterrows():

    print('\r', cnt, "/", total, end='')
    cnt += 1

    job_title = row['job_title']
    job_country = row['job_country']
    job_location = row['job_location']

    # 创建文档内容
    document = f"Job Title: {job_title}, Country: {job_country}, Location: {job_location}"

    # 生成嵌入向量
    embedding = model.encode(document)

    # 插入到 Chroma 数据库中，存储整行数据
    collection.add(
        documents=[document],
        metadatas=[row.to_dict()],  # 存储整行数据为元数据
        ids=[f"job_{index}"]
    )

print("数据已成功存储到Chroma  数据库中。")
