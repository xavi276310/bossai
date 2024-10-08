import chromadb
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# 1. 初始化 Chroma 客户端，并指定持久化存储路径
chroma_client = chromadb.PersistentClient(path="/Users/mac/PycharmProjects/job_extended")  # 替换为你的持久化路径
collection = chroma_client.get_collection(name="job_collection")  # 加载已存储的集合

# 2. 加载同样的 SentenceTransformer 模型，用于问题向量化
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. 定义函数来根据 JSON 数据查询 Chroma 数据库
def query_database_from_json(input_json):
    job_title_keywords = input_json["job_title_keyword"]
    job_country = input_json["job_country"]
    job_locations = input_json["job_location"]
    quantity = input_json.get("quantity", 1)  # 获取 quantity 参数，默认返回 1 条

    # 构建查询的文本内容
    query_text = f"Job Title: {', '.join(job_title_keywords)}, Country: {job_country}, Location: {', '.join(job_locations)}"

    # 将问题向量化
    query_embedding = model.encode([query_text])[0]

    # 在 Chroma 数据库中查找最匹配的结果
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],  # 使用问题的向量
        n_results=quantity  # 返回与 quantity 相匹配的结果数量
    )

    # 获取最匹配的元数据（即整行数据）
    if results['metadatas']:
        best_matches = results["metadatas"][0]  # 获取匹配的所有元数据
        return best_matches
    else:
        return "未找到匹配的结果。"

# 4. 输入 JSON 数据（作为 Python 字典）
input_json = {
    "job_title_keyword": [
        "Operations Agent"
    ],
    "job_country": "Philippines",
    "job_location": [
        # "Pasig"
    ],
    "quantity": 10
}

# 5. 查询数据库并返回结果
result = query_database_from_json(input_json)

# 6. 输出查询结果
print("查询结果：")
for res in result:
    print(res)  # 输出每条匹配的整行数据


