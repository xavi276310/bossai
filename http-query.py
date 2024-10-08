import chromadb
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import warnings

warnings.filterwarnings("ignore")

# 初始化 Flask 应用
app = Flask(__name__)

# 1. 初始化 Chroma 客户端，并指定持久化存储路径
chroma_client = chromadb.PersistentClient(path="/Users/mac/PycharmProjects/job_extended")  # 替换为你的持久化路径
collection = chroma_client.get_collection(name="job_collection")  # 加载已存储的集合

# 2. 加载同样的 SentenceTransformer 模型，用于问题向量化
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. 定义函数来根据 JSON 数据查询 Chroma 数据库
def query_database_from_json(input_json):
    job_title_keywords = input_json["job_title_keyword"]
    job_country = input_json["job_country"]
    job_locations = input_json["job_location"]  # 默认为空列表
    quantity = input_json["quantity"]  # 获取 quantity 参数，默认返回 1 条

    if job_title_keywords is None:
        job_title_keywords = []
    if job_locations is None:
        job_locations = []

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

# 4. 定义 Flask 路由，处理客户端的 POST 请求
@app.route('/query', methods=['POST'])
def query():
    try:
        # 从请求中获取 JSON 数据
        input_json = request.get_json()

        # 查询数据库并返回结果
        result = query_database_from_json(input_json)

        # 返回 JSON 格式的查询结果
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# 启动 Flask 服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
