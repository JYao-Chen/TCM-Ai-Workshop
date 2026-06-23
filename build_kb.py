# -*- coding: utf-8 -*-
# @Time : 2026/6/23
# @Author : Yao
# @File : build_kb.py
# @Description : 构建本地向量知识库

import os
import csv
import pickle
import asyncio
import numpy as np
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError


# ==================== 配置区域 ====================
# 通义百炼 OpenAI 兼容接口配置
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "text-embedding-v4"

# 批处理配置
BATCH_SIZE = 10  # 每批嵌入条数
CONCURRENCY_LIMIT = 8  # 并发数
MAX_RETRIES = 3  # 重试次数

# 切块配置
MAX_CHUNK_LENGTH = 800  # 最大块长度（字符数）
OVERLAP_LENGTH = 100  # 重叠长度（字符数）

# 文件路径
CSV_PATH = "./output/medical_cases.csv"
KB_PATH = "./output/kb.pkl"


# ==================== 文本切块 ====================
def split_into_chunks(text: str, max_length: int = MAX_CHUNK_LENGTH, overlap: int = OVERLAP_LENGTH) -> list:
    """
    将文本切分成带重叠的块

    策略：
    1. 如果文本长度 <= max_length，直接返回
    2. 否则按句号、问号、感叹号等标点切分成句子
    3. 贪心合并句子，直到超过 max_length
    4. 相邻块保留 overlap 长度的重叠
    """
    text = text.strip()

    # 如果文本较短，直接返回
    if len(text) <= max_length:
        return [text]

    # 按标点切分成句子（保留标点）
    import re
    sentences = re.split(r'(?<=[。！？；\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 贪心合并句子
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # 如果单个句子就超过限制，强制切分
        if sentence_length > max_length:
            # 先保存当前块
            if current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_length = 0

            # 强制切分长句子
            for i in range(0, sentence_length, max_length - overlap):
                end = min(i + max_length, sentence_length)
                chunks.append(sentence[i:end])
            continue

        # 如果加上当前句子会超过限制
        if current_length + sentence_length > max_length:
            # 保存当前块
            chunks.append(''.join(current_chunk))

            # 计算重叠：从当前块的末尾保留 overlap 长度的文本
            overlap_text = ''.join(current_chunk)
            if len(overlap_text) > overlap:
                # 保留末尾的 overlap 长度
                overlap_start = len(overlap_text) - overlap
                # 找到重叠部分的起始句子
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_len
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    # 保存最后一块
    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks


# ==================== 嵌入 API ====================
async def embed_batch(
    client: AsyncOpenAI,
    texts: list,
    semaphore: asyncio.Semaphore,
    max_retries: int
) -> list:
    """
    批量嵌入文本，带重试
    """
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts
                )

            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            return embeddings

        except APITimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 + attempt * 2)
            else:
                print(f"嵌入超时，批次大小: {len(texts)}")
                return None
        except APIConnectionError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 + attempt * 2)
            else:
                print(f"连接失败: {e}")
                return None
        except RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(5 + attempt * 5)
            else:
                print(f"达到速率限制")
                return None
        except APIStatusError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 + attempt * 2)
            else:
                print(f"API 错误 {e.status_code}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 + attempt * 2)
            else:
                print(f"未知错误: {e}")
                return None

    return None


# ==================== 主流程 ====================
async def build_knowledge_base():
    """构建知识库"""

    # 检查 API Key
    if not DASHSCOPE_API_KEY:
        print("错误：未设置环境变量 DASHSCOPE_API_KEY")
        return

    # 1. 读取 CSV
    print(f"读取 CSV: {CSV_PATH}")
    cases = []

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get('医案内容', '').strip()

            # 跳过无效内容
            if content in ['无', '解析错误', '']:
                continue

            cases.append({
                'id': row.get('id', ''),
                'custom_id': row.get('custom_id', ''),
                'txt_filename': row.get('txt_filename', ''),
                'content': content,
                'case_type': row.get('医案类型', '')
            })

    print(f"有效医案数: {len(cases)}")

    # 2. 切块
    print("切分文本块...")
    chunks = []
    chunk_id = 0

    for case in cases:
        content = case['content']
        content_chunks = split_into_chunks(content)

        for chunk_idx, chunk_text in enumerate(content_chunks):
            chunks.append({
                'chunk_id': chunk_id,
                'original_id': case['id'],
                'custom_id': case['custom_id'],
                'txt_filename': case['txt_filename'],
                'case_type': case['case_type'],
                'chunk_index': chunk_idx,
                'total_chunks': len(content_chunks),
                'content': chunk_text
            })
            chunk_id += 1

    print(f"切块后总数: {len(chunks)}")

    # 3. 批量嵌入
    print("开始嵌入...")
    client = AsyncOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        timeout=60.0
    )

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # 准备批次
    batches = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_texts = [c['content'] for c in batch_chunks]
        batches.append((i, batch_chunks, batch_texts))

    # 嵌入所有批次
    all_embeddings = []

    async def process_batch(batch_info):
        batch_idx, batch_chunks, batch_texts = batch_info
        embeddings = await embed_batch(client, batch_texts, semaphore, MAX_RETRIES)
        return batch_idx, embeddings

    # 使用 tqdm 显示进度
    tasks = [process_batch(batch) for batch in batches]

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="嵌入进度"):
        batch_idx, embeddings = await task

        if embeddings:
            all_embeddings.extend([(batch_idx + i, emb) for i, emb in enumerate(embeddings)])
        else:
            # 嵌入失败，使用零向量
            batch_size = len(batches[batch_idx // BATCH_SIZE][1])
            embedding_dim = 1024  # text-embedding-v4 的维度
            for i in range(batch_size):
                all_embeddings.append((batch_idx + i, [0.0] * embedding_dim))

    # 按索引排序
    all_embeddings.sort(key=lambda x: x[0])
    embeddings_list = [emb for _, emb in all_embeddings]

    # 4. L2 归一化
    print("L2 归一化...")
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

    # 计算 L2 范数
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    # 避免除以零
    norms = np.maximum(norms, 1e-10)
    # 归一化
    embeddings_matrix = embeddings_matrix / norms

    # 5. 构建知识库
    print("构建知识库...")
    knowledge_base = {
        'embeddings': embeddings_matrix,
        'chunks': chunks,
        'metadata': {
            'total_chunks': len(chunks),
            'embedding_model': EMBEDDING_MODEL,
            'embedding_dim': embeddings_matrix.shape[1]
        }
    }

    # 6. 保存为 pickle
    print(f"保存到: {KB_PATH}")
    with open(KB_PATH, 'wb') as f:
        pickle.dump(knowledge_base, f)

    print(f"\n知识库构建完成!")
    print(f"入库条数: {len(chunks)}")
    print(f"向量维度: {embeddings_matrix.shape[1]}")

    # 7. 检索自测
    print("\n" + "=" * 60)
    print("检索自测: 「脾胃虚弱怎么辨证」")
    print("=" * 60)

    query = "脾胃虚弱怎么辨证"

    # 嵌入查询
    query_response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    query_embedding = np.array(query_response.data[0].embedding, dtype=np.float32)

    # L2 归一化
    query_norm = np.linalg.norm(query_embedding)
    query_embedding = query_embedding / max(query_norm, 1e-10)

    # 计算余弦相似度（点积）
    similarities = np.dot(embeddings_matrix, query_embedding)

    # 获取 top-3
    top_k = 3
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    print(f"\n查询: {query}")
    print(f"\nTop-{top_k} 结果:")

    for rank, idx in enumerate(top_indices, 1):
        chunk = chunks[idx]
        similarity = similarities[idx]

        print(f"\n[{rank}] 相似度: {similarity:.4f}")
        print(f"    来源: {chunk['txt_filename']}")
        print(f"    类型: {chunk['case_type']}")
        print(f"    块序号: {chunk['chunk_index'] + 1}/{chunk['total_chunks']}")
        print(f"    内容预览: {chunk['content'][:100]}...")


if __name__ == "__main__":
    asyncio.run(build_knowledge_base())
