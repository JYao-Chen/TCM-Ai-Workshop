# -*- coding: utf-8 -*-
# @Time : 2026/6/23
# @Author : Yao
# @File : app.py
# @Description : 中医医案 RAG 问答网页

import os
import pickle
import numpy as np
import streamlit as st
from openai import OpenAI


# ==================== 配置 ====================
KB_PATH = "./output/kb.pkl"
EMBEDDING_MODEL = "text-embedding-v4"

MODEL_CONFIGS = {
    "qwen3.7-plus": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "qwen3.7-plus",
    },
    "杏林": {
        "base_url": "https://ai.tcmcds.com/v1",
        "api_key_env": "XINLIN_API_KEY",
        "model": "XinLin",
    },
}

RETRIEVE_TOP_K = 5


# ==================== 工具函数 ====================
@st.cache_resource
def load_kb():
    """加载知识库"""
    with open(KB_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_client(model_name):
    """获取 OpenAI 客户端（按模型缓存）"""
    cfg = MODEL_CONFIGS[model_name]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        return None, f"未设置环境变量 {cfg['api_key_env']}"
    return OpenAI(api_key=api_key, base_url=cfg["base_url"], timeout=120.0), None


@st.cache_resource
def get_embedding_client():
    """获取嵌入专用客户端（始终使用通义百炼）"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return None, "未设置环境变量 DASHSCOPE_API_KEY"
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60.0
    ), None


def get_embedding(client, text):
    """获取文本嵌入向量"""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def retrieve(kb, query, embedding_client, k=RETRIEVE_TOP_K):
    """
    检索相关医案
    返回: [(chunk_dict, similarity_score), ...]
    """
    query_vec = get_embedding(embedding_client, query)
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 1e-10:
        query_vec = query_vec / query_norm

    # 点积 = 余弦相似度（向量已 L2 归一化）
    similarities = kb["embeddings"] @ query_vec
    top_indices = np.argsort(similarities)[-k:][::-1]

    results = []
    for idx in top_indices:
        results.append((kb["chunks"][idx], float(similarities[idx])))
    return results


def build_rag_prompt(results, question):
    """拼接 RAG prompt"""
    refs = []
    seen = set()
    for chunk, sim in results:
        key = (chunk["txt_filename"], chunk.get("content", ""))
        if key not in seen:
            seen.add(key)
            refs.append(
                f"【来源：{chunk['txt_filename']}】（{chunk.get('case_type', '')}）\n{chunk['content']}"
            )
    refs_text = "\n\n---\n\n".join(refs)

    prompt = f"""你是一位资深中医专家，需要根据提供的医案资料回答用户问题。

【参考资料】
{refs_text}

【回答要求】
1. 只依据上述参考资料作答，在观点句末标注【来源：文献名】
2. 若资料不足以回答，如实说明「现有资料不足以完整回答该问题」
3. 不编造资料中没有的信息
4. 保持中医专业术语的准确性

【用户问题】
{question}"""
    return prompt


# ==================== 侧边栏 ====================
def setup_sidebar():
    """配置侧边栏，返回 (model_name, rag_mode)"""
    st.sidebar.title("⚙️ 设置")

    # 模型选择（只显示模型名）
    model_name = st.sidebar.radio(
        "选择模型",
        list(MODEL_CONFIGS.keys()),
        key="model_name",
    )

    # 切换模型时清空对话历史
    if st.session_state.get("_last_model") != model_name:
        st.session_state.messages = []
        st.session_state.current_refs = []
        st.session_state._last_model = model_name

    # RAG 模式开关
    rag_mode = st.sidebar.toggle("RAG 模式（检索医案）", value=True, key="rag_mode")

    # 刷新连接按钮
    if st.sidebar.button("🔄 刷新连接"):
        get_client.clear()
        st.rerun()

    return model_name, rag_mode


# ==================== 引用栏 ====================
def show_references(rag_mode, refs=None):
    """右侧引用栏"""
    st.markdown("### 📚 引用文献")

    if not rag_mode:
        st.info("💬 直接回答模式\n\n使用模型自身知识回答，不检索医案库。")
        return

    if not refs:
        st.info("提出问题后将显示引用文献。")
        return

    for chunk, sim in refs:
        pct = f"{sim * 100:.1f}%"
        st.markdown(f"**{chunk['txt_filename']}**")
        st.markdown(f"*{chunk.get('case_type', '')}*")
        st.markdown(f"相似度：{pct}")
        st.progress(min(sim, 1.0))

        with st.expander("📖 查看全文"):
            st.markdown(chunk["content"])

        st.divider()


# ==================== 渲染历史消息 ====================
def render_history():
    """渲染已存储的对话历史"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if msg.get("has_thinking") and msg.get("reasoning"):
                    with st.expander("💭 思考过程", expanded=False):
                        st.markdown(
                            f'<div style="color: #888; font-size: 0.9em;">{msg["reasoning"]}</div>',
                            unsafe_allow_html=True,
                        )
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])


# ==================== 流式响应 ====================
def stream_response(client, model_name, messages, reasoning_ph, content_ph):
    """
    流式生成响应，实时展示思考过程和正式回答。
    返回 (full_reasoning, full_content)
    """
    cfg = MODEL_CONFIGS[model_name]

    stream = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        max_tokens=16000,
        temperature=0.7,
        stream=True,
        extra_body={"enable_thinking": True},
    )

    full_reasoning = ""
    full_content = ""
    thinking_expander = None
    thinking_placeholder = None  # 用于替换思考内容
    content_started = False

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        # 思考内容
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            full_reasoning += reasoning
            # 首次收到思考内容时创建 expander 和 placeholder
            if thinking_expander is None:
                thinking_expander = reasoning_ph.expander("💭 思考过程", expanded=True)
                thinking_placeholder = thinking_expander.empty()
            # 用 placeholder 替换内容（不是追加）
            thinking_placeholder.markdown(
                f'<div style="color: #888; font-size: 0.9em;">{full_reasoning}</div>',
                unsafe_allow_html=True,
            )

        # 正式回答内容
        content = delta.content
        if content:
            full_content += content
            # 首次收到正式内容时折叠思考 expander
            if not content_started:
                content_started = True
                if thinking_expander is not None:
                    # 替换为折叠状态的 expander
                    reasoning_ph.empty().expander("💭 思考过程", expanded=False).markdown(
                        f'<div style="color: #888; font-size: 0.9em;">{full_reasoning}</div>',
                        unsafe_allow_html=True,
                    )
            # 先清空再写入（避免追加）
            content_ph.empty().markdown(full_content + "▌")

    # 完成：去掉光标
    content_ph.empty().markdown(full_content)
    return full_reasoning, full_content


# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="中医医案 RAG 问答", page_icon="🏥", layout="wide")

    # 初始化 session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_refs" not in st.session_state:
        st.session_state.current_refs = []

    # 侧边栏
    model_name, rag_mode = setup_sidebar()

    # 标题栏显示当前模式
    mode_text = "🔍 RAG 模式" if rag_mode else "💬 直接回答"
    st.title(f"🏥 中医医案 RAG 问答 {mode_text}")

    # 主布局：左侧对话 3/4，右侧引用 1/4
    chat_col, ref_col = st.columns([3, 1])

    # ---- 渲染历史对话 ----
    with chat_col:
        render_history()

    # ---- 用户输入 ----
    if prompt := st.chat_input("请输入您的问题…"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_col:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 获取客户端
        client, err = get_client(model_name)
        if err:
            with chat_col:
                with st.chat_message("assistant"):
                    st.error(f"❌ {err}")
            st.stop()

        # RAG 检索
        refs = []
        if rag_mode:
            try:
                # 获取嵌入专用客户端（始终使用通义百炼）
                embedding_client, emb_err = get_embedding_client()
                if emb_err:
                    with chat_col:
                        with st.chat_message("assistant"):
                            st.error(f"❌ {emb_err}")
                    st.stop()

                kb = load_kb()
                refs = retrieve(kb, prompt, embedding_client, k=RETRIEVE_TOP_K)
                user_prompt = build_rag_prompt(refs, prompt)
            except Exception as e:
                with chat_col:
                    with st.chat_message("assistant"):
                        st.error(f"❌ 检索失败：{e}")
                st.stop()
        else:
            user_prompt = prompt

        st.session_state.current_refs = refs

        # 构建 API 消息
        api_messages = []
        for msg in st.session_state.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        # 最后一条用 RAG 拼接后的 prompt 替换
        api_messages[-1]["content"] = user_prompt

        # 流式展示
        with chat_col:
            with st.chat_message("assistant"):
                reasoning_ph = st.empty()
                content_ph = st.empty()

                try:
                    full_reasoning, full_content = stream_response(
                        client, model_name, api_messages, reasoning_ph, content_ph
                    )
                except Exception as e:
                    content_ph.error(f"❌ 生成失败：{e}")
                    st.stop()

        # 保存到历史
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_content,
                "reasoning": full_reasoning,
                "has_thinking": bool(full_reasoning),
            }
        )
        st.rerun()

    # ---- 引用栏（右侧）----
    with ref_col:
        show_references(rag_mode, st.session_state.get("current_refs", []))


if __name__ == "__main__":
    main()
