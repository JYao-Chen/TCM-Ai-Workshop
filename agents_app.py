# -*- coding: utf-8 -*-
# @Time : 2026/6/23
# @Author : Yao
# @File : agents_app.py
# @Description : 中医辨证多智能体系统（Streamlit 可视化前端 - 流式输出版）

import os
import re
import json
import pickle
import numpy as np
import streamlit as st
from typing import Optional, Callable
from openai import OpenAI


# ==================== 配置 ====================
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
MODEL_NAME = "qwen3.6-flash"
EMBEDDING_MODEL = "text-embedding-v4"
KB_PATH = "./output/kb.pkl"


# ==================== 预设病例 ====================
PRESET_CASES = {
    "肝郁气滞（胁肋胀痛）": """患者张某，男，45岁，公司职员。
主诉：胁肋胀痛反复发作3月，加重1周。
现病史：患者3月前因工作压力大出现胁肋部胀痛，情绪抑郁时加重，嗳气频作，纳食减少。近1周因琐事恼怒，症状明显加重，胁肋胀痛呈窜痛，胸闷不舒，善太息，食欲不振，大便不调。
刻下症：胁肋胀痛，窜痛不定，情绪抑郁，胸闷善太息，纳呆，大便偏溏，日行2次。
舌脉：舌淡红，苔薄白，脉弦。""",

    "面瘫/口眼歪斜": """患者王某，女，52岁，退休教师。
主诉：右侧面部口眼歪斜20余日。
现病史：患者发病第3天至外院门诊就诊，予激素、维生素B1、甲钴胺等口服，连续服用20余日效果不佳，期间曾行针刺治疗，均未见明显好转。
刻下症：右侧额纹变浅，右侧眼睑用力可基本闭合，白睛稍红，右侧饮食夹食，嘴角向左侧喎斜，心中烦闷，自觉头闷沉重感，口干，纳可，寐欠安，二便调。
既往史：有原发性高血压、高脂血症等病史。
舌脉：舌红苔黄腻，脉细滑。""",

    "糖尿病肾病": """患者李某，男，58岁，退休工人。
主诉：发现血糖升高10年，泡沫尿半年。
现病史：患者10年前确诊2型糖尿病，长期口服降糖药治疗，血糖控制尚可。半年前出现尿液泡沫增多，未予重视。近1月出现双下肢轻度浮肿，乏力明显，口干多饮，夜尿2-3次/夜。
刻下症：口干多饮，乏力倦怠，双下肢轻度浮肿，腰膝酸软，夜尿频多，纳寐尚可。
辅助检查：尿常规示PRO 2+，24h尿蛋白定量0.8g。
舌脉：舌胖暗，苔薄白，脉沉细。""",

    "慢性肺源性心脏病": """患者赵某，男，78岁，退休干部。
主诉：反复胸闷气喘20年，再发伴下肢水肿半月。
现病史：患者20年前开始出现咳喘，每在秋冬季节或气候变化时加重，期间曾多次住院治疗，诊断为慢性阻塞性肺疾病、肺源性心脏病。半月前受凉后症状再发，胸闷气喘明显，动则加重，伴双下肢水肿。
刻下症：胸闷气喘，动则尤甚，咳嗽痰多，色白质稀，畏寒肢冷，双下肢凹陷性水肿，小便量少，纳差。
既往史：慢性阻塞性肺疾病病史20年，冠心病史10年。
舌脉：舌淡胖，苔白滑，脉沉细。"""
}


# ==================== 知识库 ====================
@st.cache_resource
def load_kb():
    with open(KB_PATH, "rb") as f:
        return pickle.load(f)


def search_kb(query: str, top_k: int = 3) -> list:
    """检索相似医案（同步版本）"""
    kb_data = load_kb()
    embeddings = kb_data["embeddings"]
    chunks = kb_data["chunks"]

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        timeout=60.0
    )
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 1e-10:
        query_vec = query_vec / norm

    similarities = embeddings @ query_vec
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "source": chunk["txt_filename"],
            "similarity": float(similarities[idx]),
            "content": chunk["content"],
            "case_type": chunk.get("case_type", "")
        })
    return results


# ==================== JSON 解析 ====================
def parse_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON"""
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return {"error": "无法解析 JSON", "raw": text}


# ==================== Agent 基类（同步流式） ====================
class BaseAgent:
    """Agent 基类 - 同步流式版本"""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=120.0
        )

    def run(self, msg: str, status_container=None, reasoning_placeholder=None, content_placeholder=None) -> dict:
        """
        同步流式调用 LLM
        status_container: st.status 容器，用于控制折叠
        reasoning_placeholder: 用于显示思考过程的 st.empty() 占位符
        content_placeholder: 用于显示正式内容的 st.empty() 占位符
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg}
        ]

        stream = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=8000,
            temperature=0.3,
            stream=True,
            extra_body={"enable_thinking": True}
        )

        full_reasoning = ""
        full_content = ""
        content_started = False

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # 思考内容
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                full_reasoning += reasoning
                if reasoning_placeholder:
                    reasoning_placeholder.markdown(
                        f'<div style="color: #888; font-size: 0.9em;">{full_reasoning}▌</div>',
                        unsafe_allow_html=True
                    )

            # 正式内容
            content = delta.content
            if content:
                full_content += content
                if not content_started:
                    content_started = True
                    # 首次收到正式内容时，折叠 status 容器
                    if status_container:
                        status_container.update(expanded=False)
                    # 思考内容去掉光标
                    if reasoning_placeholder and full_reasoning:
                        reasoning_placeholder.markdown(
                            f'<div style="color: #888; font-size: 0.9em;">{full_reasoning}</div>',
                            unsafe_allow_html=True
                        )

                if content_placeholder:
                    content_placeholder.markdown(full_content + "▌")

        # 完成：去掉光标
        if content_placeholder:
            content_placeholder.markdown(full_content)

        return parse_json(full_content)


# ==================== 专业 Agent ====================
class RetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__("🔍 检索Agent", """你是一位中医文献检索专家。根据病例信息，从相似医案中提炼关键信息。

输出严格 JSON 格式：
{
    "query": "提取的关键症状",
    "similar_cases": [
        {"source": "文献来源", "similarity": 0.85, "content_summary": "医案摘要", "relevance": "关联性说明"}
    ],
    "reference_summary": "综合参考意见"
}""")

    def run(self, case: str, similar_cases: list, status_container=None, reasoning_ph=None, content_ph=None):
        cases_text = "\n".join([f"- {c['source']}（{c['similarity']:.2f}）：{c['content'][:200]}..." for c in similar_cases])
        return super().run(f"病例：\n{case}\n\n相似医案：\n{cases_text}\n\n请分析并输出 JSON。", status_container, reasoning_ph, content_ph)


class DiagnosisAgent(BaseAgent):
    def __init__(self):
        super().__init__("🩺 辨证Agent", """你是一位资深中医辨证专家。根据病例和参考资料进行中医辨证分析。

输出严格 JSON 格式：
{
    "symptoms_analysis": "症状分析",
    "etiology": "病因",
    "pathogenesis": "病机",
    "disease_location": "病位",
    "syndrome": "证型",
    "evidence": ["辨证依据1", "辨证依据2"]
}""")

    def run(self, case: str, reference: str, status_container=None, reasoning_ph=None, content_ph=None):
        return super().run(f"病例：\n{case}\n\n参考资料：\n{reference}\n\n请进行中医辨证分析并输出 JSON。", status_container, reasoning_ph, content_ph)


class PrescriptionAgent(BaseAgent):
    def __init__(self):
        super().__init__("💊 处方Agent", """你是一位中医处方专家。根据辨证结果制定治疗方案。

输出严格 JSON 格式：
{
    "treatment_principle": "治法",
    "formula_name": "方剂名称",
    "herbs": [{"name": "药名", "dosage": "剂量", "function": "功效"}],
    "contraindications_check": "配伍禁忌检查",
    "formula_explanation": "方解"
}""")

    def run(self, diagnosis: dict, review_feedback: Optional[str] = None, status_container=None, reasoning_ph=None, content_ph=None):
        msg = f"""辨证结果：
- 证型：{diagnosis.get('syndrome', '')}
- 病机：{diagnosis.get('pathogenesis', '')}
- 病位：{diagnosis.get('disease_location', '')}"""
        if review_feedback:
            msg += f"\n\n审校意见：{review_feedback}"
        msg += "\n\n请制定治疗方案并输出 JSON。"
        return super().run(msg, status_container, reasoning_ph, content_ph)


class ReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("✅ 审校Agent", """你是一位中医审校专家。审核辨证和处方方案的合理性。

输出严格 JSON 格式：
{
    "passed": true/false,
    "consistency_check": "一致性检查",
    "safety_check": "安全性检查",
    "issues": ["问题1"],
    "suggestions": "改进建议",
    "final_verdict": "最终结论"
}""")

    def run(self, diagnosis: dict, prescription: dict, status_container=None, reasoning_ph=None, content_ph=None):
        msg = f"辨证结果：\n{json.dumps(diagnosis, ensure_ascii=False, indent=2)}\n\n处方方案：\n{json.dumps(prescription, ensure_ascii=False, indent=2)}\n\n请审核并输出 JSON。"
        return super().run(msg, status_container, reasoning_ph, content_ph)


# ==================== 编排器 ====================
class Orchestrator:
    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds
        self.retrieval_agent = RetrievalAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.prescription_agent = PrescriptionAgent()
        self.review_agent = ReviewAgent()

    def orchestrate(self, case: str, progress_callback=None) -> dict:
        """
        编排执行
        progress_callback: callable(agent_name, status, round_num) -> (status_container, reasoning_ph, content_ph)
        """
        result = {"rounds": [], "final": {}}

        for round_num in range(1, self.max_rounds + 1):
            round_result = {"round": round_num}

            # 1. 检索
            similar_cases = search_kb(case, top_k=3)
            if progress_callback:
                status_container, reasoning_ph, content_ph = progress_callback("🔍 检索Agent", "running", round_num)
            else:
                status_container, reasoning_ph, content_ph = None, None, None
            round_result["retrieval"] = self.retrieval_agent.run(case, similar_cases, status_container, reasoning_ph, content_ph)
            if progress_callback:
                progress_callback("🔍 检索Agent", "complete", round_num)

            # 2. 辨证
            if progress_callback:
                status_container, reasoning_ph, content_ph = progress_callback("🩺 辨证Agent", "running", round_num)
            else:
                status_container, reasoning_ph, content_ph = None, None, None
            round_result["diagnosis"] = self.diagnosis_agent.run(
                case, round_result["retrieval"].get("reference_summary", ""), status_container, reasoning_ph, content_ph
            )
            if progress_callback:
                progress_callback("🩺 辨证Agent", "complete", round_num)

            # 3. 处方
            review_feedback = result["rounds"][-1]["review"].get("suggestions", "") if round_num > 1 else None
            if progress_callback:
                status_container, reasoning_ph, content_ph = progress_callback("💊 处方Agent", "running", round_num)
            else:
                status_container, reasoning_ph, content_ph = None, None, None
            round_result["prescription"] = self.prescription_agent.run(
                round_result["diagnosis"], review_feedback, status_container, reasoning_ph, content_ph
            )
            if progress_callback:
                progress_callback("💊 处方Agent", "complete", round_num)

            # 4. 审校
            if progress_callback:
                status_container, reasoning_ph, content_ph = progress_callback("✅ 审校Agent", "running", round_num)
            else:
                status_container, reasoning_ph, content_ph = None, None, None
            round_result["review"] = self.review_agent.run(
                round_result["diagnosis"], round_result["prescription"], status_container, reasoning_ph, content_ph
            )
            if progress_callback:
                progress_callback("✅ 审校Agent", "complete", round_num)

            result["rounds"].append(round_result)

            # 如果审校通过，结束
            if round_result["review"].get("passed", False):
                result["final"] = {**round_result, "total_rounds": round_num}
                break

        # 如果所有轮次都未通过，使用最后一轮结果
        if not result["final"]:
            result["final"] = {**result["rounds"][-1], "total_rounds": len(result["rounds"])}

        return result


# ==================== 结果展示 ====================
def display_retrieval(result: dict):
    st.markdown(f"**关键症状**：{result.get('query', '')}")
    st.markdown("**相似医案**：")
    for case in result.get("similar_cases", []):
        st.markdown(f"- {case.get('source', '')}（相似度 {case.get('similarity', 0):.2f}）")
        st.markdown(f"  - 摘要：{case.get('content_summary', '')}")
        st.markdown(f"  - 关联：{case.get('relevance', '')}")
    st.markdown(f"**参考意见**：{result.get('reference_summary', '')}")


def display_diagnosis(result: dict):
    st.markdown(f"**症状分析**：{result.get('symptoms_analysis', '')}")
    st.markdown(f"**病因**：{result.get('etiology', '')}")
    st.markdown(f"**病机**：{result.get('pathogenesis', '')}")
    st.markdown(f"**病位**：{result.get('disease_location', '')}")
    st.markdown(f"🔴 **证型**：**:red[{result.get('syndrome', '')}]**")
    with st.expander("📋 辨证依据"):
        for ev in result.get("evidence", []):
            st.markdown(f"- {ev}")


def display_prescription(result: dict):
    st.markdown(f"**治法**：{result.get('treatment_principle', '')}")
    st.markdown(f"🟢 **方剂**：**:green[{result.get('formula_name', '')}]**")
    st.markdown("**药物组成**：")
    for herb in result.get("herbs", []):
        st.markdown(f"- {herb.get('name', '')} {herb.get('dosage', '')}（{herb.get('function', '')}）")
    st.markdown(f"**配伍禁忌**：{result.get('contraindications_check', '')}")
    with st.expander("📖 方解"):
        st.markdown(result.get("formula_explanation", ""))


def display_review(result: dict):
    passed = result.get("passed", False)
    st.markdown(f"{'✅ **审校通过**' if passed else '❌ **审校未通过**'}")
    st.markdown(f"**一致性检查**：{result.get('consistency_check', '')}")
    st.markdown(f"**安全性检查**：{result.get('safety_check', '')}")
    with st.expander("⚠️ 问题与警告"):
        for issue in result.get("issues", []):
            st.warning(issue)
    with st.expander("💡 建议"):
        st.markdown(result.get("suggestions", ""))
    st.markdown(f"**最终结论**：{result.get('final_verdict', '')}")


# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="中医辨证多智能体系统", page_icon="🏥", layout="wide")
    st.title("🏥 中医辨证多智能体系统")

    # 初始化 session_state
    if "result" not in st.session_state:
        st.session_state.result = None
    if "running" not in st.session_state:
        st.session_state.running = False

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        st.markdown(f"**模型**：{MODEL_NAME}")
        max_rounds = st.slider("最大轮次", 1, 3, 2)
        st.divider()
        st.header("🤖 Agent 角色")
        st.markdown("🔍 **检索Agent**：检索相似医案")
        st.markdown("🩺 **辨证Agent**：中医辨证分析")
        st.markdown("💊 **处方Agent**：制定治疗方案")
        st.markdown("✅ **审校Agent**：审核方案合理性")

    # 主布局
    left_col, right_col = st.columns([1, 1])

    # 左侧：病例输入
    with left_col:
        st.subheader("📋 病例输入")

        # 预设病例按钮
        btn_cols = st.columns(2)
        case_keys = list(PRESET_CASES.keys())
        for i, col in enumerate(btn_cols):
            for j in range(2):
                idx = i * 2 + j
                if idx < len(case_keys):
                    key = case_keys[idx]
                    if col.button(f"📋 {key}", key=f"preset_{idx}"):
                        st.session_state.case_input = PRESET_CASES[key]

        case_input = st.text_area("病例内容", value=st.session_state.get("case_input", ""), height=300)

        if st.button("🚀 开始辨证论治", disabled=not case_input.strip() or st.session_state.running, type="primary", use_container_width=True):
            st.session_state.case_input = case_input
            st.session_state.running = True
            st.rerun()

    # 右侧：协作过程（流式显示）
    with right_col:
        st.subheader("🔄 协作过程")

        if st.session_state.running:
            case_input = st.session_state.case_input

            # 创建状态容器（用于流式显示）
            status_containers = {}
            placeholders = {}

            for agent_name in ["🔍 检索Agent", "🩺 辨证Agent", "💊 处方Agent", "✅ 审校Agent"]:
                status_containers[agent_name] = st.container()
                placeholders[agent_name] = {
                    "status_container": None,
                    "reasoning": None,
                    "content": None
                }

            st.info("🚀 正在执行辨证论治流程...")

            # 进度回调函数
            def progress_callback(agent_name: str, status: str, round_num: int = 1):
                """在流式过程中更新显示，返回 (status_container, reasoning_ph, content_ph)"""
                if status == "running":
                    # 创建 status 容器（自动展开，思考时可见）
                    status_container = st.status(f"{agent_name}（轮次{round_num}）思考中...", expanded=True)
                    with status_container:
                        reasoning_ph = st.empty()
                        content_ph = st.empty()
                    placeholders[agent_name]["status_container"] = status_container
                    placeholders[agent_name]["reasoning"] = reasoning_ph
                    placeholders[agent_name]["content"] = content_ph
                    return status_container, reasoning_ph, content_ph

                elif status == "complete":
                    # 完成后标记完成（status 已在 Agent 内部自动折叠）
                    status_obj = placeholders[agent_name].get("status_container")
                    if status_obj:
                        status_obj.update(label=f"✅ {agent_name}（轮次{round_num}）完成", state="complete")
                    return None, None, None

            # 执行编排
            orchestrator = Orchestrator(max_rounds=max_rounds)
            result = orchestrator.orchestrate(case_input, progress_callback)

            st.session_state.result = result
            st.session_state.running = False
            st.rerun()

        elif st.session_state.result:
            st.success("✅ 辨证论治完成！")

    # 下方结果区
    if st.session_state.result:
        st.divider()
        st.subheader("📊 辨证论治结果")

        final = st.session_state.result["final"]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🩺 辨证")
            display_diagnosis(final["diagnosis"])

        with col2:
            st.markdown("### 💊 处方")
            display_prescription(final["prescription"])

        with col3:
            st.markdown("### ✅ 审校")
            display_review(final["review"])

        st.divider()

        # 检索到的相似医案
        with st.expander("🔍 检索到的相似医案"):
            display_retrieval(final["retrieval"])

        st.markdown(f"**协作轮次**：{final.get('total_rounds', 1)} 轮")


if __name__ == "__main__":
    main()
