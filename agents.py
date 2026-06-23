# -*- coding: utf-8 -*-
# @Time : 2026/6/23
# @Author : Yao
# @File : agents.py
# @Description : 中医辨证多智能体系统（命令行版）

import os
import re
import json
import pickle
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any
import numpy as np
from openai import AsyncOpenAI


# ==================== 配置 ====================
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
MODEL_NAME = "qwen3.6-flash"
EMBEDDING_MODEL = "text-embedding-v4"
KB_PATH = "./output/kb.pkl"


# ==================== 日志系统 ====================
@dataclass
class LogEntry:
    round_num: int
    agent_name: str
    action: str
    input_data: Any
    output_data: Any
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class AgentLogger:
    """Agent 日志系统"""

    def __init__(self):
        self.entries: list[LogEntry] = []

    def log(self, round_num: int, agent_name: str, action: str, input_data: Any, output_data: Any):
        entry = LogEntry(
            round_num=round_num,
            agent_name=agent_name,
            action=action,
            input_data=input_data,
            output_data=output_data
        )
        self.entries.append(entry)

    def display(self):
        """打印时间线"""
        print("\n" + "=" * 80)
        print("📋 协作轨迹时间线")
        print("=" * 80)

        for entry in self.entries:
            print(f"\n[{entry.timestamp}] 轮次{entry.round_num} | {entry.agent_name} | {entry.action}")
            print(f"  输入: {json.dumps(entry.input_data, ensure_ascii=False)[:200]}...")
            print(f"  输出: {json.dumps(entry.output_data, ensure_ascii=False)[:200]}...")

        print("\n" + "=" * 80)


# ==================== 知识库 ====================
class KnowledgeBase:
    """向量知识库"""

    def __init__(self, kb_path: str = KB_PATH):
        with open(kb_path, "rb") as f:
            kb_data = pickle.load(f)
        self.embeddings = kb_data["embeddings"]
        self.chunks = kb_data["chunks"]
        self.client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=60.0
        )

    async def search(self, query: str, top_k: int = 3) -> list:
        """检索相似医案"""
        # 嵌入查询
        resp = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_vec = np.array(resp.data[0].embedding, dtype=np.float32)

        # L2 归一化
        norm = np.linalg.norm(query_vec)
        if norm > 1e-10:
            query_vec = query_vec / norm

        # 计算余弦相似度
        similarities = self.embeddings @ query_vec
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "source": chunk["txt_filename"],
                "similarity": float(similarities[idx]),
                "content": chunk["content"],
                "case_type": chunk.get("case_type", "")
            })
        return results


# ==================== Agent 基类 ====================
class BaseAgent:
    """Agent 基类"""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=120.0
        )

    def _parse_json(self, text: str) -> dict:
        """从 LLM 输出中提取 JSON"""
        # 1. 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass

        # 2. 提取 ```json``` 块
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        # 3. 提取 {} 块
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        return {"error": "无法解析 JSON", "raw": text}

    async def run(self, msg: str, stream_callback: Optional[Callable] = None) -> dict:
        """调用 LLM"""
        self.messages.append({"role": "user", "content": msg})

        full_reasoning = ""
        full_content = ""

        stream = await self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.messages,
            max_tokens=8000,
            temperature=0.3,
            stream=True,
            extra_body={"enable_thinking": True}
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # 思考内容
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                full_reasoning += reasoning
                if stream_callback:
                    await stream_callback("reasoning", reasoning, full_reasoning, full_content)

            # 正式内容
            content = delta.content
            if content:
                full_content += content
                if stream_callback:
                    await stream_callback("content", content, full_reasoning, full_content)

        self.messages.append({"role": "assistant", "content": full_content})
        return self._parse_json(full_content)


# ==================== 专业 Agent ====================
class RetrievalAgent(BaseAgent):
    """检索 Agent"""

    def __init__(self):
        system_prompt = """你是一位中医文献检索专家。根据病例信息，从相似医案中提炼关键信息。

输出严格 JSON 格式：
{
    "query": "提取的关键症状",
    "similar_cases": [
        {
            "source": "文献来源",
            "similarity": 0.85,
            "content_summary": "医案摘要",
            "relevance": "与本病例的关联性说明"
        }
    ],
    "reference_summary": "综合参考意见"
}"""
        super().__init__("🔍 检索Agent", system_prompt)

    async def run(self, case: str, similar_cases: list, stream_callback: Optional[Callable] = None) -> dict:
        cases_text = "\n".join([
            f"- {c['source']}（相似度{c['similarity']:.2f}）：{c['content'][:200]}..."
            for c in similar_cases
        ])
        msg = f"""病例：
{case}

检索到的相似医案：
{cases_text}

请分析并输出 JSON。"""
        return await super().run(msg, stream_callback)


class DiagnosisAgent(BaseAgent):
    """辨证 Agent"""

    def __init__(self):
        system_prompt = """你是一位资深中医辨证专家。根据病例和参考资料进行中医辨证分析。

输出严格 JSON 格式：
{
    "symptoms_analysis": "症状分析",
    "etiology": "病因",
    "pathogenesis": "病机",
    "disease_location": "病位",
    "syndrome": "证型",
    "evidence": ["辨证依据1", "辨证依据2"]
}"""
        super().__init__("🩺 辨证Agent", system_prompt)

    async def run(self, case: str, reference: str, stream_callback: Optional[Callable] = None) -> dict:
        msg = f"""病例：
{case}

参考资料：
{reference}

请进行中医辨证分析并输出 JSON。"""
        return await super().run(msg, stream_callback)


class PrescriptionAgent(BaseAgent):
    """处方 Agent"""

    def __init__(self):
        system_prompt = """你是一位中医处方专家。根据辨证结果制定治疗方案。

输出严格 JSON 格式：
{
    "treatment_principle": "治法",
    "formula_name": "方剂名称",
    "herbs": [
        {"name": "药名", "dosage": "剂量", "function": "功效"}
    ],
    "contraindications_check": "配伍禁忌检查",
    "formula_explanation": "方解"
}"""
        super().__init__("💊 处方Agent", system_prompt)

    async def run(self, diagnosis: dict, review_feedback: Optional[str] = None, stream_callback: Optional[Callable] = None) -> dict:
        msg = f"""辨证结果：
- 证型：{diagnosis.get('syndrome', '')}
- 病机：{diagnosis.get('pathogenesis', '')}
- 病位：{diagnosis.get('disease_location', '')}"""

        if review_feedback:
            msg += f"\n\n审校意见：{review_feedback}"

        msg += "\n\n请制定治疗方案并输出 JSON。"
        return await super().run(msg, stream_callback)


class ReviewAgent(BaseAgent):
    """审校 Agent"""

    def __init__(self):
        system_prompt = """你是一位中医审校专家。审核辨证和处方方案的合理性。

输出严格 JSON 格式：
{
    "passed": true/false,
    "consistency_check": "辨证与处方一致性检查",
    "safety_check": "安全性检查",
    "issues": ["问题1", "问题2"],
    "suggestions": "改进建议",
    "final_verdict": "最终结论"
}"""
        super().__init__("✅ 审校Agent", system_prompt)

    async def run(self, diagnosis: dict, prescription: dict, stream_callback: Optional[Callable] = None) -> dict:
        msg = f"""辨证结果：
{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

处方方案：
{json.dumps(prescription, ensure_ascii=False, indent=2)}

请审核并输出 JSON。"""
        return await super().run(msg, stream_callback)


# ==================== 编排器 ====================
class Orchestrator:
    """多智能体编排器"""

    def __init__(self, kb: KnowledgeBase, max_rounds: int = 2):
        self.kb = kb
        self.max_rounds = max_rounds
        self.logger = AgentLogger()
        self.retrieval_agent = RetrievalAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.prescription_agent = PrescriptionAgent()
        self.review_agent = ReviewAgent()

    async def orchestrate(self, case: str, stream_callback: Optional[Callable] = None) -> dict:
        """编排执行"""
        result = {"rounds": [], "final": {}}

        # 第 1 轮：检索 + 辨证 + 处方 + 审校
        for round_num in range(1, self.max_rounds + 1):
            round_result = {"round": round_num}

            # 1. 检索
            similar_cases = await self.kb.search(case, top_k=3)
            retrieval_input = {"case": case, "similar_cases": len(similar_cases)}
            retrieval_result = await self.retrieval_agent.run(case, similar_cases, stream_callback)
            self.logger.log(round_num, "🔍 检索Agent", "检索相似医案", retrieval_input, retrieval_result)
            round_result["retrieval"] = retrieval_result

            # 2. 辨证
            diagnosis_input = {"case": case, "reference": retrieval_result.get("reference_summary", "")}
            diagnosis_result = await self.diagnosis_agent.run(case, retrieval_result.get("reference_summary", ""), stream_callback)
            self.logger.log(round_num, "🩺 辨证Agent", "中医辨证", diagnosis_input, diagnosis_result)
            round_result["diagnosis"] = diagnosis_result

            # 3. 处方
            review_feedback = None
            if round_num > 1:
                review_feedback = result["rounds"][-1]["review"].get("suggestions", "")

            prescription_input = {"diagnosis": diagnosis_result, "review_feedback": review_feedback}
            prescription_result = await self.prescription_agent.run(diagnosis_result, review_feedback, stream_callback)
            self.logger.log(round_num, "💊 处方Agent", "制定处方", prescription_input, prescription_result)
            round_result["prescription"] = prescription_result

            # 4. 审校
            review_input = {"diagnosis": diagnosis_result, "prescription": prescription_result}
            review_result = await self.review_agent.run(diagnosis_result, prescription_result, stream_callback)
            self.logger.log(round_num, "✅ 审校Agent", "审校方案", review_input, review_result)
            round_result["review"] = review_result

            result["rounds"].append(round_result)

            # 如果审校通过，结束
            if review_result.get("passed", False):
                result["final"] = {
                    "retrieval": retrieval_result,
                    "diagnosis": diagnosis_result,
                    "prescription": prescription_result,
                    "review": review_result,
                    "total_rounds": round_num
                }
                break

        # 如果所有轮次都未通过，使用最后一轮结果
        if not result["final"]:
            last_round = result["rounds"][-1]
            result["final"] = {
                "retrieval": last_round["retrieval"],
                "diagnosis": last_round["diagnosis"],
                "prescription": last_round["prescription"],
                "review": last_round["review"],
                "total_rounds": len(result["rounds"])
            }

        return result


# ==================== 主程序 ====================
async def main():
    """命令行主程序"""
    if not DASHSCOPE_API_KEY:
        print("错误：未设置环境变量 DASHSCOPE_API_KEY")
        return

    print("=" * 80)
    print("🏥 中医辨证多智能体系统")
    print("=" * 80)

    # 示例病例
    case = """患者张某，男，45岁，公司职员。
主诉：胁肋胀痛反复发作3月，加重1周。
现病史：患者3月前因工作压力大出现胁肋部胀痛，情绪抑郁时加重，嗳气频作，纳食减少。近1周因琐事恼怒，症状明显加重，胁肋胀痛呈窜痛，胸闷不舒，善太息，食欲不振，大便不调。
刻下症：胁肋胀痛，窜痛不定，情绪抑郁，胸闷善太息，纳呆，大便偏溏，日行2次。
舌脉：舌淡红，苔薄白，脉弦。"""

    print(f"\n📋 病例：\n{case}\n")

    # 初始化
    kb = KnowledgeBase()
    orchestrator = Orchestrator(kb, max_rounds=2)

    # 流式回调
    async def stream_callback(msg_type, chunk, full_reasoning, full_content):
        if msg_type == "reasoning":
            print(".", end="", flush=True)
        else:
            pass

    # 执行编排
    print("🚀 开始辨证论治...\n")
    result = await orchestrator.orchestrate(case, stream_callback)

    # 打印日志
    orchestrator.logger.display()

    # 打印最终结果
    print("\n" + "=" * 80)
    print("📊 最终结果")
    print("=" * 80)
    print(json.dumps(result["final"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
