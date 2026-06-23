# Claude Code 现场 Demo 提示词（直接复制）

第六章「用 Claude Code 动手做」的三步演示，所有提示词都可直接粘进 Claude Code。
数据：`./data` 下约 100 篇中医期刊 OCR txt；参考脚本：`./data/case_creat_openai_async.py`。
模型（通义百炼，OpenAI 兼容）：抽取/多智能体用 `qwen3.6-flash`，问答用 `qwen3.7-plus`，嵌入用 `text-embedding-v4`。
向量库：**不装任何向量数据库**，用 `numpy` 把向量存成本地 `kb.pkl`（pickle 一个 dict），检索靠点积算余弦相似度——零配置、最适合现场演示。

---

## 第零步 · 新建 venv 环境（一次性）

可以直接让 Claude Code 用中文做：

> 帮我在当前目录新建一个 Python 虚拟环境：用 `python -m venv .venv` 创建并激活它，然后 pip 升级后安装本次演示需要的包：`openai`、`numpy`、`pandas`、`streamlit`、`tqdm`（不要装 chromadb / faiss 等向量数据库）；装完用 `pip list` 确认，并把 `.venv` 写进 `.gitignore`。

或自己在终端敲（等价）：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U openai numpy pandas streamlit tqdm
```

环境变量先设好：

```bash
export DASHSCOPE_API_KEY="<你的百炼 key>"
# base_url 统一用：https://dashscope.aliyuncs.com/compatible-mode/v1
```

---

## 第一步 · 期刊 txt → 抽医案 → CSV　（extract_cases.py）

> 我在 ./data 下有约 100 篇中医期刊文章的 OCR txt。参考同目录 case_creat_openai_async.py 的结构，用 Python 写 extract_cases.py（异步并发 + 重试 + JSONL 断点续传，只用 openai SDK + tqdm + 标准库，不要框架）：
>
> ① 用通义百炼 OpenAI 兼容接口：base_url=https://dashscope.aliyuncs.com/compatible-mode/v1、api_key 读环境变量 DASHSCOPE_API_KEY、模型 qwen3.6-flash；
> ② 递归读 ./data 下所有 .txt（utf-8，失败回退 gb18030）；预清洗：全角转半角、去 OCR 多余空白、合并断行，但不改动中医术语原文；
> ③ 抽取提示词照搬 case_creat_openai_async.py 里那套：判断有无医案，按「医案N类型 / 医案N内容」格式抽出完整医案，区分 现代 / 古代 / 名医医案，多则用 # 分隔，无则输出「无」；
> ④ 用 asyncio.Semaphore 控并发（设为 30）、失败重试 3 次，结果先写 JSONL 以支持断点续传；用 tqdm 显示处理进度条（已完成 / 总数、实时速率）；
> ⑤ 解析后导出 medical_cases.csv，列为 id, custom_id, txt_filename, 医案内容, 医案类型；最后打印共抽出多少则、各类型计数、解析失败数。
>
> 用前 5 篇跑通，再放开全部。

产出：`medical_cases.csv`（每行一则医案）。

---

## 穿插 · 用 git 存档（对 Claude 用中文说即可）

> 把当前目录初始化成 git 仓库，写个 .gitignore 忽略 __pycache__ / .env / .venv / kb.pkl / 大数据文件，然后做第一次提交。

> 抽完医案了：git status 看看改了什么，把「期刊抽医案脚本 + medical_cases.csv」用一句中文提交信息提交。

> 搭完问答前端了：再提交一次，提交信息说明新增了 RAG 问答前端。

> 这一步搞砸了：帮我回到上一个 commit 的状态。

---

## 第二步之一 · 医案嵌入入库　（build_kb.py）

> 用 Python 写 build_kb.py，把 medical_cases.csv 做成本地向量库（只用 openai SDK + numpy + tqdm + 标准库，不装 chromadb / faiss 等任何向量数据库）：
>
> ① 读 medical_cases.csv，跳过「医案内容」为「无」或「解析错误」的行；
> ② 每则医案为一条；过长的（>800 字）按语义切成带重叠的块，保留 txt_filename / 医案类型 / 块序号作元数据；
> ③ 用通义百炼嵌入：base_url 同上、api_key=DASHSCOPE_API_KEY、模型 text-embedding-v4，每批 10 条批量嵌入，并发设为 8、带重试；用 tqdm 显示嵌入进度条；
> ④ 把所有向量堆成一个 numpy 矩阵（先做 L2 归一化，这样检索时点积就是余弦相似度），连同原文与元数据打包成一个 dict，用 pickle 存成本地文件 kb.pkl；
> ⑤ 跑完打印入库条数，并用「脾胃虚弱怎么辨证」做一次检索自测，打印 top-3 的来源与相似度。
>
> 先确认切块与字段方案，再写、再跑通。

产出：本地向量库文件 `kb.pkl`（问答与多智能体共用，`pickle.load` 即可读，无需起服务）。

---

## 第二步之二 · RAG 对话前端　（app.py，streamlit run app.py）

要求：支持双模型切换（通义千问 + 杏林） · RAG/直接回答切换开关 · 流式逐字输出 · 引用栏显示文献 + 相似度 + 可展开查看全文 · 思考过程可折叠且字体浅灰 · 思考完自动折叠再出正式回答 · 只依据资料并标来源。

环境变量：
```bash
export DASHSCOPE_API_KEY="<通义百炼 key>"    # 用于 qwen3.7-plus + text-embedding-v4
export XINLIN_API_KEY="<杏林 key>"           # 用于中医专业模型
```

> 用 Streamlit 写单文件 app.py 做中医医案 RAG 问答网页（只用 streamlit + openai + numpy，读 kb.pkl；检索和拼 prompt 自己写，不要 LangChain / 向量数据库）：
>
> ① 自己写 retrieve(q, k=5)：pickle.load(kb.pkl) 拿到向量矩阵与原文，用 text-embedding-v4（通义百炼）把问题嵌入，与矩阵做余弦相似度（numpy 点积 + argsort 取 top-k），返回 top-k 医案、来源 txt_filename、相似度分数；
> ② 左侧主区是对话，右侧固定一个「引用栏」：列出本次回答引用了哪些文献（txt_filename + 医案类型）及各自相似度（百分比 + 进度条），每条文献下方有「📖 查看全文」expander，点击可展开查看完整医案原文；
> ③ **支持两个对话模型切换**：侧边栏用 radio 选择，只显示模型名（`qwen3.7-plus` 和 `杏林`），不显示描述文字。配置如下：
>    - **qwen3.7-plus**：base_url=`https://dashscope.aliyuncs.com/compatible-mode/v1`，api_key 读 `DASHSCOPE_API_KEY`
>    - **杏林**：base_url=`https://ai.tcmcds.com/v1`，api_key 读 `XINLIN_API_KEY`，model name=`XinLin`
>    切换模型时自动清空对话历史；
> ④ 两个模型都开启思考（enable_thinking），流式返回：把 reasoning_content 放进可折叠的「💭 思考过程」实时滚动，正式答案 content 在下方逐字流式显示；
> ⑤ **思考过程的文字样式要与正式回答区分**：用浅灰色（#888）、略小字号（0.9em），通过 HTML `<div style="color: #888; font-size: 0.9em;">` 包裹实现；
> ⑥ **思考完开始正式回答时自动折叠**：思考阶段 expander 保持展开，一旦收到 content 的第一个 token，立即将思考 expander 折叠（expanded=False），让用户专注看正式回答；
> ⑦ **RAG / 直接回答切换**：侧边栏加一个 toggle 开关「RAG 模式（检索医案）」，默认开启。开启时先检索医案库再回答；关闭时跳过检索，直接把用户问题发给模型回答（用模型自身知识）。页面标题栏显示当前模式（RAG 模式 / 直接回答），右侧引用栏在关闭时也相应提示；
> ⑧ 自己拼 prompt：命中医案作参考资料，要求只依据资料作答、句末标【来源：文献名】，资料不足就如实说明；
> ⑨ base_url / api_key 读环境变量；检索或调用失败要友好提示。侧边栏加一个「🔄 刷新连接」按钮，点击清除缓存的 OpenAI 客户端（解决切换环境变量后 404 报错的问题）；
>
> 写完 streamlit run app.py 跑起来。建议用以下问题演示：
> - **正例**（能检索到相关医案）：「面瘫口眼歪斜如何辨证」「慢性肺源性心脏病 胸闷气喘」「老年痴呆 记忆力减退」
> - **反例**（数据库无相关内容，AI 会如实说明资料不足）：「半夏泻心汤主治什么证」

产出：`app.py`，浏览器里的流式问答页面，支持双模型切换 + RAG/直接回答切换。

**🎯 现场演示查询速查（直接复制到输入框）：**

| 类型 | 查询 | Top-1 相似度 | 命中来源 |
|:----:|------|:------------:|---------|
| ✅ 正例 | 面瘫口眼歪斜如何辨证 | 0.73 | 凃晋文从伏邪论治特发性面神经麻痹经验 |
| ✅ 正例 | 慢性肺源性心脏病 胸闷气喘 | 0.73 | 洪广祥温通并用治疗慢性肺源性心脏病经验 |
| ✅ 正例 | 老年痴呆 记忆力减退 | 0.68 | 刘祖贻基于五脏藏神理论治疗老年期痴呆经验 |
| ✅ 正例 | 糖尿病肾病怎么治疗 | 0.70 | 吕仁和教授"六对论治"糖尿病肾病经验 |
| ✅ 正例 | 慢性疲劳综合征 乏力 | 0.69 | 陈金水基于"少火生气"理论治疗慢性疲劳综合征经验 |
| ✅ 正例 | 荨麻疹 风疹块 瘙痒 | 0.69 | 董幼祺教授治疗小儿荨麻疹经验撷析 |
| ❌ 反例 | 半夏泻心汤主治什么证 | — | 数据库无相关内容，AI 会如实说明 |

---

## 第三步 · 多智能体应用　（agents.py + agents_app.py）

要求：在医案与 kb.pkl 之上协作完成一次辨证论治；命令行版展示协作轨迹；另写 Streamlit 前端可视化展示。

环境变量：
```bash
export DASHSCOPE_API_KEY="<通义百炼 key>"
```

> 在现有医案与 kb.pkl 之上，用纯 Python 写中医辨证多智能体系统（只用 openai SDK + numpy + streamlit + 标准库；调度 / 消息 / 记忆全自己写，不要 AutoGen / CrewAI 等框架）。分两个文件：
>
> **文件 1：agents.py（命令行版）**
>
> ① **最小 Agent 基类 BaseAgent**：角色系统提示词 + run(msg) 方法，自管消息列表 messages、调一次 LLM（模型 `qwen3.6-flash`，base_url=`https://dashscope.aliyuncs.com/compatible-mode/v1`，api_key 读 `DASHSCOPE_API_KEY`，开启 `enable_thinking`）；内置 `_parse_json()` 从 LLM 输出中提取 JSON（尝试直接解析 → 提取 ```json``` 块 → 提取 `{}` 块）；
>
> ② **四个专业 Agent**（各自有独立系统提示词，均要求严格 JSON 输出）：
>
> | Agent | 图标 | 输入 | 输出 JSON 字段 |
> |-------|:---:|------|----------------|
> | **检索Agent** | 🔍 | 病例 + kb.pkl 向量检索结果（numpy 点积 top-3） | `query`, `similar_cases[{source, similarity, content_summary, relevance}]`, `reference_summary` |
> | **辨证Agent** | 🩺 | 病例 + 检索参考意见 | `symptoms_analysis`, `etiology`, `pathogenesis`, `disease_location`, `syndrome`, `evidence[]` |
> | **处方Agent** | 💊 | 辨证结果（证型+病机），若有审校意见也附上 | `treatment_principle`, `formula_name`, `herbs[{name, dosage, function}]`, `contraindications_check`, `formula_explanation` |
> | **审校Agent** | ✅ | 辨证 + 处方方案 | `passed`(布尔), `consistency_check`, `safety_check`, `issues[]`, `suggestions`, `final_verdict` |
>
> ③ **KnowledgeBase 类**：`pickle.load(kb.pkl)` 读向量矩阵和文档，`search(query, top_k=3)` 用 `text-embedding-v4` 嵌入问题、numpy 点积算余弦相似度、argsort 取 top-k；
>
> ④ **orchestrate(case) 编排**：检索 → 辨证 → 处方 → 审校，审校 `passed=false` 则带意见回炉处方，**最多 2 轮**；每个 Agent 的 `run()` 方法支持 `stream_callback` 参数，流式返回 `(msg_type, chunk, full_reasoning, full_content)`；
>
> ⑤ **AgentLogger 日志系统**：`@dataclass LogEntry`（round_num, agent_name, action, input_data, output_data, timestamp）；每个 Agent 调用时记录一条；`display()` 方法打印时间线（含输入/输出/轮次）；
>
> ⑥ 用示例病例（胁肋胀痛、肝郁气滞类）在 `__main__` 跑通，打印完整协作轨迹 + 最终结构化结果。
>
> ---
>
> **文件 2：agents_app.py（Streamlit 前端，端口 8502）**
>
> ① 复用 agents.py 中的全部 Agent 类和编排器逻辑（直接在同一文件内重写一份，保持单文件独立运行）；
>
> ② **界面布局**：
>    - **左侧列**：
>      - **4 个预设病例按钮**（2×2 排列），点击自动填入对应病例：
>        1. 「📋 肝郁气滞（胁肋胀痛）」- 45岁男性，情绪抑郁胁痛
>        2. 「📋 面瘫/口眼歪斜」- 52岁女性，右侧面部歪斜20日
>        3. 「📋 糖尿病肾病」- 58岁男性，血糖升高10年+泡沫尿
>        4. 「📋 慢性肺源性心脏病」- 78岁男性，胸闷气喘20年
>      - 文本框（可自由输入或编辑预设病例）
>      - 「🚀 开始辨证论治」按钮（病例为空时禁用）
>    - **右侧列**：「🔄 协作过程」区，**流式可视化**每个 Agent 的输出（见下方详细说明）
>    - **下方结果区**：三列 `st.columns(3)` 分别展示：🩺 辨证（证型/病机/病位 + 详情 expander）· 💊 处方（治法/方剂/药物组成 + 方解 expander）· ✅ 审校（通过/未通过 + 详情 expander）
>    - **底部**：「🔍 检索到的相似医案」expander + 协作轮次统计
>
> ③ **侧边栏**：显示模型名、最大轮次、四个 Agent 角色说明（图标 + 名字 + 一句话描述）
>
> ④ **进度反馈**：运行时用 `st.progress()` + `st.info()` 实时显示当前 Agent 状态（如「🩺 辨证Agent 正在进行中医辨证...」）
>
> ⑤ **右侧协作过程 - 流式可视化**（核心功能）：
>    - 每个 Agent 调用前，用 `st.status(label, expanded=True)` 创建状态容器（可控制展开/折叠）
>    - Agent 的 LLM 调用改为 **流式**（`stream=True`），通过 `stream_callback` 实时回传内容
>    - **思考过程**（`reasoning_content`）：用浅灰色（`#888`）、略小字号（`0.9em`）通过 HTML `<div>` 包裹，实时滚动显示，末尾带 `▌` 光标
>    - **自动折叠**：一旦收到正式内容（`content`）的第一个 token，调用 `status.update(expanded=False)` 折叠容器，思考过程变为 `<details>` 可点击展开
>    - **正式输出**（`content`）：逐字流式显示，末尾带 `▌` 光标，完成后去掉光标
>    - **完成后**：`status.update(state="complete")`，并在下方用 `display_structured_result()` 显示结构化结果（非原始 JSON）：
>      - 🔍 检索Agent：关键症状 → 相似医案列表（来源/相似度/摘要/关联）→ 参考意见
>      - 🩺 辨证Agent：症状分析 → 病因 → 病机 → 病位 → **证型（红色高亮）** → 辨证依据列表
>      - 💊 处方Agent：治法 → **方剂（绿色高亮）** → 药物组成表 → 配伍禁忌 → 方解 expander
>      - ✅ 审校Agent：**✅/❌ 审校结果** → 一致性检查 → 安全性检查 → 问题警告 → 建议 expander
>
> ⑥ 用 `st.session_state` 保存结果和编排器实例，避免 rerun 时丢失；
>
> ⑦ 写完分别跑通：`python agents.py` 命令行输出协作轨迹，`streamlit run agents_app.py --server.port 8502` 启动前端。

产出：
- `agents.py` — 命令行版多智能体辨证论治流水线
- `agents_app.py` — Streamlit 可视化前端（端口 8502）

**📋 预设病例内容（直接复制到代码中）：**

```python
preset_cases = {
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
```
