# -*- coding: utf-8 -*-
# @Time : 2026/6/23
# @Author : Yao
# @File : extract_cases.py
# @Description : 从中医期刊文本中抽取医案（异步并发 + 重试 + JSONL 断点续传）

import json
import os
import re
import csv
import time
import asyncio
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError


# ==================== 配置区域 ====================
# 通义百炼 OpenAI 兼容接口配置
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
MODEL_NAME = "qwen3.6-flash"

# 并发和重试配置
CONCURRENCY_LIMIT = 30
MAX_RETRIES = 3
TEMPERATURE = 0.3
MAX_TOKENS = 16000

# 文件路径配置
INPUT_DIR = "./data"
OUTPUT_DIR = "./output"
JSONL_OUTPUT = os.path.join(OUTPUT_DIR, "extraction_results.jsonl")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "medical_cases.csv")


# ==================== 文本预处理 ====================
def fullwidth_to_halfwidth(text: str) -> str:
    """全角字符转半角字符（保留中文字符）"""
    result = []
    for char in text:
        code = ord(char)
        # 全角空格转半角
        if code == 0x3000:
            code = 0x0020
        # 其他全角字符转半角
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)


def clean_ocr_text(text: str) -> str:
    """
    预清洗 OCR 文本：
    1. 全角转半角
    2. 去除多余空白（保留单个空格）
    3. 合并断行（根据上下文判断）
    注意：不改动中医术语原文
    """
    # 1. 全角转半角
    text = fullwidth_to_halfwidth(text)

    # 2. 去除每行首尾空白
    lines = text.splitlines()
    lines = [line.strip() for line in lines]

    # 3. 合并断行（简单策略：如果一行不以标点结尾，且下一行不以标点开头，则合并）
    merged_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i]

        # 如果当前行为空，跳过
        if not current_line:
            i += 1
            continue

        # 检查是否需要与下一行合并
        while i + 1 < len(lines):
            next_line = lines[i + 1]

            # 如果下一行为空，停止合并
            if not next_line:
                break

            # 判断是否需要合并：当前行不以标点结尾，且下一行不以标点或数字开头
            end_puncts = set('。！？；：，、）】》…—·')
            start_puncts = set('。，、；：！？）】》…—·')

            current_ends_with_punct = current_line[-1] in end_puncts if current_line else False
            next_starts_with_punct = next_line[0] in start_puncts if next_line else False
            next_starts_with_digit = next_line[0].isdigit() if next_line else False

            # 如果当前行以标点结尾，或下一行以标点/数字开头，不合并
            if current_ends_with_punct or next_starts_with_punct or next_starts_with_digit:
                break

            # 合并（加一个空格）
            current_line = current_line + ' ' + next_line
            i += 1

        merged_lines.append(current_line)
        i += 1

    # 4. 去除多余空白（多个连续空格合并为一个）
    text = '\n'.join(merged_lines)
    text = re.sub(r' +', ' ', text)

    return text.strip()


def read_txt_file(file_path: str) -> str:
    """读取文本文件，支持 utf-8 和 gb18030 编码"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gb18030') as f:
                return f.read()
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
            return ""
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return ""


# ==================== 提示词构建 ====================
def format_prompt(title: str, content: str) -> str:
    """根据文章数据格式化 Prompt，用于提取中医医案信息"""
    prompt = f"""
        下面是一段经过OCR以后的中医期刊文章文本，需要对其中的信息进行理解整合，抽取需要的医案信息。
        任务背景：
            1. 任务来源和要求：PDF文件OCR转换的文本，要求经过下面处理后提取出其中的医案信息。
            2. 文献特点：中医期刊论文，以中文为主，存在着一些格式凌乱的问题。
        处理要求（按先后顺序排序，逐步完成以下步骤）：
            一. 文章范围界定
            - 根据给定的标题{title}，判断目标文章的实际范围，只提取与文章相关的内容。
            - 因为OCR识别原因，同一版面上可能存在多篇文章片段，如前一篇文章的参考文献、后一篇文章的开头等。
            二. 提取医案
            * 1.判断文章内是否有医案，医案可能是文章的验案举隅、经验描述、典型病例等部分出现，请仔细阅读文章。如果有则进行下面的步骤，如果没有直接输出：无。
            * 2.判断医案类型：默认"现代医案"，如果是用文言文描述或者描述古代诊疗方法，则为"古代医案"。如果题目或者全文内容如出现医生姓名，就再加上"名医医案"。
            * 3.医案需要提取完整，有些医案可能较长、有些医案包含按语，都需要进行提取。如果相关医案有小标题，可以不用提取。
            * 4.有些期刊内存在多重医案，需要全部进行提取。
            * 5.医案需要抽取完整。不对医案本身的内容进行任何改变、不随意增添删改中医药术语表达。
            * 6.因为OCR识别质量的影响，有些医案没有办法准确提取完整、语义不连续，需要完全保留原文的医案部分。
            三. 质量控制：
            - OCR识别后可能存在大量错字、异体字，请全部保留，不要做任何改变。
            - 保持中医专业内容的准确性、专业性，只提取不改变。
            - 有些医案可能因为OCR识别的原因，语义变得不连贯，也只输出原样。
            - 有些医案开头可能有序号，将其去掉不需要保留。
            - 确保每一则提取出来的医案都是完整、独立的医案。
            四.输出格式：
            1.如果文章中没有医案，则输出：无。
            2.如果文章中有一则医案，则输出形式为：
                医案1类型：现代医案
                医案1内容：
                    男， 37 岁。 2019 年 8 月 23 日初诊。 因"颈项及腰 骶部僵硬疼痛....
                [完]
            3.如果文章中有多则医案，用一个#号隔开，输出形式为（以两则为例）：
                医案1类型：现代医案、名医医案
                医案1内容：
                    患者，女，54 岁，主因"右耳听力下降伴耳鸣耳闷 8 d"于 2022 年 6 月 27 日就诊于北京中医药大学第....
                    按语：本例患者为中年女性，因情志因素导致突 发性聋，出现听力下降及伴随症状。白鹏教授认为....
                #
                医案2类型：古代医案
                医案2内容：
                    予所谓燥疫者，古无是名，非敢炫玉，冒自出新，乃以彼年见证，燥气之中，实有疫气乘之....
                [完]
            4、严格按照以上格式要求输出，不要附加任何其他内容和分析，只提取医案，不要做任何改变。

以下是文章内容：
{content}
    """
    return prompt


# ==================== API 调用 ====================
async def process_file_async(
    file_item: dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    max_retries: int
) -> dict:
    """异步处理单个文件，包括 API 调用和重试"""
    file_id = file_item.get("custom_id", "")
    file_name = file_item.get("file_name", "")
    title = file_item.get("title", "")
    content = file_item.get("content", "")

    prompt = format_prompt(title, content)

    success = False
    response_content = ""
    last_error = ""

    for attempt in range(max_retries):
        try:
            async with semaphore:
                completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )

            response_content = completion.choices[0].message.content
            success = True
            break

        except APITimeoutError:
            last_error = f"请求超时 (第 {attempt + 1} 次尝试)"
            await asyncio.sleep(1 + attempt * 2)
        except APIConnectionError as e:
            last_error = f"连接失败 (第 {attempt + 1} 次尝试): {e}"
            await asyncio.sleep(1 + attempt * 2)
        except RateLimitError:
            last_error = f"达到速率限制 (第 {attempt + 1} 次尝试)"
            await asyncio.sleep(5 + attempt * 5)
        except APIStatusError as e:
            last_error = f"API 状态错误 {e.status_code} (第 {attempt + 1} 次尝试)"
            if not (500 <= e.status_code < 600):
                break
            await asyncio.sleep(2 + attempt * 2)
        except Exception as e:
            last_error = f"未知错误 (第 {attempt + 1} 次尝试): {e}"
            break

    if not success:
        response_content = f"ERROR: {last_error}"

    return {
        "custom_id": file_id,
        "file_name": file_name,
        "response": response_content,
    }


# ==================== 文件读写 ====================
def write_jsonl_line(filepath: str, result: dict):
    """将单条结果写入 JSONL 文件"""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"写入文件 {filepath} 失败: {e}")
        return False


def load_processed_ids(jsonl_path: str) -> set:
    """从 JSONL 文件加载已处理的文件 ID"""
    processed_ids = set()
    if not os.path.exists(jsonl_path):
        return processed_ids

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    file_id = item.get("custom_id")
                    if file_id:
                        processed_ids.add(file_id)
                except json.JSONDecodeError:
                    continue
        print(f"从 JSONL 文件加载了 {len(processed_ids)} 个已处理的文件 ID")
    except Exception as e:
        print(f"加载 JSONL 文件时出错: {e}")

    return processed_ids


# ==================== 结果解析 ====================
def parse_prediction(raw_prediction: str) -> dict:
    """解析模型返回的预测结果，提取医案信息"""
    prediction = raw_prediction.strip()

    # 如果返回"无"，表示文章中没有医案
    if prediction == "无" or prediction == "无。" or prediction.startswith("ERROR:"):
        return {"status": "no_cases", "cases": []}

    # 使用正则表达式匹配医案内容
    pattern = re.compile(
        r'医案\d+类型：(?P<type>.*?)\n医案\d+内容：\n(?P<content>.*?)(?=\n#|\n医案\d+类型：|\n\[完\]|$)',
        re.DOTALL
    )

    cases = []
    matches = pattern.finditer(prediction)
    for match in matches:
        case_type = match.group('type').strip()
        case_content = match.group('content').strip()
        cases.append({
            "type": case_type,
            "content": case_content
        })

    if cases:
        return {"status": "success", "cases": cases}
    else:
        # 如果没有匹配到任何医案
        if "无" in prediction:
            return {"status": "no_cases", "cases": []}
        else:
            return {"status": "parsing_error", "cases": [], "raw": prediction}


def generate_csv(jsonl_path: str, csv_path: str) -> dict:
    """从 JSONL 文件生成最终 CSV 文件"""
    extracted_rows = []
    id_counter = 0
    stats = {
        "total_files": 0,
        "files_with_cases": 0,
        "files_without_cases": 0,
        "parsing_errors": 0,
        "total_cases": 0,
        "case_types": {}
    }

    try:
        # 读取 JSONL 文件
        results_by_file = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    file_name = item.get('file_name')
                    if file_name:
                        results_by_file[file_name] = item
                except json.JSONDecodeError:
                    continue

        stats["total_files"] = len(results_by_file)

        # 解析每个文件的结果
        for file_name, data in results_by_file.items():
            custom_id = data.get('custom_id', '')
            response_text = data.get('response', '')

            parsed = parse_prediction(response_text)

            if parsed["status"] == "success" and parsed["cases"]:
                stats["files_with_cases"] += 1
                for case in parsed["cases"]:
                    case_type = case["type"]
                    case_content = case["content"]

                    extracted_rows.append({
                        'id': str(id_counter),
                        'custom_id': custom_id,
                        'txt_filename': file_name,
                        '医案内容': case_content,
                        '医案类型': case_type
                    })

                    # 统计医案类型
                    stats["case_types"][case_type] = stats["case_types"].get(case_type, 0) + 1
                    stats["total_cases"] += 1
                    id_counter += 1

            elif parsed["status"] == "no_cases":
                stats["files_without_cases"] += 1
                extracted_rows.append({
                    'id': str(id_counter),
                    'custom_id': custom_id,
                    'txt_filename': file_name,
                    '医案内容': '无',
                    '医案类型': ''
                })
                id_counter += 1

            else:  # parsing_error
                stats["parsing_errors"] += 1
                extracted_rows.append({
                    'id': str(id_counter),
                    'custom_id': custom_id,
                    'txt_filename': file_name,
                    '医案内容': '解析错误',
                    '医案类型': ''
                })
                id_counter += 1

        # 写入 CSV 文件
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'custom_id', 'txt_filename', '医案内容', '医案类型']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in extracted_rows:
                writer.writerow(row)

        return stats

    except Exception as e:
        print(f"生成 CSV 时发生错误: {e}")
        return stats


# ==================== 主流程 ====================
async def run_extraction(max_files: int = None):
    """
    主提取流程

    Args:
        max_files: 最大处理文件数量，None 表示处理所有文件
    """
    # 检查 API Key
    if not DASHSCOPE_API_KEY:
        print("错误：未设置环境变量 DASHSCOPE_API_KEY")
        return

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 扫描输入目录
    print(f"扫描输入目录: {INPUT_DIR}")
    files_to_process = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                file_content = read_txt_file(file_path)

                if file_content:
                    # 预清洗 OCR 文本
                    cleaned_content = clean_ocr_text(file_content)
                    file_name = os.path.splitext(file)[0]

                    files_to_process.append({
                        "file_name": file_name,
                        "title": file_name,
                        "content": cleaned_content,
                    })

    print(f"共发现 {len(files_to_process)} 个 TXT 文件")

    # 限制处理文件数量
    if max_files is not None and max_files > 0:
        files_to_process = files_to_process[:max_files]
        print(f"限制处理前 {max_files} 个文件")

    if not files_to_process:
        print("错误：未找到任何可处理的文件")
        return

    # 分配 custom_id
    for idx, file_item in enumerate(files_to_process):
        file_item["custom_id"] = str(idx)

    # 加载已处理的文件 ID（断点续传）
    processed_ids = load_processed_ids(JSONL_OUTPUT)

    # 过滤出未处理的文件
    files_to_process = [f for f in files_to_process if f["custom_id"] not in processed_ids]

    if not files_to_process:
        print("所有文件都已处理完毕")
    else:
        print(f"需要处理 {len(files_to_process)} 个新文件")

        # 初始化 AsyncOpenAI 客户端
        client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=60.0,
        )

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        # 创建异步任务
        tasks = [
            process_file_async(file_item, client, semaphore, MAX_RETRIES)
            for file_item in files_to_process
        ]

        # 使用 tqdm 显示进度
        start_time = time.time()

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理文件"):
            result = await task

            # 立即写入 JSONL（支持断点续传）
            write_jsonl_line(JSONL_OUTPUT, result)

        elapsed_time = time.time() - start_time
        print(f"\n处理完成，耗时: {elapsed_time:.2f} 秒")

    # 生成最终 CSV
    print(f"\n生成 CSV 文件: {CSV_OUTPUT}")
    stats = generate_csv(JSONL_OUTPUT, CSV_OUTPUT)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("提取统计")
    print("=" * 60)
    print(f"总文件数: {stats['total_files']}")
    print(f"有医案的文件: {stats['files_with_cases']}")
    print(f"无医案的文件: {stats['files_without_cases']}")
    print(f"解析错误: {stats['parsing_errors']}")
    print(f"总医案数: {stats['total_cases']}")

    if stats['case_types']:
        print("\n医案类型分布:")
        for case_type, count in sorted(stats['case_types'].items()):
            print(f"  {case_type}: {count}")

    print(f"\n结果已保存到: {CSV_OUTPUT}")


if __name__ == "__main__":
    import sys

    # 检查是否需要限制文件数量（用于测试）
    max_files = None
    if len(sys.argv) > 1:
        try:
            max_files = int(sys.argv[1])
            print(f"测试模式：仅处理前 {max_files} 个文件")
        except ValueError:
            print(f"无效的参数: {sys.argv[1]}，应为整数")
            sys.exit(1)

    # 运行提取流程
    asyncio.run(run_extraction(max_files=max_files))
