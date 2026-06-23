# -*- coding: utf-8 -*-
# @Time : 2025/4/18 上午10:27
# @Author : Yao
# @File : case_creat_openai_async.py
# @Description : 从中医期刊文本中抽取医案（第一阶段）

import json
import os
import time
import ast # 用于安全地解析字符串形式的列表/字典
import asyncio # Added asyncio
import csv # <-- 引入 csv 模块
import re # <-- 引入 re 模块
from tqdm.asyncio import tqdm # Use tqdm's async version
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError # Use AsyncOpenAI
import concurrent.futures # Added concurrent.futures

# --- OpenAI Async Client 初始化 ---
# 使用环境变量配置 API，避免敏感信息硬编码
# 在运行前设置环境变量:
#   export OPENAI_API_KEY="your-api-key-here"
#   export OPENAI_BASE_URL="https://your-api-endpoint/v1"  (可选)
aclient = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    timeout=60.0, # 设置请求超时时间 (秒)
)

def format_prompt(item_data):
    """根据文章数据格式化Prompt，用于提取中医医案信息"""
    # 获取文件名作为标题和文章内容
    title = item_data.get("title", "")
    content = item_data.get("content", "")

    # 创建医案提取的提示词
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

def parse_prediction(raw_prediction):
    """尝试解析模型返回的预测结果，提取医案信息"""
    # 这是CPU密集型操作，保持同步
    prediction = raw_prediction.strip()

    # 如果返回"无"，表示文章中没有医案
    if prediction == "无" or prediction == "无。":
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
        # 如果没有匹配到任何医案，可能是格式问题或真的没有医案
        if "无" in prediction:
            return {"status": "no_cases", "cases": []}
        else:
            return {"status": "parsing_error", "cases": [], "raw": prediction}

async def process_file_async(file_item, aclient: AsyncOpenAI, semaphore: asyncio.Semaphore, max_retries: int, model_name: str, temperature: float, max_tokens: int, reasoning_effort: str = None):
    """异步处理单个文件，包括API调用和重试"""
    file_id = file_item.get("custom_id", f"unknown_{time.time()}")
    file_name = file_item.get("file_name", "")
    prompt_content = format_prompt(file_item)

    success = False
    raw_prediction_content = "Error: No response"
    request_id = None
    response_id = None
    last_error = f"Error: Max retries ({max_retries}) reached"

    for attempt in range(max_retries):
        try:
            # 使用信号量限制并发API调用
            async with semaphore:
                request_kwargs = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt_content}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if reasoning_effort:
                    request_kwargs["reasoning_effort"] = reasoning_effort
                completion = await aclient.chat.completions.create(**request_kwargs)
                # 尝试捕获请求/响应ID，兼容不同SDK结构
                response_id = getattr(completion, "id", None) or (completion.get("id") if isinstance(completion, dict) else None)
                request_id = getattr(completion, "request_id", None) or (completion.get("request_id") if isinstance(completion, dict) else None)
                if request_id is None:
                    resp_obj = getattr(completion, "response", None)
                    try:
                        if resp_obj and hasattr(resp_obj, "headers"):
                            headers = resp_obj.headers
                            request_id = headers.get("x-request-id") or headers.get("X-Request-Id")
                    except Exception:
                        pass
            raw_prediction_content = completion.choices[0].message.content
            success = True
            break # 成功，退出重试循环
        except APITimeoutError:
            last_error = f"Error: Request Timeout (Attempt {attempt + 1})"
            await asyncio.sleep(1 + attempt * 2)
        except APIConnectionError as e:
            last_error = f"Error: Connection Failed (Attempt {attempt + 1}): {e}"
            await asyncio.sleep(1 + attempt * 2)
        except RateLimitError:
            last_error = f"Error: Rate Limit Exceeded (Attempt {attempt + 1})"
            wait_time = 5 + attempt * 5
            await asyncio.sleep(wait_time)
        except APIStatusError as e:
            body_text = ""
            try:
                body_text = e.response.text if hasattr(e.response, "text") else str(e.response)
            except Exception:
                body_text = str(e.response)
            last_error = f"Error: API Status {e.status_code} (Attempt {attempt + 1}): {body_text}"
            if not (500 <= e.status_code < 600):
                break
            else:
                await asyncio.sleep(2 + attempt * 2)
        except Exception as e:
            last_error = f"Error: Unknown (Attempt {attempt + 1}): {e}"
            break

    if not success:
        raw_prediction_content = last_error

    return {
        "custom_id": file_id,
        "file_name": file_name,
        "response": raw_prediction_content,
        "request_id": request_id,
        "response_id": response_id
    }

def write_line_to_file(filepath: str, result: dict):
    """同步函数：将单个结果字典写入 JSONL 文件。"""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
        return True
    except Exception as e:
        print(f"\n[File Writer Error] 写入文件 {filepath} 失败: {e} for result {result}")
        return False

def write_batch_to_file(filepath: str, batch: list):
    """同步函数：将一批结果字典写入 JSONL 文件。"""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            for result in batch:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
        return len(batch)
    except Exception as e:
        print(f"\n[File Writer Error] 批量写入文件 {filepath} 失败: {e}")
        return 0

def read_txt_file(file_path):
    """读取文本文件内容"""
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

def load_retry_targets_from_csv(csv_path, retry_values=None):
    """从已生成的CSV中读取需要重试的文件名"""
    retry_values = set(retry_values) if retry_values else {"无", "解析错误"}
    target_files = set()
    encodings = ["utf-8", "gb18030"]
    last_exc = None
    for enc in encodings:
        try:
            with open(csv_path, 'r', encoding=enc) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get('医案内容') in retry_values:
                        fname = row.get('txt_filename') or row.get('file_name')
                        if fname:
                            target_files.add(fname)
            if target_files:
                print(f"从CSV {csv_path} (编码 {enc}) 读取到 {len(target_files)} 个需要重试的文件。")
            else:
                print(f"重试CSV {csv_path} (编码 {enc}) 中未找到需要重试的记录。")
            break
        except FileNotFoundError:
            print(f"警告：未找到CSV文件 {csv_path}，将跳过重试加载。")
            break
        except UnicodeDecodeError as e:
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            break
    if last_exc:
        print(f"警告：读取重试CSV {csv_path} 时出错: {last_exc}")
    return target_files

async def run_openai_eval_async(input_dir, output_prediction_file, model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=4000, max_retries=3, concurrency_limit=10, write_batch_size=50, batch_delay_seconds=1.0, max_files=None, retry_from_csv=None, retry_values=None, reasoning_effort=None):
    """异步从输入目录读取文本文件，调用OpenAI API，使用单独线程批量保存结果，支持断点续传和重试

    Args:
        input_dir: 输入目录路径
        output_prediction_file: 输出文件路径
        model_name: 使用的模型名称
        temperature: 模型温度参数
        max_tokens: 最大token数
        max_retries: 最大重试次数
        concurrency_limit: 并发限制
        write_batch_size: 写入批次大小
        batch_delay_seconds: 批次间延迟秒数
        max_files: 最大处理文件数量，None表示处理所有文件
        retry_from_csv: 如果提供，将仅重试CSV中"医案内容"为指定值的文件
        retry_values: 在CSV中视为需要重试的"医案内容"取值，默认 {"无","解析错误"}
        reasoning_effort: 传递给模型的推理力度参数，例如 "minimal" / "medium"，None 表示不传递
    """
    files_to_process = []
    retry_target_files = None
    retry_values = retry_values or {"无", "解析错误"}

    if retry_from_csv:
        retry_target_files = load_retry_targets_from_csv(retry_from_csv, retry_values)
        if retry_target_files:
            print(f"重试模式开启：仅处理 {len(retry_target_files)} 个在CSV中标记为 {retry_values} 的文件。")
        else:
            retry_target_files = None
            print("重试模式未找到目标文件，按常规流程处理全部文件。")

    try:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    file_content = read_txt_file(file_path)
                    if file_content:
                        file_name = os.path.splitext(file)[0]
                        if retry_target_files is not None and file_name not in retry_target_files:
                            continue
                        files_to_process.append({
                            "file_name": file_name,
                            "title": file_name,
                            "content": file_content
                        })
                    else:
                        print(f"警告：无法读取或文件为空 {file_path}")
    except Exception as e:
        print(f"错误：扫描输入目录 {input_dir} 时出错: {e}")
        return

    if max_files is not None and isinstance(max_files, int) and max_files > 0:
        original_count = len(files_to_process)
        files_to_process = files_to_process[:max_files]
        print(f"已限制处理文件数量为 {max_files} (总共发现 {original_count} 个文件)")

    if not files_to_process:
        print("错误：未能从输入目录中加载任何有效文件。")
        return

    output_dir = os.path.dirname(output_prediction_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    processed_ids = set()
    intermediate_jsonl_path = output_prediction_file + ".jsonl"
    existing_custom_ids = {}
    max_existing_custom_numeric_id = -1

    if os.path.exists(intermediate_jsonl_path):
        try:
            with open(intermediate_jsonl_path, "r", encoding="utf-8") as f_jsonl:
                for line_num, line in enumerate(f_jsonl):
                    try:
                        item = json.loads(line)
                        item_id = item.get('custom_id')
                        item_file_name = item.get('file_name') or item.get('txt_filename')
                        if item_file_name and item_id:
                            existing_custom_ids[item_file_name] = item_id
                            try:
                                max_existing_custom_numeric_id = max(max_existing_custom_numeric_id, int(item_id))
                            except (ValueError, TypeError):
                                pass
                        if item_id:
                            if retry_target_files is not None and item_file_name in retry_target_files:
                                continue
                            if "response" in item:
                                processed_ids.add(item_id)
                    except json.JSONDecodeError:
                        print(f"警告：无法解析 .jsonl 文件中的行 {line_num + 1}: {line.strip()}")
            print(f"成功从 .jsonl 文件加载了 {len(processed_ids)} 个先前处理的文件 ID。")
        except Exception as e:
            print(f"警告: 从 .jsonl 文件加载状态时发生错误: {e}。将从头开始。")
            processed_ids = set()
    else:
        print("未找到 .jsonl 文件，将从头开始处理所有文件。")
        processed_ids = set()

    next_custom_id = max_existing_custom_numeric_id + 1 if max_existing_custom_numeric_id >= 0 else 0
    for file_item in files_to_process:
        file_name = file_item.get("file_name", "")
        if file_name in existing_custom_ids and existing_custom_ids[file_name] is not None:
            file_item["custom_id"] = existing_custom_ids[file_name]
        else:
            file_item["custom_id"] = str(next_custom_id)
            next_custom_id += 1

    files_to_process = [f for f in files_to_process if f.get("custom_id") not in processed_ids]
    total_files_to_process = len(files_to_process)

    if total_files_to_process == 0:
        print("所有文件都已处理完毕。")
        generate_csv_from_jsonl(intermediate_jsonl_path, output_prediction_file.replace('.json', '.csv'))
        return

    start_time = time.time()
    print(f"开始处理 {total_files_to_process} 个新文件 (模型: {model_name}, 温度: {temperature}, MaxTokens: {max_tokens}, 最大重试: {max_retries}, 并发数: {concurrency_limit})...")

    semaphore = asyncio.Semaphore(concurrency_limit)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        print("无法获取当前事件循环，将创建一个新的事件循环。")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    executor = concurrent.futures.ThreadPoolExecutor()

    results_this_run = []
    writer_futures = []

    def chunk_list(data, size):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    api_batch_size = concurrency_limit
    with tqdm(total=total_files_to_process, desc="Processing Files") as pbar:
        for i, batch_files in enumerate(chunk_list(files_to_process, api_batch_size)):
            pbar.set_description(f"提交批次 {i+1}/{ (total_files_to_process + api_batch_size - 1) // api_batch_size } ({len(batch_files)} 个文件)")
            tasks_in_batch = [
                asyncio.create_task(
                    process_file_async(
                        f_item,
                        aclient,
                        semaphore,
                        max_retries,
                        model_name,
                        temperature,
                        max_tokens,
                        reasoning_effort
                    )
                )
                for f_item in batch_files
            ]

            batch_results = await asyncio.gather(*tasks_in_batch)

            pbar.set_description(f"处理批次 {i+1} 结果...")
            batch_to_write_to_file = []
            for result in batch_results:
                if result:
                    results_this_run.append(result)
                    batch_to_write_to_file.append(result)
                    pbar.update(1)

            if batch_to_write_to_file:
                writer_future = loop.run_in_executor(
                    executor,
                    write_batch_to_file,
                    intermediate_jsonl_path,
                    batch_to_write_to_file
                )
                writer_futures.append(writer_future)

            total_batches = (total_files_to_process + api_batch_size - 1) // api_batch_size
            if i < total_batches - 1:
                pbar.set_description(f"批次 {i+1} 完成。暂停 {batch_delay_seconds}s...")
                await asyncio.sleep(batch_delay_seconds)
            else:
                pbar.set_description("最终批次完成。")

    print(f"\n所有 API 任务完成。等待 {len(writer_futures)} 个文件写入操作完成...")
    if writer_futures:
        done, pending = await asyncio.wait(writer_futures, return_when=asyncio.ALL_COMPLETED)

        successful_writes = 0
        failed_writes = 0
        for fut in done:
            try:
                result_count = fut.result()
                if result_count > 0:
                    successful_writes += result_count
            except Exception as exc:
                print(f'后台写入任务产生异常: {exc}')
                failed_writes += 1

        print(f"文件写入完成。 成功写入约 {successful_writes} 条记录，失败 {failed_writes} 个批次。")
        if pending:
            print(f"警告：仍有 {len(pending)} 个写入任务待处理？这不应该发生。")
    else:
        print("没有需要等待的文件写入操作。")

    executor.shutdown(wait=True)
    print("线程池已关闭。")

    end_time = time.time()
    elapsed_time = end_time - start_time

    csv_path = output_prediction_file.replace('.json', '.csv')
    generate_csv_from_jsonl(intermediate_jsonl_path, csv_path)

    print(f"\n处理总结。总文件数: {len(files_to_process) + len(processed_ids)}, 本次处理: {len(files_to_process)}, 总耗时: {elapsed_time:.2f} 秒")
    print(f"结果已保存为JSONL格式: {intermediate_jsonl_path}")
    print(f"结果已保存为CSV格式: {csv_path}")

def generate_csv_from_jsonl(jsonl_file_path, csv_file_path):
    """从JSONL文件生成CSV文件，处理医案信息并生成最终CSV结果"""
    extracted_rows = []
    id_counter = 0

    pattern = re.compile(
        r'医案\d+类型：(?P<type>.*?)\n医案\d+内容：\n(?P<content>.*?)(?=\n#|\n医案\d+类型：|\n\[完\]|$)',
        re.DOTALL
    )

    try:
        latest_data_by_file = {}
        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonlfile:
            for line in jsonlfile:
                if line.strip():
                    try:
                        data = json.loads(line)
                        key = data.get('file_name') or data.get('txt_filename') or data.get('custom_id')
                        if key:
                            latest_data_by_file[key] = data
                        else:
                            latest_data_by_file[f"row_{len(latest_data_by_file)}"] = data
                    except json.JSONDecodeError:
                        continue

        for data in latest_data_by_file.values():
            custom_id = data.get('custom_id', '')
            file_name = data.get('file_name', '')
            content_text = data.get('response', '')

            matches = list(pattern.finditer(content_text))

            if matches:
                for match in matches:
                    case_type = match.group('type').strip()
                    case_content = match.group('content').strip()

                    case_id = str(id_counter)
                    id_counter += 1

                    extracted_rows.append({
                        'id': case_id,
                        'custom_id': custom_id,
                        'txt_filename': file_name,
                        '医案内容': case_content,
                        '医案类型': case_type
                    })
            elif "无" in content_text or content_text.strip() in ["无", "无。"]:
                case_id = str(id_counter)
                id_counter += 1

                extracted_rows.append({
                    'id': case_id,
                    'custom_id': custom_id,
                    'txt_filename': file_name,
                    '医案内容': '无',
                    '医案类型': ''
                })
            else:
                case_id = str(id_counter)
                id_counter += 1

                extracted_rows.append({
                    'id': case_id,
                    'custom_id': custom_id,
                    'txt_filename': file_name,
                    '医案内容': '解析错误',
                    '医案类型': ''
                })

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = ['id', 'custom_id', 'txt_filename', '医案内容', '医案类型']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in extracted_rows:
                writer.writerow(row)

        print(f"已提取 {len(extracted_rows)} 则医案并保存到 {csv_file_path}")
        return len(extracted_rows)

    except Exception as e:
        print(f"生成CSV时发生错误: {e}")
        return 0

if __name__ == "__main__":
    # === 配置区域 ===
    # 数据路径配置
    data_dir = "./data/input_texts"  # 输入文本目录（包含.txt文件）

    # 输出路径配置
    output_dir = "./output/case_extraction"
    os.makedirs(output_dir, exist_ok=True)
    output_pred_path = os.path.join(output_dir, "medical_cases.json")

    # 模型配置
    target_model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    target_temperature = 0.3
    target_max_tokens = 16000
    retry_count = 3
    concurrency = 10  # 根据API限制调整
    batch_write_size = 20
    target_batch_delay = 1.0
    max_file_count = None  # None=处理所有文件，或设置为整数限制数量

    # 重试配置（可选）
    retry_source_csv = None  # 设置为CSV路径以仅重试失败项，例如: "./output/case_extraction/medical_cases.csv"
    retry_status_values = ["无", "解析错误"]
    reasoning_effort_enabled = False
    reasoning_effort_level = "minimal"

    # === 运行程序 ===
    print("=" * 80)
    print("中医医案信息提取程序")
    print("=" * 80)
    print(f"\nAPI Base URL: {os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print(f"模型: {target_model}")
    print(f"输入目录: {data_dir}")
    print(f"输出文件: {output_pred_path}")
    print("=" * 80)

    asyncio.run(run_openai_eval_async(
        data_dir,
        output_pred_path,
        model_name=target_model,
        temperature=target_temperature,
        max_tokens=target_max_tokens,
        max_retries=retry_count,
        concurrency_limit=concurrency,
        write_batch_size=batch_write_size,
        batch_delay_seconds=target_batch_delay,
        max_files=max_file_count,
        retry_from_csv=retry_source_csv,
        retry_values=retry_status_values,
        reasoning_effort=reasoning_effort_level if reasoning_effort_enabled else None
    ))
