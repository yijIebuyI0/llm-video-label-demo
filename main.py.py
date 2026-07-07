import os
import json
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# 1. 读取 .env 里的 API Key
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def build_prompt(row):
    """构造给 LLM 的输入文本。注意：不要把人工标签放进去。"""

    title = str(row.get("视频标题", "")).strip()
    content_summary = str(row.get("视频内容摘要", "")).strip()
    comment_summary = str(row.get("评论摘要", "")).strip()

    prompt = f"""
你是一名短视频种草内容分析师。请根据以下视频信息进行结构化打标。

【视频标题】
{title}

【视频内容摘要】
{content_summary}

【评论摘要】
{comment_summary}

请严格输出 JSON，字段如下：
{{
  "视频类型": "",
  "商品露出形式": "",
  "达人类型": "",
  "是否高潜素材": "",
  "判断理由": ""
}}

字段取值规则：

1. 视频类型只能从以下选项中选择一个：
["颜值", "时尚穿搭", "生活方式", "户外旅行", "运动健身", "日常分享", "测评类", "其他"]

2. 商品露出形式只能从以下选项中选择一个：
["露出", "展示", "介绍", "试穿/试用", "对比测评", "其他"]

3. 达人类型只能从以下选项中选择一个：
["时尚", "旅行", "日常分享", "颜值达人", "测评", "运动健身", "母婴亲子", "其他"]

4. 是否高潜素材只能从以下选项中选择一个：
["是", "否", "不确定"]

5. 判断理由不超过50字。

注意：
- 只能根据视频标题、视频内容摘要、评论摘要判断；
- 不要编造没有出现的信息；
- 只输出 JSON，不要输出任何解释性文字。
"""
    return prompt.strip()


def call_llm(prompt):
    """调用 DeepSeek API，让模型返回 JSON。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是严谨的短视频内容分析助手，只输出JSON格式结果。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    text = response.choices[0].message.content

    # 防止模型偶尔输出 ```json 包裹
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


def compare_label(manual_value, llm_value):
    """简单比较人工标签和 LLM 标签是否一致。"""

    manual_value = str(manual_value).strip()
    llm_value = str(llm_value).strip()

    if manual_value == "" or manual_value.lower() == "nan":
        return "无人工标签"

    if manual_value == llm_value:
        return "一致"

    if manual_value in llm_value or llm_value in manual_value:
        return "部分一致"

    return "不一致"


def main():
    input_file = "video_demo.xlsx"
    output_file = "video_demo_labeled.xlsx"

    # 2. 读取 Excel
    df = pd.read_excel(input_file)

    results = []

    # 3. 逐条处理
    for idx, row in df.iterrows():
        title = row.get("视频标题", "")
        print(f"正在处理第 {idx + 1} 条：{title}")

        prompt = build_prompt(row)

        try:
            label = call_llm(prompt)
        except Exception as e:
            print(f"第 {idx + 1} 条调用失败：{e}")
            label = {
                "视频类型": "其他",
                "商品露出形式": "其他",
                "达人类型": "其他",
                "是否高潜素材": "不确定",
                "判断理由": f"调用失败：{str(e)[:40]}"
            }

        results.append(label)

        time.sleep(0.5)

    # 4. LLM 输出结果转成 DataFrame
    label_df = pd.DataFrame(results)

    # 5. 加前缀，避免和人工标签重名
    label_df = label_df.add_prefix("LLM_")

    # 6. 合并回原始表
    out = pd.concat([df, label_df], axis=1)

    # 7. 如果有人工标签，则做一致性对比
    if "人工_视频类型" in out.columns and "LLM_视频类型" in out.columns:
        out["视频类型_一致性"] = out.apply(
            lambda r: compare_label(r["人工_视频类型"], r["LLM_视频类型"]),
            axis=1
        )

    if "人工_商品露出形式" in out.columns and "LLM_商品露出形式" in out.columns:
        out["商品露出形式_一致性"] = out.apply(
            lambda r: compare_label(r["人工_商品露出形式"], r["LLM_商品露出形式"]),
            axis=1
        )

    if "人工_达人类型" in out.columns and "LLM_达人类型" in out.columns:
        out["达人类型_一致性"] = out.apply(
            lambda r: compare_label(r["人工_达人类型"], r["LLM_达人类型"]),
            axis=1
        )

    # 8. 保存结果
    out.to_excel(output_file, index=False)

    print("处理完成！")
    print(f"结果文件已保存为：{output_file}")


if __name__ == "__main__":
    main()