# main.py —— AI 智能文本分析与生成工具
# 第四阶段：接入 DeepSeek 大模型，实现真正的 AI 分析

# ---------------------------------------------------------------
# 导入需要的库
# ---------------------------------------------------------------
import os                          # 标准库：读取环境变量（存放 API Key 的安全位置）
import re                          # 标准库：正则表达式，用来提取文本
from collections import Counter    # 标准库：计数器，统计词频

import streamlit as st             # 网页框架
import pandas as pd                # 数据处理库，画图表需要
from openai import OpenAI          # OpenAI 官方库（DeepSeek 兼容这个协议，直接复用）


# ---------------------------------------------------------------
# 工具函数：从文本中提取"词"（英文按单词、中文按单字）
# ---------------------------------------------------------------
def extract_words(text):
    # 提取英文单词（转小写统一计数）
    english_words = re.findall(r'[a-zA-Z]+', text.lower())
    # 提取中文单字（\u4e00-\u9fff 是汉字的 Unicode 范围）
    chinese_chars = re.findall(r'[一-鿿]', text)
    return english_words + chinese_chars


# ---------------------------------------------------------------
# 预设的提示词模板：不同的分析类型，对应不同的"任务说明书"
# 用字典（dict）来存，键是用户看到的名字，值是提示词内容
# 以后想加新类型，只要在这个字典里加一条就行，不用改其他代码
# ---------------------------------------------------------------
PROMPTS = {
    "综合深度分析": """请对下面的文本进行深度分析，用中文回答，使用 Markdown 格式（## 作为小节标题）。要求每个结论都给出依据，可适当引用原文片段。

## 一、情感倾向
- 总体判断：积极 / 消极 / 中性 / 混合
- 给出你的把握程度（高 / 中 / 低）并简要说明
- 从文本中找出 2-3 处支撑判断的依据（可引用原文）

## 二、内容总结
- 用 3-5 句话概括文本核心内容
- 再用要点列表（2-4 条）列出文本传递的关键信息

## 三、关键词提取
- 提取 5-8 个关键词
- 每个关键词后面用一句话说明它在文本中的作用或重要性""",

    "快速摘要": """请用中文为下面的文本写一份简明摘要，使用 Markdown 格式：

## 一句话概括
用一句话说清这段文字讲了什么。

## 核心要点
- 用 3 条以内的要点列出最关键的信息

## 阅读建议
一句话说明这段文字适合谁读、读的时候要重点关注什么。""",

    "只看情感倾向": """请只分析下面文本的情感倾向，用中文回答，使用 Markdown 格式：

## 情感判断
- 总体倾向：积极 / 消极 / 中性 / 混合
- 把握程度：高 / 中 / 低
- 情绪强度：强烈 / 中等 / 微弱

## 判断依据
- 引用 2-3 处原文片段，逐一说明它们为什么体现了这种情感

## 情感变化
如果文本中情感有明显转折，指出在哪里转折、如何变化；如果没有转折就说明情感始终一致。""",

    "只提关键词": """请只为下面文本提取关键词，用中文回答，使用 Markdown 格式：

## 关键词清单
提取 5-8 个最能代表文本主题的关键词，每个关键词后面用一句话说明它的重要性。

## 主题归类
把上面的关键词归成 2-3 个主题类别，说明这段文本主要在谈哪几件事。""",
}

# 自定义提问时使用的提示词前缀
CUSTOM_PREFIX = """请严格根据下面的文本回答用户的问题。如果文本中没有相关信息，请直接说明"文本中未提及"，不要编造。用中文回答，使用 Markdown 格式。"""


# ---------------------------------------------------------------
# 工具函数：调用 DeepSeek 大模型做 AI 分析
# 参数 text             ：用户输入的文本
# 参数 analysis_type    ：分析类型（PROMPTS 字典里的键名）
# 参数 custom_question  ：自定义提问的内容（只有选"自定义提问"时才用）
# 返回：模型生成的分析结果（字符串）
# ---------------------------------------------------------------
def ai_analyze(text, analysis_type, custom_question=""):
    # 第一步：创建一个"客户端"对象，它负责和网络上的 DeepSeek 服务器通信
    # api_key  从环境变量读取，而不是写死在代码里（安全习惯！）
    # base_url 告诉库"我们要连的是 DeepSeek 的服务器"，不是 OpenAI 官方
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # 第二步：根据用户选择的类型，组装出最终的提示词
    if analysis_type == "自定义提问":
        # 自定义模式：问题 + 文本一起交给 AI
        prompt = f"""{CUSTOM_PREFIX}

用户的问题：{custom_question}

待分析文本：
{text}"""
    else:
        # 预设模式：从字典里取出对应模板，再把文本拼在后面
        # 用字符串相加而不是 f-string，是为了避免文本里的花括号引发格式错误
        prompt = PROMPTS[analysis_type] + "\n\n待分析文本：\n" + text

    # 第三步：真正发起调用
    # model     : 用哪个模型，deepseek-chat 是通用对话模型
    # messages  : 对话历史列表。system 是"人设设定"，user 是用户的请求
    # temperature: 随机度，0~1 之间。越低回答越稳定，分析类任务用低值更合适
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            # system 消息是"人设"：告诉 AI 它是谁、该用什么风格说话
            {"role": "system", "content": "你是一位资深的文本分析专家，擅长挖掘文本的情感、结构与要点。回答条理清晰、论证充分，每个观点都要有原文依据，使用中文。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    # 第四步：从返回结果里取出 AI 说的话
    return response.choices[0].message.content


# ---------------------------------------------------------------
# 设置浏览器标签页信息
# ---------------------------------------------------------------
st.set_page_config(
    page_title="我的第一个 AI 文本分析工具",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------
# 页面大标题
# ---------------------------------------------------------------
st.title("我的第一个 AI 文本分析工具")

# ---------------------------------------------------------------
# 侧边栏：放"分析设置"
# with st.sidebar: 表示下面这些组件都放进左侧边栏里
# ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 分析设置")

    # st.radio 单选按钮组
    # 选项 = 字典里所有键名 + "自定义提问"
    options = list(PROMPTS.keys()) + ["自定义提问"]
    analysis_type = st.radio("选择分析类型", options)

    # 只有选了"自定义提问"，才显示输入框让用户填问题
    custom_question = ""
    if analysis_type == "自定义提问":
        custom_question = st.text_input(
            "你的问题",
            placeholder="例如：这段文字适合发给客户吗？",
        )

    st.divider()
    # st.caption 显示灰色小字说明
    st.caption("💡 分析结果由 DeepSeek 大模型生成，仅供参考。")

# ---------------------------------------------------------------
# 多行文本输入框
# ---------------------------------------------------------------
user_text = st.text_area(
    label="输入文本",
    placeholder="请粘贴你的文本",
    height=200,
)

# ---------------------------------------------------------------
# 按钮：点击后开始分析
# ---------------------------------------------------------------
if st.button("开始分析"):

    if user_text.strip():

        # ============================================================
        # 第一部分：基础统计
        # ============================================================
        total_chars = len(user_text)
        line_count = len(user_text.splitlines())
        paragraphs = [p for p in user_text.split("\n\n") if p.strip()]
        para_count = len(paragraphs)

        st.success("分析完成！")

        col1, col2, col3 = st.columns(3)
        col1.metric("字符数", total_chars)
        col2.metric("行数", line_count)
        col3.metric("段落数", para_count)

        st.divider()

        # ============================================================
        # 第二部分：词频统计 + 柱状图
        # ============================================================
        st.subheader("📊 词频 Top 10")

        words = extract_words(user_text)

        if words:
            counter = Counter(words)
            top10 = counter.most_common(10)
            df = pd.DataFrame(top10, columns=["词", "次数"]).set_index("词")
            st.bar_chart(df)

            with st.expander("📋 查看完整数据表"):
                st.dataframe(df, use_container_width=True)
        else:
            st.info("没有识别到英文单词或中文字符。")

        st.divider()

        # ============================================================
        # 第三部分：AI 智能分析
        # ============================================================
        # 标题会跟着侧边栏的选择变化，让用户知道当前在做什么
        st.subheader(f"🤖 AI 智能分析 · {analysis_type}")

        # 先检查环境变量里有没有配置 Key
        if not os.environ.get("DEEPSEEK_API_KEY"):
            st.error("未找到 DEEPSEEK_API_KEY 环境变量，请先配置后再使用 AI 分析功能。")
        elif analysis_type == "自定义提问" and not custom_question.strip():
            # 选了自定义提问却没填问题，提示用户补充
            st.warning("你在侧边栏选择了「自定义提问」，请先填写你的问题。")
        else:
            # st.spinner 会在调用期间显示一个"转圈圈"的加载提示
            with st.spinner("AI 正在分析中，请稍候（通常几秒钟）..."):
                try:
                    # 把文本、分析类型、自定义问题一起传给函数
                    result = ai_analyze(user_text, analysis_type, custom_question)
                    # st.markdown 能渲染 Markdown 格式（标题、加粗、列表等）
                    st.markdown(result)
                except Exception as e:
                    # 如果调用出错（网络问题、余额不足等），把错误显示出来
                    st.error(f"AI 分析失败：{e}")

        st.divider()

        # ============================================================
        # 第四部分：折叠面板显示原文
        # ============================================================
        with st.expander("📄 查看原文"):
            st.text(user_text)

    else:
        st.warning("请先粘贴一些文本，再点击「开始分析」。")
