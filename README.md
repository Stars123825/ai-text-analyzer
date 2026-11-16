# AI 智能文本分析与生成工具

一个基于 Streamlit + DeepSeek 大模型的文本分析 Web 应用。粘贴一段文字，即可获得词频统计、情感分析、内容摘要与关键词提取。

## 功能

- **基础统计**：字符数、行数、段落数
- **词频分析**：中英文词频 Top 10 柱状图 + 完整数据表
- **AI 智能分析**（DeepSeek 大模型）
  - 综合深度分析（情感 + 总结 + 关键词）
  - 快速摘要
  - 只看情感倾向
  - 只提关键词
  - 自定义提问

## 技术栈

| 组件 | 用途 |
|------|------|
| Python 3.12 | 运行环境 |
| Streamlit | 网页界面 |
| OpenAI SDK | 调用大模型（DeepSeek 兼容 OpenAI 协议） |
| pandas | 数据处理与图表 |
| DeepSeek API | 文本分析能力 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

前往 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册并创建 API Key，然后把它设为环境变量：

**Windows PowerShell（永久生效）：**

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "你的APIKey", "User")
```

**macOS / Linux：**

```bash
export DEEPSEEK_API_KEY="你的APIKey"
```

> 设置完成后需要重新打开终端才会生效。

### 3. 启动应用

```bash
python -m streamlit run main.py
```

浏览器会自动打开 http://localhost:8501，如果没有请手动访问。

## 使用说明

1. 在文本框中粘贴待分析的文本
2. 在左侧边栏选择分析类型（或选择「自定义提问」后填写你的问题）
3. 点击「开始分析」

## 项目结构

```
ai-text-analyzer/
├── main.py            # 主程序：界面 + 分析逻辑
├── requirements.txt   # 依赖清单
├── .gitignore         # 排除缓存与敏感文件
└── README.md          # 本文件
```

## 安全说明

API Key 通过环境变量读取，**不会写入代码或提交到仓库**。请勿把密钥直接硬编码在源码中。

## License

MIT
