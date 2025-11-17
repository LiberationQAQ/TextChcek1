# 文本错误检查系统使用说明

## 功能简介

本系统基于LangChain实现，提供多维度的文本错误检查功能，包括：
- 基础错误检查（语法、拼写、标点、逻辑、事实等）
- 专业错误检查（语法、事实、逻辑、风格等多维度）
- 领域特定检查（学术、新闻、商业等）
- 批量检查功能
- 交互式检查功能

## 安装依赖

```bash
# 使用提供的Python解释器安装依赖
C:/Users/32212/python-sdk/python3.13.2/python.exe -m pip install pandas langchain==0.1.17 langchain-community>=0.0.36
```

## 设置API密钥

本系统支持使用OpenAI官方API、能用AI API或硅基流动API来执行文本错误检查。

## 使用硅基流动API

硅基流动提供了高性能的AI模型服务，您可以按照以下步骤配置：

1. 打开 `LangChain.py` 文件
2. 在文件开头找到API配置部分：

```python
# 硅基流动API配置
USE_SILIFLOW_AI = False  # 设置为True使用硅基流动AI，False使用其他API
SILIFLOW_API_KEY = "your-siliflow-api-key-here"  # 替换为您的硅基流动API密钥
SILIFLOW_API_BASE = "https://api.siliflow.com/v1"  # 硅基流动API地址
SILIFLOW_MODEL = "siliflow-llm-7b"  # 硅基流动模型名称
```

3. 将 `SILIFLOW_API_KEY` 的值替换为您的硅基流动API密钥
4. 将 `USE_SILIFLOW_AI` 设置为 `True`
5. 确保其他API的 `USE_*` 参数设置为 `False`

## 使用智谱AI API

智谱AI提供了强大的中文AI模型，您可以按照以下步骤配置：

1. 打开 `LangChain.py` 文件
2. 在文件开头找到API配置部分：

```python
# AI API配置
USE_ZHIPU_AI = True  # 设置为True使用智谱AI，False使用其他API
ZHIPU_API_KEY = "your-api-key-here"  # 替换为您的智谱AI API密钥
ZHIPU_MODEL = "glm-4.5-flash"  # 智谱AI模型名称
```

3. 将 `ZHIPU_API_KEY` 的值替换为您的智谱AI API密钥
4. 将 `USE_ZHIPU_AI` 设置为 `True`
5. 确保其他API的 `USE_*` 参数设置为 `False`

## 使用能用AI API

能用AI提供了与OpenAI兼容的API接口，您可以按照以下步骤配置：

1. 打开 `LangChain.py` 文件
2. 在文件开头找到API配置部分：

```python
# 能用AI API配置（兼容旧配置）
USE_NENGYONG_AI = False  # 设置为True使用能用AI，False使用其他API
NENGYONG_API_KEY = "your-api-key-here"  # 替换为您的能用AI密钥
NENGYONG_API_BASE = "https://ai.nengyongai.cn/v1"  # 能用AI API地址
```

3. 将 `NENGYONG_API_KEY` 的值替换为您的能用AI密钥
4. 将 `USE_NENGYONG_AI` 设置为 `True`
5. 确保其他API的 `USE_*` 参数设置为 `False`

## 使用OpenAI官方API

如果您希望使用OpenAI官方API，请按照以下步骤配置：

### 方法1：环境变量（推荐）

在命令行中设置环境变量：

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

然后在 `LangChain.py` 文件中将 `USE_NENGYONG_AI` 设置为 `False`：

```python
USE_NENGYONG_AI = False
```

### 方法2：直接修改代码

在 `LangChain.py` 文件中：
1. 将 `USE_NENGYONG_AI` 设置为 `False`
2. 找到 `get_openai_instance` 函数，修改 `OpenAI` 实例化代码添加api_key参数：

```python
def get_openai_instance(temperature=0):
    if USE_NENGYONG_AI:
        return OpenAI(
            temperature=temperature,
            openai_api_key=NENGYONG_API_KEY,
            openai_api_base=NENGYONG_API_BASE
        )
    else:
        return OpenAI(temperature=temperature, openai_api_key="your-api-key-here")
```

## 运行系统

```bash
# 使用提供的Python解释器运行
C:/Users/32212/python-sdk/python3.13.2/python.exe LangChain.py
```

## 系统结构

- `LangChain.py`: 主程序文件，包含所有检查器实现
- `test_error.py`: 测试脚本（可选）
- `USAGE.md`: 使用说明

## 主要类说明

1. **ErrorCheckOutputParser**: 解析LLM响应，提取结构化错误信息
2. **BasicErrorChecker**: 基础错误检查器
3. **ProfessionalErrorChecker**: 专业错误检查器
4. **DomainSpecificChecker**: 领域特定检查器
5. **InteractiveErrorChecker**: 交互式错误检查器
6. **BatchErrorChecker**: 批量错误检查器
7. **TextErrorCheckSystem**: 整合所有功能的主系统

## 自定义扩展

您可以通过以下方式扩展系统功能：

1. 在`DomainSpecificChecker._get_domain_knowledge()`中添加新的领域知识
2. 自定义`ErrorCheckOutputParser`来支持不同的输出格式
3. 在`ProfessionalErrorChecker._setup_chains()`中添加新的检查维度

## 注意事项

1. 确保您的OpenAI API密钥有足够的配额
2. 处理长文本时会自动分块处理
3. 对于敏感内容，请确保遵守OpenAI的使用政策

## 常见问题

**Q: 运行时提示"Did not find openai_api_key"怎么办？**
A: 请按照上述说明设置OpenAI API密钥。

**Q: 为什么检查结果为空？**
A: 可能是因为文本没有明显错误，或者OpenAI API响应格式不符合预期。

**Q: 如何提高检查准确性？**
A: 可以尝试调整OpenAI的temperature参数，或者提供更具体的领域信息。