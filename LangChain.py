"""
文本错误检查系统
基于LangChain实现的多维度文本错误检查工具
包含语法检查、事实核查、逻辑验证、领域专业检查等功能
"""

import json
import pandas as pd
import requests
from typing import Dict, List, Any, Optional
import os
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI, BaseLLM
from langchain.schema import BaseOutputParser, LLMResult, Generation
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain.text_splitter import CharacterTextSplitter
from zai import ZhipuAiClient

# AI API配置
USE_ZHIPU_AI = False  # 设置为True使用智谱AI，False使用其他API
ZHIPU_API_KEY = "e94d262fd7e640658de466f8ca64c0d8.fyc1XV2hH76ZE2lv"  # 替换为您的智谱AI API密钥
ZHIPU_MODEL = "glm-4.5-flash"  # 智谱AI模型名称

# 能用AI API配置（兼容旧配置）
USE_NENGYONG_AI = False  # 设置为True使用能用AI，False使用其他API
NENGYONG_API_KEY = "sk-BclU8ezEFdfe46j4pN0kaOKbY0tOubuYXby2Q6kYq6YKUsV9"  # 替换为您的能用AI密钥
NENGYONG_API_BASE = "https://ai.nengyongai.cn/v1"  # 能用AI API地址（添加了/v1后缀）

# 硅基流动API配置
USE_SILIFLOW_AI = True  # 设置为True使用硅基流动AI，False使用其他API
SILIFLOW_API_KEY = "sk-kqyukqomgufpgvnbfiqycafpkjafemneruzinufhappbcgcl"  # 替换为您的硅基流动API密钥
SILIFLOW_API_BASE = "https://api.siliconflow.cn/v1"  # 硅基流动API地址（修正为正确的地址）
SILIFLOW_MODEL = "Qwen/Qwen2-7B-Instruct"  # 硅基流动模型名称

# 配置OpenAI实例的函数
class ZhipuAILM(BaseLLM):
    """
    智谱AI LLM包装类，兼容langchain的调用接口
    """
    temperature: float = 0
    api_key: str = None
    model: str = None
    client: Any = None  # 声明client字段
    
    def __init__(self, temperature=0, api_key=None, model=None, **kwargs):
        """
        初始化智谱AI LLM
        
        Args:
            temperature: 模型生成温度
            api_key: 智谱AI API密钥
            model: 智谱AI模型名称
        """
        # 传递参数给父类构造函数
        super().__init__(
            temperature=temperature,
            api_key=api_key or ZHIPU_API_KEY,
            model=model or ZHIPU_MODEL,
            **kwargs
        )
        # 初始化客户端
        self.client = ZhipuAiClient(api_key=self.api_key)
    
    @property
    def _llm_type(self) -> str:
        """
        返回LLM类型
        """
        return "zhipuai"
    
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """
        调用智谱AI生成文本（实现BaseLLM的抽象方法）
        """
        results = []
        
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,          # 最大输出 tokens
                temperature=self.temperature,           # 控制输出的随机性
                **kwargs
            )
            
            result = response.choices[0].message.content
            
            # 处理停止条件
            if stop:
                for stop_token in stop:
                    if stop_token in result:
                        result = result.split(stop_token)[0]
            
            results.append(result)
        
        # 构建LLMResult对象
        return LLMResult(
            generations=[[Generation(text=result)] for result in results]
        )
    
    def _call(self, prompt: str, stop: list = None, run_manager=None, **kwargs) -> str:
        """
        调用智谱AI模型生成文本（为了兼容旧版本）
        """
        result = self._generate([prompt], stop=stop, **kwargs)
        return result.generations[0][0].text
    
    @property
    def _identifying_params(self) -> dict:
        """
        返回用于标识模型的参数
        """
        return {
            "temperature": self.temperature,
            "api_key": self.api_key,
            "model": self.model
        }

class SiliflowAILM(BaseLLM):
    """
    硅基流动AI LLM包装类，兼容langchain的调用接口
    """
    temperature: float = 0
    api_key: str = None
    model: str = None
    api_base: str = None
    client: Any = None  # 声明client字段
    
    def __init__(self, temperature=0, api_key=None, model=None, api_base=None, **kwargs):
        """
        初始化硅基流动AI LLM
        
        Args:
            temperature: 模型生成温度
            api_key: 硅基流动AI API密钥
            model: 硅基流动AI模型名称
            api_base: 硅基流动AI API地址
        """
        # 传递参数给父类构造函数
        super().__init__(
            temperature=temperature,
            api_key=api_key or SILIFLOW_API_KEY,
            model=model or SILIFLOW_MODEL,
            api_base=api_base or SILIFLOW_API_BASE,
            **kwargs
        )
        
    @property
    def _llm_type(self) -> str:
        """
        返回LLM类型
        """
        return "siliflowai"
    
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """
        调用硅基流动AI生成文本（实现BaseLLM的抽象方法）
        """
        import requests
        results = []
        
        for prompt in prompts:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4096,
                "temperature": self.temperature,
                **kwargs
            }
            
            # 发送请求到硅基流动API
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )
            
            # 解析响应
            response_data = response.json()
            result = response_data["choices"][0]["message"]["content"]
            
            # 处理停止条件
            if stop:
                for stop_token in stop:
                    if stop_token in result:
                        result = result.split(stop_token)[0]
            
            results.append(result)
        
        # 构建LLMResult对象
        return LLMResult(
            generations=[[Generation(text=result)] for result in results]
        )
    
    def _call(self, prompt: str, stop: list = None, run_manager=None, **kwargs) -> str:
        """
        调用硅基流动AI模型生成文本（为了兼容旧版本）
        """
        result = self._generate([prompt], stop=stop, **kwargs)
        return result.generations[0][0].text
    
    @property
    def _identifying_params(self) -> dict:
        """
        返回用于标识模型的参数
        """
        return {
            "temperature": self.temperature,
            "api_key": self.api_key,
            "model": self.model,
            "api_base": self.api_base
        }

def get_openai_instance(temperature=0):
    """
    创建并返回LLM实例，根据配置决定使用智谱AI、能用AI、硅基流动AI还是OpenAI官方API
    
    Args:
        temperature: 模型生成温度
        
    Returns:
        LLM实例
    """
    if USE_SILIFLOW_AI:
        return SiliflowAILM(
            temperature=temperature,
            api_key=SILIFLOW_API_KEY,
            model=SILIFLOW_MODEL,
            api_base=SILIFLOW_API_BASE
        )
    elif USE_ZHIPU_AI:
        return ZhipuAILM(
            temperature=temperature,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL
        )
    elif USE_NENGYONG_AI:
        return OpenAI(
            temperature=temperature,
            openai_api_key=NENGYONG_API_KEY,
            openai_api_base=NENGYONG_API_BASE,
            model="gpt-3.5-turbo-instruct",
            max_tokens=2000
        )
    else:
        return OpenAI(
            temperature=temperature,
            max_tokens=2000
        )

class ErrorCheckOutputParser(BaseOutputParser):
    """
    错误检查输出解析器
    用于解析LLM的响应，提取结构化的错误信息
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """
        解析LLM响应文本，提取错误信息

        Args:
            text: LLM的响应文本

        Returns:
            结构化的错误信息字典
        """
        try:
            # 替换转义换行符为实际换行符
            text = text.replace("\\n", "\n")
            
            # 尝试解析JSON格式的响应
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                
                # 尝试修复可能不完整的JSON响应
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # 尝试在字符串末尾添加缺失的括号
                    for i in range(3):
                        try:
                            fixed_json = json_str + """}
                        ]}"""[i:]
                            return json.loads(fixed_json)
                        except json.JSONDecodeError:
                            continue
                    raise
            else:
                # 如果没有JSON格式标记，尝试直接解析整个文本
                try:
                    return json.loads(text.strip())
                except json.JSONDecodeError:
                    # 如果没有JSON格式，返回原始响应，不默认标记为有错误
                    return {"raw_response": text, "has_errors": False}
        except Exception as e:
            # 解析失败时，返回原始响应，不默认标记为有错误
            return {"raw_response": text, "has_errors": False, "parse_error": str(e)}


class BasicErrorChecker:
    """
    基础错误检查器
    提供语法、拼写、标点等基础错误检查功能
    """

    def __init__(self, llm=None):
        """
        初始化基础错误检查器

        Args:
            llm: 语言模型实例，如果为None则创建新的OpenAI实例
        """
        self.llm = llm or get_openai_instance()
        self.chain, self.parser = self._setup_chain()

    def _setup_chain(self):
        """
        设置错误检查链和解析器

        Returns:
            tuple: (检查链, 输出解析器)
        """
        error_check_template = """
        请仔细检查以下【待检查文本】中的错误，仅检查这部分内容，不要检查其他任何内容！
        错误类型包括：
        1. 语法错误
        2. 拼写错误  
        3. 标点符号错误
        4. 逻辑不一致
        5. 事实错误
        6. 表达不清的地方

        【待检查文本】：
        {text}

        【回复格式要求】：
        请严格按照以下JSON格式输出结果，不要添加任何其他解释或说明：
        ```json
        {{
            "has_errors": true/false,
            "errors": [
                {{
                    "type": "错误类型",
                    "position": "错误位置描述（相对于待检查文本的位置）", 
                    "original": "原文本中的错误部分",
                    "suggestion": "修改建议",
                    "reason": "错误原因"
                }}
            ],
            "corrected_text": "修正后的完整文本"
        }}
        ```
        """

        prompt = PromptTemplate(
            input_variables=["text"],
            template=error_check_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        parser = ErrorCheckOutputParser()

        return chain, parser

    def check(self, text: str) -> Dict[str, Any]:
        """
        执行基础错误检查

        Args:
            text: 要检查的文本

        Returns:
            包含错误信息的字典
        """
        result = self.chain.run(text)
        parsed_result = self.parser.parse(result)
        return parsed_result


class ProfessionalErrorChecker:
    """
    专业错误检查器
    提供多维度专业检查：语法、事实、逻辑、表达清晰度
    """

    def __init__(self, llm=None):
        """
        初始化专业错误检查器

        Args:
            llm: 语言模型实例
        """
        self.llm = llm or get_openai_instance()
        self.chains = self._setup_chains()

    def _setup_chains(self):
        """
        设置多维度检查链

        Returns:
            SequentialChain: 顺序执行链
        """
        # 语法检查
        grammar_template = """
        检查以下文本的语法和拼写错误：
        {text}

        回复格式：
        错误数量: [数字]
        具体错误: [错误描述]
        """
        grammar_prompt = PromptTemplate(
            input_variables=["text"],
            template=grammar_template
        )
        grammar_chain = LLMChain(
            llm=self.llm,
            prompt=grammar_prompt,
            output_key="grammar_check"
        )

        # 事实检查
        fact_check_template = """
        检查以下文本中的事实错误或不准确信息：
        {text}

        回复格式：
        事实问题: [问题描述]
        """
        fact_check_prompt = PromptTemplate(
            input_variables=["text"],
            template=fact_check_template
        )
        fact_chain = LLMChain(
            llm=self.llm,
            prompt=fact_check_prompt,
            output_key="fact_check"
        )

        # 逻辑检查
        logic_template = """
        检查以下文本中的逻辑不一致或矛盾之处：
        {text}

        回复格式：
        逻辑问题: [问题描述]
        """
        logic_prompt = PromptTemplate(
            input_variables=["text"],
            template=logic_template
        )
        logic_chain = LLMChain(
            llm=self.llm,
            prompt=logic_prompt,
            output_key="logic_check"
        )

        # 表达清晰度检查
        clarity_template = """
        评估以下文本的表达清晰度，找出表达不清的地方：
        {text}

        回复格式：
        清晰度问题: [问题描述]
        """
        clarity_prompt = PromptTemplate(
            input_variables=["text"],
            template=clarity_template
        )
        clarity_chain = LLMChain(
            llm=self.llm,
            prompt=clarity_prompt,
            output_key="clarity_check"
        )

        # 综合评估
        summary_template = """
        基于以下检查结果，给出综合评估和改进建议：

        语法检查: {grammar_check}
        事实检查: {fact_check}
        逻辑检查: {logic_check}
        清晰度检查: {clarity_check}

        原始文本: {text}

        回复格式：
        ```json
        {{
            "overall_score": "分数/10",
            "main_issues": ["主要问题1", "主要问题2"],
            "improvement_suggestions": ["建议1", "建议2"],
            "corrected_version": "改进后的文本"
        }}
        ```
        """
        summary_prompt = PromptTemplate(
            input_variables=["grammar_check", "fact_check", "logic_check", "clarity_check", "text"],
            template=summary_template
        )
        summary_chain = LLMChain(
            llm=self.llm,
            prompt=summary_prompt,
            output_key="final_assessment"
        )

        return SequentialChain(
            chains=[grammar_chain, fact_chain, logic_chain, clarity_chain, summary_chain],
            input_variables=["text"],
            output_variables=["grammar_check", "fact_check", "logic_check", "clarity_check", "final_assessment"],
            verbose=False
        )

    def check(self, text: str) -> Dict[str, Any]:
        """
        执行全面专业错误检查

        Args:
            text: 要检查的文本

        Returns:
            包含多维度检查结果的字典
        """
        result = self.chains({"text": text})
        
        # 使用与BasicErrorChecker相同的解析逻辑处理换行符
        for key, value in result.items():
            if isinstance(value, str):
                # 替换转义换行符为实际换行符
                result[key] = value.replace("\\n", "\n")
        
        try:
            # 解析最终的JSON评估
            if "```json" in result["final_assessment"]:
                json_str = result["final_assessment"].split("```json")[1].split("```")[0].strip()
                result["final_assessment_parsed"] = json.loads(json_str)
        except json.JSONDecodeError:
            result["final_assessment_parsed"] = {"raw": result["final_assessment"]}

        return result


class DomainSpecificChecker:
    """
    领域特定错误检查器
    针对不同领域（学术、技术、商业、医疗等）进行专业检查
    """

    def __init__(self, domain="general", llm=None):
        """
        初始化领域特定检查器

        Args:
            domain: 领域类型 (academic, technical, business, medical, general)
            llm: 语言模型实例
        """
        self.llm = llm or get_openai_instance()
        self.domain = domain
        self.domain_knowledge = self._get_domain_knowledge()

    def _get_domain_knowledge(self) -> Dict[str, Any]:
        """
        获取领域特定知识配置

        Returns:
            领域知识配置字典
        """
        knowledge = {
            "academic": {
                "name": "学术论文",
                "checks": ["引用格式", "学术规范", "术语准确性", "论证逻辑", "文献引用"]
            },
            "technical": {
                "name": "技术文档",
                "checks": ["技术术语", "代码示例", "步骤逻辑", "准确性", "兼容性说明"]
            },
            "business": {
                "name": "商业文档",
                "checks": ["数据准确性", "商业术语", "逻辑连贯性", "专业性", "合规性"]
            },
            "medical": {
                "name": "医疗健康",
                "checks": ["医学术语", "事实准确性", "安全信息", "专业规范", "剂量准确性"]
            },
            "general": {
                "name": "通用文档",
                "checks": ["语法正确性", "事实准确性", "逻辑一致性", "表达清晰度"]
            }
        }
        return knowledge.get(self.domain, knowledge["general"])

    def check(self, text: str) -> Dict[str, Any]:
        """
        执行领域特定错误检查

        Args:
            text: 要检查的文本

        Returns:
            领域特定检查结果字典
        """
        template = """
        作为{domain_name}领域的专家，请检查以下文本的专业性错误：

        需要特别关注的方面：
        {checks}

        文本内容：
        {text}

        请指出：
        1. 专业术语使用是否正确
        2. 领域特定事实是否准确  
        3. 是否符合领域规范和标准
        4. 其他专业性问题

        回复格式：
        ```json
        {{
            "domain_issues": [
                {{
                    "type": "问题类型",
                    "description": "详细描述",
                    "suggestion": "改进建议",
                    "severity": "高/中/低"
                }}
            ],
            "domain_specific_score": "领域专业性评分/10",
            "overall_assessment": "总体评价"
        }}
        ```
        """

        prompt = PromptTemplate(
            input_variables=["domain_name", "checks", "text"],
            template=template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        checks_str = "\n".join([f"- {check}" for check in self.domain_knowledge["checks"]])

        result = chain.run({
            "domain_name": self.domain_knowledge["name"],
            "checks": checks_str,
            "text": text
        })

        # 替换转义换行符为实际换行符，与BasicErrorChecker保持一致
        result = result.replace("\\n", "\n")

        # 解析结果
        try:
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                return {"raw_response": result, "domain_issues": []}
        except json.JSONDecodeError:
            return {"raw_response": result, "domain_issues": []}


class InteractiveErrorChecker:
    """
    交互式错误检查器
    提供用户交互式检查和实时反馈功能
    """

    def __init__(self, llm=None):
        """
        初始化交互式检查器

        Args:
            llm: 语言模型实例
        """
        self.llm = llm or get_openai_instance()
        self.memory = ConversationBufferMemory()
        self.tools = self._setup_tools()

    def _setup_tools(self) -> List[Tool]:
        """
        设置检查工具集

        Returns:
            工具列表
        """
        tools = [
            Tool(
                name="grammar_check",
                func=self._grammar_check,
                description="检查语法和拼写错误"
            ),
            Tool(
                name="style_suggest",
                func=self._style_suggest,
                description="提供写作风格改进建议"
            ),
            Tool(
                name="clarify_meaning",
                func=self._clarify_meaning,
                description="澄清模糊表达"
            ),
            Tool(
                name="fact_verify",
                func=self._fact_verify,
                description="验证事实准确性"
            )
        ]

        return tools

    def _grammar_check(self, text: str) -> str:
        """语法检查工具"""
        prompt = f"检查以下文本的语法错误，给出具体修改建议：{text}"
        return self.llm(prompt)

    def _style_suggest(self, text: str) -> str:
        """风格建议工具"""
        prompt = f"为以下文本提供写作风格改进建议：{text}"
        return self.llm(prompt)

    def _clarify_meaning(self, text: str) -> str:
        """澄清含义工具"""
        prompt = f"找出以下文本中表达不清的地方，并提供更清晰的表达方式：{text}"
        return self.llm(prompt)

    def _fact_verify(self, text: str) -> str:
        """事实验证工具"""
        prompt = f"验证以下文本中的事实准确性：{text}"
        return self.llm(prompt)

    def interactive_check(self, text: str) -> Dict[str, Any]:
        """
        交互式错误检查

        Args:
            text: 要检查的文本

        Returns:
            检查结果字典
        """
        print(f"检查文本: {text}")
        print("\n请选择检查类型:")
        print("1. 语法检查")
        print("2. 风格建议")
        print("3. 表达澄清")
        print("4. 事实验证")
        print("5. 全面检查")

        choice = input("请输入选择 (1-5): ")

        if choice == "1":
            result = self._grammar_check(text)
            return {"check_type": "grammar", "result": result}
        elif choice == "2":
            result = self._style_suggest(text)
            return {"check_type": "style", "result": result}
        elif choice == "3":
            result = self._clarify_meaning(text)
            return {"check_type": "clarity", "result": result}
        elif choice == "4":
            result = self._fact_verify(text)
            return {"check_type": "fact", "result": result}
        elif choice == "5":
            results = {}
            results["grammar"] = self._grammar_check(text)
            results["style"] = self._style_suggest(text)
            results["clarity"] = self._clarify_meaning(text)
            results["fact"] = self._fact_verify(text)
            return {"check_type": "comprehensive", "results": results}
        else:
            return {"check_type": "invalid", "result": "无效选择"}


class BatchErrorChecker:
    """
    批量文本错误检查器
    支持批量处理多个文本，自动分块处理长文本
    """

    def __init__(self, llm=None):
        """
        初始化批量检查器

        Args:
            llm: 语言模型实例
        """
        self.llm = llm or get_openai_instance()
        self.text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.basic_checker = BasicErrorChecker(llm)

    def check_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        批量检查文本列表

        Args:
            texts: 文本列表

        Returns:
            包含检查结果的DataFrame
        """
        results = []

        for i, text in enumerate(texts):
            print(f"检查第 {i + 1}/{len(texts)} 个文本...")

            # 如果文本过长，进行分割处理
            if len(text) > 1000:
                chunks = self.text_splitter.split_text(text)
                chunk_results = []

                for j, chunk in enumerate(chunks):
                    print(f"  处理分块 {j + 1}/{len(chunks)}...")
                    chunk_result = self.basic_checker.check(chunk)
                    chunk_results.append(chunk_result)

                # 合并分块结果
                combined_result = self._combine_chunk_results(chunk_results, text)
                results.append(combined_result)
            else:
                result = self.basic_checker.check(text)
                results.append(result)

        # 转换为DataFrame
        df_data = []
        for i, result in enumerate(results):
            df_data.append({
                "text_id": i + 1,
                "text_preview": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "has_errors": result.get("has_errors", False),
                "error_count": len(result.get("errors", [])),
                "errors": str(result.get("errors", [])),
                "corrected_text": result.get("corrected_text", "")[:200] + "..." if len(
                    result.get("corrected_text", "")) > 200 else result.get("corrected_text", "")
            })

        return pd.DataFrame(df_data)

    def _combine_chunk_results(self, chunk_results: List[Dict], original_text: str) -> Dict[str, Any]:
        """
        合并分块检查结果

        Args:
            chunk_results: 分块结果列表
            original_text: 原始文本

        Returns:
            合并后的结果字典
        """
        all_errors = []
        for result in chunk_results:
            all_errors.extend(result.get("errors", []))

        # 生成合并后的修正文本
        corrected_parts = []
        for result in chunk_results:
            if "corrected_text" in result:
                corrected_parts.append(result["corrected_text"])

        combined_corrected = " ".join(corrected_parts) if corrected_parts else original_text

        return {
            "text": original_text,
            "has_errors": len(all_errors) > 0,
            "error_count": len(all_errors),
            "errors": all_errors,
            "corrected_text": combined_corrected,
            "chunk_count": len(chunk_results)
        }


class TextErrorCheckSystem:
    """
    文本错误检查系统
    整合所有检查功能的统一接口
    """

    def __init__(self, llm=None):
        """
        初始化文本错误检查系统

        Args:
            llm: 语言模型实例
        """
        self.llm = llm or get_openai_instance()
        self.basic_checker = BasicErrorChecker(llm)
        self.pro_checker = ProfessionalErrorChecker(llm)
        self.batch_checker = BatchErrorChecker(llm)

    def comprehensive_check(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """
        执行综合错误检查

        Args:
            text: 要检查的文本
            domain: 领域类型

        Returns:
            综合检查结果字典
        """
        print("=== 执行综合错误检查 ===")

        results = {}

        # 基础检查
        print("1. 执行基础检查...")
        results["basic_check"] = self.basic_checker.check(text)

        # 专业检查
        print("2. 执行专业检查...")
        results["professional_check"] = self.pro_checker.check(text)

        # 领域特定检查
        print("3. 执行领域特定检查...")
        domain_checker = DomainSpecificChecker(domain, self.llm)
        results["domain_check"] = domain_checker.check(text)

        # 生成综合报告
        print("4. 生成综合报告...")
        results["summary"] = self._generate_summary(results, text)

        return results

    def _generate_summary(self, results: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """
        生成综合检查报告

        Args:
            results: 各检查模块结果
            original_text: 原始文本

        Returns:
            综合报告字典
        """
        basic_errors = len(results["basic_check"].get("errors", []))
        pro_score = results["professional_check"].get("final_assessment_parsed", {}).get("overall_score", "未知")
        domain_score = results["domain_check"].get("domain_specific_score", "未知")

        # 评估文本质量
        if basic_errors == 0 and "10" in str(pro_score):
            quality = "优秀"
        elif basic_errors <= 2 and "8" in str(pro_score):
            quality = "良好"
        elif basic_errors <= 5:
            quality = "一般"
        else:
            quality = "需要改进"

        return {
            "overall_quality": quality,
            "total_errors": basic_errors,
            "professional_score": pro_score,
            "domain_score": domain_score,
            "recommendation": self._get_recommendation(quality, basic_errors)
        }

    def _get_recommendation(self, quality: str, error_count: int) -> str:
        """
        根据检查结果生成改进建议

        Args:
            quality: 质量评级
            error_count: 错误数量

        Returns:
            改进建议字符串
        """
        if quality == "优秀":
            return "文本质量很好，无需修改"
        elif quality == "良好":
            return "文本质量良好，建议进行小幅优化"
        elif quality == "一般":
            return "文本需要改进，建议重点关注语法和表达清晰度"
        else:
            return "文本需要大幅修改，建议重新组织内容和修正错误"


def main():
    """
    主函数 - 演示文本错误检查系统的使用
    """
    print("=" * 60)
    print("文本错误检查系统演示")
    print("=" * 60)
    
    # 测试系统初始化
    print("\n正在初始化系统...")
    try:
        system = TextErrorCheckSystem()
        print("✓ 系统初始化成功")
    except Exception as e:
        print(f"✗ 系统初始化失败: {str(e)}")
        print("提示: 请确保已设置OpenAI API密钥")
        return

    # 测试文本
    test_texts = [
        "昨天我去了公园，看到了很多美丽的花儿。它们的颜色很漂亮，有红色、蓝色和绿色。",
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "太阳从西边升起，然后我吃了早饭。水的沸点是50度，这个温度很适合泡茶。"
    ]

    # 演示1: 基础检查
    print("\n1. 基础错误检查演示:")
    print("-" * 30)
    print(f"测试文本: {test_texts[0]}")
    try:
        basic_result = system.basic_checker.check(test_texts[0])
        print(f"✓ 检查完成")
        print(f"是否有错误: {basic_result.get('has_errors', False)}")
        print(f"错误数量: {len(basic_result.get('errors', []))}")
    except Exception as e:
        print(f"✗ 检查失败: {str(e)}")
        print("提示: 请确保已设置有效的OpenAI API密钥")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行演示
    main()

    # 交互式检查示例 (已启用)
    print("\n交互式检查示例:")
    interactive_checker = InteractiveErrorChecker()
    test_text = "这是一个需要检查的文本。"
    result = interactive_checker.interactive_check(test_text)
    print(f"检查结果: {result}")