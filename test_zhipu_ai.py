#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试使用新的智谱AI调用形式
"""

import sys
from zai import ZhipuAiClient
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_zhipu_ai_direct():
    """
    直接测试智谱AI客户端
    """
    print("直接测试智谱AI客户端...")
    try:
        client = ZhipuAiClient(api_key="e94d262fd7e640658de466f8ca64c0d8.fyc1XV2hH76ZE2lv")  # 请填写您自己的 API Key
        response = client.chat.completions.create(
            model="glm-4.5-flash",
            messages=[
                {"role": "user", "content": "你好，请介绍一下自己的主要功能和特点。"}
            ],
            enable_thinking_deeply=True,
            max_tokens=4096,          # 最大输出 tokens
            temperature=0.6           # 控制输出的随机性
        )
        # 获取完整回复
        print("直接调用智谱AI成功:")
        content = response.choices[0].message.content
        print(f"回复: {content}")  # 显示完整回复内容
        return True
    except Exception as e:
        print(f"直接调用智谱AI失败: {e}")
        return False

def test_integrated_system():
    """
    测试集成到现有系统中的智谱AI
    """
    print("\n测试集成到现有系统中的智谱AI...")
    try:
        from LangChain import TextErrorCheckSystem, get_openai_instance
        
        # 先测试get_openai_instance函数
        print("测试get_openai_instance函数...")
        llm = get_openai_instance()
        print(f"LLM实例类型: {type(llm)}")
        print(f"LLM实例: {llm}")
        
        # 创建文本错误检查系统实例
        print("\n创建TextErrorCheckSystem实例...")
        checker = TextErrorCheckSystem()
        print(f"TextErrorCheckSystem实例创建成功: {checker}")
        
        # 从文件中读取测试文本
        input_file = "test_input.txt"
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                test_text = f.read().strip()
            print(f"从文件 {input_file} 读取测试文本成功")
            print(f"测试文本: {test_text}")
        except FileNotFoundError:
            print(f"错误：文件 {input_file} 不存在")
            # 创建一个默认的测试文本文件
            default_text = "智能健身手环的核心价值在于健康数据监测，而心率监测是所有监测功能的基础。为什么这么说？因为所有健康问题都与心率相关，比如高血压、糖尿病等慢性疾病，都会直接导致心率异常。只要手环能精准监测心率，就说明它的传感器性能出色，那么它的睡眠监测、运动数据统计等功能自然也不会差。"
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(default_text)
            print(f"已创建默认测试文件 {input_file}")
            test_text = default_text
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return False
        print("\n注意：由于这只是测试接口兼容性，不会实际调用API（需要有效的API密钥）")
        print("如果配置了正确的API密钥，可以取消下面的注释进行完整测试")
        
        # 取消下面的注释进行完整测试
        import json
        result = checker.comprehensive_check(test_text)
        print("综合检查结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        return True
    except Exception as e:
        print(f"测试集成系统失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 智谱AI调用形式测试 ===")
    
    # 测试直接调用
    # test_zhipu_ai_direct()
    
    # 测试集成系统
    test_integrated_system()
    
    print("\n=== 测试完成 ===")