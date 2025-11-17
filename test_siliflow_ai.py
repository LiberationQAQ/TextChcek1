#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试使用硅基流动API
"""

import sys
import os
import json

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_siliflow_ai_direct():
    """
    直接测试硅基流动AI客户端
    """
    print("直接测试硅基流动AI客户端...")
    
    # 从LangChain.py导入配置
    try:
        from LangChain import SILIFLOW_API_KEY, SILIFLOW_API_BASE, SILIFLOW_MODEL
    except ImportError as e:
        print(f"导入配置失败: {e}")
        return False
    
    if SILIFLOW_API_KEY == "your-siliflow-api-key-here":
        print("请在LangChain.py中设置有效的硅基流动API密钥")
        return False
    
    try:
        import requests
        
        headers = {
            "Authorization": f"Bearer {SILIFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": SILIFLOW_MODEL,
            "messages": [
                {"role": "user", "content": "你好，请介绍一下自己的主要功能和特点。"}
            ],
            "max_tokens": 4096,
            "temperature": 0.6
        }
        
        print(f"正在调用硅基流动API...")
        print(f"API地址: {SILIFLOW_API_BASE}/chat/completions")
        print(f"使用模型: {SILIFLOW_MODEL}")
        
        response = requests.post(
            f"{SILIFLOW_API_BASE}/chat/completions",
            headers=headers,
            json=data
        )
        
        # 检查响应状态
        if response.status_code == 200:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            print("\n✓ 直接调用硅基流动AI成功:")
            print(f"回复: {content}")
            return True
        else:
            print(f"\n✗ 调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ 直接调用硅基流动AI失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_system():
    """
    测试集成到现有系统中的硅基流动AI
    """
    print("\n测试集成到现有系统中的硅基流动AI...")
    try:
        from LangChain import TextErrorCheckSystem, USE_SILIFLOW_AI
        
        # 检查是否已启用硅基流动AI
        if not USE_SILIFLOW_AI:
            print("⚠️  注意：当前未启用硅基流动API")
            print("请在LangChain.py中将USE_SILIFLOW_AI设置为True")
            return False
        
        # 初始化系统
        print("正在初始化文本错误检查系统...")
        system = TextErrorCheckSystem()
        print("✓ 系统初始化成功")
        
        # 测试文本
        test_texts = [
            "昨天我去了公园，看到了很多美丽的花儿。它们的颜色很漂亮，有红色、蓝色和绿色。",
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "太阳从西边升起，然后我吃了早饭。水的沸点是50度，这个温度很适合泡茶。"
        ]
        
        # 演示基础检查
        print("\n1. 基础错误检查演示:")
        print("-" * 30)
        print(f"测试文本: {test_texts[0]}")
        
        basic_result = system.basic_checker.check(test_texts[0])
        print(f"✓ 检查完成")
        print(f"是否有错误: {basic_result.get('has_errors', False)}")
        print(f"错误数量: {len(basic_result.get('errors', []))}")
        
        if basic_result.get('errors'):
            print("\n发现的错误:")
            for error in basic_result.get('errors', []):
                print(f"- {error.get('type', '未知错误')}: {error.get('description', '')}")
        
        # 演示专业检查
        print("\n2. 专业错误检查演示:")
        print("-" * 30)
        print(f"测试文本: {test_texts[1]}")
        
        professional_result = system.pro_checker.check(test_texts[1])
        print(f"✓ 检查完成")
        
        # 解析专业检查结果
        if "final_assessment_parsed" in professional_result:
            final_assessment = professional_result["final_assessment_parsed"]
            if isinstance(final_assessment, dict):
                print(f"\n综合评估: {final_assessment.get('overall_quality', '')}")
                print(f"总分: {final_assessment.get('overall_score', '')}/100")
                
                if "dimensions" in final_assessment:
                    print("\n各维度评分:")
                    for dimension, score in final_assessment["dimensions"].items():
                        print(f"- {dimension}: {score}/25")
            else:
                print(f"\n综合评估: {final_assessment}")
        elif "final_assessment" in professional_result:
            final_assessment = professional_result["final_assessment"]
            if isinstance(final_assessment, dict):
                print(f"\n综合评估: {final_assessment.get('overall_quality', '')}")
                print(f"总分: {final_assessment.get('total_score', '')}/100")
                
                if "dimensions" in final_assessment:
                    print("\n各维度评分:")
                    for dimension, score in final_assessment["dimensions"].items():
                        print(f"- {dimension}: {score}/25")
            else:
                print(f"\n综合评估: {final_assessment}")
        else:
            print(f"\n未找到评估结果: {professional_result}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试集成系统失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 硅基流动API调用测试 ===")
    
    # 测试直接调用
    test_siliflow_ai_direct()
    
    # 测试集成系统
    test_integrated_system()
    
    print("\n=== 测试完成 ===")