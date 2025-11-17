import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LangChain import ProfessionalErrorChecker, get_openai_instance
import json

# 创建专业错误检查器实例
print("正在创建ProfessionalErrorChecker实例...")
try:
    llm = get_openai_instance()
    checker = ProfessionalErrorChecker(llm)
    print("✓ 实例创建成功")
except Exception as e:
    print(f"✗ 实例创建失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 测试文本
test_text = "太阳从西边升起，然后我吃了早饭。水的沸点是50度，这个温度很适合泡茶。"

# 执行检查
print(f"\n正在检查文本: {test_text}")
try:
    result = checker.check(test_text)
    print("✓ 检查完成")
    
    # 打印原始结果
    print("\n原始结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 特别检查final_assessment_parsed
    print("\nfinal_assessment_parsed:")
    if "final_assessment_parsed" in result:
        print(json.dumps(result["final_assessment_parsed"], ensure_ascii=False, indent=2))
    else:
        print("未找到final_assessment_parsed字段")
        
except Exception as e:
    print(f"✗ 检查失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()