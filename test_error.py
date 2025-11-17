import sys
import traceback

try:
    import LangChain
    print("模块导入成功")
    
    # 测试基本功能
    system = LangChain.TextErrorCheckSystem()
    print("系统初始化成功")
    
    test_text = "这是一个简单的测试文本。"
    result = system.basic_checker.check(test_text)
    print("基础检查完成:", result)
    
except Exception as e:
    print(f"发生错误: {type(e).__name__}: {e}")
    traceback.print_exc()
