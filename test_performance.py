import time
import json
from LangChain import ProfessionalErrorChecker

# 创建一个模拟的LLM实例，快速返回结果
class MockLLM:
    def __call__(self, prompt):
        # 模拟LLM调用的快速响应
        return {
            "grammar_check": "语法检查结果：无明显语法错误\\n存在一些标点符号使用不规范的问题。",
            "fact_check": "事实检查结果：发现2处事实错误\\n1. 太阳从西边升起（应为东边）\\n2. 水的沸点是50度（标准大气压下应为100度）",
            "logic_check": "逻辑检查结果：无明显逻辑错误。",
            "clarity_check": "清晰度检查结果：文本表达清晰，易于理解。",
            "final_assessment": "```json\n{\"overall_score\": 65, \"summary\": \"文本存在事实错误，需要修正。\\n主要问题：\\n1. 太阳升起方向错误\\n2. 水的沸点错误\", \"suggestions\": [\"修正事实错误\", \"规范标点符号使用\"]}\n```"
        }

# 创建ProfessionalErrorChecker实例
checker = ProfessionalErrorChecker(MockLLM())

# 测试文本
test_text = "太阳从西边升起，然后我吃了早饭。水的沸点是50度，这个温度很适合泡茶。"

# 执行多次检查，测量平均时间
iterations = 10
start_time = time.time()

for i in range(iterations):
    print(f"执行第 {i+1}/{iterations} 次检查...")
    result = checker.check(test_text)
    
    # 验证换行符处理是否正确
    if "final_assessment_parsed" in result:
        summary = result["final_assessment_parsed"].get("summary", "")
        if "\n" in summary:
            print("  ✓ 换行符处理正确")
        else:
            print("  ✗ 换行符处理错误")

end_time = time.time()
avg_time = (end_time - start_time) / iterations

print(f"\n平均执行时间: {avg_time:.4f} 秒")
print(f"总执行时间: {end_time - start_time:.4f} 秒")

# 输出最终检查结果的格式
print("\n最终检查结果示例:")
print(json.dumps(result, ensure_ascii=False, indent=2))