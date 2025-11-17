import LangChain
import json

# 创建错误检查系统实例
system = LangChain.TextErrorCheckSystem()

# 测试文本
test_text = "这是一个简的测试文本，看看是否语法错误。"

# 执行基础检查
result = system.basic_checker.check(test_text)
print("基础检查结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))
print()

# 执行专业检查
result = system.pro_checker.check(test_text)
print("专业检查结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))
print()

# 执行领域特定检查（例如学术领域）
domain_checker = LangChain.DomainSpecificChecker(domain="academic")
result = domain_checker.check(test_text)
print("学术领域检查结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))
print()

# 执行综合检查
result = system.comprehensive_check(test_text, domain="general")
print("综合检查结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))