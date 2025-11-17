import json

# 模拟LLM返回的结果格式，包含转义换行符
simulated_result = {
    "text": "智能健身手环选购指南：别再为选择发愁\\n\\n如今，健康管理成为全民关注的焦点...",
    "grammar_check": "\n错误数量: 2\\n具体错误: 1. \"销量逐年攀升\"应为\"销量逐年增长\"\n2. \"市面上的手环品牌繁多\"应为\"市面上的手环品牌众多\"",
    "fact_check": "\n事实问题: 1. 文中提到所有健康问题都与心率相关...\n2. 文中提到手环的心率监测准确性是判断手环性能的唯一标准...",
    "logic_check": "\n逻辑问题: 文中提到心率监测是判断手环性能的唯一标准...",
    "clarity_check": "\n清晰度问题: 文章结构不清晰，缺乏明确的标题和段落分隔...",
    "final_assessment": "\n        {\n            \"overall_score\": \"6/10\",\n            \"main_issues\": [\"事实问题\", \"逻辑问题\"],\n            \"improvement_suggestions\": [\"建议1: 文章结构不清晰...\", \"建议2: 文中提到心率监测是判断手环性能的唯一标准...\"],\n            \"corrected_version\": \"智能健身手环选购指南：别再为选择发愁\\n\\n如今，健康管理成为全民关注的焦点...\"\n        }"
}

# 模拟修复后的处理逻辑
def process_result(result):
    for key, value in result.items():
        if isinstance(value, str):
            # 替换转义换行符为实际换行符
            result[key] = value.replace("\\n", "\n")
    return result

# 处理结果
processed_result = process_result(simulated_result)

# 输出处理前后的对比
print("=== 处理前 ===")
print(json.dumps(simulated_result, ensure_ascii=False, indent=2))
print("\n=== 处理后 ===")
print(json.dumps(processed_result, ensure_ascii=False, indent=2))