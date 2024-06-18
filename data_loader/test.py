import json

# 读取JSON文件
with open('../data/mydata/mydata.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 去除空行和无效的JSON对象
valid_lines = [line.strip() for line in lines if line.strip()]

# 检查每一行是否为有效的JSON对象
valid_json_objects = []
for line in valid_lines:
    try:
        json_obj = json.loads(line)
        valid_json_objects.append(json_obj)
    except json.JSONDecodeError:
        print(f"Invalid JSON object found and skipped: {line}")

# 如果需要，修复JSON结构
if valid_json_objects:
    with open('../data/mydata_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(valid_json_objects, f, ensure_ascii=False, indent=2)

print(f"Fixed JSON file saved as '/mnt/data/mydata_fixed.json'")
