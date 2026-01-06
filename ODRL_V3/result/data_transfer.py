import json
import os

def convert_jsonl_to_json(jsonl_file_path: str, json_file_path: str) -> None:
    """
    将 JSONL 文件转换为包含 JSON 对象数组的 JSON 文件。

    Args:
        jsonl_file_path (str): 输入的 JSONL 文件的路径。
        json_file_path (str): 输出的 JSON 文件的路径。

    Raises:
        FileNotFoundError: 如果输入的 JSONL 文件不存在。
        Exception: 如果在转换过程中发生其他错误。
    """
    # 检查输入文件是否存在
    if not os.path.exists(jsonl_file_path):
        raise FileNotFoundError(f"错误：找不到输入文件 '{jsonl_file_path}'")

    data = []
    print(f"开始读取 JSONL 文件: {jsonl_file_path}...")

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 去除行首和行尾的空白字符
                line = line.strip()
                if line:  # 确保不是空行
                    try:
                        # 解析当前行
                        json_object = json.loads(line)
                        data.append(json_object)
                    except json.JSONDecodeError:
                        print(f"警告：跳过第 {line_num} 行，因为它不是一个有效的 JSON 对象。")

        print(f"成功解析 {len(data)} 个 JSON 对象。")

        print(f"开始写入 JSON 文件: {json_file_path}...")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            # indent=4 参数会让输出的 JSON 文件格式化，更易于阅读
            # ensure_ascii=False 可以确保中文字符正确显示，而不是被转义成 ASCII 码
            json.dump(data, f, ensure_ascii=False, indent=4)

        print("转换成功！")
        print(f"输出文件已保存至: {json_file_path}")

    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        raise

# --- 使用示例 ---
if __name__ == '__main__':
    # 定义输入和输出文件名
    input_file = r'ODRL_V3\result\data_for_sft\sft_train_data.jsonl'
    output_file = r'ODRL_V3\result\data_for_sft\sft_train_data.json'

    # 调用转换函数
    try:
        convert_jsonl_to_json(input_file, output_file)
    except Exception as e:
        print(f"主程序捕获到错误: {e}")