import json
import os

def merge_and_renumber_with_prefix(file_paths, output_file_path):
    """
    根据文件在列表中的位置，为合并后的JSON条目添加不同的前缀，并进行连续编号。

    - 前3个文件使用 'su_' 前缀。
    - 中间3个文件使用 'cu_j_' 前缀。
    - 最后3个文件使用 'cu_p_' 前缀。

    Args:
        file_paths (list): 包含9个输入JSON文件路径的列表。顺序至关重要。
        output_file_path (str): 输出的合并后JSON文件的路径。
    """
    if len(file_paths) != 9:
        print(f"错误：需要9个文件路径，但只提供了 {len(file_paths)} 个。请检查 'input_file_list'。")
        return

    merged_data = {}
    usecase_counter = 1

    print("开始处理文件...")

    # 使用 enumerate 获取文件索引，以确定使用哪个前缀
    for index, file_path in enumerate(file_paths):
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告：文件 '{file_path}' 不存在，已跳过。")
            continue

        # 根据文件索引确定key的前缀
        if 0 <= index < 3:
            key_prefix = "su_"
        elif 3 <= index < 6:
            key_prefix = "cu_j_"
        else:  # 6 <= index < 9
            key_prefix = "cu_p_"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"成功读取文件 (索引 {index}, 前缀 '{key_prefix}'): '{file_path}'")

                # 确保文件内部的usecase按数字顺序处理
                sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[-1]))

                # 遍历排序后的键，使用新前缀和全局计数器生成新条目
                for key in sorted_keys:
                    new_key = f"{key_prefix}{usecase_counter}"
                    merged_data[new_key] = data[key]
                    usecase_counter += 1

        except json.JSONDecodeError:
            print(f"错误：文件 '{file_path}' 不是一个有效的JSON格式，已跳过。")
        except Exception as e:
            print(f"处理文件 '{file_path}' 时发生未知错误: {e}")

    # 检查是否合并了任何数据
    if not merged_data:
        print("没有处理任何数据，不会生成输出文件。")
        return

    # 将合并后的数据写入到输出文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: '{output_dir}'")

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        
        print("\n处理完成！")
        print(f"所有文件已成功合并到: '{output_file_path}'")
        print(f"总共合并了 {len(merged_data)} 条记录。")

    except Exception as e:
        print(f"写入到输出文件 '{output_file_path}' 时发生错误: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    
    # 根据您提供的路径列表进行配置
    input_file_list = [
        r"ODRL_V3\data_preparation\simple_usecases.json",
        r"ODRL_V3\result\generated_usecases\final_usecases\gen_simple_usecases1.json",
        r"ODRL_V3\result\generated_usecases\final_usecases\gen_simple_usecases2.json",
        r"ODRL_V3\data_preparation\complex_usecases_juxtaposition.json",
        r"ODRL_V3\result\generated_usecases\final_usecases\gen_complex_usecases_juxtaposition1.json",
        r"ODRL_V3\result\generated_usecases\final_usecases\gen_complex_usecases_juxtaposition2.json",
        r"ODRL_V3\data_preparation\complex_usecases_progression.json",
        r"ODRL_V3\result\generated_usecases\final_usecases\gen_complex_usecases_progression1.json",
        r"ODRL_V3\result\generated_usecases\final_usecases\gen_complex_usecases_progression2.json",
    ]

    # 根据您提供的输出路径进行配置
    output_file_name = r"ODRL_V3\data_preparation\merged_final_usecases.json"

    # 调用主函数来执行合并操作
    merge_and_renumber_with_prefix(input_file_list, output_file_name)