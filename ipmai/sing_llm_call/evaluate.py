import os
import shutil

def check_failed(file_path):
    """检查文件是否包含 [generation failed]"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            return '[generation failed]' in content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False  # 如果文件不存在或错误，返回False

# 指定根目录
root_dir = '/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call'

# 存储结果
results = {}

# 遍历 model_path 文件夹
for model_path in os.listdir(root_dir):
    model_path_path = os.path.join(root_dir, model_path)
    if not os.path.isdir(model_path_path):
        continue  # 跳过非目录
    
    # 找到 pred_test
    pred_test_path = os.path.join(model_path_path, 'pred_test')
    if not os.path.isdir(pred_test_path):
        continue  # 没有 pred_test，跳过
    
    # 收集失败的子文件夹名和路径
    failed_folders = []
    failed_paths = []  # 用于删除
    
    # 遍历 pred_test 下的子文件夹
    for subfolder in os.listdir(pred_test_path):
        subfolder_path = os.path.join(pred_test_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # 跳过非目录
        
        md_file = os.path.join(subfolder_path, 'generated.md')
        if os.path.isfile(md_file) and check_failed(md_file):
            failed_folders.append(subfolder)  # 收集失败的文件夹名
            failed_paths.append(subfolder_path)  # 收集路径用于删除
    
    if failed_folders:  # 只在有失败时存储
        results[model_path] = {
            'failed_folders': failed_folders
        }
        
        # 删除失败的文件夹
        for path in failed_paths:
            try:
                # shutil.rmtree(path)  # 递归删除整个文件夹
                print(f"已删除失败文件夹: {path}")
            except Exception as e:
                print(f"删除 {path} 失败: {e}")

# 输出结果
for model_path, data in results.items():
    print(f'对于 {model_path} (model_path 文件夹):')
    print(f'  pred_test 下包含 [generation failed] 的子文件夹名: {", ".join(data["failed_folders"])}')
    print(f'  失败个数: {len(data["failed_folders"])}')
    print('---')