# 读取txt文件，获取需要比对的部分
def read_txt_file(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 读取scp文件，获取路径信息
def read_scp_file(scp_path):
    with open(scp_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 提取路径中的文件名部分（去掉扩展名）
def extract_filename(path):
    return path.split('/')[-1].split('.')[0]

# 删除与txt文件匹配的条目
def filter_scp(scp_file, txt_file):
    txt_paths = read_txt_file(txt_file)
    scp_lines = read_scp_file(scp_file)

    # 从txt文件中提取文件名（去掉扩展名）
    txt_files = set(extract_filename(path) for path in txt_paths)

    # 过滤掉与txt文件中路径匹配的scp路径
    filtered_scp = [
        line for line in scp_lines
        if extract_filename(line) not in txt_files
    ]

    return filtered_scp

# 写回过滤后的scp文件
def write_filtered_scp(filtered_scp, scp_output_path):
    with open(scp_output_path, 'w') as f:
        f.write("\n".join(filtered_scp))

# 使用示例
scp_file = "/train20/intern/permanent/kwli2/udkws/PLCL_udkws_re/loader/google/train_up_1word.scp"
txt_file = "/train20/intern/permanent/kwli2/phonmachnet/dataset/google_speech_commands/speech_commands/testing_list.txt"
scp_output_file = "/train20/intern/permanent/kwli2/udkws/PLCL_udkws_re/loader/google/filtered_scp_file.scp"

# 过滤
filtered_scp = filter_scp(scp_file, txt_file)

# 写回新文件
write_filtered_scp(filtered_scp, scp_file)

print(f"Filtered SCP file written to {scp_output_file}")
