import csv
import time

# 定义输入和输出文件路径
input_csv = 'silab_data03.csv'  # 输入的CSV文件路径
output_csv = 'output.csv'  # 输出的CSV文件路径


def process_data_line(timestamp, data):
    # 将第二列的数据每15个字符为一组，再每组分成3个5字符的列
    data=data[-150:]
    processed_data = []
    for i in range(0, len(data), 15):  # 每15个字符一组
        group = data[i:i + 15]
        if len(group) == 15:
            cols = []  # 用来存储每组的3个列
            # 使用for循环将每组数据按5个字符划分为列
            for j in range(0, 15, 5):
                cols.append(group[j:j + 5])
            # 将时间戳和列数据合并成一行
            processed_data.append([timestamp] + cols)
    return processed_data


def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='',encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        # 写入列标题
        writer.writerow(['Timestamp', '刹车', '油门', '方向盘'])
        # 处理每一行数据
        for row in reader:
            if len(row) < 2:
                continue  # 跳过没有数据的行
            timestamp = row[0]  # 获取时间戳（第一列）
            data = row[1]  # 获取第二列的数据（连续字符串）
            processed_lines = process_data_line(timestamp, data)
            # 将处理后的每行数据写入输出文件
            for line in processed_lines:
                writer.writerow(line)

# 执行处理
# process_csv(input_csv, output_csv)
#
# print(f"数据处理完成，结果已保存至: {output_csv}")
