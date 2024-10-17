import csv


# 定义一个函数来处理数据
def format_data(data):
    data=data[-150:]
    return ' '.join([data[i:i + 5] for i in range(0, len(data), 5)])


# 读取 CSV 文件
with open('silab_data03.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = []

    for row in reader:
        if len(row) > 1:  # 确保第二列存在
            row[1] = format_data(row[1])  # 格式化第二列
        rows.append(row)

# 写入新的 CSV 文件
with open('formatted_file.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

print("数据已格式化并写入新文件。")



if __name__ == '__main__':
    # 示例数据
    data = "20241009164856,-0.12-0.510.000-0.12-0.040.000-0.12-0.040.000-0.12-0.040.000-0.12-0.040.000-0.12-0.040.000-0.12-0.040.000-0.12-0.040.000-0.12-0.041.054-0.12-0.041.054"

    # 分割时间戳和数据
    timestamp, raw_data = data.split(",", 1)

    # 每 15 个字符为一组数据
    data_chunks = [raw_data[i:i + 15] for i in range(0, len(raw_data), 15)]

    # 按每组数据的 5 个字符一列划分
    formatted_data = []
    for chunk in data_chunks:
        formatted_data.append([chunk[i:i + 5] for i in range(0, 15, 5)])

    # 输出处理后的数据
    print("Timestamp:", timestamp)
    for row in formatted_data:
        print([timestamp] + row)  # 将时间戳和对应的三列数据合并

