import argparse


def train():
    print("Training the model...")


def test():
    print("Testing the model...")


def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="A script to process data.")

    # 添加参数
    parser.add_argument('--mode', choices=['train', 'test'], required=True,help='Mode of operation: train or test')

    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    # 解析命令行参数
    args = parser.parse_args()

    # 根据传入的参数决定执行哪一段代码
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()


if __name__ == "__main__":
    main()
