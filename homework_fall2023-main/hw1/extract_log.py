import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os


def export_tfevents_to_csv(file_path, output_csv):
    # 1. 加载事件文件
    # size_guidance={ 'scalars': 0 } 表示加载所有的标量数据，不进行抽样
    ea = EventAccumulator(file_path, size_guidance={'scalars': 0})
    ea.Reload()

    # 2. 获取所有可用的指标标签 (Tags)
    tags = ea.Tags()['scalars']
    print(f"检测到以下指标: {tags}")

    all_data = []

    # 3. 遍历每个标签并提取数据
    for tag in tags:
        events = ea.Scalars(tag)
        for event in events:
            all_data.append({
                'step': event.step,
                'tag': tag,
                'value': event.value
            })

    # 4. 转换为 DataFrame 并进行透视处理（让每个指标占一列）
    df = pd.DataFrame(all_data)
    df_pivoted = df.pivot(index='step', columns='tag', values='value').reset_index()

    # 5. 保存为 CSV
    df_pivoted.to_csv(output_csv, index=False)
    print(f"\n提取完成！数据已保存至: {output_csv}")


# 使用示例
if __name__ == "__main__":
    # 请确保文件名与你本地的文件名一致
    input_file = 'events.out.tfevents.1767656982.PC-20240724DZUS'
    output_file = 'training_log_data.csv'

    if os.path.exists(input_file):
        export_tfevents_to_csv(input_file, output_file)
    else:
        print("未找到日志文件，请检查路径。")