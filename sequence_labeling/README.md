# 概述
sequence_labeling模块用于支持序列标注。其中，
config；模型运行时的参数配置，每个模型对应1个json文件
Datasets：运行数据，包括原始数据(data.input, data.output)，训练集(data.input.train, data.output.train)，验证集(data.input.eval, data.output.eval)，测试集(data.input.test, data.output.test)，标签字典(tags)，字/词典(vocab)，模型。
dl：Deep Learning模型
sl：Statistic Learning模型，待开发
utils：目前只有评价模型(evaluate.py)
data_loader.py: 生成批处理数据
data_processor.py：生成训练集、验证集、测试集、字典
main.py：模型运行主入口

# 开发和运行
IDE工具：pycharm
环境: python 3.8, pytorch, seqeval.metrics
运行：
step1: 运行data_processor.py，生成相关数据（tags.json需要人工编辑）
step2: 在Terminal中进入sequence_labeling目录
step3: 在Terminal中输入指令，形如 “python main.py --phase 配置文件名”运行模型
       例如：python main.py --phase cmed.dl.gru.norm

# 数据
input、output文件中的数据一一对应。
目前data.input、data.output中的数据为CCKS的片段。
以字符为单元，以‘ ’（空格）切分。

# 标注集
BIO。为了适应各种模型，增加了"[PAD]"、"UNK"、"SOS"、"EOS"、[CLS]"、"[SEP]。
