# 运行
python main.py --phase cmed.dl.base_model.norm

# 修改2022.04.02
1. 所有模块运行所需参数均从config中调用
2. 模型可按批处理数据
3. 问题：有意义的Tag只有BIO三种。但是，为了实现在一批中长短不一字符串的处理，增加了"<PAD>"。LSTM的输出种类应该设为3还是4？如果设为3，运行时出现IndexError: Target 3 is out of bounds.错误。

# 修改2022.03.28
1. 增加评价模块utils.evaluate。
2. dataProcessor模块中增加split_data()，其调用save()将数据保存至文件。
3. dataLoader模块的data_generator()增加参数run_mode，其值为“train”、“val”或“test”。增加该参数的原因在于，baseModel模块的run_Model()（原为train_model()）在不同阶段运行不同数据文件中的数据。为此，以run_mode标识处理阶段，并作为需调用的数据文档名称最后一部分的标识。
4. baseModel模块的run_Model()增加参数run_mode，原因见上述说明。
5. baseModel模块增加index_to_tag()，其功能为将输出的预测结果转换为BIO形式的标签。注意，utils.evaluate评价指标计算函数的输入为BIO或BIEOS形式的标签。