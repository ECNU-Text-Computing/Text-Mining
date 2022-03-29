# Text Mining
This project aims to construct a comprehensive machine learning and deep learning framework for text mining, 
including text classification, text matching, text generation, information extraction, and so on.

## Run
### To pre-process the data processor:
* Use **show_json_data** to briefly review the data:
`python3 Data_Processor.py --phase show_json_data`
* Use **extract_abs_label** to extract input and output from the data:
`python3 Data_Processor.py --phase extract_abs_label`
* Use **save_abs_label** to save the clean input and output to a clean path:
`python3 Data_Processor.py --phase save_abs_label`
* Use **split_data** to split the clean data to *N* folds:
`python3 Data_Processor.py --phase split_data+aapr.dl.mlp.norm`
* Use **get_vocab** to get the vocabulary/word dictionary from the corpus/dataset:
`python3 Data_Processor.py --phase get_vocab+aapr.dl.mlp.norm`

### To run deep leanring models:
`python3 main.py --phase aapr.dl.mlp.norm > aapr.dl.mlp.norm.log`
`python3 main.py --phase aapr.dl.textcnn.norm > aapr.dl.textcnn.norm.log`

### To run machine learning models:
`python3 main.py --phase aapr.ml.lr.tf > aapr.ml.lr.tf.log`

## To be continued...
风雨过后一定会有美好的天空
天晴之后总会有彩虹
战胜疫情一定有始有终
孤独尽头不一定惶恐

愿疫情早日退散。