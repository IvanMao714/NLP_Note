# HuggingFace

## 安装

### 环境

- python=3.6
- pytorch=1.10

### 安装transformers

```python
#pip安装
#pip install transformers

#conda安装
#conda install -c huggingface transformers
```

### 安装datasets

```python
#pip安装
#pip install datasets

#conda安装
#conda install -c huggingface -c conda-forge datasets
```

## **tokenizer**

### 加载预训练字典和分词方法

```py
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]
```

### 编码

```
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],
    
    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,

)

print(out)

tokenizer.decode(out)


```

执行后结果

> ```
> [101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 511, 102, 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0]
> ```
>
> ```
> '[CLS] 选 择 珠 江 花 园 的 原 因 就 是 方 便 。 [SEP] 笔 记 本 的 键 盘 确 实 爽 。 [SEP] [PAD] [PAD] [PAD]'
> ```

### 增强的编码函数

```py
out = tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,
    add_special_tokens=True,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'])
```

执行后结果

> ```
> input_ids : [101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 511, 102, 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0]
> token_type_ids : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
> special_tokens_mask : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
> attention_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
> length : 30
> ```
>
> ```
> '[CLS] 选 择 珠 江 花 园 的 原 因 就 是 方 便 。 [SEP] 笔 记 本 的 键 盘 确 实 爽 。 [SEP] [PAD] [PAD] [PAD]'
> ```

### 批量编码句子

```py
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=15,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'][0]), tokenizer.decode(out['input_ids'][1])
```

执行后结果

> ```
> input_ids : [[101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 102], [101, 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0]]
> token_type_ids : [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
> special_tokens_mask : [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]
> length : [15, 12]
> attention_mask : [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
> ```
>
> ```
> ('[CLS] 选 择 珠 江 花 园 的 原 因 就 是 方 便 [SEP]',
>  '[CLS] 笔 记 本 的 键 盘 确 实 爽 。 [SEP] [PAD] [PAD] [PAD]')
> ```

### 批量编码成对的句子

```py
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'][0])
```

执行后结果

> ```
> input_ids : [[101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 511, 102, 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0], [101, 2791, 7313, 1922, 2207, 511, 1071, 800, 4638, 6963, 671, 5663, 511, 102, 791, 1921, 2798, 4761, 6887, 6821, 741, 6820, 3300, 5018, 127, 1318, 117, 4696, 3300, 102]]
> token_type_ids : [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
> special_tokens_mask : [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
> length : [27, 30]
> attention_mask : [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
> ```
>
> ```
> '[CLS] 选 择 珠 江 花 园 的 原 因 就 是 方 便 。 [SEP] 笔 记 本 的 键 盘 确 实 爽 。 [SEP] [PAD] [PAD] [PAD]'
> ```

### 获取字典

```py
zidian = tokenizer.get_vocab()

type(zidian), len(zidian), '月光' in zidian,
```

> ```
> (dict, 21128, False)
> ```

```py
#添加新词
tokenizer.add_tokens(new_tokens=['月光', '希望'])

#添加新符号
tokenizer.add_special_tokens({'eos_token': '[EOS]'})

zidian = tokenizer.get_vocab()

type(zidian), len(zidian), zidian['月光'], zidian['[EOS]']
```

执行后

> ```
> (dict, 21131, 21128, 21130)
> ```

## **Datasets**

### 加载数据

```py
from datasets import load_dataset

#注意：如果你的网络不允许你执行这段的代码，则直接运行【从磁盘加载数据】即可，我已经给你准备了本地化的数据文件
dataset = load_dataset(path='seamew/ChnSentiCorp')

#保存数据集到磁盘
#注意：运行这段代码要确保【加载数据】运行是正常的，否则直接运行【从磁盘加载数据】即可
dataset.save_to_disk(dataset_dict_path='./data/ChnSentiCorp')

#从磁盘加载数据
from datasets import load_from_disk
dataset = load_from_disk('./data/ChnSentiCorp')

dataset
```

### 数据集函数

#### sort

```py
#未排序的label是乱序的
print(dataset['label'][:10])

#排序之后label有序了
sorted_dataset = dataset.sort('label')
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])
```

#### shuffle

```py
#打乱顺序
shuffled_dataset = sorted_dataset.shuffle(seed=42)

shuffled_dataset['label'][:10]
```

#### select

```py
dataset.select([0, 10, 20, 30, 40, 50])
```

#### filter

```py
def f(data):
    return data['text'].startswith('选择')


start_with_ar = dataset.filter(f)

len(start_with_ar), start_with_ar['text']
```

#### train_test_split

```py
#train_test_split, 切分训练集和测试集
dataset.train_test_split(test_size=0.1)
```

#### shard

```py
#把数据切分到4个桶中,均匀分配
dataset.shard(num_shards=4, index=0)
```

#### rename_column

```py
#rename_column
dataset.rename_column('text', 'textA')
```

#### remove_columns

```py
#remove_columns
dataset.remove_columns(['text'])
```

#### map

```py
def f(data):
    data['text'] = 'My sentence: ' + data['text']
    return data


datatset_map = dataset.map(f)

datatset_map['text'][:5]
```

#### set_format

```py
dataset.set_format(type='torch', columns=['label'])
dataset[0]
```

#### 导出/加载

```py
#导出为csv格式
dataset = load_dataset(path='seamew/ChnSentiCorp', split='train')
dataset.to_csv(path_or_buf='./data/ChnSentiCorp.csv')

#加载csv格式数据
csv_dataset = load_dataset(path='csv',
                           data_files='./data/ChnSentiCorp.csv',
                           split='train')
csv_dataset[20]
```

```py
#导出为json格式
dataset = load_dataset(path='seamew/ChnSentiCorp', split='train')
dataset.to_json(path_or_buf='./data/ChnSentiCorp.json')

#加载json格式数据
json_dataset = load_dataset(path='json',
                            data_files='./data/ChnSentiCorp.json',
                            split='train')
json_dataset[20]
```

## **Metric**

### 列出评价指标

```py
from datasets import list_metrics

#列出评价指标
metrics_list = list_metrics()

len(metrics_list), metrics_list
```

执行后结果

> ```
> (34,
>  ['accuracy',
>   'bertscore',
>   'bleu',
>   'bleurt',
>   'cer',
>   'chrf',
>   'code_eval',
>   'comet',
>   'competition_math',
>   'coval',
>   'cuad',
>   'f1',
>   'gleu',
>   'glue',
>   'google_bleu',
>   'indic_glue',
>   'matthews_correlation',
>   'mauve',
>   'meteor',
>   'pearsonr',
>   'precision',
>   'recall',
>   'rouge',
>   'sacrebleu',
>   'sari',
>   'seqeval',
>   'spearmanr',
>   'squad',
>   'squad_v2',
>   'super_glue',
>   'ter',
>   'wer',
>   'wiki_split',
>   'xnli'])
> ```

```py
from datasets import load_metric

#加载一个评价指标
metric = load_metric('glue', 'mrpc')

print(metric.inputs_description)

#计算一个评价指标
predictions = [0, 1, 0]
references = [0, 1, 1]

final_score = metric.compute(predictions=predictions, references=references)

final_score
```