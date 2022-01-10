import fastseq
from ipywidgets import IntProgress
import tqdm 
from datasets import load_dataset
import lawrouge
 
import datasets
import random
import pandas as pd
 
from datasets import dataset_dict
import datasets
from IPython.display import display, HTML
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
 
import warnings
from pathlib import Path
from typing import List, Tuple, Union
 
import fire
from torch import nn
 
import jieba
import numpy as np
 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.utils import logging
import os
os.environ["WANDB_DISABLED"] = "true"
import tensorflow as tf
import torch

print(tf.test.is_gpu_available(),torch.cuda.is_available())
print(torch.__version__)
 
dataset = load_dataset('json', data_files='CNews_sum.json', field='data')
 
def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id":"0"
    }
dataset = dataset["train"].map(flatten, remove_columns=["title", "content"]) # , remove_columns=["title", "content"]
 
 
TokenModel = "bert-base-chinese"
 
from transformers import AutoTokenizer, BertConfig
tokenizer = AutoTokenizer.from_pretrained(TokenModel)
 
config = BertConfig.from_pretrained(TokenModel)
 
model_checkpoint = "D:/bart-large-chinese" 
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = "" # BART-12-3
print(model_checkpoint)
 
max_input_length = 512 # input, source text 注意长度，复旦BART中文预训练模型使用的bert tokenizer
max_target_length = 128 # summary, target text
 
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
 
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)
 
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
 
 
 
raw_datasets = dataset
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
 
train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()
train_data_txt, test_data_tex = train_data_txt.train_test_split(test_size=0.1).values()
# 装载数据
dd = datasets.DatasetDict({"train":train_data_txt,"validation": validation_data_txt,"test":test_data_tex }) 
 
raw_datasets = dd
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
 
 
def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
 
 
 
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
 
 
 
logger = logging.get_logger(__name__)
 
def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]) -> None:
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())
 
 
LAYERS_TO_COPY = {
    # maps  num layers in teacher -> num_layers in student -> which teacher layers to copy.
    # 12: bart, 16: pegasus, 6: marian/Helsinki-NLP
    12: {
        1: [0],  # This says that if the teacher has 12 layers and the student has 1, copy layer 0 of the teacher
        2: [0, 6],
        3: [0, 6, 11],      # the first, 7th and 12th decode layers
        4: [0, 4, 8, 11],
        6: [0, 2, 4, 7, 9, 11],
        9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
        12: list(range(12)),
    },
    16: {  # maps  num layers in student -> which teacher layers to copy
        1: [0],
        2: [0, 15],
        3: [0, 8, 15], 
        4: [0, 5, 10, 15],
        6: [0, 3, 6, 9, 12, 15],
        8: [0, 2, 4, 6, 8, 10, 12, 15],
        9: [0, 1, 3, 5, 7, 9, 11, 13, 15],
        12: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        16: list(range(16)),
    },
    6: {1: [0], 2: [0, 5], 3: [0, 2, 5], 4: [0, 1, 3, 5], 6: list(range(6))},
}
LAYERS_TO_SUPERVISE = {
    # maps  num layers in student -> which teacher layers to copy.
    6: {1: [5], 2: [3, 5], 3: [1, 4, 5], 4: [1, 2, 4, 5]},
    12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
    16: {1: [15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15]},
}
 
def create_student_by_copying_alternating_layers(
    teacher: Union[str, PreTrainedModel],
    save_path: Union[str, Path] = "student",
    e: Union[int, None] = None,
    d: Union[int, None] = None,
    copy_first_teacher_layers=False,
    e_layers_to_copy=None,
    d_layers_to_copy=None,
    **extra_config_kwargs
) -> Tuple[PreTrainedModel, List[int], List[int]]:
    
    _msg = "encoder_layers and decoder_layers cannot be both None-- you would just have an identical teacher."
    assert (e is not None) or (d is not None), _msg
    if isinstance(teacher, str):
        AutoTokenizer.from_pretrained(teacher).save_pretrained(save_path)  # purely for convenience
        teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher).eval()
    else:
 
        assert isinstance(teacher, PreTrainedModel), f"teacher must be a model or string got type {type(teacher)}"
    init_kwargs = teacher.config.to_diff_dict()
 
    try:
        teacher_e, teacher_d = teacher.config.encoder_layers, teacher.config.decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"encoder_layers": e, "decoder_layers": d})
    except AttributeError:  # T5
        teacher_e, teacher_d = teacher.config.num_layers, teacher.config.num_decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"num_layers": e, "num_decoder_layers": d})
 
    # Kwargs to instantiate student: teacher kwargs with updated layer numbers + **extra_config_kwargs
    init_kwargs.update(extra_config_kwargs)
 
    # Copy weights
    student_cfg = teacher.config_class(**init_kwargs)
    student = AutoModelForSeq2SeqLM.from_config(student_cfg)
    # Start by copying the full teacher state dict this will copy the first N teacher layers to the student.
    info = student.load_state_dict(teacher.state_dict(), strict=False)
    assert info.missing_keys == [], info.missing_keys  # every student key should have a teacher keys.
 
    if copy_first_teacher_layers:  # Our copying is done. We just log and save
        e_layers_to_copy, d_layers_to_copy = list(range(e)), list(range(d))
        logger.info(
            f"Copied encoder layers {e_layers_to_copy} and decoder layers {d_layers_to_copy}. Saving them to {save_path}"
        )
        student.save_pretrained(save_path)
        return student, e_layers_to_copy, d_layers_to_copy
 
    # Decide which layers of the teacher to copy. Not exactly alternating -- we try to keep first and last layer.
    if e_layers_to_copy is None:
        e_layers_to_copy: List[int] = pick_layers_to_copy(e, teacher_e)
    if d_layers_to_copy is None:
        d_layers_to_copy: List[int] = pick_layers_to_copy(d, teacher_d)
 
    try:
        copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)
        copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
    except AttributeError:  # For t5, student.model.encoder.layers is called student.encoder.block
        copy_layers(teacher.encoder.block, student.encoder.block, e_layers_to_copy)
        copy_layers(teacher.decoder.block, student.decoder.block, d_layers_to_copy)
    logger.info(
        f"Copied encoder layers {e_layers_to_copy} and decoder layers {d_layers_to_copy}. Saving them to {save_path}"
    )
    student.config.init_metadata = dict(
        teacher_type=teacher.config.model_type,
        copied_encoder_layers=e_layers_to_copy,
        copied_decoder_layers=d_layers_to_copy,
    )
    student.save_pretrained(save_path)
    # Save information about copying for easier reproducibility
 
    return student, e_layers_to_copy, d_layers_to_copy
 
def pick_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
            )
        return list(range(n_student))
 
batch_size = 2
args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=2,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,  # demo
    per_device_eval_batch_size=batch_size,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=500,
    save_total_limit=3,
)
 
 
# SFT 训练，看自己
# model, list_en, list_de = create_student_by_copying_alternating_layers(model, 'trian.pth', 12, 3)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
 
 
# 这里用的是中文lawrouge 至于字符级还是词级计算看自己调整 这里是字符级
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]
    # Rouge with jieba cut
    # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
    # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]
 
    labels_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in labels]
    length = len(prediction_lens)
 
    print(decoded_preds)
    print(decoded_labels)
    rouge = lawrouge.Rouge()
 
    result = rouge.get_scores(decoded_preds, decoded_labels,avg=True)
    # print(result)
    print(result)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}
 
    result = {key: value * 100 for key, value in result.items()}
 
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
 
train_result = trainer.train()
print(train_result)
 
trainer.save_model()
metrics = train_result.metrics
trainer.log_metrics("train",metrics)
trainer.save_metrics("train",metrics)
trainer.save_state()
 
import torch
model.load_state_dict(torch.load('./results/pytorch_model.bin'))
 
def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
 
    attention_mask = inputs.attention_mask.to(model.device)
  
    outputs = model.generate(input_ids, attention_mask=attention_mask,max_length=128)
    print(outputs)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str
 
_,x = generate_summary("20日凌晨,寒风刺骨,两名年纪相仿的婴儿相继被狠心的父母遗弃在翔安的两个角落,一个在莲花总院厕所里,一个在东园社区一榕树下。两名婴儿被发现时间相距不过10分钟,莲河边防派出所民警连夜走访,未寻得婴儿家属。目前,一名婴儿已被送往福利院,另一名暂时安置在村民家中。据悉,经医生初步检查,两名婴儿均身体健康,无残疾、无疾病。记者陈佩珊通讯员蔡美娟林才龙",model)
print(x)
print(len(x[0]))
 
'''
tensor([[ 102,  101, 1336, 7305, 5425, 2128,  697, 1399, 2399, 5279, 4685,  820,
         4638, 2048, 1036, 4685, 5326, 6158, 6890, 2461, 1762,  697,  702, 6235,
         5862,  117,  671,  782, 1762, 5813, 5709, 2600, 7368, 1329, 2792, 7027,
          117,  671,  702, 1762,  691, 1736, 4852, 1277,  671, 3525, 3409,  678,
          511,  102]], device='cuda:0')
['厦 门 翔 安 两 名 年 纪 相 仿 的 婴 儿 相 继 被 遗 弃 在 两 个 角 落, 一 人 在 莲 花 总 院 厕 所 里, 一 个 在 东 园 社 区 一 榕 树 下 。']
91
'''
 
eval_results = trainer.evaluate()
print(eval_results)