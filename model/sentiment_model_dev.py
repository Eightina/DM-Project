# -*- coding: utf-8 -*-
"""
@author: Orion

Only for local env training with CUDA
"""
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, Value, ClassLabel, Features, concatenate_datasets
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer
# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# model & tokenizer
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# training_args
training_args = TrainingArguments("test-trainer",num_train_epochs=3,per_device_train_batch_size=8)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

if __name__ == "__main__":
    #load data
    traindf = pd.read_excel('./data/Task-2/train.xlsx')
    testdf=pd.read_excel('./data/Task-2/test.xlsx')
    #clean data
    traindf.drop_duplicates(subset='text',inplace=True)
    traindf.shape,testdf.shape,traindf['label'].value_counts()
    ros = RandomOverSampler()
    train_x, train_y = ros.fit_resample(np.array(traindf['text']).reshape(-1, 1), np.array(traindf['label']).reshape(-1, 1))
    traindf_balance = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['text', 'label'])
    traindf_balance['label'].value_counts()
    
    #data preprocessing
    #shuffle training dataset
    traindf=traindf_balance.sample(frac=1)
    traindf['label'].replace(-1, 0, inplace=True)

    testdf=testdf[['text']]

    testds_features=Features({'text': Value(dtype='string', id = None)})
    testds=Dataset.from_dict(mapping={"text": testdf['text'].to_list()},features=testds_features)


    trainds_features = Features({'text': Value(dtype='string', id = None), 'label': ClassLabel(num_classes=2 ,id=None)})
    trainds = Dataset.from_dict(mapping={"text": traindf['text'].to_list(), 'label': traindf['label'].to_list()},
                                features=trainds_features)
    #whole training dataset
    trainds_org = trainds.shuffle(seed=42)


    #split
    cv_fold=5
    slice_ds=[]
    for i in range(cv_fold):
        slice_ds.append(trainds_org.shard(num_shards=cv_fold,index=i))
        
    #training    
    tokenized_train_org_datasets=trainds_org.map(tokenize_function, batched=True)
    tokenized_test_datasets=testds.map(tokenize_function, batched=True)
    #cross_validation cv_fold=5
    total_f1_score=0
    for i in range(cv_fold):
        valds=slice_ds[i]
        if i==0:
            trainds=concatenate_datasets(slice_ds[i+1:cv_fold])
        elif i==cv_fold-1:
            trainds=concatenate_datasets(slice_ds[0:i])
        else:
            temp=slice_ds[0:i]
            for j in range(i+1,cv_fold):
                temp.append(slice_ds[j])
            trainds=concatenate_datasets(temp)

        tokenized_train_datasets = trainds.map(tokenize_function, batched=True)
        tokenized_eval_datasets=valds.map(tokenize_function, batched=True)

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_eval_datasets,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        val_result=trainer.evaluate()
        total_f1_score+=val_result['eval_f1']
    #average f1 score
    f1_score_cv=total_f1_score/cv_fold
    print(f1_score_cv)
    
    
    #train on whole training datatest
    trainer_final = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_org_datasets,
        # eval_dataset=tokenized_eval_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer_final.train()
    #predict on test dataset
    pred=trainer_final.predict(tokenized_test_datasets)
    #change logits to probability
    label_probability=torch.softmax(torch.tensor(pred[0]),1)
    #get labels
    labels=label_probability.argmax(axis=1).reshape(-1,1)
    labels[:5]
    
    test_to_submit=pd.read_excel('./data/Task-2/test_to-submit.xlsx')
    test_to_submit['label']=labels
    test_to_submit.loc[test_to_submit['label']==0,'label']=-1
    # test_to_submit
    test_to_submit.to_excel('./data/Task-2/test_to-submit_answers1.xlsx')
    # save basic model
    trainer_final.save_model('./test-trainer/disroberta_model')
    
    #quantinize
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    model_checkpoint = "./test-trainer/disroberta_model"
    save_directory = "./test-trainer/onnx_disroberta/"
    # Load a model from transformers and export it to ONNX
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Define the quantization methodology
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    # Apply dynamic quantization on the model
    quantizer.quantize(save_dir=save_directory, quantization_config=qconfig)

    # Save the onnx model and tokenizer
    # ort_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)