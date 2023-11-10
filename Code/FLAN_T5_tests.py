
#For flan-t5-base
import pickle
import random
from torch.utils.data import Dataset, DataLoader
import os
from typing import Sequence, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Dataset_load(Dataset):
        def __init__(self, input_data, output_data, tokenizer,max_length=256):
            self.input_data = input_data
            self.output_data = output_data
            self.tokenizer = tokenizer
            
            self.max_length = max_length

        def __getitem__(self, idx):

                inp_list = self.input_data[idx]
                out_data = self.output_data[idx]

                input_encodings = self.tokenizer(inp_list, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
                output_encodings = self.tokenizer(out_data, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

                return (input_encodings["input_ids"][0], input_encodings["attention_mask"][0],  output_encodings["input_ids"][0])

        def __len__(self):
            return len(self.input_data)

class Datasetval_load(Dataset):
        def __init__(self, input_data, output_data, tokenizer, max_length=256):
            self.input_data = input_data
            self.output_data = output_data
            self.tokenizer = tokenizer
            
            self.max_length = max_length

        def __getitem__(self, idx):

            input_encodings = self.tokenizer(self.input_data[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            output_encodings = self.tokenizer(self.output_data[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            return (input_encodings["input_ids"][0], input_encodings["attention_mask"][0],  output_encodings["input_ids"][0])

        def __len__(self):
            return len(self.input_data)



if __name__ == '__main__':
    import os
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    import logging
    import evaluate
    metric = evaluate.load("sacrebleu")    
    from transformers import AutoModelForSeq2SeqLM
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer, AdamW,BartTokenizer
    import torch
    import torch.nn as nn
    import os
    import numpy as np
    import random
    from nltk.translate.bleu_score import corpus_bleu
    import random
    from torch.utils.data import Dataset, DataLoader
    from transformers import Seq2SeqTrainer, TrainingArguments,Seq2SeqTrainingArguments
    
    name_of_file = 'name_for_model_run'

    #Dataset names
    with open("1000_X.txt", "r",encoding='utf-8') as f:
        train_input_data = [line.strip() for line in f.readlines()]
    with open("1000_Y.txt", "r",encoding='utf-8') as f:
        train_output_data = [line.strip() for line in f.readlines()]
    with open("Validation_X.txt", "r",encoding='utf-8') as f:
        val_input_data = [line.strip() for line in f.readlines()]
    with open("Validation.txt", "r",encoding='utf-8') as f:
        val_output_data = [line.strip().replace('\u200c','') for line in f.readlines()]


    try:
            os.mkdir(name_of_file)
    except:
            pass

    logging.basicConfig(filename=name_of_file+'/log.log', level=logging.WARNING)
    logging.debug('This will get logged')
    

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    model.config.max_length = 256

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    steps_done = 0

    temp = list(zip(train_input_data, train_output_data))
    random.shuffle(temp)
    train_input_data, train_output_data = zip(*temp)

    train_input_data, train_output_data = list(train_input_data), list(train_output_data)

    
    def modified_bleu(truth: List[str], pred: List[str]) -> float:

            references = [sentence.split() for sentence in truth]
            candidates = [sentence.split() for sentence in pred]
            references = [r + max(0, 4 - len(r)) * [''] for r in references]
            candidates = [c + max(0, 4 - len(c)) * [''] for c in candidates]
          
            refs = [[r] for r in references]
            return corpus_bleu(refs, candidates)
    
    def postprocess_text(preds, labels):
        preds = [list(pred.split('</s>'))[0].replace('<unk>','') for pred in preds]
        labels = [list(label.split('</s>'))[0].replace('<unk>','') for label in labels]

        return preds, labels
    
    
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        global steps_done
        steps_done+=200
 
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_preds_save = tokenizer.batch_decode(predictions, skip_special_tokens=False)

        new_file=open("temp_output.txt",mode="w",encoding="utf-8")
        for lin in decoded_preds_save:
            new_file.write(lin + '\n')
        new_file.close()

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds2, decoded_labels2 = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        result["bleu_modified"]=modified_bleu(decoded_labels,decoded_preds)*100

        result = {k: round(v, 4) for k, v in result.items()}
        print('pred:', decoded_preds2[0])
        print(decoded_labels2[0])
        print('pred:', decoded_preds2[1])
        print(decoded_labels2[1])
        print('pred:', decoded_preds2[2])
        print(decoded_labels2[2])
        with open(f"{name_of_file}/output_{steps_done}", "wb") as fp:  
                pickle.dump(decoded_preds, fp)
        logging.warning('result: ' + str(steps_done) + ' ' + str(result['bleu']))
        return result

    
    
    train_dataset = Dataset_load(train_input_data, train_output_data, tokenizer)
    val_dataset = Datasetval_load(val_input_data, val_output_data, tokenizer)

    

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='output_dir'+name_of_file,
        num_train_epochs=100,
        per_device_train_batch_size=1,
        per_device_eval_batch_size = 4,
        save_steps=1000,
        learning_rate=0.00005,
        evaluation_strategy = 'steps',
        predict_with_generate = True,

        eval_steps = 200,
        fp16=False,        
        dataloader_num_workers=0,
        logging_steps=50,

        gradient_accumulation_steps=4,
        eval_accumulation_steps = 1,

    )





    trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset =val_dataset,
            compute_metrics=compute_metrics,
            data_collator=lambda data: {'input_ids': torch.stack([item[0] for item in data]),
                                        'attention_mask': torch.stack([item[1] for item in data]),
                                        'labels': torch.stack([torch.where(item[2] != tokenizer.pad_token_id, item[2], -100) for item in data])},
        )


    trainer.train()


