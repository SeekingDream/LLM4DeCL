import os
import json

import datasets
from datasets import Dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm

from src.run_code import run_sql


def read_schema(schema_file):
    with open(schema_file, 'r') as f:
        schema = f.readlines()
        return schema


def make_sql_dataset(ori_data_list):
    db_root_dir = '/home/simin/Project/KnowPrompt/Dataset/spider/database'
    new_dataset = []
    err_db_list = []
    for data in tqdm(ori_data_list):
        db_id = data['db_id']
        query = data['query']
        try:
            db_path = os.path.join(db_root_dir, '%s/%s.sqlite' % (db_id, db_id))
            schema_file = os.path.join(db_root_dir, '%s/schema.sql' % db_id)
            schema = read_schema(schema_file)
            query_res = run_sql(db_path, query)
            # data['schema'] = schema
            # data['query_res'] = query_res
            new_data = {
                'db_id': data['db_id'],
                'query': data['query'],
                'question': data['question'],
                'schema': ''.join(schema),
                'query_res': ''.join([str(r) + '\n' for r in query_res])
            }
            new_dataset.append(new_data)
        except:
            err_db_list.append(db_id)
    return new_dataset, err_db_list


def preprocess_spider():

    f = open('Dataset/spider/train_spider.json', 'r')
    train_data = json.load(f)
    f.close()
    f = open('Dataset/spider/train_others.json', 'r')
    other_train_data = json.load(f)
    f.close()
    train_data.extend(other_train_data)
    f = open('Dataset/spider/dev.json', 'r')
    dev_data = json.load(f)
    f.close()

    my_train_dataset, err_db_list = make_sql_dataset(train_data)
    print(len(err_db_list))
    my_dev_dataset, err_db_list = make_sql_dataset(dev_data)
    print(len(err_db_list))

    # create a Dataset object
    my_train_dataset = Dataset.from_list(my_train_dataset)
    my_dev_dataset = Dataset.from_list(my_dev_dataset)

    dataset = DatasetDict({
        'train': my_train_dataset,
        'test': my_dev_dataset
    })

    # publish the dataset to Hugging Face
    dataset_name = "CM/spider"
    dataset.push_to_hub(dataset_name)


def preprocess_codex_glue_code_to_code_trans():
    dataset = datasets.load_dataset('code_x_glue_cc_code_to_code_trans')
    dataset_name = "CM/codexglue_codetrans"
    dataset.push_to_hub(dataset_name)


def preprocess_codex_glue_code_to_text():
    cache_dir = './Dataset/huggingface_codexglue'
    task_name = 'code_x_glue_ct_code_to_text'
    langlist = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
    for lang in langlist:
        dataset = datasets.load_dataset(
            task_name, lang,  cache_dir=cache_dir)
        dataset_name = "CM/codexglue_code2text_" + lang
        dataset.push_to_hub(dataset_name)


if __name__ == '__main__':

    preprocess_codex_glue_code_to_code_trans()
    # preprocess_spider()
    # preprocess_codex_glue_code_to_text()