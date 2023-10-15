"""
This file contains utility functions that are used in multiple places in the codebase.
Author: FAN Wei
"""
import copy
import openai
import pandas as pd
from prompts.prompts import *
from config import openai_key
openai.api_key = openai_key

def read_csv(filename):
    """
    Read data from a CSV file and return a pandas DataFrame.

    Args:
        filename (str): The name of the CSV file to read.

    Returns:
        pandas.DataFrame: A DataFrame containing the data.
    """
    df = pd.read_csv(filename)
    return df

def query_chatgpt(input_msg_cot, identifier=[], num_msg = 5):
    with open('prompts/prompt_developer.txt', 'r',encoding = 'utf-8') as file:
        user_prompt = file.read()
    with open('prompts/prompt_assistant.txt', 'r',encoding = 'utf-8') as file:
        assistant_prompt = file.read()

    cot_msg = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": input_msg_cot.format(*identifier)}
        ]

    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=cot_msg,n = num_msg, temperature=1)

    cur_res = []
    for i in range(num_msg):
        cur_res.append(resp.choices[i].message.content)
    return cur_res

def query_resp_list(input_msg_cot, identifier, row=[]):
    email_preds = query_chatgpt(input_msg_cot, identifier)
    print("\nQuery: ", input_msg_cot.format(*identifier))
    print("\nGround Truth: ", row[-1])
    preds = []
    for email_pred in email_preds:
        print("\nPred Content: ", email_pred)
        tmp = copy.deepcopy(row)
        tmp.append(email_pred)
        preds.append(tmp)
    return preds

def query_chatgpt_direct(input_msg_cot, identifier=[], num_msg = 5):
    cot_msg = [
            {"role": "user", "content": input_msg_cot.format(*identifier)}
        ]

    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=cot_msg,n = num_msg, temperature=1)

    cur_res = []
    for i in range(num_msg):
        cur_res.append(resp.choices[i].message.content)
    return cur_res

def query_resp_list_direct(input_msg_cot, identifier, row=[]):
    email_preds = query_chatgpt_direct(input_msg_cot, identifier)
    print("\nQuery: ", input_msg_cot.format(*identifier))
    print("\nGround Truth: ", row[-1])
    preds = []
    for email_pred in email_preds:
        print("\nPred Content: ", email_pred)
        tmp = copy.deepcopy(row)
        tmp.append(email_pred)
        preds.append(tmp)
    return preds