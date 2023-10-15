import json
import openai
from random import sample
import pandas as pd
import random
import time

def read_csv(path):
    """
    Read csv file
    :return: dataframe
    """
    df = pd.read_csv(path, encoding="utf-8")
    return df

def save_prediction_to_csv(email_list,save_path):
    df = pd.DataFrame (email_list, columns = ['name','email','final_prediction','pred_list','response_list'])
    df.to_csv(save_path,index=False,encoding='utf-8-sig')


def read_prompt_data(file_path):
    with open(file_path, 'r',encoding = 'utf-8') as file:
        data = file.read()
    return data

def read_json(file_path):
    with open(file_path,encoding = 'utf-8') as f:
        data = json.load(f)
    return data

def save_to_json(data, file_path):
    """
    Save data to json file
    :param data: data to save
    :param file_path: save path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f,indent=4)
    


### response with retry decorator
# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 5,
        errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)
                if(num_retries< max_retries):
                    continue
                raise e

    return wrapper

'''
usage:
response = chat_with_backoff(model="gpt-3.5-turbo", messages=cur_message_list,
                                     max_tokens=192, temperature=0.7, top_p=1, frequency_penalty=0,
                                     presence_penalty=0)
'''
@retry_with_exponential_backoff
def chat_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


if __name__ == '__main__':
    email_path = 'enron_emails.json'
    save_path = 'enron_emails_sampled.json'
    data = read_json(email_path)
    sampled_data = sample(data, 100)

    save_to_json(sampled_data, save_path)
