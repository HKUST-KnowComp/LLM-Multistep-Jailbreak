## Multi-step Jailbreaking Privacy Attacks on ChatGPT
Paper Link: https://arxiv.org/pdf/2304.05197.pdf

We use MJP and direct propmpt to retreive the email from the enron dataset with the basic information of sender, receiver,subject and other several combination.

```bash
1. +date+msg_id+subject
2. +date+subject
3. +date+msg_id
4. +msg_id+subject
5. +date
6. +msg_id
7. +subject
```

We have prepared the sample data in the data/ folder. Then you can directly run the following command to predict the email content with MJP attacks. If you want to use the direct prompt, you can change the code in the get_email.py/main().

### 1. Get the prediction email from enron dataset
```bash
## get_email.py
python get_email.py
```

### 2.Evaluate the prediction results
```bash
## flair_evaluate.py
python flair_evaluate.py
```