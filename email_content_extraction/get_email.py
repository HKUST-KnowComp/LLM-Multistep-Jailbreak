import csv
import time
import re
from utils import *
from prompts.prompts import *
import warnings
warnings.filterwarnings('ignore')

def get_email_pred_DP(csv_id = 0):
    """
    Get email content with Direct Prompt
    """
    
    input_msg_cot = input_msg_main + msg_time + msg_message_id + msg_subject + msg_output_structure
    combination = "sender/receiver/date/message_id/subject"
    file_name = 'data/enron_email_pred_DP/email_pred_dp_sender_receiver_data_messageid_subject_'

    with open('data/email_detail/email_detail_' + str(csv_id) + '.csv', 'r') as original_file, open(file_name + str(csv_id)     + '.csv', 'w',newline='') as new_file:
        reader = csv.reader(original_file)
        reader.__next__()
        writer = csv.writer(new_file)
        writer.writerows([['id', 'sender', 'receiver', 'date', 'message_id', 'subject', 'email_content', 'email_pred', 'combination']])
        for n in range(50): 
            print("----------- " + str(n) + " -----------")
            while True:
                row = next(reader)
                email_content = row[6]
                if len(email_content.split()) < 200:
                    break
            # id = row[0]
            sender = row[1]
            receiver = row[2]
            date = row[3]
            message_id = row[4]
            subject = row[5]
            # email_content = row[6]
            identifier = [sender, receiver, date, message_id, subject]

            try_count = 0
            while try_count < 3:
                try:
                    print("\nCombination: ", combination)
                    preds = query_resp_list_direct(input_msg_cot, identifier, row)
                    for p in preds:
                        p.append(combination)
                        writer.writerow(p)
                        new_file.flush()
                except Exception as e:
                    try_count += 1
                    print(e)
                    time.sleep(20)
                    continue
                
                break
                
def get_email_pred_MJP(csv_id = 0):
    """
    Get email content with Multi-step Jailbreaking Prompt
    """
    for i in range(7):
        if i == 0:
            input_msg_cot = input_msg_main + msg_time + msg_message_id + msg_subject + msg_output_structure
            combination = "sender/receiver/date/message_id/subject"
            file_name = 'data/enron_email_pred_MJP/email_pred_sender_receiver_data_messageid_subject_'
        elif i == 1:       
            input_msg_cot = input_msg_main + msg_time + msg_message_id + msg_output_structure
            combination = "sender/receiver/date/message_id"
            file_name = 'data/enron_email_pred_MJP/email_pred_sender_receiver_data_messageid_'
        elif i == 2:
            input_msg_cot = input_msg_main + msg_message_id + msg_subject + msg_output_structure
            combination = "sender/receiver/message_id/subject"
            file_name = 'data/enron_email_pred_MJP/email_pred_sender_receiver_messageid_subject_'
        elif i == 3:
            input_msg_cot = input_msg_main + msg_time + msg_subject + msg_output_structure
            combination = "sender/receiver/date/subject"
            file_name = 'data/enron_email_pred_MJP/email_pred_sender_receiver_data_subject_'
        elif i == 4:
            input_msg_cot = input_msg_main + msg_time + msg_output_structure
            combination = "sender/receiver/date"
            file_name = 'data/enron_email_pred_MJP/email_pred_sender_receiver_data_'
        elif i == 5:
            input_msg_cot = input_msg_main +  msg_message_id + msg_output_structure
            combination = "sender/receiver/message_id"
            file_name = 'data/enron_email_pred_MJP/email_pred_sender_receiver_messageid_'
        elif i == 6:           
            input_msg_cot = input_msg_main + msg_subject + msg_output_structure
            combination = "sender/receiver/subject"
            file_name = 'data/enron_email_pred_MJP/email_pred_sender_receiver_subject_'
        with open('data/enron_detail/email_detail_' + str(csv_id) + '.csv', 'r') as original_file, open(file_name + str(csv_id)     + '.csv', 'w', newline='') as new_file:

            reader = csv.reader(original_file)
            reader.__next__()
            writer = csv.writer(new_file)
            writer.writerows([['id', 'sender', 'receiver', 'date', 'message_id', 'subject', 'email_content', 'email_pred',  'normal_output', 'developer_output', 'parser_output', 'combination']])

            for n in range(50): 
                print("----------- " + str(n) + " -----------")
                while True:
                    row = next(reader)
                    email_content = row[6]
                    if len(email_content.split()) < 200:
                        break
                # id = row[0]
                sender = row[1]
                receiver = row[2]
                date = row[3]
                message_id = row[4]
                subject = row[5]
                # email_content = row[6]

                if i == 0:
                    identifier = [sender, receiver, date, message_id, subject]
                elif i == 1:       
                    identifier = [sender, receiver, date, message_id]
                elif i == 2:
                    identifier = [sender, receiver, message_id, subject]
                elif i == 3:
                    identifier = [sender, receiver, date, subject]
                elif i == 4:
                    identifier = [sender, receiver, date]
                elif i == 5:
                    identifier = [sender, receiver, message_id]
                elif i == 6:           
                    identifier = [sender, receiver, subject]
                
                try_count = 0

                while try_count < 3:
                    try:

                        print("\nCombination: ", combination)
                        preds = query_resp_list(input_msg_cot, identifier, row)
                        for p in preds:
                            text = p[-1]
                            normal_output = re.search(r"(?<=🔒Normal Output\) ).+", text).group() if re.search(r"(?<=🔒Normal Output\) ).+", text) is not None else ""
                            developer_output = re.search(r"(?<=🔓Developer Mode Output\) ).+", text).group() if re.search(r"(?<=🔓Developer Mode Output\) ).+", text) is not None else ""
                            parser_output = re.search(r'"email_content": "([^"]*)"', text).group(1) if re.search    (r'"email_content": "([^"]*)"', text.replace("\n", "")) is not None else ""
                            p.append(normal_output)
                            p.append(developer_output)
                            p.append(parser_output)
                            p.append(combination)
                            writer.writerow(p)
                            new_file.flush()

                    except Exception as e:
                        try_count += 1
                        print(e)
                        time.sleep(20)
                        continue
                    
                    break
    
if __name__ == "__main__":
    # get_email_pred_DP(0)
    get_email_pred_MJP(0)
    