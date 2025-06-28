import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Project 1: Bar Inventory Manager

def load_inventory():
    try:
        with open('inventory.txt', 'r+') as inventory_file:
            inventory_dict = {key: int(value) for key, value in [line.split(',') for line in inventory_file]}
            return inventory_dict
    except FileNotFoundError:
        with open('inventory.txt', 'w') as inventory_file:
            print("The inventory is empty")


delivery_1 = [('orange', 10), ('tonic water', 10), ('napkins', 20), ('lime', 10), ('chips', 20)]

def update_stock(new_delivery):
    inventory = load_inventory()
    inventory.update({tuple[0]: inventory.get(tuple[0], 0) + tuple[1] for tuple in new_delivery})
    print(inventory)
    with open('inventory.txt', 'r+') as inventory_file:
        inventory_file.writelines([f'{key},{inventory[key]}\n' for key in inventory])

        
# Project 2: Customer Feedback Sanitizer

import re

def rm_whitespace(text):
    return text.strip()
def rm_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text
def rm_doublespace(text):
    for i in range(5):
        text.replace('  ', ' ')
def letter_formatting(text, cap = True, low = False, up = False):
    if low:
        return text.lower()
    elif up:
        return text.upper()
    else:
        return text[0].capitalize() + text[1:]
def apply_funcs(list, *funcs):
    for item in list:
        if(item.replace('\n','') == ''): continue
        if type(funcs[0]) == tuple: funcs = funcs[0]
        for func in funcs:
            item = func(item)
        yield item + '\n'
    
def get_feedback_formatted(*funcs):
    with open('feedback.txt') as feedback_file:
        with open('feedback_formatted.txt', 'w') as feedback_f:
            feedback_f.writelines([item for item in apply_funcs(feedback_file, funcs)])


def get_unique_words():
    with open('feedback.txt') as feedback_file:
        formatted_lines = apply_funcs(feedback_file.readlines(), rm_whitespace, rm_punctuation, letter_formatting)
        return set(' '.join([word for line in formatted_lines for word in line.rstrip().split(' ')]).lower().split(' '))



# Project 3: Sales Data Aggregator
import collections

def get_products():
    with open('products.txt') as products_file:
        return {line.split(',')[0]:float(line.split(',')[1]) for line in products_file}
def get_sales():
    with open('sales.txt') as sales_file:
        return [item for item in sales_file.__next__().split(',')]
    
def get_sales_occurencies():
    sales = get_sales()
    return sorted([[item_name, sales.count(item_name)] for item_name in {item for item in sales}], key=lambda x: x[1], reverse=True)

# print(get_sales_occurencies()[:3])

def calculate_revenue():
    products = get_products()
    sales_occurencies = {item[0]: item[1] for item in get_sales_occurencies()}
    revenue = sum([sales_occurencies[item] * products[item] for item in sales_occurencies])
    return revenue

# print(calculate_revenue())


# Project 4: Log File Anomaly Detector

# We have a large server log file (access.log) that is too big to load into memory. I need a script that can process this file line-by-line to count how many requests resulted in an error (HTTP status codes 4xx or 5xx) and identify when they occurred.

# a) how many requests resulted in an error
# b) identify when they occurred

import datetime
import itertools

def get_err_report():
    with open('big_log.txt') as big_log:
        err_requests_gen = (line for line in big_log if line[-4] == '5' or line[-4] == '4')
        num_of_errors = len(list(err_requests_gen))
        err_by_datestr_to_err = {item[item.index('[') + 1:item.index(']')]:item[item.index('"'):] for item in err_requests_gen if '[' in item}
        err_by_date_to_err = {datetime.datetime.strptime(item, '%d/%b/%Y:%H:%M:%S'):err_by_datestr_to_err[item].rstrip() for item in err_by_datestr_to_err}
        # print([item for item in err_requests_gen])
        report = '\n'.join([f'{err_by_date_to_err[date]} at {date.strftime("%d, %b, %Y")}' for date in err_by_date_to_err])
        return f'''{num_of_errors} error requests occured:
{report}
        '''
# print(get_err_report())


def get_ip_groups_gen():
    with open('big_log.txt') as big_log:
        big_log = [item for item in big_log if '[' in item]
        gen = itertools.groupby(big_log, lambda x: x[:x.index('[')-3])
        return gen

# print([[obj for obj in item[1]] for item in get_ip_groups_gen()])



# Project 5: Multi-Department Data Unifier

with open('sales_ids.txt', 'w', encoding='utf-8') as file:
    file.write('C001,C002,C003,C004')
with open('marketing_ids.txt', 'w', encoding='latin-1') as file:
    file.write('C002,C005,C006,C004')
with open('support_ids.txt', 'w', encoding='utf-8') as file:
    file.write('C003,C004,C007')

with open('sales_ids.txt') as sales_file, \
    open('marketing_ids.txt', encoding='latin-1') as marketing_file, \
    open('support_ids.txt',) as support_file:

    file_list = [sales_file, marketing_file, support_file]
    file_set_list = [set(file.readlines()[0].split(',')) for file in file_list]
    cust_unique_sales = file_set_list[0] - file_set_list[1] - file_set_list[2]
    cust_unique_marketing = file_set_list[1] - file_set_list[0] - file_set_list[2]
    cust_unique_support = file_set_list[2] - file_set_list[0] - file_set_list[1]
    cust_unique_dict = {
        'sales': cust_unique_sales,
        'marketing': cust_unique_marketing, 
        'support': cust_unique_support
    }
    cust_common = file_set_list[0] & file_set_list[1] & file_set_list[2]
    cust_all_unique = file_set_list[0] | file_set_list[1] | file_set_list[2]


    with open('master_customer_list.txt', 'w') as final_file:
        final_file.write(','.join(cust_all_unique))
    
    # print([cust_unique_dict[item] for item in cust_unique_dict], cust_common, cust_all_unique)



import random
from collections import Counter
import time

# Create a large list of 1,000,000 numbers
data = [random.randint(1, 100) for _ in range(1_000_000_0)]
unique_items = set(data)

# --- Method 1: list.count() in a loop ---
start_time = time.time()
counts_loop = {item: data.count(item) for item in unique_items}
end_time = time.time()
print(f"Using list.count() in a loop took: {end_time - start_time:.4f} seconds")

# --- Method 2: collections.Counter ---
start_time = time.time()
counts_counter = {}
for item in data:
    counts_counter[item] = counts_counter.get(item, 0) + 1
end_time = time.time()
print(f"Using collections.Counter took:   {end_time - start_time:.4f} seconds")

# --- Method 2: collections.Counter ---
start_time = time.time()
counts_counter = Counter(data)
end_time = time.time()
print(f"Using collections.Counter took:   {end_time - start_time:.4f} seconds")


