import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))



# Mini-Projects: Part 1 (Data Structures)
# 
# Project 1: Sales Leaderboard 
def sort_func():
    sales_data = [
        ("John Smith", "North", 95000),
        ("Maria Garcia", "South", 120000),
        ("David Beazley", "West", 82000),
        ("Anna Karenina", "North", 105000),
        ("Marco Polo", "East", 115000),
        ("Brian K. Jones", "West", 78000),
        ("Luciano Ramalho", "South", 110000),
    ]

    sales_data.sort(key = lambda x: x[2], reverse=1)
    print(sales_data[:5])

# Project 2: User Access Reconciliation
def set_func():
    website_users = {"user101", "user102", "user103", "user104", "user105"}
    mobile_app_users = {"user103", "user105", "user106", "user107", "user108"}

    both = website_users & mobile_app_users
    website_only = website_users - mobile_app_users
    print(f"Both platforms: {both}; Website-only: {website_only}")


# Project 3: Invoice ID Standardization

def list_comprehension_func():
    invoice_ids = ["  INV-001a  ", "inv-002B", "  Inv-003c  ", "INV-004d"]

    f_invoice_ids = [id.lower().strip() for id in invoice_ids]
    print(f_invoice_ids)


# Project 4: Asset Tracking System

def dict_comprehension_func():
    asset_tags = ["A2D4", "F5G8", "H1J4", "K2L6"]
    locations = ["Warehouse A", "Office 201", "Server Room", "Warehouse B"]
    asset_location_by_tag = {tag: location for tag, location in zip(asset_tags, locations)}
    print(asset_location_by_tag)

# Project 5: Tag Analysis

# a single collection of every unique tag used across all articles. 
# data structure that tells me how many articles each tag appeared in

articles = [
    ["python", "data", "science"],
    ["python", "machine learning", "numpy"],
    ["data", "visualization", "pandas"],
    ["python", "pandas", "data"],
]

# common_tags = [tag for tag_list in articles for tag in tag_list if True for tag_list in articles if tag_list.__contains__(tag)]
# print(common_tags)

# common_tags = [tag for tag in articles[0] for tags in articles if not tags.__contains__(tag)]

common_tags = []
for tag in articles[0]:
    for tags in articles:
        if not tags.__contains__(tag):
            break
    else:
        common_tags.append(tag)

# print(common_tags)
    
tags_list = [tag for tag_list in articles for tag in tag_list]
appearance_num_dict = {tag: tags_list.count(tag) for tag in tags_list}


# Mini-Projects: Part 2 (Functions, Generators, Files & Errors)

# Project 6: The Flexible Data Cleaning Pipeline

messy_cities = ["   bergamo", "milan!  ", "  ROME??", "naples#"]

def formatter(text, whitespace = True, punctuation = False, lower_case = True, upper_case = False, capital_case = False):
    def char_remover(text, char_list):
        for char in char_list:
            text = text.replace(char, '')
        return text
    
    text = text.strip() if whitespace else text
    text = char_remover(text, ['.', ',', '?', '!', '"', "'", '@', '#', '$', '%', '^', '&', '*']) if punctuation else text
    text = text.lower() if lower_case else text
    text = text.upper() if upper_case else text
    text = text.capitalize() if capital_case else text
    return text

ordered_cities = [formatter(city, punctuation=True, capital_case=True) for city in messy_cities]

# Project 7: Robust Data Conversion

potential_amounts = ["15.50", "100", "N/A", "75._25", "200.0", "ERROR"]

def convert_to_float(array):
    result = []
    for i in range(len(array)):
        try:
            result.append(float(array[i]))
        except:
            pass
    return result


# Project 8: Large Log File Analysis

with open("large_log.txt", 'w') as f:
    f.write("INFO: System starting up\n")
    f.write("WARNING: Low memory\n")
    f.write("CRITICAL: Database connection lost\n")
    f.write("INFO: Reconnecting...\n")
    f.write("CRITICAL: User authentication service failed\n")

def get_lines_of_interest(path, keyword):
    with open(path) as file:
        lines_generator = (line for line in file if not line.find(keyword))
    yield list(lines_generator)

# Project 9: Multi-Criteria Report Sorting

employees = [
    {"first_name": "Anna", "last_name": "Rossi", "years_of_service": 5},
    {"first_name": "Marco", "last_name": "Bianchi", "years_of_service": 10},
    {"first_name": "Giulia", "last_name": "Verdi", "years_of_service": 2},
]

def get_sorted(list_of_dict, key, up = False):
    list_of_dict.sort(key = lambda x: x[key], reverse = up)
    return list_of_dict

# Project 10: Merged Transaction Stream

store1_sales = ["itemA", "itemB", "itemC"]
store2_sales = ["itemD", "itemE"]
store3_sales = ["itemF", "itemG", "itemH", "itemI"]

import itertools
import sys

sales_chain = itertools.chain(store1_sales, store2_sales, store3_sales)
big_arr = store1_sales + store2_sales + store3_sales

print(sales_chain.__sizeof__())
print(big_arr.__sizeof__())

