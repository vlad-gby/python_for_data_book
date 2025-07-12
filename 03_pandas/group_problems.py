## Group 1: Data Ingestion & Inspection
# (Goal: Load data from an external source and understand its basic structure, content, and quality without making any changes.)

# ### Project 1.1: Initial Exploration of Book Sales Data
# Scenario: You've just been handed a CSV file containing sales data from a small bookstore.

# Your Task: Your manager wants a high-level, first-look report. You need to load the data and provide a summary of its structure, the first few entries, and the types of data it contains.

# Dataset Description: Create a DataFrame that simulates this data. It must have columns for Title (string), Author (string), Genre (string), PublicationYear (integer), and CopiesSold (integer). Include at least 15 rows of data.

# Potential Functions: pd.read_csv, .head(), .info(), .shape

import pandas as pd
import numpy as np

sales = pd.read_csv('datasets_groups/book_sales.csv')
first_entries = sales.head()
# info = sales.info()
shape = sales.shape
# print(shape)



# ### Project 1.2: Cafe Transaction Analysis
# Scenario: The point-of-sale system at a local cafe exports its daily transaction log as a Python list of dictionaries. You need to load this data and analyze product popularity.

# Your Task: Load the data from the provided Python list. Then, determine which products are sold most frequently, list all unique products sold, and get a statistical summary of the transaction prices.

cafe_transactions = [
    {'Timestamp': '08:05', 'Item': 'Espresso', 'Price': 2.50},
    {'Timestamp': '08:06', 'Item': 'Cornetto', 'Price': 1.50},
    {'Timestamp': '08:10', 'Item': 'Cappuccino', 'Price': 3.00},
    {'Timestamp': '08:11', 'Item': 'Espresso', 'Price': 2.50},
    {'Timestamp': '08:15', 'Item': 'Espresso', 'Price': 2.50},
    {'Timestamp': '08:22', 'Item': 'Cornetto', 'Price': 1.50},
    {'Timestamp': '08:25', 'Item': 'Latte', 'Price': 3.50},
    {'Timestamp': '08:31', 'Item': 'Muffin', 'Price': 2.75},
    {'Timestamp': '08:35', 'Item': 'Cappuccino', 'Price': 3.00},
    {'Timestamp': '08:42', 'Item': 'Espresso', 'Price': 2.50},
    {'Timestamp': '08:43', 'Item': 'Cornetto', 'Price': 1.50},
    {'Timestamp': '08:50', 'Item': 'Latte', 'Price': 3.50},
    {'Timestamp': '08:55', 'Item': 'Espresso', 'Price': 2.50},
    {'Timestamp': '09:01', 'Item': 'Cappuccino', 'Price': 3.00},
    {'Timestamp': '09:03', 'Item': 'Muffin', 'Price': 2.75},
    {'Timestamp': '09:12', 'Item': 'Espresso', 'Price': 2.50},
    {'Timestamp': '09:18', 'Item': 'Cornetto', 'Price': 1.50},
    {'Timestamp': '09:24', 'Item': 'Latte', 'Price': 3.50},
    {'Timestamp': '09:30', 'Item': 'Espresso', 'Price': 2.50},
    {'Timestamp': '09:33', 'Item': 'Americano', 'Price': 2.75},
    {'Timestamp': '09:40', 'Item': 'Cornetto', 'Price': 1.50},
    {'Timestamp': '09:45', 'Item': 'Cappuccino', 'Price': 3.00},
    {'Timestamp': '09:51', 'Item': 'Espresso', 'Price': 2.50},
    {'Timestamp': '09:58', 'Item': 'Americano', 'Price': 2.75}
]

cafe_transactions_df = pd.DataFrame(cafe_transactions)
sorted_3_most_sold = cafe_transactions_df.value_counts(['Item']).iloc[:3]
unique_prods = cafe_transactions_df['Item'].unique()
summary = cafe_transactions_df['Price'].describe()

# print(sorted_3_most_sold)


# ### Project 1.3: User Account Quality Audit
# Scenario: You've been given a raw text data dump of recent user sign-ups. The data is messy and uses a pipe | as a separator. You need to perform a quick quality check.

# Your Task: Load the data from the provided raw text string. Check for missing information in the records and verify the uniqueness of the UserID.

users = pd.read_csv('datasets_groups/users.csv', sep='|')
is_id_unique = users['UserID'].is_unique
missing_data = users.isna().sum()
# print(missing_data)


# ### Project 2.1: Filtering a Customer List for a Marketing Campaign
# Scenario: A marketing team wants to target a very specific group of customers for a new campaign in Germany.

# Your Task: From the provided customer data, create a final DataFrame that contains only the customers who meet all of the following criteria:

# They are from 'Germany'.

# Their age is between 30 and 40 (inclusive).

# Their last_name is missing.


sales = pd.read_csv('datasets_groups/customer_data.csv')
cust_of_interest = sales.loc[(sales['country'] == 'Germany') & 
                             (sales['age'] <= 40) & 
                             (sales['age'] >= 30) & 
                             (sales['last_name'].isna())]
# print(cust_of_interest)


# ### Project 2.2: Repairing Warehouse Inventory Data
# Scenario: You receive an inventory list from a warehouse as a Python list of dictionaries. You notice some products are listed twice and some counts are missing.

# Your Task: Clean the inventory list to produce a final, accurate DataFrame.

# Load the data from the Python list.

# Remove any completely duplicate rows.

# Some products have a missing quantity. Fill these NaN values with 0.

# A few products have a missing category. Drop these rows entirely, as they cannot be properly categorized.

# Ensure the quantity column is an integer data type.

inventory_list = [
    {'product_id': 'A101', 'product_name': 'Keyboard', 'category': 'Electronics', 'quantity': 150.0},
    {'product_id': 'A102', 'product_name': 'Mouse', 'category': 'Electronics', 'quantity': 200.0},
    {'product_id': 'B201', 'product_name': 'T-Shirt', 'category': 'Apparel', 'quantity': 300.0},
    {'product_id': 'A102', 'product_name': 'Mouse', 'category': 'Electronics', 'quantity': 200.0}, # Duplicate Row
    {'product_id': 'C301', 'product_name': 'Coffee Mug', 'category': 'Kitchenware', 'quantity': np.nan}, # Missing Quantity
    {'product_id': 'B202', 'product_name': 'Jeans', 'category': 'Apparel', 'quantity': 250.0},
    {'product_id': 'D401', 'product_name': 'Unknown Item', 'category': np.nan, 'quantity': 100.0}, # Missing Category
    {'product_id': 'C301', 'product_name': 'Coffee Mug', 'category': 'Kitchenware', 'quantity': np.nan}  # Duplicate with NaN
]

inv = pd.DataFrame(inventory_list)
inv_cleaned = inv.drop_duplicates().dropna(subset=['category']).fillna(0)
inv_cleaned['quantity'] = inv_cleaned['quantity'].astype(int)
# print(inv_cleaned)



# ### Project 2.3: Processing Raw Server Logs
# Scenario: You are given a small snippet of a raw server log as a single block of text. You need to parse it and extract only the critical error messages from a specific time window.

# Your Task:

# Load the multi-line string data into a DataFrame.

# Filter the DataFrame to select only the rows where the LogLevel is 'ERROR'.

# From that result, further filter to select only the errors that occurred between 02:00:00 and 03:00:00.

import re
import io
import datetime

with open('datasets_groups/sys_log.csv') as file:
    log = file.readlines()
    new_log = []
    for line in log:
        new_log.append(re.sub(r'\s{2,}', '*', line)) 

df = pd.read_csv(io.StringIO('\n'.join(new_log)), sep='*')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
errors = df[df['LogLevel'] == 'ERROR']
errors_of_interest = errors[(errors['Timestamp'].dt.time > pd.to_datetime('02:00:00').time()) &
                            (errors['Timestamp'].dt.time < pd.to_datetime('03:00:00').time())]

# print(errors_of_interest)



# ### Project 3.1: E-commerce Order Profitability Analysis
# Scenario: You have a raw dataset of individual order items from an online store.

# Your Task: The objective is to analyze the profitability of each order item. The final DataFrame must contain two new columns: profit and profit_margin_percent. The result should be sorted to show the 5 most profitable order items at the top.

data = pd.read_csv('datasets_groups/orders_data.csv')
profit_col = (data['sale_price'] - data['cost_price']) * data['quantity']
prof_marg_col = (data['sale_price'] - data['cost_price']) * data['quantity'] / data['sale_price'] / data['quantity']

final_df = pd.DataFrame()
final_df['profit'] = profit_col
final_df['profit_margin_percent'] = prof_marg_col

# print(final_df)



### Project 3.2: Standardizing User Data
# Scenario: You have raw user data from an old system with messy column names and inconsistent country information.

# Your Task: The goal is to produce a clean, production-ready DataFrame. The final DataFrame must have snake_case column names (user_id, full_name, country), standardized three-letter country codes ('USA', 'ITA', 'DEU'), and use the user_id as its index.

user_data = [
    [101, 'John Smith', 'USA'],
    [102, 'Anna Rossi', 'italia'],
    [103, 'Peter Jones', 'U.S.A.'],
    [104, 'Maria Garcia', 'Spain'],
    [105, 'Hans Schmidt', 'Germany'],
    [106, 'Luca Conti', 'ITALY'],
    [107, 'David Brown', 'United States']
]
messy_columns = ['USER ID', 'FULL NAME', 'COUNTRY OF ORIGIN']

def get_nice_cols(cols):
    new_cols = []
    for col in cols:
        new_cols.append(col.replace(' ', '_').lower())
    return new_cols
new_cols = get_nice_cols(messy_columns)

def get_country_codes(country):
    country = country.lower().replace('.', '').replace(',', '')
    match country:
        case 'usa' | 'united states':
            return 'USA'
        case 'italy' | 'italia':
            return 'ITA'
        case 'germany':
            return "DEU"
        case 'spain':
            return "SPA"


user_data_df = pd.DataFrame(user_data, columns=new_cols)
user_data_df['country_of_origin'] = user_data_df['country_of_origin'].map(get_country_codes)
user_data_df = user_data_df.set_index('user_id')

# print(user_data_df)


### Project 3.3: Processing and Enriching Weather Data
# Scenario: You have a dataset of daily weather recordings where the date is a string and the temperature is in Fahrenheit.

# Your Task: The objective is to enrich the weather data. The final DataFrame must include the temperature in Celsius and the corresponding day of the week for each recording.

# Provided Data (Python Dictionary):

weather_data = {
    'date_str': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19'],
    'temp_fahrenheit': [35.6, 33.8, 41.0, 32.0, 37.4],
    'condition': ['Cloudy', 'Sunny', 'Rain', 'Sunny', 'Cloudy']
}

data = pd.DataFrame(weather_data)
data['temp_celsius'] = data['temp_fahrenheit'].apply(lambda x : (x - 32) * 5 / 9)
data['week_day'] = pd.to_datetime(data['date_str']).dt.day_name()
data = data.iloc[:, [0, 4, 1, 3, 2]]

# print(data)




# ### Project 4.1: Analyzing Regional Sales Performance
# Scenario: You are given a raw transaction log from an e-commerce platform.

# Your Task: Create a summary report that shows the performance of each sales Region. The final report must show the total Revenue, the average Quantity per order, and the number of unique Products sold for each region.

log = pd.read_csv('datasets_groups/sales_log_data.csv')
log['Revenue'] = log['Price'] * log['Quantity']

summary = log.groupby('Region').agg(tot_rev = ('Revenue', 'sum'),
                                 avg_quantity = ('Quantity', 'mean'),
                                 unique_prods = ('Product', 'nunique'))

# print(summary)


### Project 4.2: User Engagement by Signup Cohort
# Scenario: You have activity data from a mobile app. You want to see if users who signed up in different months behave differently.

# Your Task: The objective is to compare user engagement by their sign-up month. The final DataFrame should be indexed by month and show the average_sessions, max_time_spent, and total_users for each monthly cohort.

user_activity_data = {
    'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'signup_date': ['2025-01-15', '2025-01-22', '2025-02-05', '2025-02-10', '2025-02-18', '2025-03-01', '2025-03-11', '2025-03-14', '2025-03-20', '2025-01-28'],
    'sessions': [10, 15, 5, 8, 9, 22, 18, 12, 14, 11],
    'time_spent_min': [120, 200, 45, 90, 110, 300, 250, 150, 180, 130]
}

data = pd.DataFrame(user_activity_data)
data['signup_date'] = pd.to_datetime(data['signup_date'])
data['month'] = data['signup_date'].dt.month

groups = data.groupby('month').agg(average_sessions = ('sessions', 'mean'),
                                    max_time_spent = ('time_spent_min', 'max'),
                                    total_users = ('user_id', 'count'))

# print(groups)


### Project 4.3: Product Preference by Customer Segment
# Scenario: A marketing team wants to understand the relationship between different customer segments and the product categories they purchase from.

# Your Task: Create a summary frequency table (a cross-tabulation) that shows the number of purchases for each combination of customer_segment and product_category. The rows should be the segments, and the columns should be the categories.

import io

data_str = """transaction_id~customer_segment~product_category
1~New~Electronics
2~Loyal~Apparel
3~VIP~Books
4~New~Apparel
5~Loyal~Electronics
6~New~Electronics
7~VIP~Electronics
8~Loyal~Books
9~New~Apparel
10~Loyal~Apparel
11~VIP~Books
12~New~Electronics"""

data_df = pd.read_csv(io.StringIO(data_str), sep='~')
reshaped_count = pd.crosstab(data_df['customer_segment'], data_df['product_category'])



# ## Group 5: Joining & Reshaping

# ### Project 5.1: Combining Customer and Order Data
# Scenario: An online store keeps its customer information and order details in two separate files.

# Your Task: Create a single, unified DataFrame that links each order to the customer's name and country.

custs = pd.read_csv('datasets_groups/customers.csv')
orders = pd.read_csv('datasets_groups/orders.csv')

unified_df = custs.merge(orders, on='customer_id')
unified_df = unified_df[['customer_id', 'name', 'country']]

# print(unified_df)


# ### Project 5.2: Reshaping "Wide" Sensor Data to "Long" Format
# Scenario: An environmental monitoring system outputs its data in a "wide" format, where each month's reading is in a separate column. This format is difficult to use for plotting.

# Your Task: Transform the provided "wide" DataFrame into a "long" (or "tidy") format. The final DataFrame should have only three columns: sensor_id, month, and reading.

wide_sensor_data = {
    'sensor_id': ['SensorA', 'SensorB', 'SensorC'],
    'Jan_2025': [15.2, 14.8, 15.5],
    'Feb_2025': [16.1, 15.5, 16.3],
    'Mar_2025': [18.5, 17.9, 18.8]
}

data = pd.DataFrame(wide_sensor_data)
data = pd.melt(data, id_vars='sensor_id', var_name='month', value_name='reading')

# print(data)


# ### Project 5.3: Creating a Sales Summary Pivot Table
# Scenario: You have a long, raw list of sales transactions. A manager wants a compact, spreadsheet-style summary showing total sales for each product across different regions.

# Your Task: Create a pivot table from the provided transaction data. The final table should have each unique Product as a row, each unique Region as a column, and the sum of Sales as the values in the cells. Fill any missing combinations with 0.

transaction_data = [
    {'Region': 'North', 'Product': 'Keyboard', 'Sales': 225},
    {'Region': 'South', 'Product': 'Mouse', 'Sales': 127},
    {'Region': 'North', 'Product': 'Monitor', 'Sales': 300},
    {'Region': 'West', 'Product': 'Mouse', 'Sales': 72},
    {'Region': 'South', 'Product': 'Webcam', 'Sales': 100},
    {'Region': 'North', 'Product': 'Keyboard', 'Sales': 75},
    {'Region': 'West', 'Product': 'Monitor', 'Sales': 560},
    {'Region': 'South', 'Product': 'Keyboard', 'Sales': 225},
    {'Region': 'North', 'Product': 'Mouse', 'Sales': 102}
]

data = pd.DataFrame(transaction_data)
pivot = data.pivot_table(index='Product', columns='Region', aggfunc='sum', values='Sales').fillna(0)

# print(pivot)


# ### Project 6.1: Consolidating Monthly Sales Reports
# Scenario: You have received three separate monthly sales reports as different data files. You also have a separate file with product category information.

# Your Task: Create a single, master DataFrame for the entire quarter. This master table should include the Revenue for each transaction, and it must also be enriched with the correct Category for each product sold. Finally, provide a simple summary of total sales per category.

import io

jan_sales = pd.read_csv('datasets_groups/jan_sales.csv')
from feb_sales_data import feb_sales_data
feb_sales = pd.DataFrame(feb_sales_data)
with open('datasets_groups/mar_sales_data.txt') as f:
    mar_sales = pd.read_csv(io.StringIO(''.join(f.readlines())), sep='|')
prod_lookup = pd.read_csv('datasets_groups/product_lookup.csv')

quarter_sales = pd.concat([jan_sales, feb_sales, mar_sales]).reset_index(drop=1)
quarter_sales['Revenue'] = quarter_sales['Quantity'] * quarter_sales['Price']
quarter_sales = quarter_sales.merge(prod_lookup, on = 'ProductID')

sales_per_cat = quarter_sales.groupby('Category')[['Quantity', 'Price', 'Revenue']].sum()

# print(sales_per_cat)



### Project 6.2: Advanced User Activity Pivot Table
# Scenario: You have a log of user activities, and you need to create a complex summary report.

# Your Task: Create a single pivot table that shows user engagement metrics. The table's rows should be the User_Segment, the columns should be the Device_Type, and the values should display both the average session duration AND the total number of unique actions for each combination. The report must   


user_activity = pd.read_csv('datasets_groups/user_activity.csv')
pivot = user_activity.pivot_table(index='User_Segment', columns='Device_Type', 
                                  values=['Session_Duration_Min', 'Action'],
                                  aggfunc={'Session_Duration_Min': 'mean',
                                           'Action': 'nunique'},
                                  margins=1)
pivot_adequate_naming = user_activity.groupby('User_Segment').agg(avg_session_duration = ('Session_Duration_Min', 'mean'),
                                                                  nunique_actions = ('Action', 'nunique'))
#won't do the totals - you'll show me if there is a concise way of making those. Pivot is not a good way of doing it

# print(pivot_adequate_naming)


### Project 6.3: Tidying and Reshaping Messy Health Data
# Scenario: You have received patient data from a clinical trial in a "wide" format that is difficult to analyze. The data needs to be cleaned, reshaped into a "long" format, and then pivoted back into a clean summary table.

# Your Task:

# Load the data and clean the column names (e.g., remove special characters, make them lowercase).

# "Melt" the DataFrame to transform it from a wide format to a long format with columns for patient_id, visit, and measurement.

# Create two new columns from the measurement column: one for the metric (e.g., 'hr' or 'bp') and one for the value.

# Create a final, clean pivot table where the index is patient_id, the columns are the metric, and the values are the average measurement for that metric across all visits.

patient_data = {
    'Patient-ID': ['P01', 'P02', 'P03'],
    'Visit 1 (HR)': [72, 68, 75],
    'Visit 1 (BP)': [120, 115, 125],
    'Visit 2 (HR)': [75, 70, 78],
    'Visit 2 (BP)': [122, 118, 128]
}

data = pd.DataFrame(patient_data)
data.columns = data.columns.str.lower().str.replace('[()]', '', regex=True).str.replace('[ -]', '_', regex=True)
data = data.melt(id_vars='patient_id', var_name='visit', value_name='measurement')
data['metric'] = data['visit'].str.extract(pat=r'(hr|bp)')
data['visit'] = data['visit'].str.extract(pat=r'(\d)')
data['metric'] = data['metric'].str.upper()
data = data.iloc[:, [0, 1, 3, 2]]

pivot = data.pivot_table(index='patient_id', columns='metric', values='measurement')

# print(pivot)


### Project 6.4: Analyzing Voting Patterns
# Scenario: You have data from a local election with voter demographics and their vote.

# Your Task: Create a cross-tabulation that shows the voting distribution (how many people voted 'Yes' vs. 'No') across different Age_Group and District combinations. The final table must show percentages relative to each district's total vote.

voting_data = """VoterID,Age_Group,District,Vote
1,18-30,North,Yes
2,31-50,South,No
3,51+,North,No
4,18-30,North,Yes
5,31-50,West,Yes
6,51+,South,No
7,18-30,West,No
8,31-50,North,Yes
9,51+,West,Yes"""

data = pd.read_csv(io.StringIO(voting_data))
voting_distribution = pd.crosstab([data['Age_Group'], data['District']], data['Vote'], normalize='index') * 100
voting_distribution['Yes'] = voting_distribution['Yes'].astype(int).astype(str) + '%'
voting_distribution['No'] = voting_distribution['No'].astype(int).astype(str) + '%'

# print(voting_distribution)


### Project 6.5: Merging Aggregated Data Back to Original Source
# Scenario: You have a DataFrame of employees and their sales. You need to calculate each employee's performance relative to their department's average.

# Your Task:

# Calculate the average sales for each Department.

# Join this aggregated information back to the original employee DataFrame.

# Create a new column, performance_vs_avg, that shows how much each employee's sales are above or below their department's average.

employee_sales = [
    {'Employee_ID': 'E101', 'Name': 'Alice', 'Department': 'Sales', 'Sales': 50000},
    {'Employee_ID': 'E102', 'Name': 'Bob', 'Department': 'Sales', 'Sales': 60000},
    {'Employee_ID': 'E201', 'Name': 'Charlie', 'Department': 'Marketing', 'Sales': 25000},
    {'Employee_ID': 'E103', 'Name': 'David', 'Department': 'Sales', 'Sales': 45000},
    {'Employee_ID': 'E202', 'Name': 'Eve', 'Department': 'Marketing', 'Sales': 30000}
]

data = pd.DataFrame(employee_sales)
avg_for_dep = data[['Department', 'Sales']].groupby('Department').transform('mean')
data['Dep_Avg'] = avg_for_dep.round()
data['Performance_vs_Avg'] = data['Sales'] - data['Dep_Avg']

# print(data)








