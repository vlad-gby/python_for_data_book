import numpy as np

# ADDITIONAL PRACTICE TASKS

# Calculate the total number of products sold each day.
# Calculate the average sales for each product over the entire week.

# 7 days, 4 products
sales_data = np.array([
    [10, 15, 12, 18],  # Day 0
    [14, 11, 9,  16],  # Day 1
    [8,  20, 13, 22],  # Day 2
    [17, 14, 11, 20],  # Day 3
    [12, 18, 10, 19],  # Day 4
    [20, 22, 18, 25],  # Day 5
    [18, 19, 15, 23]   # Day 6
])

sum_for_day = np.sum(sales_data, axis=1)
product_averages = np.average(sales_data, axis = 0)

# print(sum_for_day, product_averages)


# 5 students, 3 tests
student_scores = np.array([
    [75, 82, 88],
    [91, 94, 89],
    [65, 70, 72],
    [88, 85, 90],
    [78, 81, 84]
])

# Bonus points for each of the 3 tests
bonus_points = np.array([5, 3, 7])

scores_with_bonus = student_scores + bonus_points
# print(scores_with_bonus)



rng = np.random.default_rng(seed=101)
pressure_readings = rng.integers(0, 101, size=20)

# Create a boolean mask to identify all the "unsafe" readings.
# Use the mask to create a new array containing only the unsafe values.

unsafe_readings = (pressure_readings < 20) | (pressure_readings > 80)
unsafe_values = pressure_readings[unsafe_readings]
# print(unsafe_values)



# Prices for parts with IDs 0 through 9
master_price_list = np.array([10.50, 22.00, 5.75, 8.00, 32.50, 14.99, 55.00, 3.25, 18.49, 25.00])

# The customer wants parts with these specific IDs
shopping_list_ids = np.array([8, 2, 0, 5])
numer_of_parts = rng.integers(1, 11, size=shopping_list_ids.shape)
shopping_list_ids = np.array([master_price_list[shopping_list_ids], numer_of_parts])
prices = shopping_list_ids[0] * shopping_list_ids[1]
# print(shopping_list_ids, prices)

# Use the shopping_list to instantly retrieve the prices of all the items they want to buy.


# Reshape the 1D wind_log into a 2D array where each row represents a day and each column represents an hour (7x24).
# Transpose the resulting array so that each row represents an hour and each column represents a day (24x7).

rng = np.random.default_rng(seed=42)
# 7 days * 24 hours = 168 readings
wind_log = rng.random(168) * 40 # Wind speed in km/h
wind_log = wind_log.reshape(7, 24)
# print(wind_log.T)



# Unifying Task: Group 1


# Use the customer's order_ids to retrieve the price and stock information for the ordered books from the inventory matrix.
# From this retrieved list, identify which books are actually in stock (stock > 0).
# Calculate the total cost of the items that are in stock.
# For all items in the order, create a "status" array: if the stock is 0, the status should be 0; otherwise, the status should be 1.


# Inventory for 10 books. Columns: [price, stock_quantity]
inventory = np.zeros((10, 2))
inventory[:, 0] = rng.random(10) * 20 + 10 # Prices between 10 and 30
inventory[:, 1] = rng.integers(0, 10, size=10) # Stock levels between 0 and 14

# A customer order
order_ids = np.array([7, 3, 9, 3, 2, 2])

price_and_stock = inventory[order_ids]
in_stock = price_and_stock[price_and_stock[:, 1] > 0]
in_stock_total = in_stock[:, 0].sum()
status = np.where(price_and_stock[:, 1] > 0, 1, 0)

