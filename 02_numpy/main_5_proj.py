import numpy as np


# Final Project 1: Monte Carlo Pizza Shop Simulation
# Scenario:
# You want to decide if opening a small pizza shop in Bergamo is financially viable. You decide to run a Monte Carlo simulation to model one year (365 days) of profit. Profit depends on several random variables: the number of customers each day and the variable cost of ingredients per customer.

# Dataset Creation:

# Create a 1D array daily_customers for 365 days. The number of customers per day should be random integers between 50 and 150.
# Create a 1D array cost_per_customer. This should be drawn from a normal distribution with a mean of €8.50 and a standard deviation of €1.50.
# Assume a fixed price per customer of €17.00 and fixed overhead costs (rent, etc.) of €150 per day.
# Your Task:

# Calculate the daily revenue, daily variable costs, and then the daily profit.
# Calculate the cumulative_profit over the year.
# Find the average_daily_profit and the total_profit for the year.
# Determine the number of days the shop was profitable (profit > 0).
# Find the worst_day_loss and best_day_profit.
# Using np.where, create a "performance" array where days with profit > €500 are labeled 2 (great), days with profit between €0 and €500 are labeled 1 (good), and days with a loss are labeled 0 (bad).
# Save the daily_profit and cumulative_profit arrays to a single compressed .npz file.
# Concepts to Synthesize: Random Generation, Vectorized Arithmetic, Broadcasting, Statistical Methods (sum, mean, min, max), Boolean Indexing, np.where, File I/O.

rng = np.random.default_rng(seed=1234)

daily_customers = rng.integers(50, 151, size=356)
cost_per_customer = rng.standard_normal(size=daily_customers.size) * 3 + 8.5 #greater deviation, as no loss days with dev1.5
revenue_per_customer = 17
daily_rent = 150

daily_revenue = daily_customers * revenue_per_customer
daily_costs = daily_customers * cost_per_customer + daily_rent
daily_profit = daily_revenue - daily_costs
cum_profit = daily_profit.cumsum()
average_daily_profit = np.average(daily_profit)
total_profit = average_daily_profit * 365

days_in_profit = (daily_profit[daily_profit > 0]).size
worst_day_profit = daily_profit.min()
best_day_profit = daily_profit.max()

performance = np.select([daily_profit < 0, daily_profit < 900], ['loss', 'good'], 'great')
unique = np.unique(performance, return_counts=True)

np.savez('profitability.npz', daily = days_in_profit, cum_profit = cum_profit)




# Final Project 2: Image Processing - Applying a Vignette Filter
# Scenario:
# You need to write a function that applies a photographic "vignette" effect to an image. A vignette darkens the corners of an image, drawing focus to the center. This can be done by creating a brightness image based on each pixel's distance from the center.

# Dataset Creation:

# Create a simulated 3-channel color image named image. It should be a 3D NumPy array of shape (200, 300, 3) (200 pixels high, 300 wide, 3 channels for R,G,B).
# The values should be floating-point numbers between 0.0 and 1.0 (you can use rng.random).
# Your Task:

# Find the center coordinates of the image (e.g., center_y, center_x).
# Use np.linspace or np.arange to create two 1D arrays representing the y and x coordinates of the pixels.
# Use np.meshgrid to create two 2D coordinate matrices, Y and X, from these coordinate vectors.
# Calculate a distance matrix. For each pixel, its distance from the center is sqrt((X - center_x)**2 + (Y - center_y)**2).
# Create a 2D vignette_mask from this distance matrix. The mask's values should fade from 1.0 at the center to a smaller value (e.g., 0.2) at the edges. A simple way to do this is to normalize the distance to be between 0 and 1 and then apply a function like 1 - normalized_distance.
# The vignette_mask is 2D, but your image is 3D. You need to make the mask compatible for multiplication. Reshape the (200, 300) mask into a (200, 300, 1) array.
# Apply the filter by multiplying the 3D image array by the reshaped 3D vignette_mask. Broadcasting will handle the rest.
# Concepts to Synthesize: np.meshgrid, np.linspace, Vectorized Arithmetic, ufuncs (sqrt), Reshaping, Broadcasting.

import matplotlib.pyplot as plt
from PIL import Image

rng = np.random.default_rng(seed=6534)
image = np.array(Image.open('02_numpy/img.jpg'))
# rng.random(size=(200, 300, 3))

x_values = np.arange(0, image.shape[1])
y_values = np.arange(0, image.shape[0])

all_x, all_y = np.meshgrid(x_values, y_values)
mask = np.sqrt((all_x - (image.shape[1] / 2))**2 + (all_y - (image.shape[0] / 2))**2).reshape(image.shape[0], image.shape[1], 1)
mask = mask / mask.max()

image = (image / 255) - mask / 3
image = np.where(image < 0, 0, image)

right_img = Image.fromarray((image * 255).astype(np.uint8))
right_img.save('new_img.jpg')

# plt.imshow(image)
# plt.show()


# Final Project 3 (Recompiled): Climate Data Analysis
# Scenario: You have obtained a dataset representing 40 years of monthly temperature anomalies (deviations from the average). The data is in a single NumPy array where rows represent years and columns represent months.

# Dataset:

# Your task begins with a single NumPy array named temperature_anomalies of shape (40, 12). You will need to generate this data yourself. It should contain floating-point values representing temperature deviations, ideally with a slight warming trend over the years to make the analysis interesting.
# Your Task:

# Analyze this data to find compelling evidence of climate trends. Your final output should be a summary that answers these questions:
# Which of the 40 years was the warmest on record?
# Which 3 years were the coldest on record?
# Which calendar month (e.g., July, August) shows the strongest warming trend over the 40-year period?


temp_anomalies = rng.standard_normal(size=(40, 12)) + np.linspace(0, 1, 40*12).reshape(40, 12)
y_averages = np.average(temp_anomalies, axis=1)
warmest_y = y_averages.argmax()
coldest_3_y = np.argsort(y_averages)[:3]

temp_ano_1_mavg = np.average(temp_anomalies[:20], axis= 0)
temp_ano_2_mavg = np.average(temp_anomalies[20:], axis= 0)
biggest_month_of_change = (temp_ano_2_mavg - temp_ano_1_mavg).argmax() + 1



# Final Project 4 (Recompiled): E-commerce Transaction Analysis
# Scenario:
# You are given the raw transaction data from a small online shop. The data consists of three separate 1D arrays: one listing the customer_id for each purchase, another listing the product_id of the item purchased, and a final lookup array containing the price for each product.

# Dataset:

# Your task begins with three 1D NumPy arrays you create:
# transactions_customers: A long array of customer IDs (e.g., 1000 entries).
# transactions_products: A long array of the same length, with the product ID for each transaction.
# product_prices: A lookup array where the index corresponds to the product ID.
# Your Task:

# Process this raw transaction data to identify the shop's single most valuable customer. "Most valuable" is defined as the customer who has spent the most money in total. Present the ID of the top customer and their total spending.


cust_ids = rng.integers(0, 643, size=1123)
prod_ids = rng.integers(0, 54, size=1123)
prod_prices = rng.integers(12, 45, size=55)
prods = prod_prices[prod_ids]

cust_spendings = np.bincount(cust_ids, prods)
# print(cust_spendings.argmax(), 'spent', cust_spendings.max())
# print('transactions:', prods[cust_ids == 638])



# Final Project 5 (Recompiled): Factory Optimization Problem
# Scenario:
# Your factory produces two advanced components: "Component A" and "Component B". The production process is constrained by the availability of two rare raw materials: "Adamantium" and "Vibranium".

# To produce one unit of Component A, you need 3 grams of Adamantium and 5 grams of Vibranium.
# To produce one unit of Component B, you need 4 grams of Adamantium and 2 grams of Vibranium.
# For today's production run, you have exactly 145 grams of Adamantium and 130 grams of Vibranium in stock.
# Your Mission:

# Determine the exact integer number of Component A and Component B you must produce to use up your entire stock of both materials perfectly, with no leftovers.


adam = 145
vibr = 130
matrix = np.array([[3, 4],
                   [5, 2]])
b = [145, 130]
result = np.linalg.inv(matrix).dot(b)



