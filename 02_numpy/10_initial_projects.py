import numpy as np


# Project 1: Bar Inventory Revenue Calculation



products_sold = [22, 15, 8, 31, 25]  
prices = [12.50, 11.00, 15.75, 9.00, 10.25] 

products_sold_np = np.array(products_sold)
prices_np = np.array(prices)
revenue_by_product = np.array(products_sold_np * prices_np, dtype=np.float64)
# print(revenue_by_product)


# Project 2: Environmental Sensor Data Correction

# The first 5 readings are known to be calibration noise and must be discarded
# section of the data from index 20 to 25 is corrupted and should be the average of the valid readings that come just before it.


np.random.seed(42) 
temperature_data = 20 + 5 * np.random.randn(50) 

temperature_data[20:25] = temperature_data[20:25] + 15

temperature_data_new = temperature_data[5:].copy()
temperature_data_new[15:20] = temperature_data_new[10:15].mean()
# print(temperature_data_new)


# Project 3: Customer Segmentation with Boolean Indexing

# 1) only the customer named 'Maria', and 2) customers named 'Marco' OR 'Sofia'.
# 2) customers named 'Marco' OR 'Sofia'.

customer_names = np.array(['Paolo', 'Maria', 'Marco', 'Sofia', 'Luca', 'Maria'])

# Each row corresponds to a customer, columns are 'Groceries' and 'Electronics' spending
purchase_data = np.array([
    [150, 80],
    [200, 45],
    [50, 120],
    [300, 200],
    [80, 20],
    [180, 90]
])

only_maria = purchase_data[customer_names == 'Maria']
only_marco_sofia = purchase_data[(customer_names == 'Marco') | (customer_names == 'Sofia')]
# print(only_maria)
# print(only_marco_sofia)


# Project 4: Simple Image Normalization and Cropping

# A simplified 10x10 grayscale image (0=black, 255=white)
image_data = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)


# 1) Normalize the pixel values from their original 0-255 integer range to a 0.0-1.0 float range. 
# 2) Crop the image to focus on the central region.

image_data_normalized = image_data / 255
image_data_normalized_centered = image_data_normalized[2:8][:, 2:8]

# print(image_data_normalized)
# print(image_data_normalized_centered, '\n')
# print(image_data_normalized[[2, 2, 7, 7],[2, 7, 2, 7]], '\n')


# Project 5: Processing Warehouse Orders with Fancy Indexing


# Master inventory: [product_ID, stock_quantity]. IDs are the index for simplicity.
inventory = np.zeros((20, 2), dtype=int)
inventory[:, 0] = np.arange(20) # Product IDs from 0 to 19
inventory[:, 1] = np.random.randint(0, 100, size=20) # Random stock levels

# A new, non-sequential order comes in for these product IDs
order_ids = [18, 2, 9, 15, 11]

def get_prods_with_ids(ids):
    return inventory[ids]

# print(inventory)
# print(get_prods_with_ids(order_ids))



# Project 1: Financial Portfolio Simulation

# You want to run a simple simulation of a financial portfolio containing two stocks over 30 days. The daily returns for each stock are assumed to follow a standard normal distribution. Your task is to generate these returns, calculate the cumulative value of each stock over time (assuming they both start at a value of 100), and identify the day with the highest portfolio value.

# Create a (30, 2) array of random daily returns for 30 days and 2 stocks.
# Assume a starting value of 100 for each stock.



stock_a = stock_b = 100
rng_a = np.random.default_rng(seed=123)
rng_b = np.random.default_rng(seed=1234)
returns_raw_a = (rng_a.standard_normal((30, 1)) / 100) + 1
returns_raw_b = (rng_b.standard_normal((30, 1)) / 100) + 1

stock_a_cumulated = stock_a * np.cumprod(returns_raw_a)
stock_b_cumulated = stock_b * np.cumprod(returns_raw_b)

total_portfolio_value = stock_a_cumulated + stock_b_cumulated

# print(f'''For stock A the final value is {round(stock_a_cumulated[-1], 2)},with day {round(stock_a_cumulated.argmax(), 2)} having the maximum value of {round(stock_a_cumulated[stock_a_cumulated.argmax()], 2)}''')
# print(f'''For stock B the final value is {round(stock_b_cumulated[-1], 2)},with day {round(stock_b_cumulated.argmax(), 2)} having the maximum value of {round(stock_b_cumulated[stock_b_cumulated.argmax()], 2)}''')


# Project 2: Signal Clipping and Analysis
# Scenario:
# You're given a raw audio signal represented as a 1D NumPy array. The signal is noisy, with several large "spikes" (outliers). Your task is to clean this signal by "clipping" any value whose absolute magnitude is greater than a certain threshold (e.g., 3.0). Any value above 3.0 should be set to 3.0, and any value below -3.0 should be set to -3.0. Finally, report the percentage of the data points that were clipped.




rng = np.random.default_rng(seed=123)
# Generate a signal with some outliers
noisy_signal = rng.standard_normal(200) * 2
noisy_signal[25:30] = 10 # Add some positive spikes
noisy_signal[150:155] = -8 # Add some negative spikes


mask_3 = noisy_signal > 3
mask_neg_3 = noisy_signal < -3

noisy_signal[mask_3] = 3
noisy_signal[mask_neg_3] = -3

# print(min(noisy_signal))
# print(f"Clipped: {(sum(mask_3) + sum(mask_neg_3)) / (len(noisy_signal) / 100)}%")

# Here i've done it in my way, as it seems better. Two masks are needed, cos we can't use just .abs, as we don't change the signal to the one value, but to both 3 and -3 instead. And i didn't find .where to be useful here neither, as i need those masks to find the percentage. with .when i don't get the array and i can't calculate the number of substitutions (sure thing i can't just calculate 3's and -3's either, as in the signal there could be others 3's that were not touched by clipper)




# Project 3: Student Exam Score Standardization
# Scenario:
# You have a matrix of exam scores where each row represents a student and each column represents a different exam. To compare student performance fairly, you need to standardize the scores. This involves two steps: 1) Calculate the average and standard deviation for each exam. 2) For each score, calculate its "z-score" using the formula: z = (score - exam_mean) / exam_std_dev. Finally, sort each student's z-scores to see their performance from worst to best.



rng = np.random.default_rng(seed=42)
# 5 students, 4 exams
student_scores = rng.integers(50, 101, size=(5, 4))

exam_means = student_scores.mean(axis=0)
exam_std_dev = student_scores.std(axis = 0)

z_scores = (student_scores - exam_means) / exam_std_dev
z_scores_filtered = np.sort(z_scores, axis = 1)
# print(z_scores_filtered)






# Project 4: 2D Random Walk Simulation & File I/O
# Scenario:
# This is a 2D version of the random walk from the book. Simulate a particle moving on a 2D grid for 1,000 steps. The particle starts at (0, 0). At each step, it can move one unit up, down, left, or right with equal probability. Track the particle's (x, y) position over time and save the final trajectory (the list of all (x, y) coordinates) to a compressed NumPy archive file.

# Dataset:

# You will generate this yourself. The main inputs are the number of steps (1000).


part_coord = (0, 0)
rng = np.random.default_rng(seed=101)
moves_nums = rng.choice([(0, 1),(1, 0),(0, -1),(-1, 0)], size=1000)
# print(moves_nums.sum(axis=0))



# Project 5: Product Similarity with Linear Algebra
# Scenario:
# You work for an e-commerce site. You represent several products as vectors where each element corresponds to a feature score (e.g., price, user rating, popularity). Given a target product, you need to find the product that is most "similar" to it. A simple way to measure similarity is the dot product. A larger dot product between two (normalized) vectors implies greater similarity. Your task is to calculate the dot product of a target product vector with all other product vectors.


import numpy as np

# 5 products, 3 features (e.g., normalized price, rating, popularity)
product_features = np.array([
    [0.9, 0.8, 0.2],
    [0.2, 0.3, 0.9],
    [0.85, 0.9, 0.3],
    [0.3, 0.2, 0.8],
    [0.5, 0.5, 0.5]
])

# Let's say our target is the first product (index 0)
target_product_vector = product_features[0]

def find_most_similar(target):
    dots = np.dot(product_features, target)
    number = dots.argmax() + 1
    print(dots, f'Product numer {number}')

# find_most_similar(target_product_vector)

print(5)


