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





# Group 2 Mini-Tasks
# Mini-Task 1: Logarithmic Scale Generation
# Scenario: For scientific and financial plots, it's common to view data on a logarithmic scale to see rates of change more clearly. Your task is to generate a set of values and transform them to a log scale.

# Dataset Creation:

# Create a 1D NumPy array named linear_scale. It should contain 200 evenly spaced points ranging from 1 to 1000 (inclusive).
# Your Task:

# Generate the linear_scale array.
# Create a new array named log_scale by applying the natural logarithm to every element in linear_scale.


scale = np.linspace(1, 1000, 200, endpoint=True)

log_scale = np.log(scale)
# print(scale[:10])
# print(log_scale[:10])


# Mini-Task 2: Physics Calculation with Fixed Values
# Scenario: You need to calculate the hypotenuse c of several right-angled triangles using the formula c = sqrt(a² + b²). For this batch of calculations, one side, b, is always a fixed length.

# Dataset Creation:

# Create a 1D NumPy array named side_a_lengths containing the floating-point values [2.5, 7.0, 10.5, 12.0].
# Create another 1D array named side_b_lengths with the same shape as side_a_lengths, but where every single element is the number 6.0.
# Your Task:

# Generate the side_a_lengths and side_b_lengths arrays.
# Using a single vectorized formula, calculate a new array hypotenuse_lengths containing the result for each triangle.


a_lengths = np.array([2.5, 7.0, 10.5, 12.0])
b_lengths = np.full(a_lengths.shape, 6.0)

c_lengths = np.sqrt(a_lengths**2 + b_lengths**2)
# print(c_lengths)



# Mini-Task 3: Working with Matrix Diagonals
# Scenario: In linear algebra, you often need to work with identity matrices or extract the diagonal elements of a matrix.

# Dataset Creation:

# Create a 2D NumPy array (a matrix) named data_matrix with a shape of (5, 5). Fill it with random integers between 10 and 50.
# Your Task:

# Generate the data_matrix.
# Create a new 1D array named main_diagonal by extracting only the diagonal elements from data_matrix.
# Create a (5, 5) identity matrix named identity_matrix (a matrix with 1.0 on the diagonal and 0.0 everywhere else).



data_matrix = rng.integers(10, 51, size=(5,5))
main_diagonal = np.diag(data_matrix)
# print(data_matrix, main_diagonal)
identity_matrix = np.diag(np.ones(5))
# print(identity_matrix)
another_identity = np.eye(5)
# print(another_identity)



# Mini-Task 4: Comparing Sensor Signals
# Scenario: You are analyzing data from two different temperature sensors placed in the same environment. To create a robust "maximum reading" signal, at each point in time, you want to take the value with the largest absolute magnitude from either sensor.

# Dataset Creation:

# Using the modern np.random.default_rng() generator, create two 1D NumPy arrays, sensor_1_data and sensor_2_data. Both should have 15 elements, drawn from a standard normal distribution (bell curve).
# Your Task:

# Generate the two sensor data arrays.
# Create a new array max_magnitude_signal where each element is the element-wise maximum of the absolute values of the two sensor arrays.

sensor_1 = rng.standard_normal(15) * 3 + 20
sensor_2 = rng.standard_normal(15) * 3 + 20
max_magnitude_signal = np.maximum(np.abs(sensor_1), np.abs(sensor_2))
# print(max_magnitude_signal)




# Unifying Task: Group 2
# Scenario:
# You are simulating a noisy signal from an electronic instrument and then processing it. The base signal is a sine wave. You need to generate this signal, add some random noise, and then apply conditional logic to process the result.

# Your Task:

# Generate the Data (Dataset Creation):

# Create a 1D array named time that represents your time axis. It should have 500 points evenly spaced from 0 to 10 * pi. (You can get pi from np.pi).
# Create the ideal signal, clean_signal, by calculating the sine of every element in the time array.
# Create a noise array of the same shape as time, with values drawn from a standard normal distribution, scaled by a factor of 0.2.
# Create the final noisy_signal by adding clean_signal and noise.
# Process the Signal:

# Create a new array called processed_signal based on noisy_signal.
# The logic should be: if a value in noisy_signal is positive, the corresponding value in processed_signal should be the natural logarithm of that value. If the value in noisy_signal is zero or negative, the corresponding value in processed_signal should be 0.



time = np.linspace(0, 10 * np.pi, 500)
ideal_signal = np.sin(time)
noise = rng.standard_normal(time.size) * 0.2
final_noisy_signal = ideal_signal + noise

# processed_signal = np.select((final_noisy_signal > 0, final_noisy_signal <= 0), (np.log(final_noisy_signal), 0))
processed_signal = np.where(np.abs(final_noisy_signal) > 0, np.log(np.abs(final_noisy_signal)), 0)
# print(processed_signal[:100])


# Group 3 Mini-Tasks
# Mini-Task 1: Simulating Dice Rolls
# Scenario: You want to simulate 100 rolls of a pair of standard six-sided dice to test the probabilities.

# Dataset Creation:

# Create two separate 1D NumPy arrays, dice1_rolls and dice2_rolls.
# Each array should contain 100 random integers representing the outcome of a single die roll (values 1 through 6).
# Your Task:

# Generate the dice1_rolls and dice2_rolls arrays.
# Create a new array roll_sums by adding the two arrays together element-wise.
# Using a boolean mask, count how many times the sum of the dice was exactly 7.


rng_dice = np.random.default_rng(seed = 12)
dice1 = rng_dice.integers(1, 7, 100)
dice2 = rng_dice.integers(1, 7, 100)
roll_sums = dice1 + dice2
sum_7 = (roll_sums == 7).sum()



# Mini-Task 2: Generating and Scaling Coordinates
# Scenario: You need to generate random spawn points for 50 objects within a square game map that is 1500x1500 units in size.

# Dataset Creation:

# Create a single 2D NumPy array named normalized_coords with a shape of (50, 2).
# The values should be random floating-point numbers uniformly distributed between 0.0 and 1.0.
# Your Task:

# Generate the normalized_coords array.
# Create a new array game_coords by scaling the normalized_coords to fit the [0, 1500) range of the game map. This should be done with a single multiplication operation.


rng = np.random.default_rng(seed=342)
norm_coord = rng.random((50, 2))
game_coord = norm_coord * 1500
# print(game_coord[:10])




# Mini-Task 3: Simulating a Population's Height
# Scenario: You're creating a synthetic dataset for a health study. You need to simulate the heights of 1,000 adult males, which typically follow a normal distribution (a bell curve).

# Dataset Creation:

# Create a 1D NumPy array named heights_cm containing 1,000 floating-point values.
# These values should be drawn from a normal distribution with a mean of 178 cm and a standard deviation of 8 cm. (Hint: Start with rng.standard_normal which has a mean of 0 and std dev of 1, then scale and shift the result).
# Your Task:

# Generate the heights_cm array.
# Calculate and print the actual mean and standard deviation of your generated sample to verify they are close to 178 and 8.
# Count how many individuals in your simulation are "very tall" (over 202 cm, which is 3 standard deviations above the mean).


rng = np.random.default_rng(seed=32)
heights_cm = rng.standard_normal(1000) * 8 + 178
n_very_tall = (heights_cm > 202).sum()
# print(n_very_tall)



# Mini-Task 4: Drawing Lottery Winners
# Scenario: You're running a lottery. From a pool of 200 tickets, you need to draw 10 unique winning tickets. A ticket cannot win more than once.

# Dataset Creation:

# Create a 1D NumPy array named ticket_pool containing the integer ticket numbers from 1001 to 1200, inclusive.
# Your Task:

# Generate the ticket_pool array.
# Create a new array winning_tickets by selecting 10 tickets from the pool. Ensure your selection is done without replacement so that each winning ticket is unique.


ticket_pool = np.arange(1001, 1201)
rng = np.random.default_rng(seed=654)
winning_tickets = np.full(10, rng.choice(ticket_pool, size=10, replace=False))
# print(winning_tickets)





# Mini-Task 5: Randomizing Experimental Groups
# Scenario: You are preparing a clinical trial. You have 30 participants, and you need to randomly assign them to two groups (Group A and Group B) of 15 participants each. The best way to do this is to shuffle the entire list of participants and then assign the first 15 to one group and the last 15 to the other.

# Dataset Creation:

# Create a 1D NumPy array named participant_ids containing the integers from 0 to 29.
# Your Task:

# Create the participant_ids array and print it to see the original order.
# Shuffle the participant_ids array in-place.
# Print the array again to show that it has been randomly reordered.


rng = np.random.default_rng(6542)
participants_ids = np.arange(0, 30)
rng.shuffle(participants_ids)
group_1 = participants_ids[:15]
group_2 = participants_ids[-15:]
# print(group_1.size, group_2.size)



# Unifying Task: Group 3
# Scenario:
# You are procedurally generating a set of 50 enemy spaceships for a video game. Each ship needs a unique ID, a faction, a shield strength, and a weapon power level.

# Your Task:

# Generate Base Data:

# Create a default_rng instance.
# Create a 1D array ship_ids containing integers from 0 to 49.
# Define a Python list of possible factions: ['Federation', 'Klingon', 'Romulan'].
# Generate Attributes using Random Methods:

# Use rng.choice to create a 1D array ship_factions by randomly assigning one of the three factions to each of the 50 ships.
# Use rng.integers to create a 1D array shield_strength with values for each ship between 500 and 2000 (inclusive).
# Use rng.random to create a 1D array weapon_power with values for each ship between 10.0 and 50.0. (Hint: rng.random gives [0, 1), so you will need to scale and shift the values).
# Create a Final, Shuffled Dataset:

# You now have four parallel 1D arrays. To ensure the final dataset is randomized, create an shuffled_indices array containing numbers from 0 to 49, and then use rng.shuffle on it.
# Use this shuffled_indices array and Fancy Indexing to create new, shuffled versions of all four of your attribute arrays (shuffled_ids, shuffled_factions, etc.). This ensures that the link between an ID and its attributes is maintained but the final order is random.


# a unique ID, a faction, a shield strength, and a weapon power level.

rng = np.random.default_rng(seed=781)

possible_factions = np.array(['Federation', 'Klingon', 'Romulan'])
ids = np.arange(0, 50)
factions = rng.choice(possible_factions, 50)
shield_strengths = rng.integers(500, 2001, size=50)
weapon_power = rng.random(50) * 40 + 10

ships = np.array([ids, factions, shield_strengths, weapon_power]).T
# print(ships[5])
rng.shuffle(ships)
# print(ships[ships[:, 0] == '5'], ships)




# Group 4 Mini-Tasks
# Mini-Task 1: Positional Statistics (argmin & argmax)
# Scenario: You have the daily maximum temperature readings for your city, Bergamo, for the 31 days of July. You need to find out on which days the lowest and highest temperatures of the month occurred.

# Dataset Creation:

# Create a 1D NumPy array named july_temps. It should have 31 floating-point values, representing temperatures from a normal distribution with a mean of 28.0 and a standard deviation of 4.0.
# Your Task:

# Generate the july_temps array.
# Use np.argmin() to find the index (day of the month, 0-30) of the coldest day.
# Use np.argmax() to find the index of the hottest day.



july_temps = rng.standard_normal(31) * 4 + 28
coldest = july_temps.argmin()
hottest = july_temps.argmax()
# print(july_temps[coldest], july_temps[hottest])



# Mini-Task 2: Sorting Along an Axis (np.sort)
# Scenario: You are analyzing the performance of 4 race car drivers over 5 laps. You want to see each driver's individual lap times sorted from fastest to slowest to quickly identify their best and worst laps.

# Dataset Creation:

# Create a 2D NumPy array named lap_times with a shape of (4, 5) (4 drivers, 5 laps).
# The values should be random floats representing lap times in seconds, for example, between 88.5 and 91.5 seconds.
# Your Task:

# Generate the lap_times array.
# Create a new array sorted_lap_times where each row (each driver's laps) is sorted independently from fastest to slowest.


lap_times = rng.standard_normal(size=(4, 5)) * 2 + 90
sorted_lap_times = np.sort(lap_times, axis=1)
# even though it's not practical, as we don't know at the end which lap was what, i was better to use tuples with lap nums, but anyway.
# probably in pandas it's built-in in some way, but in numpy it's better to work with tuple in that example




# Mini-Task 3: Indirect Sorting (np.argsort)
# Scenario: You are the manager of your bar. At the end of the day, you have the sales counts for several products. You need to create a ranked list of the product names from best-selling to worst-selling.

# Dataset Creation:

# Create a 1D NumPy array named product_names containing the string values: ['Aperol Spritz', 'Espresso', 'Cornetto', 'Prosecco', 'Gin Tonic'].
# Create a parallel 1D array named product_sales with a corresponding integer sales count for each product.
# Your Task:

# Generate the two arrays.
# Use np.argsort() on the product_sales array. This will give you the indices that would sort the sales array.
# Use these sorted indices (with fancy indexing) to reorder the product_names array to get your final ranked list.



prod_names = np.array(['Aperol Spritz', 'Espresso', 'Cornetto', 'Prosecco', 'Gin Tonic'])
prod_sales = rng.integers(2, 8, size=prod_names.size)
prod_names = prod_names[np.argsort(prod_sales)]
# print(prod_names, prod_sales)



# Mini-Task 4: Finding Unique Visitors (np.unique)
# Scenario: A local news website logs the country of origin for every visitor. You have a long log from the last hour and you need to report how many unique countries the visitors came from.

# Dataset Creation:

# Create a 1D NumPy array named visitor_log with 200 elements.
# To simulate traffic, populate this array by randomly choosing from a list of 8 country codes (e.g., ['IT', 'DE', 'US', 'GB', 'FR', 'UA', 'ES', 'CH']), allowing for many duplicates.
# Your Task:

# Generate the visitor_log array.
# Use np.unique() to get a new array containing only the unique country codes.
# Find the total number of unique countries by checking the size of this new array.



country_examples = ['IT', 'DE', 'US', 'GB', 'FR', 'UA', 'ES', 'CH']
visitor_log = rng.choice(country_examples, 200)
unique_num = np.unique(visitor_log).size
# print(unique_num)



# Mini-Tast 5: Checking Set Membership (np.in1d)
# Scenario: Your company has a list of all employee IDs. A separate list contains the IDs of employees who have registered for a special "Advanced Python" workshop. You need to quickly check which employees from the Data Science department are registered.

# Dataset Creation:

# Create a 1D array named data_science_team_ids with the integer IDs of 5 employees.
# Create a 1D array named workshop_registrants with 15 random integer IDs, some of which should overlap with the data science team's IDs.
# Your Task:

# Generate the two arrays.
# Use np.in1d() to get a boolean array that answers the question: "Is each member of the data science team registered for the workshop?"


my_employees_ids = np.arange(100)
workshop_ids = rng.choice(np.arange(0, 1000), size=999, replace=False)
my_emp_present = my_employees_ids[np.isin(my_employees_ids, workshop_ids)]
# print('yes' if my_emp_present.size == my_employees_ids.size else 'no')



# Unifying Task: Group 4
# Scenario:
# You are given a raw dataset from a multiplayer online game. It's a 2D array where each row represents a player and the columns are [player_id, score, time_played_in_hours]. You need to perform a series of analyses to identify top players and understand player behavior.

# Dataset Creation:

# Create a 2D array named player_data for 100 players.
# The first column (player_id) should contain unique IDs from 1000 to 1099.
# The second column (score) should contain random integers between 0 and 50000.
# The third column (time_played) should contain random integers between 1 and 500.
# Analysis:

# Top Player: Find the player_id of the single player with the absolute highest score.
# Leaderboard: Create a new 1D array named leaderboard_ids that contains the player_ids of all 100 players, ranked from highest score to lowest score.
# Elite Group Check: You are given a separate list of veteran_player_ids. Determine which of these veteran players appear in the top 20 of your new leaderboard.
# Efficiency Analysis: For each unique value of time_played, calculate the average score of all players who played for that exact amount of time. This will show if more time played correlates with higher scores. (This is a challenging step that requires combining several techniques).

rng = np.random.default_rng(seed=64323)
ids = rng.choice(np.arange(1000, 1100), 100, replace = False)
score = rng.integers(0, 50_001, 100)
time_played = rng.integers(1, 100, 100)

def take_score(arr):
    return arr[:, 1]

players = np.array([ids, score, time_played]).T
top_by_score = players[players[:, 1] == players[:, 1].max()]
indices = np.argsort(take_score(players))
leaderboard = players[indices]

veteran_player_ind = (rng.standard_normal(40) * 20).round()
veteran_player_ind = veteran_player_ind[veteran_player_ind < 0].astype('int')
veteran_players = leaderboard[veteran_player_ind]
veterans_in_top = veteran_players[np.isin(veteran_players[:, 0], leaderboard[-20:][:, 0])]

avg_for_time = np.array([np.arange(1, 101), np.zeros(100)]).T
# for item in avg_for_time:
#     item[1] =  players[players[:, 2] == item[0]][:, 1].mean()
# print(avg_for_time)






# Group 5 Tasks
# Task 1: Cocktail Production Cost Calculator
# Scenario: You run a bar and need to calculate the production cost for each of your signature cocktails based on the cost of their ingredients. This can be solved efficiently with a single matrix multiplication.

# Dataset Creation:

# Create a 2D NumPy array named recipes. It should have a shape of (3, 4), representing 3 cocktails and 4 ingredients. The values should be the amount of each ingredient in milliliters (e.g., a Negroni might be [30, 30, 30, 0]).
# Create a 1D NumPy array named ingredient_costs with 4 elements. The values should represent the cost per milliliter for each of the 4 ingredients (e.g., [0.05, 0.03, 0.03, 0.04]).
# Your Task:

# Using a single matrix multiplication operation (@ or np.dot), calculate a new 1D array named cocktail_costs that contains the final production cost for each of the 3 cocktails.



recipes = np.array([
    [30, 30, 30,  5],
    [60, 10,  5,  2],
    [45,  5, 15, 40]  
], dtype=np.int32)

ingredient_costs = rng.integers(25, 40, size=4) / 1000
prices_per_cocktail = recipes @ ingredient_costs
# print(prices_per_cocktail)



# Task 2: Solving for Unknowns with Matrix Inversion
# Scenario: You're trying to determine the individual prices of an espresso and a cornetto from two past orders.

# Order 1: 2 espressos + 1 cornetto cost €4.40
# Order 2: 1 espresso + 3 cornettos cost €5.70
# This can be written as a system of linear equations A * x = b, where the solution is x = inv(A) @ b.

# Dataset Creation:

# Create the (2, 2) matrix A representing the quantities of items in the orders.
# Create the (2,) vector b representing the total costs of the orders.
# Your Task:

# Calculate the inverse of matrix A.
# Use matrix multiplication to solve for x, the vector containing the individual prices of the espresso and the cornetto.



orders = np.array([[2, 1],
                  [1, 3]])
cost = np.array([4.40, 5.70])
prices = np.linalg.inv(orders) @ cost
# print(prices)




# Task 3: Saving & Loading Machine Learning Model Parameters
# Scenario: After training a machine learning model, its "knowledge" is stored in its parameters (weights and biases), which are just NumPy arrays. You need to save these parameters to disk so you can load them later for making predictions without having to retrain the model.

# Dataset Creation:

# Simulate the parameters for a neural network layer. Create two separate 2D NumPy arrays: str(weights_layer) with a shape of (128, 64), and biases_layer with a shape of (1, 64).
# Fill both arrays with random floating-point numbers from a standard normal distribution.
# Your Task:

# Save the weights_layer array to a file named weights.npy.
# Save the biases_layer array to a file named biases.npy.
# Imagine you've closed your script. Now, load the two arrays back from their files into new variables, loaded_weights and loaded_biases.
# Verify that the loaded arrays are identical to the original ones. (Hint: Use np.array_equal()).


weights_layer = rng.standard_normal((128, 64))
biases_layer = rng.standard_normal((1, 64))

np.save('weights', weights_layer)
np.save('biases', biases_layer)

load_weights = np.load('weights.npy')
load_biases = np.load('biases.npy')
# print(np.array_equal(load_weights, weights_layer))
# print(np.array_equal(load_biases, biases_layer))







