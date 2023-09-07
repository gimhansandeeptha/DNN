import random

# Two lists with the same elements
list1 = [1, 2, 3, 4, 5]
list2 = ['A', 'B', 'C', 'D', 'E']

# Combine the lists into a single list of tuples
combined_lists = list(zip(list1, list2))

# Shuffle the combined list
random.shuffle(combined_lists)

# Unpack the shuffled elements back into separate lists
shuffled_list1, shuffled_list2 = zip(*combined_lists)

# Print the shuffled lists
print(shuffled_list1)
print(shuffled_list2)
