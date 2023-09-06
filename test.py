import csv

def compare_csv_files(file1_path, file2_path, tolerance=0.000001):
    # Read data from the first CSV file
    data1 = []
    with open(file1_path, 'r') as file1:
        csv_reader = csv.reader(file1)
        for row in csv_reader:
            row_data = []
            for value in row:
                if value.strip():  # Check if the value is not empty
                    row_data.append(float(value))
            data1.append(row_data)

    # Read data from the second CSV file
    data2 = []
    with open(file2_path, 'r') as file2:
        csv_reader = csv.reader(file2)
        for row in csv_reader:
            row_data = []
            for value in row:
                if value.strip():  # Check if the value is not empty
                    row_data.append(float(value))
            data2.append(row_data)

    # Check for similarity
    if len(data1) != len(data2) or len(data1[0]) != len(data2[0]):
        return False  # Files have different dimensions

    for i in range(len(data1)):
        for j in range(len(data1[0])):
            if abs(data1[i][j] - data2[i][j]) > tolerance:
                return False  # Values at (i, j) position are not similar

    return True  # Files are similar within the given tolerance

# Example usage:
file1_path = 'D:\Gimhan Sandeeptha\Gimhan\Semester 05\Deep Neural Networks\Assignment_1 Back Propagation\Task_1\\a\\true-dw.csv'
file2_path = 'derivative_weights2.csv'
if compare_csv_files(file1_path, file2_path):
    print("The CSV files are similar.")
else:
    print("The CSV files are not similar.")
