import joblib
import pandas as pd

# Load the model from the .pkl file
model = joblib.load('model.pkl')

# Assuming 'custom_data' is your custom data in the same format as your training data
# Prepare the custom data
# Ensure it's in the same format and has undergone the same preprocessing steps as your training data
# custom_data = pd.read_csv('toBePridicted.csv')
# Feature selection
# X= data[['Description_Length', 'Priority', 'Technical_Complexity']]
# Make predictions on the custom data
data = {
    'Description_Length': [400],
    'Priority': [5],
    'Technical_Complexity': [5]
}
custom_data = pd.DataFrame(data)
print("predictions",custom_data)

predictions = model.predict(custom_data)

# Print or use the predictions as needed
print("predictions",predictions)
# def fibonacci(n):
#     """Generate Fibonacci sequence up to n."""
#     fib_sequence = [0, 1]
#     while fib_sequence[-1] <= n:
#         fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
#     return fib_sequence[:-1]

# def nearest_fibonacci(decimal_number):
#     """Find the nearest Fibonacci number to a given decimal number."""
#     fib_sequence = fibonacci(decimal_number)
#     nearest_lower = max([fib for fib in fib_sequence if fib <= decimal_number], default=float('-inf'))
#     nearest_higher = min([fib for fib in fib_sequence if fib >= decimal_number], default=float('inf'))
#     return nearest_lower if decimal_number - nearest_lower < nearest_higher - decimal_number else nearest_higher

# # Example usage:
# decimal_number = 20.5
# nearest_fib = nearest_fibonacci(decimal_number)
# print("The nearest Fibonacci number to", decimal_number, "is", nearest_fib)
