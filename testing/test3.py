def print_variable(var):
    name = [k for k, v in globals().items() if v is var][0]
    print(f"{name} = {var}")

# Example usage
my_variable = 42
print_variable(my_variable)
