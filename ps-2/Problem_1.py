# Importing necessary libraries
import numpy as np

# Function to generate the floating point representation of NumPy's float32-bit
def get_bits(number):
    """For a NumPy quantity, return bit representation
    
    Inputs:
    ------
    number : NumPy value
        value to convert into list of bits
        
    Returns:
    -------
    bits : list
       list of 0 and 1 values, highest to lowest significance
    """
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

# Determining the mantissa and exponent of representation in NumPy
value = np.float32(100.98763)
bitlist = get_bits(value)
exponent = bitlist[1:9]
mantissa = bitlist[9:32]

print("Exponent of", value,":", exponent)
print("Mantissa of", value,":", mantissa)

# Calculating difference using Python
difference = value - 100.98763

# Printing Python calculated difference
print("The difference calculated using Python is", difference)

# Printing the value determined from calculating differnce by hand (calculator)
print("The difference in values determined using a calculator is 2.75146484375e-06")