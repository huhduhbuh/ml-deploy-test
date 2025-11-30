import requests
import numpy as np
import torch

# Server URL
SERVER_URL = 'http://localhost:5000'

# Check if server is running
print("Checking server health...")
try:
    response = requests.get(f'{SERVER_URL}/health')
    print(f"Server status: {response.json()}\n")
except Exception as e:
    print(f"Error: Server not running. Make sure deploy.py is running. {e}\n")
    exit()

# Example 1: Send random sensor data (simulating a non-fall)
print("=" * 50)
print("Example 1: Random data (likely no fall)")
print("=" * 50)

sample_data_1 = np.random.rand(6, 200).tolist()

try:
    response = requests.post(
        f'{SERVER_URL}/predict',
        json={'input': sample_data_1}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']:.4f}")
    print(f"Label: {result['label']}\n")
except Exception as e:
    print(f"Error: {e}\n")

# Example 2: Send another sample
print("=" * 50)
print("Example 2: Different sensor data")
print("=" * 50)

sample_data_2 = np.random.rand(6, 200).tolist()

try:
    response = requests.post(
        f'{SERVER_URL}/predict',
        json={'input': sample_data_2}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']:.4f}")
    print(f"Label: {result['label']}\n")
except Exception as e:
    print(f"Error: {e}\n")

# Example 3: Batch predictions
print("=" * 50)
print("Example 3: Batch of 5 predictions")
print("=" * 50)

for i in range(5):
    sample_data = np.random.rand(6, 200).tolist()
    
    try:
        response = requests.post(
            f'{SERVER_URL}/predict',
            json={'input': sample_data}
        )
        result = response.json()
        print(f"Sample {i+1}: Prediction={result['prediction']:.4f}, Label={result['label']}")
    except Exception as e:
        print(f"Error on sample {i+1}: {e}")


# Example 4: True Fall scenario
print("=" * 50)
print("Example 4: True Fall scenario")
print("=" * 50)

# load the saved tensor
loaded = torch.load('fall_input.pt')  # could be a tensor of shape (1,6,200) or (6,200)

# Ensure we have a torch tensor
if isinstance(loaded, list):
    # if somehow saved as list
    arr = np.array(loaded)
else:
    arr = loaded.numpy() if isinstance(loaded, torch.Tensor) else np.array(loaded)

# Remove a leading batch dimension if present
if arr.ndim == 3 and arr.shape[0] == 1:
    arr = arr.squeeze(0)  # becomes (6, 200)

if arr.shape != (6, 200):
    raise ValueError(f"Unexpected input shape {arr.shape}, expected (6, 200)")

sample_data_4 = arr.tolist()  # convert to JSON-serializable list

# optional: print shape for debugging
print("Prepared sample_data_4 shape:", np.array(sample_data_4).shape)

try:
    response = requests.post(
        f'{SERVER_URL}/predict',
        json={'input': sample_data_4}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']:.4f}")
    print(f"Label: {result['label']}\n")
except Exception as e:
    print(f"Error: {e}\n")




print("\nDone!")