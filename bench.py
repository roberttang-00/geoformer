import pickle
import numpy as np
import time
import msgpack
import json
import rapidjson
import orjson
import pyarrow as pa

def benchmark(serialize_func, deserialize_func, data, num_iterations=1000):
    # Serialization
    start_time = time.time()
    for _ in range(num_iterations):
        serialized = serialize_func(data)
    serialize_time = (time.time() - start_time) / num_iterations

    # Deserialization
    start_time = time.time()
    for _ in range(num_iterations):
        deserialized = deserialize_func(serialized)
    deserialize_time = (time.time() - start_time) / num_iterations

    return serialize_time, deserialize_time, len(serialized)

# Sample data
data = {
    'array': np.random.randn(3, 224, 224),
    'lat': np.random.random() * 180.0,
    'lon': np.random.random() * 180.0,
    'int': np.random.randint(0, 11399)
}

# Pickle
pickle_serialize = pickle.dumps
pickle_deserialize = pickle.loads

# JSON (standard library)
json_serialize = lambda x: json.dumps({"array": x['array'].tolist(), **{k: v for k, v in x.items() if k != 'array'}}).encode()
json_deserialize = lambda x: {**json.loads(x.decode()), "array": np.array(json.loads(x.decode())['array'])}

# RapidJSON
rapidjson_serialize = lambda x: rapidjson.dumps({"array": x['array'].tolist(), **{k: v for k, v in x.items() if k != 'array'}}).encode()
rapidjson_deserialize = lambda x: {**rapidjson.loads(x.decode()), "array": np.array(rapidjson.loads(x.decode())['array'])}

# orjson
orjson_serialize = lambda x: orjson.dumps({"array": x['array'].tolist(), **{k: v for k, v in x.items() if k != 'array'}})
orjson_deserialize = lambda x: {**orjson.loads(x), "array": np.array(orjson.loads(x)['array'])}

# Apache Arrow
arrow_serialize = lambda x: pa.serialize(x).to_buffer()
arrow_deserialize = lambda x: pa.deserialize(x)

# Run benchmarks
methods = [
    ("Pickle", pickle_serialize, pickle_deserialize),
    ("RapidJSON", rapidjson_serialize, rapidjson_deserialize),
    ("orjson", orjson_serialize, orjson_deserialize),
    ("Apache Arrow", arrow_serialize, arrow_deserialize)
]

for name, serialize, deserialize in methods:
    serialize_time, deserialize_time, size = benchmark(serialize, deserialize, data)
    print(f"{name}:")
    print(f"  Serialize time: {serialize_time:.6f} seconds")
    print(f"  Deserialize time: {deserialize_time:.6f} seconds")
    print(f"  Serialized size: {size} bytes")
    print()

# Note: You'll need to install the required libraries:
# pip install msgpack rapidjson orjson pyarrow
