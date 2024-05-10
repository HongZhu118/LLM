import json
import requests
import numpy as np
import re
N = 5
D = 10
array1 = np.random.rand(N, D)
array2 = np.random.rand(N, D)

template = {
  "L": [],
  "K": [],}

prompt = f"""
        I have two existing {N} by {D} dimensional numpy array P={array1} and O={array2}.\
        Please generate two numpy array L and K with the same size of P that is totally different from O and P but can be motivated from them.\
        The numpy array L and K have elements between 0 and 1 numpy array L and K
        Respond using JSON.\nUse the following template: {json.dumps(template)}.
        """

data = {
    "prompt": prompt,
    "model": 'llama3',
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.0, "top_p": 0.99, "top_k": 100},
}

while True:
    try:
        offspring = []
        response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
        json_data = json.loads(response.text)
        r = json.dumps(json.loads(json_data["response"]), indent=2)
        float_pattern = r'\b\d+\.\d+\b'
        L = re.findall(float_pattern, r)[:N * D]
        L_values = [float(match) for match in L]
        K = re.findall(float_pattern, r)[-N * D:]
        K_values = [float(match) for match in K]
        if len(L_values) == len(K_values) == N * D:
            for i in range(N):
                offspring.append(L_values[i * 10:i * 10 + 10])
                offspring.append(K_values[i * 10:i * 10 + 10])
            break
    except:
        print("Please")
        continue
print(offspring)
print(type(L_values))
print(json.dumps(json.loads(json_data["response"]), indent=2))
