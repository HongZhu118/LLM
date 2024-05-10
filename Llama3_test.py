import re
import ollama
import numpy as np


def test_llama3(D,pop1,pop2):
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': f"""
                        I have two existing 1 by {D} dimensional numpy array P={pop1} and O={pop2}.\
                        Please return two numpy array L and K with the same size of P that is totally different from O and P but can be motivated from them.\
                        Please use the format:
                        L=<L>
                        K=<K>
                        Do not give additional explanations.If you return code, give the results of your code run and output a specific numpy array L and K
                        """
        },
    ])
    r = response['message']['content']
    float_pattern = r'\b\d+\.\d+\b'
    text = re.findall(float_pattern, r)[-80:]
    # float_array1 = np.array(text[:10], dtype=np.float64)
    # float_array2 = np.array(text[10:20], dtype=np.float64)
    float_values = [float(match) for match in text]
    return float_values


D = 10
array1 = np.random.rand(4, 10)
array2 = np.random.rand(4, 10)
result = test_llama3(D,array1, array2)
# empty_array = np.empty((2, 10))
# empty_array[0] = result0
# empty_array[1] = result1
print(result)

