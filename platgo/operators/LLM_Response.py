import re
import ollama
import numpy as np
import json
def GA_binary(D,pop1,pop2):
    template = {
      "L": [],
      "K": [],}
    while True:
        try:
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': f"""
                                I have two existing 1 by {D} dimensional numpy array P={pop1} and O={pop2}.\
                                Please return two numpy array L and K with the same size of P that is totally different from O and P but can be motivated from them.\
                                Respond using JSON.\nUse the following template: {json.dumps(template)}.
                                Do not give additional explanations.If you return code, give the results of your code run and output a specific list
                                """
                },
            ])
            r = response['message']['content']
            float_pattern = r'\b\d+\.\d+\b'
            text = re.findall(float_pattern, r)[-2*D:]
            if len(text) == 2*D:
                float_array1 = np.array(text[:D], dtype=np.float32)
                float_array2 = np.array(text[D:2*D], dtype=np.float32)
                break
        except:
            print("Error,Regenerate")
            continue
    return float_array1,float_array2

def GAhalf_binary(D,pop1,pop2):
    while True:
        try:
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': f"""
                                I have two existing 1 by {D} dimensional numpy array P={pop1} and O={pop2}.\
                                Please return one numpy array L with the same size of P that is totally different from O and P but can be motivated from them.\
                                Please use the format:
                                L=<L>
                                Do not give additional explanations.If you return code, give the results of your code run and output a specific list
                                """
                },
            ])
            r = response['message']['content']
            float_pattern = r'\b\d+\.\d+\b'
            text = re.findall(float_pattern, r)[-D:]
            if len(text) == D:
                float_array1 = np.array(text[:D], dtype=np.float32)
                break
        except:
            print("Error,Regenerate")
            continue
    return float_array1