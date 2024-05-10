# Llama3-for-evolution

## 1. 下载开源框架ollama
用于在本地运行大型语言模型，模型权重、配置和数据捆绑到一个包中，优化了设置和配置细节，包括 GPU 使用情况，从而简化了在本地运行大型模型的过程。
[下载地址](https://ollama.com/)

## 2. 使用ollama下载模型
在命令行运行以下命令安装llama3-8B：
```
ollama run llama3
```
其他模型下载参考[官方文档](https://github.com/ollama/ollama)

## 3. 安装模型需要的依赖包
```
    pip install ollama

    pip install numpy

    pip install transformers（含huggingface_hub）

    torch   ————    特别要注意CUDA版本是否与torch版本匹配。
                    # ROCM 5.6 (Linux only)
                    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6

                    # CUDA 11.8版本
                    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

                    # CUDA 12.1版本
                    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

                    # CPU only
                    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

    pip install accelerate

    pip install bitsandbytes
```

## 4. 使用ollama提供的本地接口调用llama3示例
```
# 首先需要导入ollama
import ollama


"""
使用示例：
"model"：模型名称
"content"：提示词
"""
response = ollama.chat(
"model": "llama3",
 "messages": [
 { "role": "user", "content": "why is the sky blue?" }
 ])
```
## 5. 使用llama3代替交叉、变异
### 1.测试用例（主函数）
Llama3-for-evolution\tests\GA_test.py
### 2.测试问题
Llama3-for-evolution\platgo\problems\SOP_F20.py
### 3.模型调用
在Llama3-for-evolution\platgo\operators\LLM_Response.py中调用llama3，并修改提示词content内容:

```
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
```
### 4.替换算子
在Llama3-for-evolution\platgo\operators\OperatorGA.py中调用LLM_Response.py中提供的函数
```
    Offspring = np.empty((2*N,D))
    for i in range(N):
        off1,off2 =GA_binary(D,pop1[i],pop2[i])
        Offspring[2*i] = off1
        Offspring[2*i+1] = off2
```