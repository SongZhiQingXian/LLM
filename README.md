# LLM
This project is for my team

# 关于Meta-Llama-3-8B-Instruct模型在NotebookGallery上部署的操作文档



## 一、环境准备

### 1.阿里云价值10000￥的白嫖服务器[阿里云免费试用 - 阿里云 (aliyun.com)](https://free.aliyun.com/?searchKey=交互式建模)

1. 服务器的地址选择杭州即可，其他可选可不选的就别选（隐约记得），因为我现在进不去那个界面了。

2. 进入这里[Notebook Gallery (aliyun.com)](https://pai.console.aliyun.com/?regionId=cn-hangzhou&spm=5176.12818093_-1363046575.0.0.3be916d0wmEUPl&workspaceId=197645#/dsw-gallery-workspace?category=&pageNum=1)

3. 如图：进入创建实例，实例的配置：
   24GB显存的A10
   镜像选择DSW官方镜像`modelscope:1.14.0-pytorch2.1.2-gpu-py310-cu121-ubuntu22.04`

   ![image-20240916210921911](LLM操作文档.assets/image-20240916210921911.png)

4. 创建好后即可运行，运行成功如下图
   ![image-20240916211303059](LLM操作文档.assets/image-20240916211303059.png)

5. 点击打开即可进入类似Vscode一个Web开发界面
   ![image-20240916211851948](LLM操作文档.assets/image-20240916211851948.png)

6. 余下的操作与Vscode无异

### 2.大语言模型的微调框架-LLaMA Factory的安装

1. 在Web开发平台上新建一个.ipynb文件

2. 执行下面的代码：

   - 拉取LLMA-Factory到DSW实例中：
     !git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

   - 接着，安装LLaMA-Factory依赖环境：

     ```
     !pip uninstall -y vllm
     !pip install llamafactory[metrics]==0.7.1
     !pip install accelerate==0.30.1
     ```

   - 运行如下命令，如果显示llamafactory-cli的版本，则表示安装成功
     !llamafactory-cli version

     ```
     [2024-05-08 10:25:22,857] [INFO] [real_accelerator.py:161:get_accelerator] Setting ds_accelerator to cuda (auto detect)
     Welcome to LLaMA Factory, version 0.7.1.dev0
     ```

   如图

   ![image-20240916212725747](LLM操作文档.assets/image-20240916212725747.png)

3. LLama-factory的GitHub网站[hiyouga/LLaMA-Factory: Efficiently Fine-Tune 100+ LLMs in WebUI (ACL 2024) (github.com)](https://github.com/hiyouga/LLaMA-Factory)

## 二、下载Meta-Llama-3-8B-Instruct模型到DSW

1. 执行三条代码即可：

   ```python
   !pip install modelscope==1.12.0 #（如果下载报错可以试着加入!pip install transformers==4.37.0）
   ```

   ```python
   from modelscope.hub.snapshot_download import snapshot_download
   ```

   ```python
   snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='.', revision='master')
   ```


   ![image-20240916213714260](LLM操作文档.assets/image-20240916213714260.png)

2. 模型下载好了就在左边了
   ![image-20240916213808863](LLM操作文档.assets/image-20240916213808863.png)

## 三、模型推理

1. 数据介绍(记得下载数据集放入Web开发平台哦)

   训练集：train_data

   测试集：testA_data

   ![image-20240917113332899](LLM操作文档.assets/image-20240917113332899.png)
   内部结构类似，以testA_data举例说明

   ![image-20240917113508678](LLM操作文档.assets/image-20240917113508678.png)
   database文件夹中是.sqlite也就是数据库文件

   dev.json中是输入的问题以及对应的数据库(如下图)，db_list在这里指的是中国城市.sqlite，在interaction中问了4轮的问题。

   ![image-20240917113616226](LLM操作文档.assets/image-20240917113616226.png)

2. 打开Web开发界面导入包

   ```python
   import json
   import sqlite3
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch
   import time
   ```

   如果没有的话就!pip install 包名
   ![image-20240917113901957](LLM操作文档.assets/image-20240917113901957.png)

3. 编写2个函数分别是get_schema()以及generate_sql_query()，他们的作用分别是查询某一个数据库的结构以及使用模型生成SQL的查询指令。

   ```python
   def get_schema(db_path):#查询db_path路径指定的数据库的结构，查询到的结构将告知给模型
       schema = ""
       with sqlite3.connect(db_path) as conn:
           cursor = conn.cursor()
           cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
           tables = cursor.fetchall()
           for table in tables:
               table_name = table[0]
               cursor.execute(f"PRAGMA table_info({table_name});")
               columns = cursor.fetchall()
               schema += f"CREATE TABLE {table_name} (\n"
               schema += ",\n".join([f"    {col[1]} {col[2]} {'NOT NULL' if col[3] else 'NULL'} {col[4] if col[4] is not None else 'NULL'} {'PRIMARY KEY' if col[5] else ''}" for col in columns])
               schema += "\n);\n"
       return schema
   ```

   ```python
   def generate_sql_query(user_question, messages_history):
       # 添加用户输入到对话历史
       current_input = {"role": "user", "content": user_question}
       messages_history.append(current_input)
   
       # 准备输入数据
       inputs = tokenizer.apply_chat_template(messages_history, add_generation_prompt=True, return_tensors="pt").to(model.device)
       
       # 检查 inputs 的类型
       if isinstance(inputs, dict):
           input_ids = inputs["input_ids"]
           attention_mask = inputs["attention_mask"]
       else:
           input_ids = inputs
           attention_mask = torch.ones_like(input_ids)
   
       # 生成SQL查询
       terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
       outputs = model.generate(
           input_ids,
           attention_mask=attention_mask,
           max_new_tokens=256,
           eos_token_id=terminators,
           pad_token_id=tokenizer.eos_token_id,
           do_sample=True,
           temperature=0.6,
           top_p=0.9
       )
       response = outputs[0]
       sql_query = tokenizer.decode(response, skip_special_tokens=True)
   
       # 使用 rsplit 从右侧开始分割，确保获取最后一个 "assistant" 之后的内容
       sql_query = sql_query.rsplit("assistant\n", 1)[-1].strip()
   
       # 添加助手的回复到对话历史
       current_input = {"role": "assistant", "content": sql_query}
       messages_history.append(current_input)
   
       return sql_query
   
   ```

   

4. 测试代码，检查能否输出数据库的结构：

   ```python
   db_path = 'TestA/database/中国城市.sqlite'
   print(get_schema(db_path))
   ```

   ![image-20240917143401034](LLM操作文档.assets/image-20240917143401034.png)

5. 加载模型

   ```python
   # 加载模型和分词器
   model_name = "LLM-Research/Meta-Llama-3-8B-Instruct"#这个是你自己的模型的路径
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.bfloat16,
       device_map="auto"
   )
   ```

   

6. 遍历测试集,使用generate_sql_query()进行推理

   ```python
   with open('TestA/dev.json', 'r', encoding='utf-8') as f:
       data = json.load(f)
   
   start_time = time.time()
   all_interactions_output = []
   
   for db_entry in data[:5]:
       db_list = db_entry['db_list']
       interactions = db_entry['interaction']
       
       for db_name in db_list:
           db_path = f'TestA/database/{db_name}.sqlite'
           schema = get_schema(db_path)
           
           # 准备Prompt
           prompt = f"""You are a text to SQL query translator.You can only return SQL query\n\
           Users will ask you questions in Chinese once or twice and you will generate a SQL query based on the provided SCHEMA.\n\
           SCHEMA:\n\
           {schema}\n\
           Please generate an SQL query without semicolon at the end and using \\\"string\\\" for string literals.\n\
           For example:\n\
           Input: Retrieve the names of employees whose department is "Sales" \n\
           Output: SELECT name FROM employees WHERE department = \\\"Sales\\\" \n"""
           
           messages_history = [
               {"role": "system", "content": prompt},
           ]
           
           interaction_output = []
           
           for interaction in interactions:
               user_question = interaction['question']
               question_id = interaction['question_id']
               sql_query = generate_sql_query(user_question, messages_history)
               
               interaction_output.append({
                   "question_id": question_id,
                   "question": user_question,
                   "db_id": db_name,
                   "query": " " + sql_query
               })
               
               # print(f"Question ID: {question_id}")
               # print(f"User Question: {user_question}")
               # print(f"Generated SQL Query: {sql_query}")
               print(f"{sql_query}\n")
           
           all_interactions_output.append(interaction_output)
   
   with open('test_output2.json', 'a', encoding='utf-8') as w:
       json.dump(all_interactions_output, w, ensure_ascii=False, indent=4)
       w.write('\n')
   
   end_time = time.time()
   elapsed_time = end_time - start_time
   minutes, seconds = divmod(elapsed_time, 60)
   print(f"共耗时: {minutes:.0f} 分钟 {seconds:.2f} 秒\n")
   
   ```

   

7. 运行结果类似：
   ![image-20240917153231785](LLM操作文档.assets/image-20240917153231785.png)

   

   

## 四、LLaMA-Factory框架的使用

我们使用LLaMA-Factory这个开源的专门用来微调模型的框架进行微调，它内部有一个可视化界面，全程轻量化代码操作！

### 1.运行框架

1. 运行下面的代码开启可视化界面，点击本地URL即可开启可视化界面

   ```python
   !export USE_MODELSCOPE_HUB=1 && \
   llamafactory-cli webui
   ```

   ![image-20240919152009529](LLM操作文档.assets/image-20240919152009529.png)

   ![image-20240919152048445](LLM操作文档.assets/image-20240919152048445.png)

2. 请将注意力投影至Train中的数据路径，它指的是在LLaMA-Factory/下的路径（或者需要手动输入LLaMA-Factory/data）
   观察LLaMA-Factory/data下的文件：
   ![image-20240919152553407](LLM操作文档.assets/image-20240919152553407.png)

3. 关于可视化界面的具体参数选择请参考[智码实验室 (pai-ml.com)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)

### 2.数据集准备(我已经处理好了，可以忽略，此处为备案，防止错误)

1. 明确我们的数据集格式：
   ![image-20240919153031125](LLM操作文档.assets/image-20240919153031125.png)
   写一个小脚本进行格式的转化,见压缩包里的DataTransform.ipynb

2. 调整配置文件

   在LLaMA-Factory/data中的dataset_info.json是数据集配置文件，由它告诉框架数据集的存在
   ![image-20240919153219929](LLM操作文档.assets/image-20240919153219929.png)
   我们需要按如下格式修改它：
   ![image-20240919153333599](LLM操作文档.assets/image-20240919153333599.png)
   修改后的结果：
   ![image-20240919153440682](LLM操作文档.assets/image-20240919153440682.png)

3. 接下来验证一下是否配置好数据集了，如下图所示肉眼可见的配置好了！
   ![image-20240919153627388](LLM操作文档.assets/image-20240919153627388.png)
   ![image-20240919153636795](LLM操作文档.assets/image-20240919153636795.png)

4. 划分数据集7：3，分为训练集与测试集

   写脚本进行数据集的划分，脚本见压缩包

   

## 五、有监督微调(欢迎大家集思广益)

![image-20240919184454741](LLM操作文档.assets/image-20240919184454741.png)

### 1.提示词优化(Prompt optimization) 学弟和我去做 

ChatGPT生成  

### 2.LoRa微调-高效微调的算法、我们算力乞丐的立身之本 天宇和sky去做

- LoRA方法可以在缩减训练参数量和GPU显存占用的同时，使训练后的模型具有与全量微调相当的性能。
- LoRA方法是一个主流、高效的方法，它不仅节约了显存还能够让微调后的性能不错
- 指令数据集(手动构建)：使用Train数据集进行语境学习
- 指令数据集(自动生成)：利用大模型生成指令数据集，通过一系列评判标准后加入训练集
- ![image-20240919191111515](LLM操作文档.assets/image-20240919191111515.png)

### 3.语境学习之二分对比：syw帮我想一想

- 奖励建模(Reward Modeling)：

  构建一个优质的回答与劣质的回答对比的数据集(先做做看，备份几份)：

  奖励建模阶段目标是构建一个文本质量对比模型，对于同一个提示词，SFT 模型给出的多个不同输出结果的质量进行排序。奖励模型（RM 模型）可以通过二分类模型，对输 入的两个结果之间的优劣进行判断。RM 模型与基础语言模型和 SFT 模型不同，RM 模型本身并 不能单独提供给用户使用。奖励模型的训练通常和 SFT 模型一样，使用数十块 GPU，通过几天时 间完成训练。由于 RM 模型的准确率对于强化学习阶段的效果有着至关重要的影响，因此对于该 模型的训练通常需要大规模的训练数据。Andrej Karpathy 在报告中指出，该部分需要百万量级的 对比数据标注，而且其中很多标注需要花费非常长的时间才能完成。

### *4.上下文窗口拓展之插值法(如果有需要的话)

​	大模型的输入文本通常有词元数量的限制，这会限制模型对于长文本的理解与表达能力。当涉及长时间对话或者长对话时，传统的上下文窗口大小可能无法捕捉到全局语境，从而导致信息丢失或者模糊的建模结果。

## 六、强化学习(不会、好像硬件也不够)

## 七、推理优化  学弟也可以做

使用vLLM框架进行推理优化(加速推理)

其实就是代码优化

## 八、模型评估

直接用Llama-Factory内置的评估方法以及提交的得分进行评估(比较粗糙)











