# Compiler-VCU128

# README

## 中文版

### 项目概览

本项目基于 VCU128 现有的 RTL 代码衍生，旨在对 Dense 模型进行量化与常量折叠。

- **量化**  
  - 使用 GPTQ 进行量化  
  - 默认配置：  
    - `W4A16`  
    - `groupsize=128`  
    - `act_order=False`  
    - `sym=True`  
    - `backend=Torch`

- **常量折叠**  
  - 先将已量化模型转换为软件格式的 `.bin` 文件  
  - 再将生成的 `.bin` 文件转换为硬件格式

### Quick Start

1. **环境配置**  
   - 根据 `requirements.txt` 配置所需环境，配置方法为：pip install -r requirements.txt 

2. **修改 `main.sh`**  
   - 将未量化的 Dense 模型名称写入 `main.sh` 中，并修改 `model_path`。  
   - 如有需要，也可在此处调整其他参数。

3. **生成软件格式 `.bin`**  
   - `main.sh` 中的 `--compile` 参数决定是否对量化后模型进行软件格式 `.bin` 文件生成。  
   - 默认直接针对量化后的模型生成软件格式 `.bin` 文件。

4. **量化导出 `lm_head`**  
   - 运行 `lm_head.py` 对 `lm_head` 层进行量化导出，并生成对应的软件格式 `.bin` 文件。

5. **硬件格式 `.bin` (针对 Qwen 模型)**  
   - 如果是 Qwen 模型，则进入 `hardware.py` 进行硬件格式 `.bin` 文件的生成。  
   - 请注意在此过程中根据需求修改文件位置和路径。

6. **Embedding 转换**  
   - 使用 `embedding_tokens.py` 将 `embedding_tokens` 层的数据转换为硬件格式。  
   - 生成的硬件格式数据文件会在第 5 步（`hardware.py`）针对 Qwen 模型时产生。

---

## English Version

### Project Overview

This project is derived from the existing VCU128 RTL code and aims to perform quantization and constant folding on Dense models.

- **Quantization**  
  - Method: GPTQ  
  - Default configuration:  
    - `W4A16`  
    - `groupsize=128`  
    - `act_order=False`  
    - `sym=True`  
    - `backend=Torch`

- **Constant Folding**  
  - First convert the quantized model to a software-format `.bin` file.  
  - Then convert that `.bin` file to the hardware format.

### Quick Start

1. **Environment Setup**  
   - Install the required dependencies listed in `requirements.txt` You can use pip install -r requirements.txt to set up the enviroment.
2. **Edit `main.sh`**  
   - Place the unquantized Dense model name in `main.sh` and modify `model_path`.  
   - If needed, you can adjust other parameters as well.

3. **Software-format `.bin` Generation**  
   - The `--compile` parameter in `main.sh` determines the generation of the software-format `.bin` file for the quantized model.  
   - By default, it directly generates the software-format `.bin` file from the quantized model.

4. **Quantize and Export `lm_head`**  
   - Run `lm_head.py` to export the quantized `lm_head` layer and generate its software-format `.bin` file.

5. **Hardware-format `.bin` (for Qwen models)**  
   - If you are working with a Qwen model, proceed to `hardware.py` to generate the hardware-format `.bin` file.  
   - Be sure to adjust file paths as necessary for your setup.

6. **Embedding Conversion**  
   - Use `embedding_tokens.py` to convert the `embedding_tokens` layer data into hardware format.  
   - The hardware-format data files for the Qwen model will be generated when you run `hardware.py` in the previous step.
