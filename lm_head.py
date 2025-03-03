import os
import torch
import torch.nn as nn
import pickle
import sys

import numpy as np
from tqdm import tqdm

# import quant
from transformers import AutoModelForCausalLM, AutoTokenizer

def int32_to_int4(tensor):
    # 定义掩码，提取4位
    mask = torch.tensor([0xF], dtype=torch.int32, device=tensor.device)
    # 构建移位列表
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=tensor.device)
    # 扩展移位张量维度以匹配输入张量
    shifts = shifts.view(1, 1, -1)
    # 将输入张量扩展一个维度，以匹配移位张量
    tensor_expanded = tensor.unsqueeze(-1)
    # 进行移位并应用掩码
    int4_tensor = (tensor_expanded >> shifts) & mask
    return int4_tensor

def out_txt_1(data_pt, txt_name):
    data = data_pt.detach().cpu().numpy()
    shape = data.shape

    filename = os.path.join('out_txt_glm3', txt_name)
    with open(filename, 'w') as file:
        shape_line = ' '.join(map(str, data.shape))
        file.write(f'{shape_line}\n')
        np.savetxt(file, data.flatten(), delimiter=',', newline='\n')

def out_bin(data_pt, bin_name):
    if data_pt.dtype == torch.float32:
        data_pt = data_pt.to(torch.float16)

    print(bin_name, data_pt.shape)
    with open(bin_name, 'wb') as f:
        f.write(data_pt.numpy().flatten().tobytes())

def out_txt(data_pt_src, txt_name, is_out=False, is_transpose=False):
    if is_transpose:
        data_pt = data_pt_src.transpose(0, 1)
    else:
        data_pt = data_pt_src

    data = data_pt.detach().cpu().numpy()
    shape = data.shape
    
    # print(txt_name, shape, data.mean(), data_pt.dtype)
    if is_out:
        # out_txt_1(data_pt, txt_name)
        out_bin(data_pt, txt_name.replace('.txt', '.bin'))

def in_pickle(pkl_name, is_weight=True):
    print(pkl_name)
    weight = None
    with open(pkl_name, 'rb') as f:
        weight = pickle.load(f)
    if is_weight:
        return weight.detach().cpu()
    else:
        return weight

# def quantize_group(tensor, num_bits):
#     # V0909
#     # 计算最小值和最大值
#     print(type(tensor))
#     min_val = torch.min(tensor)
#     max_val = torch.max(tensor)
    
#     # 计算量化的范围
#     max_val = torch.max(max_val, torch.abs(min_val))
#     qmin, qmax = -(2**(num_bits - 1)), (2**(num_bits - 1)) - 1
#     scale = 2 * max_val / (qmax - qmin)

#     # zero_point = qmin - min_val / scale
#     # print(max_val / scale, min_val / scale)

#     # 量化
#     quantized_tensor = (tensor / scale).to(torch.int)
#     return quantized_tensor, scale

# def quantize_group(tensor, num_bits):
#     # V0910
#     # 计算最小值和最大值
#     # print(type(tensor))
#     min_val = torch.min(tensor)
#     max_val = torch.max(tensor)
    
#     # 计算量化的范围
#     max_val = torch.max(max_val, torch.abs(min_val))
#     qmin, qmax = -(2**(num_bits - 1)), (2**(num_bits - 1)) - 1
#     scale = 2 * max_val / (qmax - qmin)

#     # zero_point = qmin - min_val / scale
#     # print(max_val / scale, min_val / scale)

#     # 量化
#     quantized_tensor = torch.round((tensor / scale)).to(torch.int)
#     quantized_tensor[quantized_tensor == 8] = 7
#     return quantized_tensor, scale


def quantize_group(tensor, num_bits):
    # V0911
    # 计算最小值和最大值
    # print(type(tensor))
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    qmin, qmax = -(2**(num_bits - 1)), (2**(num_bits - 1)) - 1
    if max_val >= torch.abs(min_val):
        scale = 2 * max_val / (2*qmax)
    else:
        max_val = torch.abs(min_val)
        scale = 2 * max_val / (-2*qmin)
    # zero_point = qmin - min_val / scale
    # print(max_val / scale, min_val / scale)

    # 量化
    quantized_tensor = (tensor / scale).to(torch.int)
    return quantized_tensor, scale

def quant_minmax(weight, _bits, _gs):
    # weight = torch.rand(4096, 4096)
    h, w = weight.shape
    print(h, w)
    qweight = torch.zeros(h, w).half()
    qscales = torch.zeros(h//_gs, w).half()
    
    for i in range(0, h, _gs):
        for j in range(w):
            print(i,j)
            block = weight[i:i+_gs, j]
            qblock, scale = quantize_group(block, _bits)
            qweight[i:i+_gs, j] = qblock
            qscales[i//_gs, j] = scale.item()

    return qweight, qscales

def read_bin(bin_name, _dtype=np.float16):
    # 从文件中读取数据
    with open(bin_name, 'rb') as f:
        data_bytes = f.read()
    
    # 将字节数据转换为NumPy数组
    data_np = np.frombuffer(data_bytes, dtype=_dtype)
    return torch.tensor(data_np)

def get_model():
    # 模型地址
    model_path = '/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq'
    print("============  Loading model ============")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

if __name__ == "__main__":
    # 加载模型
    model, tokenizer = get_model()
    print(model)
    for name, param in model.state_dict().items():
        print(name, param.shape, param.dtype)
    # OUTPUT_LAYER
    output_layer = model.lm_head.weight
    Wt, scales = quant_minmax(output_layer.transpose(0, 1).to(torch.float32), 4, 128)
    print(Wt.shape,scales.shape)

    output_dir = '/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_bin(Wt.transpose(0,1).to(torch.int32)      , os.path.join(output_dir, f'lm_head_qweight.bin'))
    out_bin(scales.transpose(0,1).to(torch.float16), os.path.join(output_dir, f'lm_head_scales.bin'))


