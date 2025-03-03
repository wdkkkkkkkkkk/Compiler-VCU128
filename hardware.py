import torch
import numpy as np
import os

device = "cuda" # the device to load the model onto

port_num  = 32
layer_num = 28

os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}")

def read_bin(bin_name, _dtype=np.float16):
    with open(bin_name, 'rb') as f:
        data_bytes = f.read()
    data_np = np.frombuffer(data_bytes, dtype=_dtype)
    return data_np

def out_bin(data_pt, bin_name):
    if data_pt.dtype == torch.float32:
        data_pt = data_pt.to(torch.float16)

    print(bin_name, data_pt.shape)
    
    
    with open(bin_name, 'wb') as f:
        f.write(data_pt.detach().cpu().numpy().flatten().tobytes())

def gen_block_para(layer,step,chin,chout,dir,file):
    #q_weight
    q_weight_bin = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/model.layers.{layer}.{step}.qweight.bin",_dtype=np.int32)
    q_weight_bin = torch.tensor(q_weight_bin,dtype=torch.int32)
    q_weight_bin = q_weight_bin & 0xF
    q_weight     = q_weight_bin.reshape(len(q_weight_bin)//8,8)
    for col in range(8):
        q_weight[:,col] = q_weight[:,col] * 2**(4*col)
    q_weight = torch.sum(q_weight,dim=1)
    q_weight = q_weight.reshape(chout,chin//8)
    print(q_weight,q_weight.shape)

    q_scale_bin  = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/model.layers.{layer}.{step}.scales.bin",_dtype=np.int16)
    q_scale_bin  = torch.tensor(q_scale_bin,dtype=torch.int32)
    print(f"checkpos1 {layer} {step} q_scale: {q_scale_bin} {q_scale_bin.shape}")
    q_scale_bin  = q_scale_bin & 0xFFFF
    print(f"checkpos2 {layer} {step} q_scale: {q_scale_bin} {q_scale_bin.shape}")
    q_scale      = q_scale_bin.reshape(len(q_scale_bin)//2,2)
    print(f"checkpos3 {layer} {step} q_scale: {q_scale_bin} {q_scale_bin.shape}")
    for col in range(2):
        q_scale[:,col] = q_scale[:,col] * 2**(16*col)
    print(f"checkpos4 {layer} {step} q_scale: {q_scale_bin} {q_scale_bin.shape}")
    q_scale      = torch.sum(q_scale,dim=1)
    print(f"{layer} {step} q_scale: {q_scale} {q_scale.shape}")
    q_scale      = q_scale.reshape(chout,chin//2//128)
    q_scale_add0 = torch.zeros((chout,(chin//2//128+8-1)//8*8),dtype=torch.int32)
    q_scale_add0[:,:chin//128//2] = q_scale
    
    scale_weight = torch.tensor([],dtype=torch.int32)
    for i in range(0,q_scale_add0.shape[1]//8):
        scale_weight = torch.cat((scale_weight,q_scale_add0[:,i*8:(i+1)*8]),dim=1)
        if((i+1)*256>q_weight.shape[1]):
            scale_weight = torch.cat((scale_weight,q_weight[:,i*256:]),dim=1)
        else:
            scale_weight = torch.cat((scale_weight,q_weight[:,i*256:(i+1)*256]),dim=1)        
    scale_weight = torch.tensor(scale_weight,dtype=torch.int32)
    print(scale_weight,scale_weight.dtype)
    
    for port in range(port_num):
        out_bin(scale_weight[port::port_num,:],f"QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}/{dir}/{file}_HBM_DDR_{str(port).zfill(2)}.bin")

def gen_outlayer_para(chin,chout,dir,file):
    #q_weight
    q_weight_bin = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/lm_head_qweight.bin",_dtype=np.int32)
    q_weight_bin = torch.tensor(q_weight_bin,dtype=torch.int32)
    q_weight_bin = q_weight_bin & 0xF
    q_weight     = q_weight_bin.reshape(len(q_weight_bin)//8,8)
    for col in range(8):
        q_weight[:,col] = q_weight[:,col] * 2**(4*col)
    q_weight = torch.sum(q_weight,dim=1)
    q_weight = q_weight.reshape(chout,chin//8)
    print(q_weight,q_weight.shape)

    q_scale_bin  = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/lm_head_scales.bin",_dtype=np.int16)
    q_scale_bin  = torch.tensor(q_scale_bin,dtype=torch.int32)
    q_scale_bin  = q_scale_bin & 0xFFFF
    q_scale      = q_scale_bin.reshape(len(q_scale_bin)//2,2)
    for col in range(2):
        q_scale[:,col] = q_scale[:,col] * 2**(16*col)
    q_scale      = torch.sum(q_scale,dim=1)
    q_scale      = q_scale.reshape(chout,chin//2//128)
    q_scale_add0 = torch.zeros((chout,(chin//2//128+8-1)//8*8),dtype=torch.int32)
    q_scale_add0[:,:chin//128//2] = q_scale
    
    scale_weight = torch.tensor([],dtype=torch.int32)
    for i in range(0,q_scale_add0.shape[1]//8):
        scale_weight = torch.cat((scale_weight,q_scale_add0[:,i*8:(i+1)*8]),dim=1)
        if((i+1)*256>q_weight.shape[1]):
            scale_weight = torch.cat((scale_weight,q_weight[:,i*256:]),dim=1)
        else:
            scale_weight = torch.cat((scale_weight,q_weight[:,i*256:(i+1)*256]),dim=1)        
    scale_weight = torch.tensor(scale_weight,dtype=torch.int32)
    print(scale_weight,scale_weight.dtype)
    
    for port in range(port_num):
        out_bin(scale_weight[port::port_num,:],f"QWEN_BLOCK_write_data_port{port_num}/OutLayer/{dir}/{file}_HBM_DDR_{str(port).zfill(2)}.bin")

def gen_block_ln(layer,step,dir,file):
    source_file = f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_fp16_weight_bin/model.layers.{layer}.{step}.weight.bin"
    target_file = f"QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}/{dir}/{file}_wt_in_DDR.bin"
    os.system(f"cp -rf {source_file} {target_file}")

def gen_outlayer_ln(dir,file):
    source_file = f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_fp16_weight_bin/model.norm.weight.bin"
    target_file = f"QWEN_BLOCK_write_data_port{port_num}/OutLayer/{dir}/{file}_wt_in_DDR.bin"
    os.system(f"cp -rf {source_file} {target_file}")

def gen_block_bias(layer,step,dir,file):
    bias_bin = read_bin(f"/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_qweight_bin/model.layers.{layer}.{step}.bias.bin",_dtype=np.int16)
    bias_bin  = torch.tensor(bias_bin,dtype=torch.int32)
    bias_bin  = bias_bin & 0xFFFF
    bias_bin  = bias_bin | 0x3C000000
    out_bin(bias_bin,f"QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}/{dir}/{file}_wt_and_bias_in_DDR.bin")

def gen_outlayer_bias(chout,dir,file):
    bias_bin  = torch.zeros(chout,dtype=torch.int32)
    bias_bin  = bias_bin | 0x3C000000
    out_bin(bias_bin,f"QWEN_BLOCK_write_data_port{port_num}/OutLayer/{dir}/{file}_wt_and_bias_in_DDR.bin")

# GEN BLOCK LAYER
for layer in range(layer_num):
    os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}")
    os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}/LN_DDR_bin")
    os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}/MVM_BN_DDR_bin")
    os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}/MVM_BN_RES_DDR_bin")
    os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}/MVM_BN_RES_write_to_HBM_bin") 
    os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/BLOCK{str(layer).zfill(2)}/MVM_BN_write_to_HBM_bin")
          
    ##GEN SCALE AND WEIGHT
    gen_block_para(layer,"self_attn.q_proj",3584,3584,"MVM_BN_write_to_HBM_bin","MVMBN0_q")
    gen_block_para(layer,"self_attn.k_proj",3584,512 ,"MVM_BN_write_to_HBM_bin","MVMBN0_k")
    gen_block_para(layer,"self_attn.v_proj",3584,512 ,"MVM_BN_write_to_HBM_bin","MVMBN0_v")
    gen_block_para(layer,"mlp.gate_proj"   ,3584,18944,"MVM_BN_write_to_HBM_bin","MVMBN1")
    
    gen_block_para(layer,"self_attn.o_proj",3584,3584 ,"MVM_BN_RES_write_to_HBM_bin","MVMBNRES0")
    gen_block_para(layer,"mlp.up_proj"     ,3584,18944 ,"MVM_BN_RES_write_to_HBM_bin","MVMBNRES1")
    gen_block_para(layer,"mlp.down_proj"   ,18944,3584 ,"MVM_BN_RES_write_to_HBM_bin","MVMBNRES2")

    ##GEN BIAS
    gen_block_bias(layer,"self_attn.q_proj","MVM_BN_DDR_bin","MVMBN0_q")
    gen_block_bias(layer,"self_attn.k_proj","MVM_BN_DDR_bin","MVMBN0_k")
    gen_block_bias(layer,"self_attn.v_proj","MVM_BN_DDR_bin","MVMBN0_v")
    gen_block_bias(layer,"mlp.gate_proj"   ,"MVM_BN_DDR_bin","MVMBN1")
    
    gen_block_bias(layer,"self_attn.o_proj","MVM_BN_RES_DDR_bin","MVMBNRES0")
    gen_block_bias(layer,"mlp.up_proj"     ,"MVM_BN_RES_DDR_bin","MVMBNRES1")
    gen_block_bias(layer,"mlp.down_proj"   ,"MVM_BN_RES_DDR_bin","MVMBNRES2")   
    
    ##GEN LN
    gen_block_ln(layer,"input_layernorm","LN_DDR_bin","LN0")
    gen_block_ln(layer,"post_attention_layernorm","LN_DDR_bin","LN1")
    

# GEN OUTLAYER
os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/OutLayer")
os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/OutLayer/LN_DDR_bin")
os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/OutLayer/MVM_BN_DDR_bin")
os.system(f"mkdir -p QWEN_BLOCK_write_data_port{port_num}/OutLayer/MVM_BN_write_to_HBM_bin")

gen_outlayer_para(3584,152064,"MVM_BN_write_to_HBM_bin","MVMBN_Argmax")
gen_outlayer_bias(152064,"MVM_BN_DDR_bin","MVMBN_Argmax")
gen_outlayer_ln("LN_DDR_bin","LN")