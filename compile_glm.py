import os
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
import torch
import time
import torch.nn as nn
from gptqmodel import GPTQModel
from gptqmodel.utils.backend import BACKEND

def get_llm(model_name, seqlen=2048):  
    
    model = GPTQModel.load(
        model_id_or_path=model_name,
        trust_remote_code=True,
        backend=BACKEND.TORCH
    )
    print(model)
    model.seqlen = seqlen
    return model

def find_layers(module, layers=[nn.Conv2d, nn.Linear, TorchQuantLinear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

# def quantize(args, sublayer):
#     weight = sublayer.weight.data
    
#     max_values = weight.view(weight.shape[0], -1, args.groupsize).abs().max(dim=2).values
#     # compute scale
#     scales = max_values / (2 ** (args.wbits - 1) - 1)
    
    
#     q_values = torch.round(weight.view(weight.shape[0], -1, args.groupsize) / scales.unsqueeze(2))
    
    
#     q_values = q_values.view(weight.shape[0], weight.shape[1])

#     return q_values, scales

def dequantize(args, sublayer):
        
    if isinstance(sublayer, TorchQuantLinear):    
        infeatures = sublayer.infeatures
        outfeatures = sublayer.outfeatures
        padded_infeatures = sublayer.padded_infeatures

        identity = torch.eye(
            infeatures, 
            device=sublayer.qweight.device,
            dtype=sublayer.scales.dtype 
        )

        with torch.no_grad():
            output = sublayer(identity.to(sublayer.scales.dtype))  

        weight = output.T[:outfeatures, :infeatures]

        max_values = weight.view(weight.shape[0], -1, args.group_size).abs().max(dim=2).values

        scales = max_values / (2 ** (args.wbits - 1) - 1)

        q_values = torch.round(weight.view(weight.shape[0], -1, args.group_size) / scales.unsqueeze(2))
        
        q_values = q_values.view(weight.shape[0], weight.shape[1])

        print(f'weight shape={weight.shape} sacles shape={scales.shape}')

        return weight, scales

    if isinstance(sublayer, nn.Linear):   
        weight = sublayer.weight.data
    
        max_values = weight.view(weight.shape[0], -1, args.group_size).abs().max(dim=2).values

        scales = max_values / (2 ** (args.wbits - 1) - 1)
        
        q_values = torch.round(weight.view(weight.shape[0], -1, args.group_size) / scales.unsqueeze(2))
        
        q_values = q_values.view(weight.shape[0], weight.shape[1])

        return q_values, scales

def out_bin(data_pt, bin_name):
    if data_pt.dtype == torch.float32:
        data_pt = data_pt.to(torch.float16)

    print(bin_name, data_pt.shape)
    
    
    with open(bin_name, 'wb') as f:
        f.write(data_pt.detach().cpu().numpy().flatten().tobytes())


def compile(args, model_name):
    model=get_llm(model_name)
    layers = model.transformer.encoder.layers
    start_time = time.time() 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        print(subset)
        print("/---------------------------------/")
        print(f"/       compiling layer {i}         /")
        print("/---------------------------------/")
        block = 'BLOCK' + str(i).zfill(2)
        out_dir = f'{args.quantize_model}/compile/{block}/step1_6'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # QKV
        query_key_value = subset['self_attention.query_key_value']
        bias = query_key_value.bias.detach()
        Wt, scales = dequantize(args, query_key_value)
        print("compiling qkv")
        MVM_BN_Wq = Wt[:4096, :]
        MVM_BN_Wk = Wt[4096:4096+256, :]
        MVM_BN_Wv = Wt[4096+256:, :]
        MVM_BN_Scaleq = scales[:4096, :]
        MVM_BN_Scalek = scales[4096:4096+256, :]
        MVM_BN_Scalev = scales[4096+256:, :]
        MVM_BN_Biasq = bias[:4096]
        MVM_BN_Biask = bias[4096:4096+256]
        MVM_BN_Biasv = bias[4096+256:]

        out_bin(MVM_BN_Wq.to(torch.int32), os.path.join(out_dir, 'MVM_BN_Wq.bin'))
        out_bin(MVM_BN_Wk.to(torch.int32), os.path.join(out_dir, 'MVM_BN_Wk.bin'))
        out_bin(MVM_BN_Wv.to(torch.int32), os.path.join(out_dir, 'MVM_BN_Wv.bin'))
        out_bin(MVM_BN_Scaleq.to(torch.float16), os.path.join(out_dir, 'MVM_BN_Scaleq.bin'))
        out_bin(MVM_BN_Scalek.to(torch.float16), os.path.join(out_dir, 'MVM_BN_Scalek.bin'))
        out_bin(MVM_BN_Scalev.to(torch.float16), os.path.join(out_dir, 'MVM_BN_Scalev.bin'))
        out_bin(MVM_BN_Biasq.to(torch.float16), os.path.join(out_dir, 'MVM_BN_Biasq.bin'))
        out_bin(MVM_BN_Biask.to(torch.float16), os.path.join(out_dir, 'MVM_BN_Biask.bin'))
        out_bin(MVM_BN_Biasv.to(torch.float16), os.path.join(out_dir, 'MVM_BN_Biasv.bin'))

        out_dir = f'{args.quantize_model}/compile/{block}/step7_12'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # MVMBNRES0
        dense = subset['self_attention.dense']
        Wt, scales = dequantize(args, dense)
        print("compiling dense")
        out_bin(Wt.to(torch.int32), os.path.join(out_dir, 'MVM_BN_RES_weight.bin'))
        out_bin(scales.to(torch.float16), os.path.join(out_dir, 'MVM_BN_RES_scales.bin'))

        # MVMBNRES1
        dense_h_to_4h = subset['mlp.dense_h_to_4h']
        Wt, scales = dequantize(args, dense_h_to_4h)
        print("compiling ht4h")
        Wts = Wt.split([13696, 13696], dim=0)
        Wt0, Wt1 = Wts[0], Wts[1]
        Scls = scales.split([13696, 13696], dim=0)
        scales0, scales1 = Scls[0], Scls[1]
        out_bin(Wt0.to(torch.int32), os.path.join(out_dir, 'MVMBNRES1_weight_0.bin'))
        out_bin(scales0.to(torch.float16), os.path.join(out_dir, 'MVMBNRES1_scales_0.bin'))
        out_bin(Wt1.to(torch.int32), os.path.join(out_dir, 'MVMBNRES1_weight_1.bin'))
        out_bin(scales1.to(torch.float16), os.path.join(out_dir, 'MVMBNRES1_scales_1.bin'))

        # MVMBNRES2
        dense_4h_to_h = subset['mlp.dense_4h_to_h']
        Wt, scales = dequantize(args, dense_4h_to_h)
        print("compiling 4hth")
        out_bin(Wt.to(torch.int32), os.path.join(out_dir, 'MVMBNRES2_weight.bin'))
        out_bin(scales.to(torch.float16), os.path.join(out_dir, 'MVMBNRES2_scales.bin'))

    output_layer = model.transformer.output_layer
    out_dir = f'{args.quantize_model}/compile/output_layer'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # output_layer = output_layer['']
    Wt, scales = dequantize(args, output_layer)
    print("compiling output_layer")
    MVM_weight_0 = Wt[:32512, :]
    MVM_weight_1 = Wt[32512:, :]
    MVM_scales_0 = scales[:32512, :]
    MVM_scales_1 = scales[32512:, :]
    out_bin(Wt.to(torch.int32), os.path.join(out_dir, 'MVM_weight.bin'))
    out_bin(MVM_weight_0.to(torch.int32), os.path.join(out_dir, 'MVM_weight_0.bin'))
    out_bin(MVM_weight_1.to(torch.int32), os.path.join(out_dir, 'MVM_weight_1.bin'))
    out_bin(scales.to(torch.float16), os.path.join(out_dir, 'MVM_scales.bin'))
    out_bin(MVM_scales_0.to(torch.float16), os.path.join(out_dir, 'MVM_scales_0.bin'))
    out_bin(MVM_scales_1.to(torch.float16), os.path.join(out_dir, 'MVM_scales_1.bin'))

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(f"Total time for compiling all layers: {elapsed_time:.2f} seconds")
        