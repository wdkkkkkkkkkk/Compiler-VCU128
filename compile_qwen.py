import os
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
import torch
import time
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel
from gptqmodel.utils.backend import BACKEND

# def get_llm(model_name, seqlen=2048):  
    
#     model = GPTQModel.load(
#         model_id_or_path=model_name,
#         trust_remote_code=True,
#         backend=BACKEND.TORCH
#     )
#     print(model)
#     model.seqlen = seqlen
#     return model

# def dequantize(sublayer):
#     infeatures = sublayer.infeatures
#     outfeatures = sublayer.outfeatures
#     padded_infeatures = sublayer.padded_infeatures
    
#     identity = torch.eye(
#         infeatures, 
#         device=sublayer.qweight.device,
#         dtype=sublayer.qweight.dtype  
#     )
    
#     with torch.no_grad():
#         output = sublayer(identity.to(sublayer.qweight.dtype))  

#     weight = output.T[:outfeatures, :infeatures]

#     print(f'weight (dequantizing): {weight} {weight.shape}')
    
#     return weight

# def dequantize(args, sublayer):
        
#     if isinstance(sublayer, TorchQuantLinear):    
#         infeatures = sublayer.infeatures
#         outfeatures = sublayer.outfeatures
#         padded_infeatures = sublayer.padded_infeatures

#         identity = torch.eye(
#             infeatures, 
#             device=sublayer.qweight.device,
#             dtype=sublayer.scales.dtype 
#         )

#         with torch.no_grad():
#             output = sublayer(identity.to(sublayer.scales.dtype))  

#         weight = output.T[:outfeatures, :infeatures]

#         print(f'weight (dequantizing): {weight} {weight.shape}')
#         print(f'max: {weight.max()} min: {weight.min()}')

#         max_values = weight.view(weight.shape[0], -1, args.group_size).abs().max(dim=2).values

#         scales = max_values / (2 ** (args.wbits - 1) - 1)

#         q_values = torch.round(weight.view(weight.shape[0], -1, args.group_size) / scales.unsqueeze(2))
        
#         q_values = q_values.view(weight.shape[0], weight.shape[1])

#         print(f'weight shape type={weight.shape} {weight.dtype} sacles shape type={scales.shape} {scales.dtype}')

#         return q_values, scales
    
#     if isinstance(sublayer, nn.Linear):   
#         weight = sublayer.weight.data
    
#         max_values = weight.view(weight.shape[0], -1, args.group_size).abs().max(dim=2).values

#         scales = max_values / (2 ** (args.wbits - 1) - 1)
        
#         q_values = torch.round(weight.view(weight.shape[0], -1, args.group_size) / scales.unsqueeze(2))
        
#         q_values = q_values.view(weight.shape[0], weight.shape[1])

#         return q_values, scales



def compile(args, model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def out_bin(data_pt, bin_name):
        if data_pt.dtype == torch.float32:
            data_pt = data_pt.to(torch.float16)

        print(bin_name, data_pt.shape)
        
        
        with open(bin_name, 'wb') as f:
            f.write(data_pt.detach().cpu().numpy().flatten().tobytes())

    for name, param in model.state_dict().items():
        print(f"name: {name}, param.shape: {param.shape}, param.dtype: {param.dtype}")
        
        output_dir = f'{args.quantize_model}/compile/qwen2_qweight_bin'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ###########################output_qweight_start###########################
        if("qweight" in name):
            qweight      = param

            print(f'{name} qwight: {qweight} ')

            print(f"Max value of qweight before quantization: {qweight.max()}")
            print(f"Min value of qweight before quantization: {qweight.min()}")
            qweight_int4 = torch.empty(qweight.shape[0], 8, qweight.shape[1], device=param.device, dtype=torch.int8)
            for i in range(8):
                # Shift right by 4*i bits and apply bitmask to extract the lowest 4 bits
                qweight_int4[:, i, :] = ((qweight >> (4 * i)) & 0xF).to(torch.int8)
            # print(f"Max value of qweight before -8: {qweight_int4.max()}")
            # print(f"Min value of qweight before -8: {qweight_int4.min()}")
            print(f"qweight_int4 (before -8): {name}, {qweight_int4} {qweight_int4.shape}")
            qweight_int4 = qweight_int4 - 8
            qweight_int4 = qweight_int4.reshape(qweight.shape[0]*8, qweight.shape[1])
            print(f"qweight_int4 (out): {name}, {qweight_int4} {qweight_int4.shape}")
            qweight_trp= qweight_int4.transpose(0, 1).to(torch.int32)
            
            out_bin(qweight_trp, os.path.join(output_dir, f'{name}.bin'))
        ###########################output_qweight_end###########################
        
        ###########################output_scales_start###########################
        if("scales" in name):
            scales      = param
            print(f'scales: {scales} {scales.shape}')
            scale_trp   = scales.transpose(0, 1)
            out_bin(scale_trp, os.path.join(output_dir, f'{name}.bin'))
        ###########################output_scales_end###########################

        ###########################output_bias_start###########################
        if("bias" in name):
            bias      = param
            out_bin(bias, os.path.join(output_dir, f'{name}.bin'))
        ###########################output_bias_end###########################

        ###########################output_no_bias_start###########################
        if("scales" in name):
            scales     = param
            bias_name  = name.replace("scales","bias")
            if ( bias_name not in list(model.state_dict().keys()) ):
                bias = bias = torch.zeros(1, scales.shape[1], device=param.device, dtype=torch.float16)
                print(bias_name+"_0", bias.shape)
                out_bin(bias, os.path.join(output_dir, f'{bias_name}.bin'))
        ###########################output_no_bias_end###########################
        

        output_dir = f'{args.quantize_model}/compile/qwen2_fp16_weight_bin'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ###########################output_layernor_weight_start###########################
        if("norm" in name):
            weights    = param
            bias       = torch.zeros(weights.shape, device=param.device, dtype=torch.float16)
            weights_bias= torch.cat([weights, bias], dim=0)
            out_bin(weights_bias, os.path.join(output_dir, f'{name}.bin'))
        ##########################output_layernor_weight_end###########################    
        
        # # ###########################output_embedding_weight_start###########################
        if("embed_tokens" in name):
            weights    = param
            print("\nembed_tokens_weight",weights.shape)
            out_bin(weights.to(torch.float16), os.path.join(output_dir, f'{name}.bin'))
        # ###########################output_layernor_weight_end###########################  

    # output_layer = model.lm_head
    # Wt, scales = dequantize(args, output_layer.to(torch.float32))
    # print(f"weight:{Wt}, {Wt.dtype}")
    # print(f"scales: {scales} {scales.dtype}")

    # out_bin(Wt.to(torch.int32)      , os.path.join(output_dir, f'lm_head_qweight.bin'))
    # out_bin(scales.to(torch.float16), os.path.join(output_dir, f'lm_head_scales.bin'))





    # model = get_llm(model_name)
    # print(model)
    # for name, param in model.state_dict().items():
    #     print(f"name: {name}, param.shape: {param.shape}, param.dtype: {param.dtype}")
    
    # layers = model.model.layers
    # output_dir = f'{args.quantize_model}/compile/qwen2_fp16_weight_bin'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # def process_special_layer(name, param):
    #     if "norm" in name:
    #         weights = param
    #         bias       = torch.zeros(weights.shape, device=param.device, dtype=torch.float16)
    #         weights_bias= torch.cat([weights, bias], dim=0)
    #         out_bin(weights_bias, os.path.join(output_dir, f'{name}.bin'))
        
    #     if "embed" in name:
    #         out_bin(param, os.path.join(output_dir, f'{name}.bin'))

    
    # for name, param in model.named_parameters():
    #     process_special_layer(name, param)

    # output_dir = f'{args.quantize_model}/compile/qwen2_weight_bin'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # for i in range(len(layers)):
    #     layer = layers[i]
    #     subset = find_layers(layer)
    #     print(subset)
    #     print("/---------------------------------/")
    #     print(f"/       compiling layer {i}         /")
    #     print("/---------------------------------/")
    #     linear_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    #     for layer_name, sublayer in subset.items():
    #         if any(t in layer_name for t in linear_types):
    #             print(f"Processing {layer_name}...")
                
    #             weight, scales = dequantize(args, sublayer)

    #             print(f'weight: {weight} {weight.shape}')
                
    #             out_bin(weight.to(torch.int32), os.path.join(output_dir, f'model.model.layers.{i}.{layer_name}.qweight.bin'))
    #             out_bin(scales, os.path.join(output_dir, f'model.model.layers.{i}.{layer_name}.scales.bin'))
    #             bias = sublayer.bias if (sublayer.bias is not None) else torch.zeros(1, scales.shape[1], device=param.device, dtype=torch.float16)
    #             out_bin(bias, os.path.join(output_dir, f'model.model.layers.{i}.{layer_name}.bias.bin'))
    #             print(f"weight:{ weight}, {weight.dtype}")
    #             print(f"scales: {scales} {scales.dtype}")
    # output_layer = model.model.lm_head
    # Wt, scales = dequantize(args, output_layer.to(torch.float32))
    # print(f"weight:{Wt}, {Wt.dtype}")
    # print(f"scales: {scales} {scales.dtype}")

    # out_bin(Wt.to(torch.int32)      , os.path.join(output_dir, f'lm_head_qweight.bin'))
    # out_bin(scales.to(torch.float16), os.path.join(output_dir, f'lm_head_scales.bin'))


# import os
# import torch
# import time
# import torch.nn as nn
# from gptqmodel import GPTQModel
# from gptqmodel.utils.backend import BACKEND
# from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

# def get_llm(model_name, seqlen=2048):  
#     model = GPTQModel.load(
#         model_id_or_path=model_name,
#         trust_remote_code=True,
#         backend=BACKEND.TORCH
#     )
#     model.seqlen = seqlen
#     return model

# def find_layers(module, layers=[nn.Conv2d, nn.Linear, TorchQuantLinear, 'QuantLinear'], name=''):
#     if type(module) in layers:
#         return {name: module}
#     res = {}
#     for name1, child in module.named_children():
#         res.update(find_layers(
#             child, layers=layers, name=name + '.' + name1 if name != '' else name1
#         ))
#     return res

# def dequantize(args, sublayer):
        
#     if isinstance(sublayer, TorchQuantLinear):    
#         infeatures = sublayer.infeatures
#         outfeatures = sublayer.outfeatures
#         padded_infeatures = sublayer.padded_infeatures

#         identity = torch.eye(
#             infeatures, 
#             device=sublayer.qweight.device,
#             dtype=sublayer.scales.dtype 
#         )

#         with torch.no_grad():
#             output = sublayer(identity.to(sublayer.scales.dtype))  

#         weight = output.T[:outfeatures, :infeatures]

#         max_values = weight.view(weight.shape[0], -1, args.group_size).abs().max(dim=2).values

#         scales = max_values / (2 ** (args.wbits - 1) - 1)

#         q_values = torch.round(weight.view(weight.shape[0], -1, args.group_size) / scales.unsqueeze(2))
        
#         q_values = q_values.view(weight.shape[0], weight.shape[1])

#         print(f'weight shape type={weight.shape} {weight.dtype} sacles shape type={scales.shape} {scales.dtype}')

#         return q_values, scales
    
#     if isinstance(sublayer, nn.Linear):   
#         weight = sublayer.weight.data
    
#         max_values = weight.view(weight.shape[0], -1, args.group_size).abs().max(dim=2).values

#         scales = max_values / (2 ** (args.wbits - 1) - 1)
        
#         q_values = torch.round(weight.view(weight.shape[0], -1, args.group_size) / scales.unsqueeze(2))
        
#         q_values = q_values.view(weight.shape[0], weight.shape[1])

#         return q_values, scales

# def out_bin(data_pt, bin_name):
#     if data_pt.dtype == torch.float32:
#         data_pt = data_pt.to(torch.float16)

#     print(bin_name, data_pt.shape)
    
    
#     with open(bin_name, 'wb') as f:
#         f.write(data_pt.detach().cpu().numpy().flatten().tobytes())

# def compile(args, model_name):
#     model = get_llm(model_name)
#     print(model)
#     for name, param in model.state_dict().items():
#         print(f"name: {name}, param.shape: {param.shape}, param.dtype: {param.dtype}")
    
#     layers = model.model.model.layers
#     output_dir = f'{args.quantize_model}/compile/qwen2_fp16_weight_bin'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     def process_special_layer(name, param):
#         if "norm" in name:
#             weights = param
#             bias       = torch.zeros(weights.shape, device=param.device, dtype=torch.float16)
#             weights_bias= torch.cat([weights, bias], dim=0)
#             out_bin(weights_bias, os.path.join(output_dir, f'{name}.bin'))
        
#         if "embed" in name:
#             out_bin(param, os.path.join(output_dir, f'{name}.bin'))

    
#     for name, param in model.named_parameters():
#         process_special_layer(name, param)

#     output_dir = f'{args.quantize_model}/compile/qwen2_weight_bin'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for i in range(len(layers)):
#         layer = layers[i]
#         subset = find_layers(layer)
#         print(subset)
#         print("/---------------------------------/")
#         print(f"/       compiling layer {i}         /")
#         print("/---------------------------------/")
#         linear_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
#         for layer_name, sublayer in subset.items():
#             if any(t in layer_name for t in linear_types):
#                 print(f"Processing {layer_name}...")
                
#                 weight, scales = dequantize(args, sublayer)
                
#                 out_bin(weight.to(torch.int32), os.path.join(output_dir, f'model.model.layers.{i}.{layer_name}.qweight.bin'))
#                 out_bin(scales, os.path.join(output_dir, f'model.model.layers.{i}.{layer_name}.scales.bin'))
#                 bias = sublayer.bias if (sublayer.bias is not None) else torch.zeros(1, scales.shape[1], device=param.device, dtype=torch.float16)
#                 out_bin(bias, os.path.join(output_dir, f'model.model.layers.{i}.{layer_name}.bias.bin'))
#                 print(f"weight:{ weight}, {weight.dtype}")
#                 print(f"scales: {scales} {scales.dtype}")
#     output_layer = model.model.lm_head
#     Wt, scales = dequantize(args, output_layer.to(torch.float32))
#     print(f"weight:{Wt}, {Wt.dtype}")
#     print(f"scales: {scales} {scales.dtype}")

#     out_bin(Wt.to(torch.int32)      , os.path.join(output_dir, f'lm_head_qweight.bin'))
#     out_bin(scales.to(torch.float16), os.path.join(output_dir, f'lm_head_scales.bin'))

