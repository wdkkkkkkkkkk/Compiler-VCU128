import numpy as np

def read_bin(bin_file, numb):
    # 读取二进制数据
    embeddings = np.fromfile(bin_file, dtype=np.uint16)
    
    # 计算每个分块的大小
    block = len(embeddings) // numb
    print(len(embeddings), block)
    for i in range(numb):
        print('writing')
        # 计算每个块的文件名
        file_name = f"Embedding_{i+1:02d}-of-{numb:02d}.bin"
        
        # 计算当前块的数据范围
        start_idx = i * block
        end_idx = (i + 1) * block
        
        # 获取当前块的嵌入数据
        chunk = embeddings[start_idx:end_idx]
        
        # 写入当前块到新文件
        chunk.tofile(file_name)

if __name__ == "__main__":
    read_bin("/data/disk0/Workspace/wdk/GPTQModel/DeepSeek-R1-Distill-Qwen-7B-W4A16-gptq/compile/qwen2_fp16_weight_bin/model.embed_tokens.weight.bin", 16)
