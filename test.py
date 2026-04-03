from transformers import AutoTokenizer
import torch

def test_tokenizer_outputs():
    # 1. 初始化分词器 (模拟 train_7b_sota.py 中的配置)
    model_name = "/tmp/pretrainmodel/Qwen2.5-7B-Instruct"  # 使用你脚本中的模型路径
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    except Exception as e:
        print(f"无法从本地加载模型，切换到远程测试模型: {e}")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, padding_side="right")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备一些测试句子（模拟不同长度）
    sentences = [
        "I am happy!",
        "This is a longer sentence for testing tokenizer padding and truncation."
    ]

    print("--- 原始输入 ---")
    for i, s in enumerate(sentences):
        print(f"句子 {i+1}: {s}")

    # 3. 执行分词 (模拟 __getitem__ 中的逻辑)
    # 我们设置一个较小的 max_length 以方便观察截断
    max_length = 20
    
    print(f"\n--- 分词处理 (max_length={max_length}) ---")
    encodings = tokenizer(
        sentences,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 4. 展示返回的字典内容
    print("\n--- Tokenizer 返回的键值对 ---")
    for key, value in encodings.items():
        print(f"\n键名: '{key}'")
        print(f"形状: {value.shape}")
        print(f"内容:\n{value}")

    # 5. 可视化解释每一个键的作用
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    print("\n--- 深入理解结果 ---")
    for i in range(len(sentences)):
        print(f"\n样本 {i+1}:")
        # 还原回文本
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        print(f"Token 列表: {tokens}")
        
        # 解释 input_ids
        print(f"Input IDs: 每个数字代表词表中的一个词/子词。PAD 被映射为 {tokenizer.pad_token_id}")
        
        # 解释 attention_mask
        active_tokens = attention_mask[i].sum().item()
        print(f"Attention Mask: 1 代表真实内容 ({active_tokens}个)，0 代表填充 (PAD)")
        
        # 模拟 labels 屏蔽 (逻辑见 train_7b_sota.py L95-105)
        labels = input_ids[i].clone()
        # 示例：假设前 5 个 token 是问题，我们要屏蔽它们
        labels[:5] = -100
        # 屏蔽 PAD 部分
        labels[attention_mask[i] == 0] = -100
        print(f"模拟训练标签 (Labels): {labels.tolist()}")
        print("(注: -100 的位置在计算 Loss 时会被忽略，模型不学习这些位置)")

if __name__ == "__main__":
    test_tokenizer_outputs()
