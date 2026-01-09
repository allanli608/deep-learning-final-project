import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import torch
import os

font_path = "src/misc/SimHei.ttf" 

# Check if you actually placed it there
if not os.path.exists(font_path):
    print(f"❌ Error: Font file not found at: {font_path}")
    print("Please manually download SimHei.ttf and place it in this directory.")
else:
    print(f"✅ Loaded local font: {font_path}")

# Load the font property
my_font = fm.FontProperties(fname=font_path)

def plot_attention_averaged(model_wrapper, text_input):
    """
    Plots the AVERAGED Cross-Attention weights across all heads.
    """
    # 1. Get Model & Tokenizer
    if hasattr(model_wrapper, "model"):
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
    else:
        model = model_wrapper.get_model()
        tokenizer = model_wrapper.get_tokenizer()
    
    device = model_wrapper.device

    # Force Eager Mode
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"
    model.config.output_attentions = True
    model.config.return_dict_in_generate = True
    
    # Tokenize
    inputs = tokenizer(text_input, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            output_attentions=True,
            return_dict_in_generate=True,
            forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"],
            max_length=20,
            early_stopping=False,
            num_beams=1
        )

    # 2. Extract & Aggregate Attention
    tgt_tokens = [tokenizer.decode(t) for t in outputs.sequences[0]]
    src_tokens = [tokenizer.decode(t) for t in inputs.input_ids[0]]
    
    final_attentions = []
    
    for step in outputs.cross_attentions:
        if step is None: continue
        # Get Last Layer: [Batch, Heads, Tgt_Len, Src_Len]
        layer_attn = step[-1].cpu() 
        
        # Average across heads (Dim 1) -> [Batch, Tgt_Len, Src_Len]
        avg_attn = layer_attn.mean(dim=1) 
        
        # Append the attention for this step
        # avg_attn[0] gets the batch item -> shape [Tgt_Len, Src_Len] (usually [1, Src])
        final_attentions.append(avg_attn[0])

    if not final_attentions: 
        print("❌ No valid attention weights found.")
        return

    # Stack along dim 0 to build the full sequence
    attention_matrix = torch.cat(final_attentions, dim=0).numpy()
    
    # 3. Visualize
    # Slicing [1:-1, 1:-1] removes start/end tokens for cleaner view
    matrix_zoomed = attention_matrix[1:-1, 1:-1]
    src_labels = src_tokens[1:-1]
    tgt_labels = tgt_tokens[1:-1]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix_zoomed, 
        xticklabels=src_labels, 
        yticklabels=tgt_labels, 
        cmap="viridis",
        annot=True, 
        fmt=".2f"
    )
    
    # Apply Font
    plt.xticks(fontproperties=my_font, rotation=45)
    plt.yticks(fontproperties=my_font, rotation=0)
    
    plt.xlabel("Source (Input)", fontproperties=my_font)
    plt.ylabel("Generated (Output)", fontproperties=my_font)
    plt.title(f"Averaged Semantic Attention: {model_wrapper.__class__.__name__}")
    plt.show()