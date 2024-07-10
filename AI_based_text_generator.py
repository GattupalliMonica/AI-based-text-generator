import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # You can also use 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to generate text
def generate_text(prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.9):
    # Encode the prompt text
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=2,
        num_return_sequences=1
    )
    
    # Decode the generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return text

# Example usage
prompt = "Once upon a time in a land far away,"
generated_text = generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9)
print("Generated Text:\n", generated_text)
