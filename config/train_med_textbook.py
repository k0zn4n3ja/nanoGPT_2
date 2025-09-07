# Configuration for training nanoGPT on medical textbook
# Based on shakespeare config but optimized for larger medical text

out_dir = 'out-med-textbook'
eval_interval = 500
eval_iters = 200
log_interval = 10

# Save checkpoints when validation loss improves
always_save_checkpoint = False

# Logging (set to True if you want to use wandb)
wandb_log = False 
wandb_project = 'med-textbook'
wandb_run_name = 'med-gpt'

# Dataset configuration
dataset = 'med_textbook'
gradient_accumulation_steps = 8  # Simulate larger batch size
batch_size = 32  # Reasonable for most GPUs
block_size = 512  # Context length for medical text

# Model architecture - medium sized GPT
n_layer = 8
n_head = 8 
n_embd = 512
dropout = 0.1  # Some dropout for better generalization

# Training hyperparameters
learning_rate = 3e-4  # Conservative learning rate
max_iters = 10000  # Adjust based on your compute budget
lr_decay_iters = 10000  # Usually equal to max_iters
min_lr = 3e-5  # learning_rate / 10
beta2 = 0.95
weight_decay = 0.1

warmup_iters = 1000

# System settings - adjust based on your hardware
# device = 'cuda'  # Use 'cpu' if no GPU available
# compile = True   # Set to False if you have issues with torch.compile
