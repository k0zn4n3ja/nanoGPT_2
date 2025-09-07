# Colab-optimized configuration for training nanoGPT on medical textbook
# Designed for T4 GPU with 16GB VRAM

out_dir = 'out-med-textbook'
eval_interval = 250  # More frequent evaluation for shorter runs
eval_iters = 100     # Faster evaluation
log_interval = 10

# Save checkpoints when validation loss improves
always_save_checkpoint = False

# Logging (set wandb_log=True if you want to track in wandb)
wandb_log = False 
wandb_project = 'med-textbook-colab'
wandb_run_name = 'med-gpt-colab'

# Dataset configuration
dataset = 'med_textbook'
gradient_accumulation_steps = 8  # Simulate larger batch size
batch_size = 16                  # Conservative for Colab T4
block_size = 256                 # Smaller context for memory efficiency

# Smaller model architecture for Colab constraints
n_layer = 6
n_head = 8 
n_embd = 384
dropout = 0.1

# Training hyperparameters
learning_rate = 1e-3   # Slightly higher for smaller model
max_iters = 3000       # Reasonable for Colab session
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.95
weight_decay = 0.1

warmup_iters = 300

# Colab-specific settings
compile = False  # Disable torch.compile for compatibility
device = 'cuda'  # Use GPU in Colab
