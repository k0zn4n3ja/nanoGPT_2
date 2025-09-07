# Training nanoGPT on Medical Textbook in Google Colab

## Step 1: Setup Environment

```python
# Install dependencies
!pip install torch numpy transformers datasets tiktoken wandb tqdm

# Clone the repository (or upload your files)
!git clone https://github.com/your-repo/nanoGPT_2.git
%cd nanoGPT_2

# Or if uploading files manually, create the directory structure:
# !mkdir -p data/med_textbook
# !mkdir -p config
```

## Step 2: Upload Your Medical Textbook

Upload your `med_textbook.txt` file to the `data/med_textbook/` directory in Colab.

## Step 3: Data Preparation

```python
# Run the data preparation script
%cd /content/nanoGPT_2
!python data/med_textbook/prepare.py
```

This will create `train.bin` and `val.bin` files in the `data/med_textbook/` directory.

## Step 4: Start Training

```python
# Train with the medical textbook configuration
!python train.py config/train_med_textbook.py

# Or with custom parameters (example for Colab T4 GPU):
!python train.py config/train_med_textbook.py --batch_size=16 --gradient_accumulation_steps=4 --max_iters=5000
```

## Step 5: Monitor Training

The training will output:
- Loss values every 10 iterations
- Validation loss every 500 iterations
- Checkpoints saved to `out-med-textbook/`

## Step 6: Generate Text (Optional)

```python
# After training, generate some medical text
!python sample.py --out_dir=out-med-textbook --num_samples=3 --max_new_tokens=500
```

## Hardware Recommendations

### For Colab Free (T4 GPU):
```python
# Smaller model configuration
!python train.py config/train_med_textbook.py --batch_size=8 --gradient_accumulation_steps=8 --n_layer=6 --n_embd=384 --max_iters=3000
```

### For Colab Pro (V100/A100):
```python
# Use the default configuration or larger:
!python train.py config/train_med_textbook.py --batch_size=32 --gradient_accumulation_steps=8
```

## Expected Timeline
- Data preparation: ~2-5 minutes
- Training (5000 iterations): ~1-3 hours depending on GPU
