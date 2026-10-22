# Character-level ABC model on O'Neill's 1850 + 1001 (~620K training chars).
# Smaller than the Session config: ~0.8M parameters, because with this little
# data a bigger model just memorises the book. Run from the nanoGPT root:
#   python train.py config/train_oneill.py

out_dir = 'out-oneill'
dataset = 'oneill'

# Apple Silicon settings
device = 'mps'        # 'cuda' on Kaggle/Colab, 'cpu' as a last resort
compile = False
dtype = 'float32'

# Small dataset: evaluate often and keep only the best-on-validation checkpoint,
# since training loss will keep falling long after the model starts memorising.
eval_interval = 100
eval_iters = 40
log_interval = 20
always_save_checkpoint = False
wandb_log = False

gradient_accumulation_steps = 1
batch_size = 32
block_size = 256      # median tune is ~220 chars including headers

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2         # more regularisation than usual, for the same reason

learning_rate = 1e-3
max_iters = 2000      # ~16M chars seen, i.e. many passes over a small corpus
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
