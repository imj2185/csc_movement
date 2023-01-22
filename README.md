# csc_movement


## Usage
### 1. Train model with experiement data 1
```bash
python train.py  --exp_number 1 --train
```

### 2. Evaluate model (trained with experiment data 1) with validation data  4 
```bash
python train.py  --exp_number 1 --val_number 4 --val_count 51 --sample_name IFF5% --test
```