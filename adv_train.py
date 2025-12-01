import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
from improved_model import AdvancedBindingModel
from dataloader import KIBADataset, collate_fn

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.000575  # Reduced from 0.005 for stability
NUM_EPOCHS = 8  # Increased for better convergence
WEIGHT_DECAY = 1e-5  # L2 regularization
GRADIENT_CLIP = 1.0  # Gradient clipping value
PATIENCE = 10  # Early stopping patience
WARMUP_EPOCHS = 3  # Learning rate warmup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = "data"
SAVE_DIR = "checkpoints"
LOG_DIR = "logs"

class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'max'
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop

def get_linear_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup + cosine decay learning rate scheduler."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def concordance_index(y_true, y_pred):
    """Calculate concordance index (C-index) for ranking evaluation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    concordant = 0
    total = 0
    
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                   (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                    concordant += 1
    
    return concordant / total if total > 0 else 0.0

def train(model, loader, optimizer, criterion, device, gradient_clip=None):
    model.train()
    running_loss = 0.0
    batch_count = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Move batch to device
        batch.ligand = batch.ligand.to(device)
        protein = batch.protein.to(device)
        target = batch.affinity.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch.ligand, protein)
        
        # Loss calculation
        loss = criterion(output.view(-1), target.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        running_loss += loss.item()
        batch_count += 1
        pbar.set_postfix({'loss': loss.item(), 'avg_loss': running_loss / batch_count})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch.ligand = batch.ligand.to(device)
            protein = batch.protein.to(device)
            target = batch.affinity.to(device)
            
            output = model(batch.ligand, protein)
            loss = criterion(output.view(-1), target.view(-1))
            
            running_loss += loss.item()
            
            predictions.extend(output.view(-1).cpu().numpy())
            targets.extend(target.view(-1).cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # Pearson Correlation
    pearson = np.corrcoef(predictions, targets)[0, 1]
    
    # R² Score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Concordance Index (for ranking)
    ci = concordance_index(targets, predictions)
    
    return {
        'loss': epoch_loss,
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson,
        'r2': r2,
        'ci': ci,
        'predictions': predictions,
        'targets': targets
    }

def plot_results(train_losses, val_losses, val_metrics, save_dir):
    """Create comprehensive visualization of training results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # RMSE
    axes[0, 1].plot(val_metrics['rmse'], color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Validation RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Pearson Correlation
    axes[0, 2].plot(val_metrics['pearson'], color='green', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Pearson R', fontsize=12)
    axes[0, 2].set_title('Validation Pearson Correlation', fontsize=14, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)
    
    # MAE
    axes[1, 0].plot(val_metrics['mae'], color='red', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('MAE', fontsize=12)
    axes[1, 0].set_title('Validation MAE', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # R² Score
    axes[1, 1].plot(val_metrics['r2'], color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('R² Score', fontsize=12)
    axes[1, 1].set_title('Validation R² Score', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # Concordance Index
    axes[1, 2].plot(val_metrics['ci'], color='brown', linewidth=2)
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('C-Index', fontsize=12)
    axes[1, 2].set_title('Validation Concordance Index', fontsize=14, fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(targets, predictions, save_path, title="Predictions vs True Values"):
    """Scatter plot of predictions vs true values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Affinity', fontsize=12)
    plt.ylabel('Predicted Affinity', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
def find_lr(model, train_loader, optimizer, criterion, device, 
            start_lr=1e-7, end_lr=10, num_iter=100):
    """Find optimal learning rate using LR range test."""
    model.train()
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    losses = []
    lrs = []
    best_loss = float('inf')
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
    
    iterator = iter(train_loader)
    
    for iteration in range(num_iter):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        
        # Training step
        batch.ligand = batch.ligand.to(device)
        protein = batch.protein.to(device)
        target = batch.affinity.to(device)
        
        optimizer.zero_grad()
        output = model(batch.ligand, protein)
        loss = criterion(output.view(-1), target.view(-1))
        loss.backward()
        optimizer.step()
        
        # Store loss and lr
        current_lr = optimizer.param_groups[0]['lr']
        losses.append(loss.item())
        lrs.append(current_lr)
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
        
        # Stop if loss explodes
        if loss.item() > 4 * best_loss:
            break
        
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    plt.savefig('lr_finder.png')
    
    # Suggest optimal LR (steepest descent point)
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]
    
    print(f"Suggested Learning Rate: {optimal_lr:.6f}")
    return optimal_lr
def mc_dropout_predict(model, ligand, protein, n_samples=10):
    """
    Predict with MC Dropout to estimate uncertainty.
    Model must have dropout layers that stay active during inference.
    """
    model.train()  # Keep dropout active
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(ligand, protein)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    return mean_pred, std_pred 

def main():
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(SAVE_DIR, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Load Datasets
    print("\nLoading datasets...")
    train_dataset = KIBADataset(os.path.join(DATA_DIR, "kiba_train.csv"))
    valid_dataset = KIBADataset(os.path.join(DATA_DIR, "kiba_valid.csv"))
    test_dataset = KIBADataset(os.path.join(DATA_DIR, "kiba_test.csv"))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize Model
    print("\nInitializing model...")
    model = AdvancedBindingModel().to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = get_linear_warmup_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    # Alternative: ReduceLROnPlateau
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001, mode='min')
    '''
    # Training metrics storage
    train_losses = []
    val_losses = []
    val_metrics = {
        'rmse': [],
        'mae': [],
        'pearson': [],
        'r2': [],
        'ci': []
    }
    
    best_val_rmse = float('inf')
    best_epoch = 0
    
    # Save hyperparameters
    config = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'weight_decay': WEIGHT_DECAY,
        'gradient_clip': GRADIENT_CLIP,
        'warmup_epochs': WARMUP_EPOCHS,
        'patience': PATIENCE,
        'device': str(DEVICE),
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Training Loop
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 80)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE, GRADIENT_CLIP)
        
        # Validate
        val_results = validate(model, valid_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step()
        # For ReduceLROnPlateau: scheduler.step(val_results['rmse'])
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_results['loss'])
        val_metrics['rmse'].append(val_results['rmse'])
        val_metrics['mae'].append(val_results['mae'])
        val_metrics['pearson'].append(val_results['pearson'])
        val_metrics['r2'].append(val_results['r2'])
        val_metrics['ci'].append(val_results['ci'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_results['loss']:.4f}")
        print(f"Val RMSE:   {val_results['rmse']:.4f}")
        print(f"Val MAE:    {val_results['mae']:.4f}")
        print(f"Val Pearson: {val_results['pearson']:.4f}")
        print(f"Val R²:     {val_results['r2']:.4f}")
        print(f"Val CI:     {val_results['ci']:.4f}")
        
        # Save best model
        if val_results['rmse'] < best_val_rmse:
            best_val_rmse = val_results['rmse']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_results,
                'config': config
            }, os.path.join(exp_dir, "best_model.pth"))
            print(f"✓ Saved new best model! (RMSE: {best_val_rmse:.4f})")
        
        # Check early stopping
        if early_stopping(val_results['rmse']):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
            break
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
    
    # Plot training curves
    print("\nGenerating plots...")
    plot_results(train_losses, val_losses, val_metrics, exp_dir)
    '''
    # Final Test Evaluation
    print("\n" + "=" * 80)
    print("Evaluating on Test Set with Best Model...")
    checkpoint = torch.load(os.path.join("checkpoints", "exp_20251201_105448", "best_model.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = validate(model, test_loader, criterion, DEVICE)
    
    print("\nTest Set Results:")
    print(f"Test Loss:    {test_results['loss']:.4f}")
    print(f"Test RMSE:    {test_results['rmse']:.4f}")
    print(f"Test MAE:     {test_results['mae']:.4f}")
    print(f"Test Pearson: {test_results['pearson']:.4f}")
    print(f"Test R²:      {test_results['r2']:.4f}")
    print(f"Test CI:      {test_results['ci']:.4f}")
    
    # Save test results
    test_metrics = {
        'loss': float(test_results['loss']),
        'rmse': float(test_results['rmse']),
        'mae': float(test_results['mae']),
        'pearson': float(test_results['pearson']),
        'r2': float(test_results['r2']),
        'ci': float(test_results['ci'])
    }
    
    with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_predictions(y_true, y_pred, save_path=None, title="Test Set Predictions"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Density for better visualization
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)

    plt.figure(figsize=(8, 8))

    # Scatter with density-based color
    sc = plt.scatter(y_true, y_pred, c=z, s=10, alpha=0.7)
    plt.colorbar(sc, label="Density")

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Prediction')

    # Axis labels & limits tightened to data
    pad = 0.5
    plt.xlim(y_true.min() - pad, y_true.max() + pad)
    plt.ylim(y_pred.min() - pad, y_pred.max() + pad)

    # Equal scale on both axes to avoid distortion
    plt.gca().set_aspect('equal', 'box')

    # Title & labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("True Affinity", fontsize=12)
    plt.ylabel("Predicted Affinity", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # Plot test predictions
    plot_predictions(
        test_results['targets'], 
        test_results['predictions'],
        os.path.join(exp_dir, 'test_predictions.png'),
        title=f"Test Set Predictions (R²={test_results['r2']:.3f}, RMSE={test_results['rmse']:.3f})"
    )
    
    print(f"\nAll results saved to: {exp_dir}")
    print("=" * 80)
if __name__ == "__main__":
    main()