
import os
import torch
import numpy as np
import argparse
import random
from data_loader import prepare_dataloaders
from models import CNNModel, LSTMModel, CNNLSTMModel
from train import train_model, evaluate_model, plot_training_history

def set_seed(seed=3407):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():

    parser = argparse.ArgumentParser(description='EEG binary classification model training')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'lstm', 'cnn_lstm'], 
                        help='Select model type: cnn, lstm, cnn_lstm')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training rounds')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seq_length', type=int, default=None, help='Sequence length (None for auto-detection)')
    parser.add_argument('--device', type=str, default='cuda', help='Training device: cuda or cpu')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed 42/3407')



    args = parser.parse_args()
    
    set_seed(args.seed)

    print("Preparing data...")

    epilepsy_dir = os.path.join('epilepsy_columns')
    normal_dir = os.path.join('normal_columns')  
    train_loader, val_loader, test_loader = prepare_dataloaders(
        epilepsy_dir=epilepsy_dir,
        normal_dir=normal_dir,
        batch_size=args.batch_size,
        max_seq_length=args.seq_length,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    
    if args.seq_length is None:
        for data, _ in train_loader:
            args.seq_length = data.shape[2]
            print(f"Detected sequence length: {args.seq_length}")
            break

    print(f"Creating {args.model} model...")
    if args.model == 'cnn':
        model = CNNModel(seq_length=args.seq_length)
    elif args.model == 'lstm':
        model = LSTMModel(input_size=1, hidden_size=128)
    elif args.model == 'cnn_lstm':
        model = CNNLSTMModel(seq_length=args.seq_length)
    

    print(f"Starting training {args.model} model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )

    print("Plotting training history...")
    plot_training_history(history)

    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device=args.device)

    # Save model
    model_path = f"{args.model}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Done!")

if __name__ == "__main__":
    main()