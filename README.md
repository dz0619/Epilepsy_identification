# Epilepsy Identification System

A deep learning-based EEG epilepsy seizure detection system using CNN, LSTM, and CNN-LSTM hybrid models for binary classification tasks.

## Project Structure

```
Epilepsy_identification/
├── main.py           # Main program entry, includes argument parsing and model training workflow
├── models.py         # Neural network model definitions (CNN, LSTM, CNN-LSTM)
├── data_loader.py    # Data loading and preprocessing module
├── train.py          # Model training, evaluation, and visualization functions
└── README.md         # Project documentation
```

## Requirements

### Python Version
- Python 3.7+

### Dependencies
```
torch>=1.8.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0
```


**Note**: This system is for research and educational purposes only and should not be used as the sole basis for medical diagnosis. Clinical applications require professional medical judgment and validation.
