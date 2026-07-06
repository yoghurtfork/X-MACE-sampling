# Data sampling for X-MACE transfer learning

This project investigates different sampling techniques for transfer learning for X-MACE. 

The workflow is
- Train a base model with a large, lower-fidelity dataset
- Use different sampling techniques to select a small, high-fidelity dataset
- Do transfer learning to fine-tune the base model
- Compare model performance

## Dependencies
Python 3.11
MACE
NumPy
ASE
scikit-learn
scikit-matter