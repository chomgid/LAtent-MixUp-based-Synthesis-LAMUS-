# LAtent Mixup-based Synthesis (LAMUS)

This repository contains the official Python package implementation of the paper **LAtent Mixup-based Synthesis (Tentative)**.

## Getting Started with LAMUS
### Running LAMUS

1. **Set Input Data**  
   Place your time-series dataset under the `TSdata/{dataset}` directory.  
   The dataset should be stored as a pickle file named `{dataset}.pkl` and formatted as either:
   - `(num_samples, sequence_length, num_channels)`, or
   - `(num_samples, sequence_length)` (automatically expanded to 3D).

2. **Run LAMUS**  
   Execute the main script `main.py`, which loads the dataset, fits the LAMUS model, and generates synthetic time-series data:
   ```bash
   python main.py --dataset electricity --centering 
   ```

3. **Output**  
    The generated synthetic data is saved as a pickle file under `TSexperiments/dstspca/{dataset}/` directory.

## LAMUS Parameters

LAMUS is implemented through the `dsts` class, which performs PCA-based decomposition, latent mixup–based synthesis, and calibration for time series data.

### Centering Strategy

LAMUS supports multiple centering strategies to control how mean structures are handled before synthesis (`--centering`):

- **sample_wise**  
  Each time series is centered by subtracting its own temporal mean.  

- **feature_wise**  
  Centering is performed across samples at each time step.  

- **double**  
  Applies both sample-wise and feature-wise centering sequentially.


### Synthesis Variants

LAMUS supports two synthesis modes controlled by the `--pca_mixup` flag.

- **PCA-Mixup (default)**  
  Performs mixup directly in the PCA latent space.

- **PCA-NN**  
  PCA representations are used only to identify neighboring samples.

## Acknowledgement

This repository contains code adapted from the TSGBench project
(https://github.com/YihaoAng/TSGBench).

While TSGBench provides a general benchmarking framework for time series
evaluation, the current implementation adapts the evaluation pipeline
by adding and replacing evaluation methods to align with our proposed time series generation (and evaluation) framework. 

## License

This project is licensed under the terms of the [MIT License](LICENSE).
