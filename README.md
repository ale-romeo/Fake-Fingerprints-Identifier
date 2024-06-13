# ML Project: Fingerprint Spoofing Detection

## Project Overview

This project focuses on a binary classification problem: fingerprint spoofing detection. The goal is to identify genuine versus counterfeit fingerprint images using a dataset with labeled samples. The dataset contains features extracted from fingerprint images and includes both genuine (True, label 1) and fake (False, label 0) samples.

## Dataset

The dataset used in this project consists of 6-dimensional feature vectors. The training data is stored in `Project/trainData.txt`, a CSV file where each row represents a sample with the first 6 values being the features and the last value representing the class label (1 or 0). The samples are not ordered.

## Project Tasks

### 1. Exploratory Data Analysis
- Load the dataset.
- Plot histograms and pair-wise scatter plots of the features.
- Analyze the distribution and overlap of classes for different feature pairs.

### 2. PCA and LDA Application
- Apply PCA to reduce dimensionality and analyze the variance captured by each principal component.
- Apply LDA to find the best linear separation between the two classes.
- Evaluate the performance of classifiers using PCA and LDA.

### 3. Logdensity Estimation
- Fit uni-variate Gaussian models to the features.
- Plot the distribution densities on top of normalized histograms.
- Analyze the goodness of fit for Gaussian models.

### 4. Training and Validation
- Split the dataset into training and validation sets.
- Train and evaluate different models: MVG, tied Gaussian, and Naive Bayes Gaussian.
- Compute log-likelihood ratios and evaluate classifier performance.

### 5. DCF and minDCF Evaluation
- Analyze classifier performance under different prior probabilities and cost scenarios.
- Compute the actual and minimum Decision Cost Function (DCF) for various applications.

### 6. Binary Logistic Regression
- Train logistic regression models with different regularization parameters.
- Evaluate the impact of regularization on classification performance.
- Analyze the effects of data preprocessing techniques like centering and PCA.

## Project Structure

ML-Project/

│

├── Project/

│ ├── trainData.txt # Training dataset

│ ├── project.py # Main project code

│ ├── models/ # Saved models

│ ├── plots/ # Saved plots

│

├── README.md # Project README file

├── Report.tex # LaTeX report


## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- LaTeX (for compiling the report)

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ML-Project.git
    cd ML-Project
    ```

2. Run the project code:
    ```bash
    python Project/project_code.py
    ```

3. Compile the LaTeX report:
    ```bash
    pdflatex report.tex
    ```

## Contributions

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

