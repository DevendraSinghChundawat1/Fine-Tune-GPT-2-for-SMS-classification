# Fine-Tune GPT-2 for SMS Classification

## Overview
This project fine-tunes a **GPT-2** model to classify SMS messages as either **Spam** or **Ham** (Not Spam). The dataset used is the **SMS Spam Collection** dataset from UCI Machine Learning Repository. The project demonstrates how to download, preprocess, and balance the dataset, and then train a GPT-2 model for binary classification.

## Project Structure
```bash
├── Fine-Tune-GPT-2-for-SMS-classification
│   ├── train.csv                   # Training dataset
│   ├── validation.csv              # Validation dataset
│   ├── test.csv                    # Test dataset
│   ├── gpt_download.py             # Script to download GPT-2 model
│   ├── utils.py                    # Utility functions for the project
│   ├── gpt_classs_finetune.py      # Main script to run the Fine-tuning and classification
│   ├── LORA.py                     # Main script to run the Fine-tuning and classification using LORA  
│   ├── requirements.txt            # Required Python packages
│   ├── tests.py                    # Test script for running the model in test mode
│   ├── .gitignore                  # Git ignore file
│   └── README.md                   # Project documentation
```

## Setup Instructions

### Clone the Repository
To get started, clone the repository:
```bash
git clone https://github.com/your-username/Fine-Tune-GPT-2-for-SMS-classification.git
cd Fine-Tune-GPT-2-for-SMS-classification
```

### Install Dependencies
Ensure that you have **Python 3.8+** installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

### Download the Dataset
The dataset will be automatically downloaded and unzipped when you run the main script. However, if you need to download it manually, follow these steps:

1. Download the dataset from [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
2. Save it as `sms_spam_collection.zip` in the project root.
3. Extract the dataset, and it will be converted to a `.tsv` file named `SMSSpamCollection.tsv`.

<!-- ### Load Pretrained Model
This project uses a fine-tuned GPT-2 model, which is saved in the `saved_model` folder. The model is loaded using the path `Fine-Tune-GPT-2-for-SMS-classification/saved_model/review_classifier.pth`. You can customize the path or fine-tune the model by running the `gpt_classs_finetune.py` script.
 -->

### Training Parameters:
- **Train/Validation/Test Split:** 70/10/20
- **Batch Size:** 8
- **Learning Rate:** Set in the optimizer inside the script
- **Model Checkpoints:** Saved in `saved_model/`

## Data Preprocessing
The script balances the **spam** and **ham** categories and performs a **random split** into training, validation, and test sets. The **GPT-2 tokenizer** is used to encode SMS messages for the model.


## Model Training and Evaluation
The script tracks and prints:
- **Training Loss**
- **Validation Loss**
- **Training Accuracy**
- **Validation Accuracy**

Training can be monitored in real-time, and loss plots are generated after training.


## Testing the Model with `tests.py`
You can test the fine-tuned model using the `tests.py` script. This script runs the `gpt_class_finetune.py` in test mode to verify that the model runs without errors.

### Running the Test
To execute the test, run:
```bash
python -m pytest tests.py
```

## Output Files
- **Model Checkpoints:** The fine-tuned model is saved as `review_classifier.pth` (excluded in `.gitignore`).
- **Loss Plots:** Saved as `loss-plot.pdf`.


### LoRA Fine-tuning (LORA.py)

The `LORA.py` script fine-tunes the GPT model using the Low-Rank Adaptation (LoRA) technique. This method introduces additional trainable parameters while keeping most of the original model's parameters frozen, allowing efficient fine-tuning with fewer trainable weights.

#### Key Highlights:
- LoRA fine-tunes linear layers by introducing rank-reduced matrices and scaling with a configurable parameter (`alpha`).
- The `replace_linear_with_lora` function replaces standard `torch.nn.Linear` layers with `LinearWithLoRA` layers.
- The script loads a pre-trained GPT-2 model (configurable between various sizes like small, medium, large, xl).
- It fine-tunes the model for SMS spam classification using the UCI dataset.

#### How to Run:
```bash
python LORA.py
```

#### Important Parameters:
- `rank`: The rank of the LoRA matrices.
- `alpha`: A scaling factor to control the impact of the LoRA layers.


## Conclusion
This project demonstrates how to fine-tune a **GPT-2 model** for **SMS classification** and how to use **PyTorch** for loading, training, evaluating, and testing the model. The fine-tuned model achieves a good balance between **spam** and **ham** classification accuracy.
