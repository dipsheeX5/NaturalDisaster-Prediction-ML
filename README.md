# NaturalDisaster-Prediction-ML
Hybrid ML model for classifying natural disaster images (e.g., earthquakes, floods) using MobileNetV2 for feature extraction and classical classifiers like SVM, XGBoost, and Neural Networks. Built with Python, TensorFlow/Keras, and Scikit-learn for reproducible disaster prediction and evaluation.


## Project Overview

This project uses a pre-trained MobileNetV2 to extract deep features (embeddings) from disaster images, then trains and compares three different classifiers:

- Support Vector Machine (SVM)
- XGBoost
- Neural Network (simple MLP)

The goal is to accurately classify images into four disaster categories while comparing traditional ML vs neural network performance on extracted features.

## Dataset

- Images organized in folders: `cyclone/`, `earthquake/`, `flood/`, `wildfire/`
- Images resized to 224×224 pixels
- Dataset source: public disaster image collections (e.g. Kaggle or custom-curated)

## Key Features

- Deep feature extraction with MobileNetV2 (transfer learning)
- Multiple classifiers: SVM, XGBoost, Neural Network
- Hyperparameter tuning (GridSearchCV)
- Stratified K-Fold cross-validation
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC)
- Model saving/loading (joblib + Keras)
- Visual comparison of predictions on sample images

## Technologies Used

- Python 3.12+
- TensorFlow / Keras (MobileNetV2)
- scikit-learn, xgboost
- numpy, pandas, matplotlib, seaborn
- OpenCV, PIL
- Jupyter Notebook

## Project Structure
Disaster_AI/
├── embeddings_cache/               # Cached MobileNetV2 embeddings
├── models/                         # Saved trained models & metadata
├── notebook_disaster_hybrid.ipynb  # Complete analysis & modeling notebook
├── README.md
└── (your image folders: cyclone/, earthquake/, flood/, wildfire/)


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Disaster_AI.git
   cd Disaster_AI

2. Install required packages:
pip install tensorflow keras scikit-learn xgboost numpy pandas matplotlib seaborn tqdm joblib opencv-python pillow
3. Place your images in the appropriate class folders
4. Open and run the notebook:
jupyter notebook notebook_disaster_hybrid.ipynb

## Results (example – update after training)

Best model: [e.g. XGBoost / Neural Network]
Test Accuracy: [XX.XX%]
Macro F1-Score: [0.XX]

## Future Improvements

Advanced data augmentation
More powerful backbone (EfficientNet, ResNet, Vision Transformer)
Model ensembling
Real-time inference deployment (Flask / FastAPI / Streamlit)

## License
MIT License
Feel free to use, modify, and contribute!