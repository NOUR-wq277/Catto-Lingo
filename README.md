🐱 Cat Emotion Detector

Developed by: Abdalrhman Badawi & Ziad Sakr
Base Model: ResNet50
Experiment Tracking: MLflow

📘 Project Overview

This project aims to classify cat emotions using deep learning on image data.
The model can identify nine emotional states:

angry, disgusted, happy, normal, relaxed, sad, scared, surprised, uncomfortable

The model was trained using a ResNet50 backbone, fine-tuned on a custom dataset of labeled cat facial expressions.
All experiments, metrics, and evaluation reports were logged using MLflow for full reproducibility.

🧱 Project Structure
Catto-Lingo/
└── cat-emotion-detector/
    ├── data/                   # Dataset or DVC tracking files
    ├── src/
    │   ├── preprocess/         # Data loading and augmentations
    │   ├── models/             # Model architectures (ResNet, etc.)
    │   ├── train.py            # Training loop
    │   ├── evaluate.py         # Model evaluation and metrics
    │   └── utils.py            # Helper functions
    ├── notebooks/              # Optional Jupyter notebooks
    ├── results/
    │   ├── confusion_matrix.png
    │   ├── metrics.csv
    │   └── model_report.txt
    ├── requirements.txt        # Dependencies
    ├── main.py                 # Entry point (training + evaluation)
    ├── README.md               # Documentation
    └── .gitignore

⚙️ Setup & Installation

Clone the repository:

git clone https://github.com/NOUR-wq277/Catto-Lingo.git
cd Catto-Lingo/cat-emotion-detector


Install dependencies:

pip install -r requirements.txt


(Optional) Configure MLflow tracking URI in main.py:

mlflow.set_tracking_uri("file:./mlruns")

🚀 Training

To train the model and log results in MLflow:

python main.py --mode train


You can customize:

--epochs → number of epochs

--batch_size → training batch size

--lr → learning rate

Example:

python main.py --mode train --epochs 30 --batch_size 16 --lr 0.0001

📊 Evaluation

To evaluate a trained model and view metrics:

python main.py --mode test


Then open MLflow UI to visualize results:

mlflow ui


Open the local MLflow dashboard at: http://localhost:5000

📈 Results Summary
Metric	Value
Accuracy	0.8915
Macro F1-Score	0.8406
Dataset Size	1051 test images

Key observations:

The model performs consistently well across most classes.

Some underrepresented emotions (like no clear emotion) show slightly lower F1 scores.

Further tuning or data balancing can improve performance.

🧠 Future Improvements

Integrate Grad-CAM visualization to explain predictions.

Use EfficientNet for potentially higher accuracy.

Balance emotion classes with augmentation or synthetic data.

🏁 Acknowledgments

This project was developed as part of a research/training effort on visual emotion recognition for animals.
Special thanks to mentors and collaborators who supported the work.
