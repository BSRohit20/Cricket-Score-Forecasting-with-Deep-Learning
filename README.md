
# Cricket Score Predictor 🏏

This project predicts the final cricket score of a team based on various match conditions using a neural network model. The model is built using TensorFlow and Keras, and an interactive UI is provided using `ipywidgets`.

---

## Features 🚀
- **Data Preprocessing**: Handles categorical data using label encoding and normalizes features using Min-Max scaling.
- **Neural Network**: A regression model with multiple dense layers optimized using the Huber loss function.
- **Interactive UI**: User-friendly dropdown menus to input match conditions and predict scores in real-time.
- **Customizable**: Built with modular components, making it easy to extend and enhance.

---

## Technologies Used 🛠️
- **Python**: Programming language.
- **Pandas & NumPy**: Data manipulation and preprocessing.
- **Scikit-learn**: Label encoding, scaling, and train-test splitting.
- **Keras & TensorFlow**: Neural network architecture and training.
- **Seaborn & Matplotlib**: Data visualization.
- **Ipywidgets**: Interactive user interface.

---

## Project Setup 🛠️

### Prerequisites
Ensure you have the following installed:
- Python (>=3.7)
- pip (Python package manager)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cricket-score-predictor.git
   cd cricket-score-predictor
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the `ipl_data.csv` file in the project directory.

---

## Usage 💻

### Training the Model
Run the script to preprocess the data, train the model, and save the results:
```bash
python train_model.py
```

### Interactive Prediction
Run the notebook to access the interactive UI:
1. Open Jupyter Notebook or JupyterLab.
2. Load `cricket_score_predictor.ipynb`.
3. Execute all cells to train the model and launch the interactive UI.
4. Use the dropdown menus to select match conditions and click "Predict Score."

---

## File Structure 📂
```
.
├── ipl_data.csv                # Dataset used for training and testing
├── train_model.py              # Script to preprocess data and train the model
├── cricket_score_predictor.ipynb # Jupyter Notebook with the interactive UI
├── README.md                   # Documentation
├── requirements.txt            # List of required Python packages
```

---

## Results 📊
- **Training Loss Visualization**: Track model performance during training.
- **Predicted Score**: Accurate prediction of cricket scores based on match conditions.

---


## Acknowledgments 🙏
- The IPL dataset is publicly available and used purely for educational purposes.
- Thanks to the open-source community for the amazing libraries used in this project.
```

### Notes:
1. Replace `yourusername` in the clone command with your GitHub username.
2. Update the file structure and dependencies in the `requirements.txt` as needed.
3. Include a `LICENSE` file in the repository if you wish to specify a license.
