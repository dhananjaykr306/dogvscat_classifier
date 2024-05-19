# dogvscat_classifier


# Cat vs Dog Classifier

This project is a Cat vs Dog classifier built using a deep learning model. The classifier is trained to distinguish between images of cats and dogs and can be used to classify new images.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.7 or later
- TensorFlow 2.x
- Keras
- NumPy
- OpenCV
- Matplotlib
- Jupyter Notebook (optional, for running the notebooks)

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/cat-vs-dog-classifier.git
    cd cat-vs-dog-classifier
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Dataset

The dataset used for training and evaluation is the [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765). Download the dataset and extract it into the `data/` directory within the project.

The directory structure should look like this:

```
cat-vs-dog-classifier/
├── data/
│   ├── PetImages/
│   │   ├── Cat/
│   │   └── Dog/
│   ├── train/
│   └── validation/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
│   └── predictions.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

## Training

1. Preprocess the data using the Jupyter Notebook `notebooks/data_preprocessing.ipynb`. This notebook will organize the images into training and validation sets.

2. Train the model using the `notebooks/model_training.ipynb` notebook or run the training script directly:

    ```sh
    python src/train.py
    ```

3. The trained model will be saved to the `models/` directory.

## Evaluation

Evaluate the model's performance using the `notebooks/model_evaluation.ipynb` notebook or the evaluation script:

```sh
python src/evaluate.py
```

This will provide metrics such as accuracy, precision, recall, and F1 score.

## Usage

To classify new images, use the `notebooks/predictions.ipynb` notebook or the prediction script:

```sh
python src/predict.py --image_path path_to_your_image.jpg
```

The script will output whether the image is predicted to be a cat or a dog.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Contact

For any questions or suggestions, please contact:

- Your Name - [your.email@example.com](mailto:dhananjaykr306@gmail.com)
- GitHub: [yourusername](https://github.com/dhananjaykr306)

---

Feel free to modify and use this README file as per your project's requirements.
