# AI Fruit-Ripeness Classifier

Computer-vision prototype that classifies the type and ripeness of a fruit from a photo. Built by Satoshi Squad for the Samsung Innovation Campus Hackathon (December 2025), as part of the Samsung Innovation Campus AI program.

## Problem

Classify fruit ripeness from images: 5 fruit types (apple, banana, mango, orange, papaya) in 3 states (green, ripe, rotten), for a total of 15 classes.

## Solution

A MobileNetV2 transfer-learning model trained on a web-collected image dataset, served through a Streamlit app. A user uploads a photo and gets the predicted fruit, its state, the model's confidence, and a chart of the top 5 predicted classes.

## Model

- **Base:** MobileNetV2 with ImageNet weights, without its top layer
- **Head:** global average pooling, a 1024-unit ReLU dense layer, and a 15-class softmax layer
- **Input:** 224 x 224 images scaled to [0, 1], batch size 16, 80/20 train/validation split
- **Stage 1:** base frozen, RMSprop, 10 epochs
- **Stage 2:** fine-tuning of the base from layer index 125 onward, SGD (learning rate 0.0001, momentum 0.9), 10 epochs
- **Loss:** categorical cross-entropy

## Reported results

The team's original notes report 98.75% training accuracy and 85.28% validation accuracy on a dataset of 1,188 images (about 79 per class). The dataset and trained model are not included in this repository, so these numbers cannot be reproduced from it alone.

## Streamlit app

`app.py` provides:

- Image upload (JPG, JPEG, PNG) with a preview
- A Predict button that shows the fruit, the ripeness state, and the confidence
- A Plotly bar chart of the top 5 predictions, with colors per fruit
- A button to choose another image

The interface text is in Spanish.

## Project structure

```
app.py                              Streamlit app
scripts/descargar_imagenes_bing.py  Downloads the dataset images (bing-image-downloader)
scripts/train_model.py              Trains and saves the model
scripts/predecir.py                 Command-line prediction
requirements.txt
```

`data/` (one folder per class) and `models/` are created locally and are not tracked.

## Run it

```bash
pip install -r requirements.txt

# 1. Download the dataset (about 120 images per class)
python scripts/descargar_imagenes_bing.py

# 2. Train the model
python scripts/train_model.py

# 3. Launch the app
streamlit run app.py

# Or predict from the command line
python scripts/predecir.py path/to/image.jpg
```

The dataset images come from web search results and are not redistributed here.

## Technologies

Python, TensorFlow, Keras, MobileNetV2, Streamlit, Plotly

## Team

| Member | Role |
|---|---|
| Nayim Rodriguez | Documentation |
| Hassan El Rada | Dataset preparation and backend |
| Li Chao Wu | Dataset preparation and backend |
| Diego Corrales | Front-end |
| Valentin Rodriguez | Documentation |

## My role

Front-end developer on a team of five. I developed the Streamlit interface with image upload, predictions, confidence scores, and Plotly visualizations, and I debugged the app and classification logic during prototype testing, fixing image-rendering issues and incorrect fruit and ripeness predictions.
