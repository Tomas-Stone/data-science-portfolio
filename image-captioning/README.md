# Image Captioning Project

This README file provides a detailed overview of the Image Captioning project, focusing on generating descriptive captions for images using a neural network architecture. Below are the objectives, features, implementation details, and results of the project.

---

## Project Overview

The project implements an image captioning model combining convolutional and recurrent neural networks. It leverages a pre-trained VGG16 model for feature extraction from images and a custom LSTM-based sequence generator for generating captions. The captions are evaluated using BLEU scores to measure their accuracy compared to the ground truth.

---

## Features

- Pre-trained VGG16 network for extracting high-level image features.
- Tokenization and embedding of textual captions for sequence modeling.
- Encoder-decoder architecture for generating captions.
- Evaluation metrics include BLEU-1 and BLEU-2 scores.
- Visualization of predictions alongside images and actual captions.

---

## Dataset

- **Dataset Name**: Flickr8k
- **Structure**:
  - `Images/`: Contains all the image files.
  - `captions.txt`: Provides image IDs and their corresponding captions.

---

## Dependencies

The following Python libraries are required:

- numpy
- tqdm
- tensorflow
- matplotlib
- nltk
- Pillow

Install these libraries using:

```bash
pip install numpy tqdm tensorflow matplotlib nltk Pillow
```

---

## Implementation Details

### 1. Feature Extraction

- **Model**: VGG16 pre-trained on ImageNet.
- **Process**:
  - Images are resized to 224x224.
  - Extract features from the penultimate layer of VGG16.
  - Features are saved in `features.pkl` for reuse.

### 2. Caption Preprocessing

- Captions from `captions.txt` are cleaned and preprocessed:
  - Converted to lowercase.
  - Special characters are removed.
  - Added start (`tagstart`) and end (`endtag`) tokens.
- Captions are tokenized using Keras's `Tokenizer`.

### 3. Data Preparation

- Dataset split into training (90%) and testing (10%) sets.
- Data generator creates batches for training using extracted features and processed captions.

### 4. Model Architecture

- **Image Encoder**:
  - Input: 4096-dimensional features from VGG16.
  - Layers: Dropout and Dense.
- **Caption Decoder**:
  - Input: Tokenized and embedded captions.
  - Layers: Dropout, LSTM.
- **Decoder Merging**:
  - Combines encoder and decoder outputs.
  - Final Dense layer predicts the next word.

### 5. Training

- Optimizer: Adam.
- Loss Function: Categorical crossentropy.
- Trained for 15 epochs with a batch size of 64.

### 6. Evaluation

- BLEU scores used to evaluate caption quality:
  - BLEU-1: Measures unigram precision.
  - BLEU-2: Measures bigram precision.

### 7. Caption Generation

- Captions are generated using greedy search.
- Predictions can be compared with actual captions for evaluation.

---

## Instructions

### 1. Setting Up the Dataset

- Ensure the following structure:
  - `BASE_DIR/Images`: Contains image files.
  - `BASE_DIR/captions.txt`: Contains the captions.

### 2. Running the Code

- Run the script to:
  - Extract image features.
  - Process captions.
  - Train the model.
  - Generate and evaluate captions.

### 3. Generating Captions

- Use the `generate_caption()` function to predict captions for a specific image:
  ```python
  generate_caption("1000268201_693b08cb0e.jpg")
  ```

### 4. Visualization

- Display the image with actual and predicted captions.

---

## Results

### Example Output

**Image ID**: 1000268201\_693b08cb0e.jpg

**Actual Captions**:

- tagstart child in pink dress is climbing up set of stairs in an entry way endtag
- tagstart girl going into wooden building endtag
- tagstart little girl climbing into wooden playhouse endtag
- tagstart little girl climbing the stairs to her playhouse endtag
- tagstart little girl in pink dress going into wooden cabin endtag

**Predicted Caption**:

- tagstart little girl in purple shirt is standing in front of wooden cabin endtag

**Evaluation Metrics**:

- BLEU-1: `0.556230`
- BLEU-2: `0.334852`

---

## Future Improvements

- Experiment with beam search for more accurate predictions.
- Use a larger dataset like MS-COCO for better generalization.
- Explore transformer-based architectures for improved results.

---

## References

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [BLEU Score](https://en.wikipedia.org/wiki/BLEU)


