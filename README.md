## Deep Learning for German Traffic Sign Recognition
In this project, I built a neural network to perform image classification on the GTSRB dataset. The model managed to achieve a test accuracy of **99.24%**. Additionally, I also deployed the model locally by creating an API for it using FastAPI.

- The  [German Traffic Sign Recognition](https://benchmark.ini.rub.de/gtsrb_dataset.html) Benchmark (GTSRB) dataset contains 39,209 training images and 12,630 test images of 43 different kinds of road signs.
- **OpenCV** was used to pre-process both the training and test images.
- Extensive hyptertuning was done to identify the optimal model architecture and other hyperparameters. The model was easily prone to overfitting, which meant I had to strike a balance between keeping the model simple enough while also making sure that the model captured the underlying patterns in the images.
- *Batch normalization* was implemented. The model was trained using *T4 GPU* from Google Colab. *Learning rate scheduling* was also implemented to halve the learning rate every 10 epochs.
- Although the model achieved close to 100% validation accuracy by 10th epoch, test accuracy at this point was only 97-98% and the validation loss continued to decrease. Hence the model was trained for 20 more epochs and it finally achieved a test accuracy of **99.24%**.
- Applying image augmentations to increase the size of the dataset or implementing regularization did not improve the test accuracy.
- To ensure uniform performance across all the classes, class-wise accuracies were computed. The results demonstrated exceptional performance across all classes, with the model even achieving a perfect **100% accuracy** for half of the classes in the test data.

### API:
- I deployed the trained model locally using FastAPI. Users can upload multiple image files as part of the POST request to the API.
- The API processes the images, uses the trained model to make predictions, and returns a JSON response. In this response, the keys represent file names, and the corresponding values are dictionaries containing information about the predicted class and confidence score.

Sample JSON Response:
```
{
    "00053.ppm": {
        "class": 9,
        "confidence": 1.0
    },
    "00083.ppm": {
        "class": 7,
        "confidence": 1.0
    },
    "00191.ppm": {
        "class": 29,
        "confidence": 1.0
    },
    "00410.ppm": {
        "class": 25,
        "confidence": 1.0
    },
    "00793.ppm": {
        "class": 3,
        "confidence": 1.0
    }
}
```
