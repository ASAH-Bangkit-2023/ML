# Asah

ASAH is an application to encourage people to manage their waste properly by giving them rewards if they successfully manage their waste properly, where they need to sort their waste first and then they can dispose of or give their waste to recycling agencies around them

## Machine Learning: Image Classification for Waste Sorting

This project utilizes machine learning to facilitate the waste sorting process. We have developed an image classification model that accurately identifies the type of waste in a given image. The model, based on MobileNetV2, is trained to categorize waste into 10 distinct classes.

The classes are:

1. Shoes
2. Metal
3. Plastic
4. Glass
5. Clothes
6. Paper
7. Trash
8. Battery
9. Biological
10. Cardboard

## Libraries Used

This project utilizes several Python libraries for data handling, machine learning, and visualization:

| Library       | Purpose       |
| ------------- |:-------------:|
| `os`          | Provides functions for interacting with the operating system. |
| `shutil`      | Used for high-level file operations. |
| `zipfile`     | Allows the reading and writing of ZIP-format archives. |
| `pathlib`     | For manipulating filesystem paths. |
| `random`      | Generates random numbers, selects random elements from lists. |
| `cv2`         | OpenCV for image and video processing. |
| `numpy`       | Enables numerical computing with powerful numerical arrays objects, and routines to manipulate them. |
| `tensorflow`  | An open-source platform for machine learning. |
| `matplotlib.pyplot` | Used for creating static, animated, and interactive visualizations in Python. |

## Documentation

### Data Acquisition

The data for this project is sourced from Kaggle. The dataset comprises images representing the 10 classes of waste mentioned above. In total, there are 15,515 images in the dataset.

### Model Training

We used TensorFlow to train our image classification model. The MobileNetV2 architecture, pre-trained on the ImageNet dataset, serves as the backbone of our model. This enables the model to extract useful features from the waste images and classify them accurately. 

### Using the Model

The trained model can be used by ASAH app users to classify their waste. Users simply need to take a picture of the waste item, and the app will tell them which category it belongs to. This assists users in sorting their waste correctly, leading to more efficient recycling and waste management. 

## Future Work

We aim to continue improving the accuracy of our waste classification model and add more classes to cover a wider variety of waste items. We also plan to develop partnerships with recycling agencies and other relevant bodies to increase the reach and impact of our app.

## Credits

This project utilizes a dataset from Kaggle: [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

ASAH is a step towards a cleaner, greener future. Let's manage our waste wisely and save our planet!
