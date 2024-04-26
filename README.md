# Facial Recognition Project

This project is a Python program that allows you to train a facial recognition model to identify individuals based on their captured images. It provides a user-friendly menu-based interface for capturing images, training the model, and identifying persons.

## Features

- Capture images for two individuals using the webcam
- Train a facial recognition model using the captured images
- Save the trained model with a personalized filename
- Identify individuals using the trained model
- Error handling and user feedback through loading bars and menus

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn
- pickle

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/facial-recognition-project.git
```

2. Install the required dependencies:

```
pip install opencv-python numpy scikit-learn
```

## Usage

1. Run the Python script:

```
python facial_recognition.py
```

2. The program will display a menu with the following options:
```
   1. Capture images for person 1
   2. Capture images for person 2
   3. Train the model
   4. Identify person
   5. Quit
```
3. Select option 1 or 2 to capture images for the respective individuals. The program will prompt you to enter the person's name and capture 100 images using the webcam.

4. After capturing images for both individuals, select option 3 to train the facial recognition model. The program will prompt you to enter names for person 1 and person 2, and it will save the trained model with a personalized filename.

5. To identify a person, select option 4. The program will display a list of available trained models. Enter the number corresponding to the model you want to use, and the program will capture an image from the webcam and identify the person based on the loaded model.

6. To exit the program, select option 5.

## Model Saving and Loading

- The trained models are saved in the same directory as the Python script.
- The model filenames are generated based on the names entered for person 1 and person 2 during the training process.
- When identifying a person, the program automatically detects the available model files and presents them as options to choose from.

## Customization

- You can adjust the number of images captured for each person by modifying the `num_images` parameter in the `capture_images` function.
- The `train_test_split` function splits the data into training and testing sets. You can modify the `test_size` parameter to change the proportion of data used for testing.
- The `KNeighborsClassifier` is used as the machine learning algorithm for facial recognition. You can experiment with different algorithms or adjust the hyperparameters to improve the model's performance.

## Limitations

- The program assumes that only two individuals are being recognized. If you want to recognize more individuals, you need to modify the code accordingly.
- The accuracy of the facial recognition model depends on the quality and quantity of the captured images. Ensure that the images are clear, well-lit, and capture various angles and expressions of the individuals.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The project utilizes the OpenCV library for image capture and processing.
- The scikit-learn library is used for training the facial recognition model.

Feel free to contribute to this project by submitting pull requests or reporting issues on the GitHub repository.
