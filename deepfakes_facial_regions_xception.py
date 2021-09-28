"""
BiDA Lab - Universidad Autónoma de Madrid
Author: Sergio Romero
Creation Date: 23/09/2021
Last Modification: 28/09/2021
-----------------------------------------------------
This code provides the implementation of the Fake Detection System using Xception Networks and based on
different Facial Region, such as eyes or mouth, among others. The detector will predict whether a face belongs
to a fake or a real one. For more information, please visit https://github.com/BiDAlab/DeepFakes_FacialRegions
"""

# Import some libraries
import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image


# This function parses the arguments
def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default='entire', help='Facial region to use: entire, eyes, mouth, nose or rest')
    parser.add_argument('--ddbb', default='UADFV', help='Database: UADFV, FaceForensics, CelebDF or DFDC')

    opt = parser.parse_args()

    assert opt.region == "entire" or opt.region == "eyes" or opt.region == "mouth" or opt.region == "nose" or opt.region == "rest", "Facial region must be: entire, eyes, mouth, nose or rest"
    assert opt.ddbb == "UADFV" or opt.ddbb == "FaceForensics" or opt.ddbb == "CelebDF" or opt.ddbb == "DFDC", "Database must be: UADFV, FaceForensics, CelebDF or DFDC"
    print(opt)
    return opt


# This function show the image with some details about the process (such as the model prediction, database used...)
def show_img(img):
    plt.imshow(img)
    plt.title("Detector: Xception\nDatabase: " + opt.ddbb + "  -  Region: " + opt.region)
    plt.xlabel("Prediction (0 - Real, 1 - Fake):\n" + str(round(final_prediction, 2)))
    plt.show()


# Main function
if __name__ == "__main__":
    # Firstly, the arguments are parsed
    opt = parser_arguments()

    # Get the current path
    path = os.getcwd()

    # Get the model name
    model_dir = path + "/models/" + opt.ddbb

    # Get the full path of the test directory
    test_dir = path + "/media/databases/" + opt.ddbb + "/" + opt.region

    # Xception detection system
    model = models.load_model(os.path.join(model_dir, 'Xception_' + opt.ddbb + "_" + opt.region + '.hdf5'))

    # Load the test image and normalize it
    img_path = glob.glob(test_dir + "/**/*.jpg", recursive=True)
    img = image.load_img(img_path[0], target_size=(200, 200))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # The model predicts between 0 and 1
    preds = model.predict(x)
    final_prediction = preds[0][0]

    # Finally the image is shown
    show_img(img)
