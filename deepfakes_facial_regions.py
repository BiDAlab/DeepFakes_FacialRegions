"""
BiDA Lab - Universidad Autónoma de Madrid
Author: Sergio Romero
Creation Date: 23/09/2021
Last Modification: 27/09/2021
-----------------------------------------------------
This code provides the implementation of two DeepFake detection systems (DSP-FWA and Capsule Networks) based on
different Facial Region, such as eyes or mouth, among others. The detector will predict whether a face belongs
to a fake or a real one. For more information, please visit https://github.com/BiDAlab/DeepFakes_FacialRegions
"""

# Import some libraries
import os
import argparse
import torch
from torchvision import transforms
from src.Capsule import model_big
from src.DSP import classifier
from torch.autograd import Variable
import torchvision.datasets as dset
import matplotlib.pyplot as plt


# This function parses the arguments
def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', default='DSP-FWA', help='CNN detector: DSP-FWA or Capsule')
    parser.add_argument('--region', default='entire', help='Facial region to use: entire, eyes, mouth, nose or rest')
    parser.add_argument('--ddbb', default='UADFV', help='Database: UADFV, FaceForensics, CelebDF or DFDC')

    opt = parser.parse_args()

    assert opt.detector == "DSP-FWA" or opt.detector == "Capsule", "CNN detector must be DSP-FWA or Capsule"
    assert opt.region == "entire" or opt.region == "eyes" or opt.region == "mouth" or opt.region == "nose" or opt.region == "rest", "Facial region must be: entire, eyes, mouth, nose or rest"
    assert opt.ddbb == "UADFV" or opt.ddbb == "FaceForensics" or opt.ddbb == "CelebDF" or opt.ddbb == "DFDC", "Database must be: UADFV, FaceForensics, CelebDF or DFDC"
    print(opt)
    return opt


# This function show the image with some details about the process (such as the model prediction, database used...)
def show_img(inputs):
    value, index = torch.max(inputs, 0)
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    inv_tensor = invTrans(value)
    plt.imshow(inv_tensor.permute(1, 2, 0).cpu())
    plt.title("Detector: " + model + "\nDatabase: " + opt.ddbb + "  -  Region: " + opt.region)
    plt.xlabel("Prediction (0 - Real, 1 - Fake):\n" + str(round(final_prediction, 2)))
    plt.show()


# Main function
if __name__ == "__main__":
    # Firstly, the arguments are parsed
    opt = parser_arguments()

    # Get the current path
    path = os.getcwd()

    # Get the model name
    model = opt.detector
    model_dir = path + "/models/" + opt.ddbb

    # Get the full path of the test directory
    test_dir = path + "/media/databases/" + opt.ddbb + "/" + opt.region

    # Add the transformations to the neural model (resize the image and normalization)
    transform_fwd = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Create the dataset in order to work with both images and neural network
    dataset_test = dset.ImageFolder(root=os.path.join(test_dir), transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1)

    # DSP-FWA detection system (from https://github.com/danmohaha/DSP-FWA)
    if model == "DSP-FWA":
        net = classifier.SPPNet(backbone=50, num_class=2)
        net = net.cuda(0)
        net.load_state_dict(torch.load(os.path.join(model_dir, 'DSP-FWA_' + opt.ddbb + "_" + opt.region + '.pt')))
        net.eval()

        inputs, labels = next(iter(dataloader_test))

        inputs = inputs.cuda(0)
        labels = labels.cuda(0)

        outputs = net(inputs)
        pred_prob = torch.softmax(outputs, dim=1)

        pred_prob = pred_prob.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        final_prediction = pred_prob[0][0]

    else:  # Capsule Network detection system (from https://github.com/nii-yamagishilab/Capsule-Forensics-v2)
        vgg_ext = model_big.VggExtractor()
        capnet = model_big.CapsuleNet(2, 0)

        capnet.load_state_dict(torch.load(os.path.join(model_dir, 'Capsule_' + opt.ddbb + "_" + opt.region + '.pt')))
        capnet.eval()

        vgg_ext.cuda(0)
        capnet.cuda(0)

        inputs, labels = next(iter(dataloader_test))

        inputs = inputs.cuda(0)
        labels = labels.cuda(0)

        input_v = Variable(inputs)

        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=False)
        output_dis = class_.data.cpu()
        pred_prob = torch.softmax(output_dis, dim=1)
        pred_prob = pred_prob.cpu().detach().numpy()
        final_prediction = pred_prob[0][0]

    # Finally the image is shown
    show_img(inputs)
