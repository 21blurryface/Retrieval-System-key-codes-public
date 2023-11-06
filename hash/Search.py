import numpy as np
import os
from torch.autograd import Variable
from hash.net import ReIDSE
import matplotlib.pyplot as plt
from PIL import Image


import torch
from hash.load import load_pic
import torch.optim.lr_scheduler
np.set_printoptions(threshold=np.inf)
def binary_output(input):
    net = ReIDSE(256)
    net.load_state_dict(torch.load('./model/0',map_location='cpu'))
    net.eval()
    output = torch.FloatTensor()
    for i, inputs in enumerate(input):
        inputs = Variable(inputs, volatile=True)
        binary, _ = net(inputs)
        output = torch.cat((output,binary.data),0)
    return torch.round(output)


def precision(binary,trn_binary):

    trn_binary = trn_binary.cpu().numpy()


    qbinary = binary.cpu().numpy() #<class 'numpy.ndarray'>
    qbinary = np.asarray(qbinary, np.int32) #<class 'numpy.ndarray'>
    qresult = np.count_nonzero(qbinary != trn_binary, axis=1) #<class 'numpy.ndarray'>
    #Hamming distance is a concept that represents the number of bits that correspond to two (same length) words that are different
    sort_indices = np.argsort(qresult) #Returns the subscript of an array sorted from smallest to largest (unordered)
    #outputfile.close()  # close file
    return sort_indices

def show(path):
    im = Image.open(path)
    plt.imshow(im)
    plt.show()
    return 0

def Sotu(filepath, count):
    train_binary = torch.load('./result/train_binary')
    pic_path = torch.load('./result/pic_path')
    pic = load_pic(filepath)
    qbinary = binary_output(pic)


    #Number of images to be returned
    sort = precision(qbinary, train_binary)
    filepath=[]
    for i in range(count):
        num = sort[i]
        fp = pic_path[num]
        filepath.append(fp)

    return filepath




