from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from .util import *
import os 
import os.path as osp
from .darknet import Darknet
from .preprocess import prep_image, inp_to_image
import random 
import pickle as pkl
import itertools

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("yolov3/dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    return img_


def prepare_model(cfg='yolov3/cfg/yolov3.cfg', weights='yolov3/yolov3.weights', reso='416'):
    #Set up the neural network
    print("Loading yolo network...")
    model = Darknet(cfg)
    model.load_weights(weights)
    print("Loaded yolo network...")
    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    
    CUDA = torch.cuda.is_available()

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    #Set the model in evaluation mode
    model.eval()
    return model, inp_dim

def yolo(images, model, inp_dim, det='det', bs=1, confidence=0.5, nms_thesh=0.4, scales='1,2,3', ):
    start = 0
    num_classes = 80
    classes = load_classes('yolov3/data/coco.names') 
    CUDA = torch.cuda.is_available()



    #Detection phase
    try:
        imlist = [images]
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    if not os.path.exists(det):
        os.makedirs(det)
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
      
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    batch_size = bs
    if (len(im_dim_list) % batch_size):
        leftover = 1
        
        
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        

    i = 0

    write = False
    model(get_test_input(inp_dim, CUDA), CUDA)
    
    start_det_loop = time.time()
    
    #Detected objects
    objs = {}
       
    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        
        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
        
#        prediction = prediction[:,scale_indices]

        
        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 
        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
     
        if type(prediction) == int:
            i += 1
            continue

        end = time.time()                

        prediction[:,0] += i*batch_size  
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]

        i += 1
        
        if CUDA:
            torch.cuda.synchronize()
    
    try:
        output
    except NameError:
        print("No detections were made")
        exit()
        
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
       
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
            
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
    detections = []

    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        x = c1[0].item()
        y = c1[1].item()
        w = int(c2[0].item())-int(c1[0].item())
        h = int(c2[1].item())-int(c1[1].item())
        detections.append((x,y,w,h))
        return x,y,w,h
    
            
    list(map(lambda x: write(x, im_batches, orig_ims), output))
    
    end = time.time()
    mapped = set(zip(detections, objs))
    torch.cuda.empty_cache()
    return mapped
