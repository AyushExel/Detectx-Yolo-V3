from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from DNModel import net
from img_process import preprocess_img, inp_to_image
import pandas as pd
import random 
import pickle as pkl

        


def arg_parse():

    parser = argparse.ArgumentParser(description='YOLOv3 ')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing input images",
                        default = "images", type = str)
    parser.add_argument("--result", dest = 'result', help = 
                        " Directory to store results ",
                        default = "result", type = str)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)

    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Detection Confidence ", default = 0.5)
    parser.add_argument("--cfg", dest = 'configfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'resolution', help = 
                        "Input resolution of the network",
                        default = "256", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()
    
    scales = args.scales
    
 
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('data/coco.names') 

    model = net(args.configfile)
    model.load_weights(args.weightsfile)
    print("Network loaded")
    
    model.DNInfo["height"] = args.resolution
    in_dim = int(model.DNInfo["height"])


    if CUDA:
        model.cuda()
    model.eval()
    
    read_dir = time.time()
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No with the name {}".format(images))
        exit()
        
    if not os.path.exists(args.result):
        os.makedirs(args.result)
            
    batches = list(map(preprocess_img, imlist, [in_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    #Explain
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
    
    
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1
        

    i = 0
    

    write = False
    
    
    objs = {}
    
    
    
    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()
        #print('batch size => ', batch.size())
        with torch.no_grad():
            prediction = model(batch, CUDA)
        

        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        
        if type(prediction) == int:
            i += 1
            continue

            
        #Add the current batch number
        prediction[:,0] += i*batch_size
        
    
            
          
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))
            
        
        

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        
        if CUDA:
            torch.cuda.synchronize()
    
    try:
        output
    except NameError:
        print("No detections were made")
        exit()
        
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(in_dim/im_dim_list,1)[0].view(-1,1)
    
    
    output[:,[1,3]] -= (in_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (in_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
    colors = pkl.load(open("pallete", "rb"))

    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img
    
            
    list(map(lambda x: write(x, im_batches, orig_ims), output))
      
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.result,x.split("\\")[-1]))
    
    list(map(cv2.imwrite, det_names, orig_ims))

    torch.cuda.empty_cache()
    
    
        
        
    
    
