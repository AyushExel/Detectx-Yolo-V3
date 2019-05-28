from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2 
from util import *



class dummyLayer(nn.Module):
    def __init__(self):
        super(dummyLayer, self).__init__()
        

class detector(nn.Module):
    def __init__(self, anchors):
        super(detector, self).__init__()
        self.anchors = anchors


def construct_cfg(configFile):
    '''
    Build the network blocks using the configuration file.
    Pre-process it to form easy to manupulate using pytorch.
    ''' 
    
    # Read and pre-process the configuration file
    
    config = open(configFile,'r')
    file = config.read().split('\n')
    
    file = [line for line in file if len(line) > 0 and line[0]!= '#']
    file = [line.lstrip().rstrip() for line in file]
    
    
    #Separate network blocks in a list 
    
    networkBlocks = [] 
    networkBlock = {}

    for x in file:
        if x[0] == '[':
            if len(networkBlock) != 0:
                networkBlocks.append(networkBlock)
                networkBlock = {}
            networkBlock["type"] = x[1:-1].rstrip()
        else:
            entity , value = x.split('=')
            networkBlock[entity.rstrip()] = value.lstrip()
    networkBlocks.append(networkBlock)
    
    return networkBlocks
    

def buildNetwork(networkBlocks):
    DNInfo = networkBlocks[0]
    modules = nn.ModuleList([])
    channels = 3
    filterTracker = [] 

    for i,x in enumerate(networkBlocks[1:]):
        seqModule  = nn.Sequential()
        if (x["type"] == "convolutional"):

            filters= int(x["filters"])
            pad = int(x["pad"])
            kernelSize = int(x["size"])
            stride = int(x["stride"])

            if pad:
                padding = (kernelSize - 1) // 2
            else:
                padding = 0
            
            activation = x["activation"]
            try:
                bn = int(x["batch_normalize"])
                bias = False
            except:
                bn = 0
                bias = True

            conv = nn.Conv2d(channels, filters, kernelSize, stride, padding, bias = bias)
            seqModule.add_module("conv_{0}".format(i), conv)

            if bn:
                bn = nn.BatchNorm2d(filters)
                seqModule.add_module("batch_norm_{0}".format(i), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                seqModule.add_module("leaky_{0}".format(i), activn)


        elif (x["type"] == "upsample"):
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            seqModule.add_module("upsample_{}".format(i), upsample)
        
        elif (x["type"] == "route"):
            x['layers'] = x["layers"].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end =0
            
            if start > 0:
                start = start - i 
            if end > 0:
                end = end - i
            
            route = dummyLayer()
            seqModule.add_module("route_{0}".format(i),route)
            if end < 0:
                filters = filterTracker[i+start] + filterTracker[i+end]
            else:
                filters = filterTracker[i+start]
        elif (x["type"] == "shortcut"):
            shortcut = dummyLayer()
            seqModule.add_module("shortcut_{0}".format(i),shortcut)
        elif (x["type"] == "yolo"):
            anchors = x["anchors"].split(',')
            anchors = [int(a) for a in anchors]
            masks = x["mask"].split(',')
            masks = [int(a) for a in masks]
            anchors = [(anchors[j],anchors[j+1]) for j in range(0,len(anchors),2)]
            anchors = [anchors[j] for j in masks]
            detectorLayer = detector(anchors)

            seqModule.add_module("Detection_{0}".format(i),detectorLayer)
        
        modules.append(seqModule)
        channels = filters
        filterTracker.append(filters)
    return (DNInfo, modules)



class net(nn.Module):
    def __init__(self, cfgfile):
        super(net, self).__init__()
        self.netBlocks = construct_cfg(cfgfile)
        self.DNInfo, self.moduleList = buildNetwork(self.netBlocks)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
                
    def forward(self, x, CUDA):
        detections = []
        modules = self.netBlocks[1:]
        layerOutputs = {}  
        
        
        written_output = 0
        #Iterate throught each module 
        for i in range(len(modules)):        
            
            module_type = (modules[i]["type"])
            #Upsampling is basically a form of convolution
            if module_type == "convolutional" or module_type == "upsample" :
                
                x = self.moduleList[i](x)
                layerOutputs[i] = x

            #Add outouts from previous layers to this layer
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]
                
                #If layer nummber is mentioned instead of its position relative to the the current layer
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = layerOutputs[i + (layers[0])]

                else:
                    #If layer nummber is mentioned instead of its position relative to the the current layer
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                        
                    map1 = layerOutputs[i + layers[0]]
                    map2 = layerOutputs[i + layers[1]]
                    
                    
                    x = torch.cat((map1, map2), 1)
                layerOutputs[i] = x
            
            #ShortCut is essentially residue from resnets
            elif  module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = layerOutputs[i-1] + layerOutputs[i+from_]
                layerOutputs[i] = x
                
            
            
            elif module_type == 'yolo':        
                
                anchors = self.moduleList[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.DNInfo["height"])
                
                #Get the number of classes
                num_classes = int (modules[i]["classes"])
                
                #Output the result
                x = x.data
                print("Size before transform => " ,x.size())
                
                #Convert the output to 2D (batch x grids x bounding box attributes)
                x = transformOutput(x, inp_dim, anchors, num_classes, CUDA)
                print("Size after transform => " ,x.size())

                
                #If no detections were made
                if type(x) == int:
                    continue

                
                if not written_output:
                    detections = x
                    written_output = 1
                
                else:
                    detections = torch.cat((detections, x), 1)
                
                layerOutputs[i] = layerOutputs[i-1]
                
        
        try:
            return detections
        except:
            return 0

            
    def load_weights(self, weightfile):
        
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        

        weights = np.fromfile(fp, dtype = np.float32)
        
        tracker = 0
        for i in range(len(self.moduleList)):
            module_type = self.netBlocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.moduleList[i]
                try:
                    batch_normalize = int(self.netBlocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                convPart = model[0]
                
                if (batch_normalize):
                    #Weights file Configuration=> bn bais->bn weights-> running mean-> running var
                    #The weights are arranged in the above mentioned order
                    bnPart = model[1]
                    
                    biasCount = bnPart.bias.numel()
                    
                    bnBias = torch.from_numpy(weights[tracker:tracker + biasCount])
                    tracker += biasCount
                    
                    bnPart_weights = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker  += biasCount
                    
                    bnPart_running_mean = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker  += biasCount
                    
                    bnPart_running_var = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker  += biasCount
                    
                    bnBias = bnBias.view_as(bnPart.bias.data)
                    bnPart_weights = bnPart_weights.view_as(bnPart.weight.data)
                    bnPart_running_mean = bnPart_running_mean.view_as(bnPart.running_mean)
                    bnPart_running_var = bnPart_running_var.view_as(bnPart.running_var)

                    bnPart.bias.data.copy_(bnBias)
                    bnPart.weight.data.copy_(bnPart_weights)
                    bnPart.running_mean.copy_(bnPart_running_mean)
                    bnPart.running_var.copy_(bnPart_running_var)
                
                else:
                    biasCount = convPart.bias.numel()
                
                    convBias = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker = tracker + biasCount
                    
                    convBias = convBias.view_as(convPart.bias.data)
                    
                    convPart.bias.data.copy_(convBias)
                    
                    
                weightfile = convPart.weight.numel()
                
                convWeight = torch.from_numpy(weights[tracker:tracker+weightfile])
                tracker = tracker + weightfile

                convWeight = convWeight.view_as(convPart.weight.data)
                convPart.weight.data.copy_(convWeight)
'''
#Test CFG:
construct = construct_cfg('cfg/yolov3.cfg')
print(construct,"/n constructed from cfg file")
'''

#TestMOdel:

num_classes = 80
classes = load_classes('data/coco.names') 

model = net('cfg/yolov3.cfg')
model.load_weights("yolov3.weights")
print("Network loaded")

test_data = torch.randn(1,3,256,256,dtype = torch.float)
test_output = model(test_data,False)

print(test_output.size())
