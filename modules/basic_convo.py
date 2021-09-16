
import os 
import re
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

from torch.utils.tensorboard import SummaryWriter

from modules.functions import load_training_labels, load_training_batch, data_to_image


writer = SummaryWriter()

class QTransConv(nn.Module):
    def __init__(self):
        super(QTransConv, self).__init__()
        
        # For this first most basic model, I'm using the same set of weights for each input
        # So, only declaring once, then using x3 in forward pass
        self.conv_input = nn.Conv2d(1, 4, 3, padding=1, stride=2)
        self.batch_input = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, stride=2)
        self.batch2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1, stride=2)
        self.batch3 = nn.BatchNorm2d(16)
        self.lin_flat = nn.Linear(1920, 1000) #Flatten and through linear 
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 250)
        
        self.lin_main = nn.Linear(250, 1)
        
        
        
        
    def convolve(self, x):
        x = torch.transpose(x, 2, 3)
        x = self.conv_input(x)
        x = self.batch_input(x)
        x = F.relu(x)
        
        x = F.max_pool2d(x, (2,2))
        
        x = F.relu(x)
        
        x = self.batch2(self.conv2(x))
        x = F.relu(x)
        
        x = F.max_pool2d(x, (2,2))
        
        x = self.batch3(self.conv3(x))
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        return F.relu(x)
        #return nn.Sigmoid()(x)
        
    def lin(self, x):
        #print("Shape!: ", x.shape)
        x = self.lin_flat(x.view(-1, 1920))
        #print("Right after flattening: ", x[0])
        x = F.leaky_relu(x)
        #print("Relu 6: ", x[0])
        x = self.lin2(x)
        x = F.leaky_relu(x)
        x = self.lin3(x)
        
        return F.relu6(x)
    
        
    def forward(self, x):
        
        
        x1 = self.convolve(x[:, 0, :, :, :])
        #x1 = self.lin(x1)
        
        x2 = self.convolve(x[:, 1, :, :, :])
        #x2 = self.lin(x2)
        
        
        x3 = self.convolve(x[:, 2, :, :, :])
        #x3 = self.lin(x3)
        
        #x12 = torch.bmm(x1, x2)
        #x123 = torch.bmm(x12, x3)
        x123 = x1 * x2 * x3
        #print("Before: ", x1.shape, x2.shape, x3.shape)
        #print("Shape now?", x123.shape)
        
        #print("multiplied: ", x123[0])
        x_combo = self.lin(x123)
        #print("Aftere shape?", x_combo.shape)
        
        #print("X1 out: ", x1[0])
        #print("X2 out: ", x2[0])
        #print("X3 out: ", x3[0])
        
        #print("Shapes...", x1.shape, x2.shape, x3.shape)
        #print(torch.cat((x1, x2, x3), 1).shape)
        #x_combo = self.lin_main(torch.cat((x1, x2, x3), 1))
        x_combo = self.lin_main(x_combo)
        #print("Combo out: ", x_combo)
        return nn.Sigmoid()(x_combo)
    
                               
    
def batch_data(data, labels, batch_size=32):
    """
    Method to batch the data b/c I'm too lazy to lood up data loader again
    """
    
    #...this method doesn't account for the leftover (will all be zeros). Last batch_size
    # has to be less.
    
    #num_batches= math.ceil(data.shape[0]/batch_size)
    
    #batches = torch.zeros(num_batches, batch_size, 3, data.shape[2], data.shape[3])
    #outputs = torch.zeros(num_batches, batch_size)
    batches = []
    outputs = []
    
    batch_tmp = torch.zeros(batch_size, 3, 1, data.shape[2], data.shape[3])
    output_tmp = torch.zeros(batch_size, 1)
    
    for i in range(data.shape[0]):
        if i%batch_size == 0 and i > 0:
            batches.append(batch_tmp.type(torch.FloatTensor))
            outputs.append(output_tmp.type(torch.FloatTensor))
            
            if i + batch_size < data.shape[0]:
                num = batch_size
            else:
                num = data.shape[0] - i
            batch_tmp = torch.zeros(batch_size, 3, 1, data.shape[2], data.shape[3])
            output_tmp = torch.zeros(batch_size, 1)
            
        for j in range(3):
            
            batch_tmp[i%batch_size][j][0] = torch.from_numpy(data[i][j])
            output_tmp[i%batch_size][0] = labels[i]
            
    batches.append(batch_tmp.type(torch.FloatTensor))
    outputs.append(output_tmp.type(torch.FloatTensor))
            
    return batches, outputs
        
def train_model(labels, batch_size=32, training_steps=25):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = QTransConv()
    model.to(device)
    
    loss_func = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999))
    
    testing_signals, test_classes = load_training_batch(labels, "train/f/f/f/")
    testing_data = data_to_image(testing_signals)
    testing_batches, test_outputs = batch_data(testing_data, test_classes)
    print("Test classes?", test_classes)
    test_best = 1
    #Input will have to be reshaped from (Batch_size, 3, width (axis0), height (axis1)) to (batch_size, 3, 1 (channel), height (axis1), width (axis0))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
                                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=3),
                                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/convnet1"),
                                record_shapes=True, profile_memory=True,
                                with_stack=True) as prof:
        counter = 0
        test_counter = 0
        for t in range(training_steps):
            print("Running training step: ", t)
            
            for f in os.listdir("train/"):
                print("On main dir: ", f)
                for f2 in os.listdir("train/" + f + "/"):
                    for f3 in os.listdir("train/" + f + "/" + f2 + "/"):
                        if f != 'f' and f2 != 'f' and f3 != 'f':
                            signals, classes = load_training_batch(labels , "train/" + f + "/" + f2 + "/" + f3 + "/")
                            data = data_to_image(signals)
                            batches, outputs = batch_data(data, classes, batch_size)
                            for i in range(len(batches)):
                                
                                model.zero_grad()
                                out = model(batches[i].to(device))
                                #print(out, "||", outputs[i])
                                #print("?", out.shape, outputs[i].shape)
                                loss = loss_func(out, outputs[i].to(device))
                                loss.backward()

                                optimizer.step()
                                writer.add_scalar("adam_convo_basic", loss, counter)
                                prof.step()
                                counter += 1
            avg_loss = 0        
            
            for i in range(len(testing_batches)):
                with torch.no_grad():
                    out = model(testing_batches[i].to(device))
                    loss = loss_func(out, test_outputs[i].to(device))
                    true_pos = 0
                    false_pos = 0
                    true_neg = 0
                    false_neg = 0
                    for idx in range(out.shape[0]):
                        guess = 0
                        if out[idx][0] > .5:
                            guess = 1
                        if guess == int(test_outputs[i][idx][0]):
                            if guess == 1:
                                true_pos += 1
                            else:
                                true_neg += 1
                        else:
                            if guess == 0:
                                false_neg += 1
                            else:
                                false_pos += 1
                    print("Testing: ", true_pos, true_neg, false_pos, false_neg)
                    avg_loss += loss.item()
                    writer.add_scalar("adam_convo_test", loss, test_counter)
                    test_counter += 1
            avg_loss = avg_loss/float(len(testing_batches))
            if avg_loss < test_best:
                test_best = avg_loss
                print("New best test loss!:", test_best)
                torch.save({'epoch': t, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, "gwave_convo_model.pt")
            
                
    writer.flush()
    writer.close()
    return model
            
            
        
    