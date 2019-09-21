# FFNN.py
#
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_fall19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys
import time

import numpy as np
from Eval import Eval

import torch
import torch.nn as nn
import torch.optim as optim

from imdb import IMDBdata

class FFNN(nn.Module):
    def __init__(self, X, Y, VOCAB_SIZE, DIM_EMB=10, NUM_CLASSES=2):
        super(FFNN, self).__init__()
        (self.VOCAB_SIZE, self.DIM_EMB, self.NUM_CLASSES) = (VOCAB_SIZE, DIM_EMB, NUM_CLASSES)
        #TODO: Initialize parameters.
        self.V = nn.Embedding(self.VOCAB_SIZE, self.DIM_EMB)
        self.g = nn.ReLU()
        self.W = nn.Linear(self.DIM_EMB, self.NUM_CLASSES)
        self.softmax = nn.LogSoftmax(dim=0)

        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)


    def forward(self, X, train=False):
        #TODO: Implement forward computation.
        v = torch.sum(self.V(X), dim=0)
        
        return self.softmax(self.W(self.g(v)))

def Eval_FFNN(X, Y, mlp):
    num_correct = 0
    for i in range(len(X)):
        logProbs = mlp.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))

def Train_FFNN(X, Y, vocab_size, n_iter):
    print("Start Training!")
    mlp = FFNN(X, Y, vocab_size)
    #TODO: initialize optimizer.
    #optimizer = optim.SGD(mlp.parameters(), lr=.01)
    optimizer = optim.Adam(mlp.parameters(), lr=1e-5)

    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            #TODO: compute gradients, do parameter update, compute loss.
            mlp.zero_grad()
            probs = mlp.forward(X[i])

            if int(Y[i]) == 1:
                gold = torch.Tensor([0,1])
            else:
                gold = torch.Tensor([1,0])
            loss = torch.neg(probs).dot(gold)
            total_loss += loss

            loss.backward()
            optimizer.step()
            
        print(f"loss on epoch {epoch} = {total_loss}")
    return mlp

if __name__ == "__main__":
    start1 = time.time()
    train = IMDBdata("%s/train" % sys.argv[1])
    train.vocab.Lock()
    #test  = IMDBdata("%s/dev" % sys.argv[1], vocab=train.vocab)
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    mlp = Train_FFNN(train.XwordList, (train.Y + 1.0) / 2.0, train.vocab.GetVocabSize(), int(sys.argv[2]))
    Eval_FFNN(test.XwordList, (test.Y + 1.0) / 2.0, mlp)
    end1 = time.time()
    print(end1-start1)
