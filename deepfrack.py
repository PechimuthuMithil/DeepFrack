# 1. Succeeding layer tiling should align with that of previous layer 
# 2. tiling should also align with number of heads
# 3. cost of layer = (Multiplication Cost + Addition Cost of one tile) * number of tiles

import os
import yaml
import numpy as np
import json
import matplotlib.pyplot as plt
import time
import math
import concurrent.futures
import threading
import sys
from multiprocessing import Process, Manager
import subprocess

# start with smallest and tile it then figure out cost for tiling all others in the same way. 
def decimal_to_binary_fixed_length(number, length):
    binary = bin(number)[2:].zfill(length)
    return binary

def LayerOrder(order):
    with open(order,'r') as file:
        data = file.read()
    lines = data.splitlines()
    layer_names = []
    num_parallel = []

    for line in lines:
        layer, num = line.split()
        layer_names.append(layer) 
        num_parallel.append(int(num))

    return layer_names, num_parallel

# square tile will not fit all - hence zero padding
def pad_tile(P,Q,K,tile_size):
    if (P%tile_size != 0):
        P = P + (tile_size - P%tile_size)
    if (Q%tile_size != 0):
        Q = Q + (tile_size - Q%tile_size)
    if (K%tile_size != 0):
        K = K + (tile_size - K%tile_size)
    return P,Q,K


# Full = 0, Mul = 1, Add = 2 (From benchmarking)
def LastLayer(n,end,k,arch_sizes,CurrAnswer,num_tiles,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small):            
    ### FOR LAST LAYER ###
        outputs_size = 0
        Trace=""
        if(layer_names[end]=='Attn'):  
                Trace+="-Attn"   
                if(TileSizes[end]>small):
                    CurrAnswer = float('inf')

                inputs_size = (TileSizes[end-1])**2 * num_tiles[n-1][2] * num_parallel[end] 
                weights_size = inputs_size
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['EWCC']
                if np.all(const >= arch_sizes):
                    Best=float('inf')
                    for l in range(num_parallel[end]+1):
                        inputs_size = (TileSizes[end-1])**2 * num_tiles[end][2] * l
                        weights_size = inputs_size
                        const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['ELBLC']
                        if np.all(const >= arch_sizes):
                            Curr= float('inf')
                            continue
                        else:
                            try:
                                MCEWCC = EWCC["1"][str(TileSizes[end])]
                                MCSLC = SLC["1"][str(TileSizes[end])]
                            except KeyError:
                                MCEWCC=float('inf')
                                MCSLC=float('inf')
                            AdditionCost = EWCC["2"][str(TileSizes[end])]
                        Curr=(MCEWCC+AdditionCost)*num_tiles[n-1][2]*num_tiles[n-1][0]*num_tiles[n-1][1]*l + (MCSLC+SLC["2"][str(TileSizes[end])])*num_tiles[n-1][2]*num_tiles[n-1][0]*num_tiles[n-1][1]*(num_parallel[end]-l)
                        if(Curr<Best):
                            Best=Curr
                            T=f':EWCC{l}'
                    CurrAnswer+=Best   
                    Trace+=T
                else:
                    try:
                        MultiplicationCost = EWCC["1"][str(TileSizes[end])]
                    except KeyError:
                        MultiplicationCost=float('inf')
                    AdditionCost = EWCC["2"][str(TileSizes[end])]
                    CurrAnswer+=(MultiplicationCost+AdditionCost)*num_tiles[n-1][2]*num_tiles[n-1][0]*num_tiles[n-1][1]*num_parallel[end]
                    Trace=':EWCCFull'
                

        elif(layer_names[end]=='QKV'):
                Trace+="-QKV"
                if(TileSizes[end]>small):
                    CurrAnswer = float('inf')

                inputs_size = (TileSizes[end-1])**2 *num_parallel[end]
                weights_size =  (TileSizes[end-1])**2 * num_tiles[n-1][2] *num_parallel[end]  

                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['ELBLC']
                if np.all(const >= arch_sizes):
                    inputs_size = (TileSizes[end-1])**2 * k
                    weights_size = (TileSizes[end-1])**2 * num_tiles[n-1][2] * k
                    const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['ELBLC']
                    if np.all(const >= arch_sizes):
                        CurrAnswer = float('inf')
                    else:
                        try:
                            MultiplicationCost = EWCC["1"][str(TileSizes[end])]
                        except KeyError:
                            MultiplicationCost=float('inf')
                        AdditionCost = EWCC["2"][str(TileSizes[end])]
                    if(CurrAnswer==float('inf')): return "",CurrAnswer
                    CurrAnswer+=(MultiplicationCost+AdditionCost)*num_tiles[n-1][2]*num_tiles[n-1][0]*num_tiles[n-1][1]*k + (SLC["1"][str(TileSizes[end])]+SLC["2"][str(TileSizes[end])])*num_tiles[n-1][2]*num_tiles[n-1][0]*num_tiles[n-1][1]*(num_parallel[end]-k)
                    Trace+=f":EWCC{k}"
                else:
                    try:
                        MultiplicationCost = EWCC["1"][str(TileSizes[end])]
                    except KeyError:
                        MultiplicationCost=float('inf')
                    AdditionCost = EWCC["2"][str(TileSizes[end])]
                CurrAnswer+=(MultiplicationCost+AdditionCost)*num_tiles[n-1][2]*num_tiles[n-1][0]*num_tiles[n-1][1]*num_parallel[end]
                Trace=':EWCCFull'

        else:
            Trace+="-"+layer_names[end]
            weights_size = 0
            inputs_size = TileSizes[end-1]**2 * k
            if(layer_names[end]=='MHead' and num_parallel[end]!=1):
                inputs_size = (TileSizes[end-1])**2 * num_tiles[n-1][2] *num_parallel[end]

            if (ChosenCached[n-1] == '1'):
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['EWCC']
                if np.all(const >= arch_sizes):
                    CurrAnswer = float('inf')
                else:
                    try:
                        MultiplicationCost = EWCC["1"][str(TileSizes[end])]
                    except KeyError:
                        MultiplicationCost=float('inf')
                    AdditionCost = EWCC["2"][str(TileSizes[end])]
                Trace+="EWCC"
            else:
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['ELBLC']
                if np.all(const >= arch_sizes):
                    CurrAnswer = float('inf')
                else:
                    try:
                        MultiplicationCost = ELBLC["1"][str(TileSizes[end])]
                    except KeyError:
                        MultiplicationCost=float('inf')
                    AdditionCost = ELBLC["2"][str(TileSizes[end])]
                Trace+="ELBLC"
            if(CurrAnswer==float('inf')): return "",CurrAnswer
            CurrAnswer+=(MultiplicationCost+AdditionCost)*num_tiles[n-1][2]*num_tiles[n-1][0]*num_tiles[n-1][1]
            
        return Trace,CurrAnswer

def middle_layers(iter,start,n,k,arch_sizes,CurrAnswer,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,num_tiles,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small): 
    end = n + start -1  
    ll = 0     
    Trace=""
    for i in range(iter,n-1):
        outputs_size = (TileSizes[start+i])**2      
        inputs_size = (TileSizes[start+i-1])**2
        weights_size = 0
        ll=0
        if(layer_names[i]=='Attn'):
                Trace+="-Attn"

                if(TileSizes[start+i]>small):
                    CurrAnswer = float('inf')
                
                inputs_size = (TileSizes[start+i-1])**2 * num_tiles[i][2] * num_parallel[start+i]    
                weights_size = inputs_size 
                outputs_size = (TileSizes[start+i])**2 * num_parallel[start+i]
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['WCC']
                if np.all(const >= arch_sizes):
                    Best=float('inf')
                    bestl=0
                    Ans=CurrAnswer
                    bestT="Not taken!"
                    for l in range(num_parallel[start]+1):
                        inputs_size = (TileSizes[start+i-1])**2 * num_tiles[i][2] * l
                        weights_size = inputs_size
                        outputs_size =  (TileSizes[start+i])**2 * num_parallel[start+i] * l
                        const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['WCC']
                        if np.all(const >= arch_sizes):
                            CurrAnswer = float('inf')
                        else:
                            try:
                                MCWCC = FullyWCC["1"][str(TileSizes[start+i])]
                                MCSLC = SLC["1"][str(TileSizes[start+i])]
                            except KeyError:
                                MCWCC=float('inf')
                                MCSLC=float('inf')
                            AdditionCost = FullyWCC["2"][str(TileSizes[start+i])]
                        if CurrAnswer==float('inf'): continue
                        CurrAnswer+=((MCWCC+AdditionCost)*l+ (MCSLC+SLC["2"][str(TileSizes[start+i])])*(num_parallel[start+i]-l))*num_tiles[i][2]*num_tiles[i][0]*num_tiles[i][1]
                        
                        if(i>=n-2):
                            ll=1
                            T,Curr=LastLayer(n,end,l,arch_sizes,CurrAnswer,num_tiles,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small)
                            CurrAnswer+=Curr
                        else:
                            T,Curr=middle_layers(i+1,start,n,l,arch_sizes,CurrAnswer,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,num_tiles,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small)
                            CurrAnswer+=Curr
                        if(CurrAnswer<Best):
                            Best=CurrAnswer
                            bestT=T 
                            bestl=l
                        CurrAnswer=Ans
                    CurrAnswer=Best
                    Trace+=f':WCC{bestl}'+bestT
                    i = n-2     
                else:
                    try:
                        MultiplicationCost = FullyWCC["1"][str(TileSizes[start+i])]
                    except KeyError:
                        MultiplicationCost=float('inf')
                    AdditionCost = FullyWCC["2"][str(TileSizes[start+i])]
                    CurrAnswer+=((MultiplicationCost+AdditionCost)*num_tiles[i][2]*num_tiles[i][0]*num_tiles[i][1]*num_parallel[start+i])      
                    Trace+=':WCCFull'              

        elif(layer_names[i]=='QKV'):
                Trace+="-QKV"
                if(TileSizes[start+i-1]>small):
                    CurrAnswer = float('inf')

                inputs_size = (TileSizes[start+i-1])**2 * num_parallel[start+i]
                weights_size =  (TileSizes[start+i-1])**2 * num_tiles[i][1]  * num_parallel[start+i]
                outputs_size = (TileSizes[start+i])**2 * num_tiles[i][1] * num_parallel[start+i]
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['WCC']
                if np.all(const >= arch_sizes):
                    inputs_size =  (TileSizes[start+i-1])**2  * k
                    weights_size = (TileSizes[start+i-1])**2 * num_tiles[i][1] * k
                    outputs_size =  (TileSizes[start+i])**2 * num_tiles[i][1] * k
                    const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['WCC']
                    if np.all(const >= arch_sizes):
                        CurrAnswer = float('inf')
                    else:
                        try:
                            MultiplicationCost = FullyWCC["1"][str(TileSizes[start+i])]
                        except KeyError:
                            MultiplicationCost=float('inf')
                        AdditionCost = FullyWCC["2"][str(TileSizes[start+i])]
                    if CurrAnswer==float('inf'): return "",CurrAnswer
                    CurrAnswer+=((MultiplicationCost+AdditionCost)*num_tiles[i][2]*num_tiles[i][0]*num_tiles[i][1]*k) + (SLC["1"][str(TileSizes[start+i])]+SLC["2"][str(TileSizes[start+i])])*num_tiles[i][2]*num_tiles[i][0]*num_tiles[i][1]*(num_parallel[start+i]-k)
                    Trace+=f':WCC{k}'

                else:
                    try:
                        MultiplicationCost = FullyWCC["1"][str(TileSizes[start+i])]
                    except KeyError:
                        MultiplicationCost=float('inf')
                    AdditionCost = FullyWCC["2"][str(TileSizes[start+i])]
                    CurrAnswer+=((MultiplicationCost+AdditionCost)*num_tiles[i][2]*num_tiles[i][0]*num_tiles[i][1]*num_parallel[start+i])
                    Trace+=':WCCFull'
                

        else: 
            k=1 
            Trace+="-"+layer_names[i]
            if(layer_names[i]=='MHead'):
                inputs_size = (TileSizes[start+i-1])**2 * num_tiles[i][1] *num_parallel[i]
                outputs_size = (TileSizes[start+i])**2 * num_tiles[i][1] *num_parallel[i]  
                

            if (ChosenCached[i] == '1'):
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['WCC']
                if np.all(const >= arch_sizes):
                    CurrAnswer = float('inf')
                else:
                    try:
                        MultiplicationCost = FullyWCC["1"][str(TileSizes[start+i])]
                    except KeyError:
                        MultiplicationCost=float('inf')
                    AdditionCost = FullyWCC["2"][str(TileSizes[start+i])]
                Trace+=":WCC"
            else:
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['LBLC']
                if np.all(const >= arch_sizes):
                    CurrAnswer = float('inf')
                else:
                    try:
                        MultiplicationCost = LBLC["1"][str(TileSizes[start+i])]
                    except KeyError:
                        MultiplicationCost=float('inf')
                    AdditionCost = LBLC["2"][str(TileSizes[start+i])]   
                if(CurrAnswer==float('inf')): return "",CurrAnswer
                CurrAnswer+=(MultiplicationCost+AdditionCost)*num_tiles[i][2]*num_tiles[i][0]*num_tiles[i][1]
                Trace+=":LBLC"
    if(not ll):
        T,Curr=LastLayer(n,end,k,arch_sizes,CurrAnswer,num_tiles,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small)    
        Trace+=T
        CurrAnswer+=Curr
    return Trace,CurrAnswer         


def StackCostGenerator(
        WeightLevel_name,
        WeightLevel_size,
        InputLevel_name,
        InputLevel_size,
        OutputLevel_name,
        OutputLevel_size,
        layers, num_parallel, layer_names,
        stack,
        SLC,
        Start,
        OutWCC,
        LBLC,
        FullyWCC,
        ELBLC,
        EWCC,
):

    cache_names = [WeightLevel_name, InputLevel_name, OutputLevel_name]
    mask = [[1 if inner_name == outer_name else 0 for inner_name in cache_names] for outer_name in cache_names]
    mask = np.array(mask)
    factor = {'SLC':np.array([0,0,0]),'LBLC':np.array([0,1,1]),'ELBLC':np.array([0,1,0]),'WCC':np.array([1,1,1]),'EWCC':np.array([1,1,0]),'OutWCC':np.array([1,0,1]),'Start':np.array([0,0,1])}
    Dataflow_types = ['SLC','LBLC','ELBLC','WCC','EWCC','OutWCC','Start']
    
    BestTilingCost = float('inf') # For the full stack
    BestTiling = []
    BestWeightCaching = 'None'
    BestTrace="None"

    # smallest dimension across layers should be max tile size
    small = 64

    if (stack[0] == stack[1]):
        with open(layers[stack[0]], 'r') as file:
            data = yaml.safe_load(file)

        inputs=[]
        weights=[]
        for data_space in data['problem']['shape']['data-spaces']:
            if (data_space['name'] == 'Inputs'):
                for dim in data_space['projection']:
                    inputs.append(dim[0][0])          
            if (data_space['name'] == 'Weights'):
                for dim in data_space['projection']:
                    weights.append(dim[0][0])

        P=data['problem']['instance'][inputs[0]]
        Q=data['problem']['instance'][weights[0]] 
        weight_mid=data['problem']['instance'][weights[1]]
        input_mid=data['problem']['instance'][inputs[1]]  

        return SLC["0"][layer_names[stack[0]]]*num_parallel[stack[0]], [(P,Q)], 'None', layer_names[stack[0]]+':SLC'

    start = stack[0]
    end = stack[1]    
    n = end - start + 1

    for tile in range(2,small+1): # all sizes from 2 

        WeightSizes = {}
        TileSizes = {}

        num_tiles=[]
        ### SCHEDULING THE LARGEST OPTIMALLY AND CHECKING FOR WEIGHT CACHING ###
        # find size of weights to be cached and tile sizes per layer then check if they can be cache
        incompatible=0
        for fileindex in range (end,start -1, -1):
            file_path = layers[fileindex]
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)

            inputs=[]
            weights=[]
            for data_space in data['problem']['shape']['data-spaces']:
                if (data_space['name'] == 'Inputs'):
                    for dim in data_space['projection']:
                        inputs.append(dim[0][0])          
                if (data_space['name'] == 'Weights'):
                    for dim in data_space['projection']:
                        weights.append(dim[0][0])

            P=data['problem']['instance'][inputs[0]]
            Q=data['problem']['instance'][weights[0]] 
            weight_mid=data['problem']['instance'][weights[1]]
            input_mid=data['problem']['instance'][inputs[1]]  

            assert input_mid == weight_mid, "Error in K size"

            P,Q,K=pad_tile(P,Q,weight_mid,tile)
            
            WeightSizes[fileindex] = Q*K
            TileSizes[fileindex] = tile
            num_tiles.append([P//tile,Q//tile,K//tile])

            ######### WEIGHT CACHING STRATEGY #######################################
        BestAnswer = float('inf') # For a tiling config of a stack
        ClubSize = math.ceil(n/20) # 20 is a constant that we chose
        q = math.ceil(n/ClubSize) # no of clubbed layers
            # assume if layer combinations is cached and check if valid for memory size

        for Q in range(2**q): # Iterating over all possible Weight Caching Patterns (WCP)
            CurrAnswer = 0
            comb = decimal_to_binary_fixed_length(Q,q) # if ChosenCached[i] = 1, then the (start + i)th layer's weights are going to be cached.
            ChosenCached = ''
            for a in range(q):
                ChosenCached += comb[a]*min(ClubSize,n-a*ClubSize)
            TotalCachedWeights = 0
            for layer in range(n):
                if (ChosenCached[layer] == '1' and layer_names[layer]!='Attn' and layer_names[layer]!='QKV'):   
                    TotalCachedWeights = WeightSizes[start+layer]*num_parallel[start+layer] # Calculate the total weights required.

            valid = False
            if (WeightLevel_name != InputLevel_name and WeightLevel_name != OutputLevel_name):
                if (TotalCachedWeights <= WeightLevel_size):
                    valid = True

            if (valid):
    

                    arch_sizes = np.array([[WeightLevel_size - TotalCachedWeights,],[InputLevel_size,],[OutputLevel_size,]])
                                            
                    ### FOR FIRST LAYER ###
                    # Only two options, Start, OutWCC//.

                    outputs_size = (TileSizes[start])**2 * num_tiles[0][1]
                    inputs_size = 0 
                    weights_size = 0
                    Trace=""

                    if(layer_names[start]=='Attn'):
                            
                            # in attn layer, there are no "weights", both operands are inputs but i cannot model that on timeloop so will pretend one is weight
                            if(TileSizes[start]>small):
                                CurrAnswer = float('inf')
                            
                            outputs_size = TileSizes[start]**2 * num_tiles[0][1]  * num_parallel[start]
                            const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['Start']
                            if np.all(const >= arch_sizes):
                                Best=float('inf')
                                for k in range(num_parallel[start]+1):
                                    outputs_size = TileSizes[start]**2 * num_tiles[0][1]  * k
                                    try:
                                        MCStart = Start["1"][str(TileSizes[start])]
                                        MCSLC = SLC["2"][str(TileSizes[start])]
                                    except:
                                        MCStart=float('inf')
                                        MCSLC=float('inf')
                                    CurrAnswer=(MCStart+Start["2"][str(TileSizes[start])])*num_tiles[0][2]*num_tiles[0][0]*num_tiles[0][1]*k + (MCSLC+SLC["2"][str(TileSizes[start])])*num_tiles[0][2]*num_tiles[0][0]*num_tiles[0][1]*(num_parallel[start]-k)
                                    T,Curr=middle_layers(1,start,n,k,arch_sizes,CurrAnswer,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,num_tiles,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small)   
                                    if(CurrAnswer+Curr<Best):
                                        Best=CurrAnswer+Curr
                                        Trace+=f'Attn:Start{k}'+T
                                CurrAnswer+=Best
                            else:
                                try:
                                    MultiplicationCost = Start["1"][str(TileSizes[start])]
                                except KeyError:
                                    MultiplicationCost=float('inf')
                                AdditionCost = Start["2"][str(TileSizes[start])]
                                CurrAnswer=(MultiplicationCost+AdditionCost)*num_tiles[0][2]*num_tiles[0][0]*num_tiles[0][1]*num_parallel[start]
                                Trace='Attn:StartFull'
                                T,Curr=middle_layers(1,start,n,num_parallel[start],arch_sizes,CurrAnswer,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,num_tiles,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small)      
                                CurrAnswer+=Curr
                                Trace+=T

                    elif(layer_names[start]=='QKV'):
                            outputs_size = (TileSizes[start])**2 * num_tiles[0][1]
                            if(TileSizes[start]>small):
                                CurrAnswer = float('inf')
                            
                            outputs_size = (TileSizes[start])**2 * num_tiles[0][1] * num_parallel[start]
                            const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['Start']
                            if np.all(const >= arch_sizes):
                                for k in range(num_parallel[start]+1):
                                    outputs_size = (TileSizes[start])**2 * num_tiles[0][1] * k
                                    try:
                                        MCStart = Start["1"][str(TileSizes[start])]
                                        MCSLC = SLC["2"][str(TileSizes[start])]
                                    except:
                                        MCStart=float('inf')
                                        MCSLC=float('inf')
                                    CurrAnswer=((MCStart+Start["2"][str(TileSizes[start])])*k + (MCSLC+SLC["2"][str(TileSizes[start])])*(num_parallel[start]-k))*num_tiles[0][2]*num_tiles[0][0]*num_tiles[0][1]
                                    Trace=f'QKV:Start{k}'
                                    T,Curr=middle_layers(1,start,n,1,arch_sizes,CurrAnswer,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,num_tiles,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small)     
                                    CurrAnswer+=Curr
                                    Trace+=T 
                            else:
                                try:
                                    MultiplicationCost = Start["1"][str(TileSizes[start])]
                                except KeyError:
                                    MultiplicationCost=float('inf')
                                AdditionCost = Start["2"][str(TileSizes[start])]
                                CurrAnswer=(MultiplicationCost+AdditionCost)*num_tiles[0][2]*num_tiles[0][0]*num_tiles[0][1]*num_parallel[start]
                                Trace='QKV:StartFull'
                                T,Curr=middle_layers(1,start,n,1,arch_sizes,CurrAnswer,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,num_tiles,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small)      
                                CurrAnswer+=Curr
                                Trace+=T
                    else:   
                        Trace=layer_names[start]
                        if(layer_names[start]=='MHead'):
                            outputs_size = (TileSizes[start])**2 * num_tiles[0][1] * num_parallel[start]

                        if (ChosenCached[0] == '1'):
                            const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['OutWCC']
                            if np.all(const >= arch_sizes):
                                CurrAnswer = float('inf')
                            else:
                                try:
                                    MultiplicationCost = OutWCC["1"][str(TileSizes[start])]
                                except KeyError:
                                    MultiplicationCost=float('inf')
                                AdditionCost = OutWCC["2"][str(TileSizes[start])]
                                Trace+=':OutWCC'
                        else:
                            const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['Start']
                            if np.all(const >= arch_sizes):
                                CurrAnswer = float('inf')
                            else:
                                try:
                                    MultiplicationCost = Start["1"][str(TileSizes[start])]
                                except KeyError:
                                    MultiplicationCost=float('inf')
                                AdditionCost = Start["2"][str(TileSizes[start])]
                                Trace+=':Start'
                        if(CurrAnswer==float('inf')): continue
                        CurrAnswer=(MultiplicationCost+AdditionCost)*num_tiles[0][2]*num_tiles[0][0]*num_tiles[0][1]
                        T,Curr=middle_layers(1,start,n,1,arch_sizes,CurrAnswer,TileSizes,WeightSizes,ChosenCached,layer_names,num_parallel,num_tiles,mask,factor,SLC,LBLC,FullyWCC,ELBLC,EWCC,small)            
                        Trace+=T     
                        CurrAnswer+=Curr          

                    
                                
            if (CurrAnswer == 0):
                    CurrAnswer = float('inf')
                        
        ###################################################################################################################
                    
            if (CurrAnswer < BestAnswer):
                    BestAnswer = CurrAnswer
                    BestCaching = ChosenCached
                    BestListofTiles = [tile,tile]
                    BestTrace= Trace

        if (BestAnswer < BestTilingCost):
            BestTilingCost = BestAnswer
            BestTiling = BestListofTiles
            if (BestCaching == "0"*n):
                BestCaching = "None"
            BestWeightCaching = BestCaching
            BestTrace=BestTrace


    return BestTilingCost, BestTiling, BestWeightCaching, BestTrace



def save_dictionary_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)

def load_dictionary_from_file(filename):
    with open(filename, 'r') as file:
        dictionary = json.load(file)
    return dictionary

def worker(worker_id, n, m, worker_costs, worker_tilings, worker_WC,worker_trace, WeightLevel_name, WeightLevel_size, InputLevel_name, InputLevel_size, OutputLevel_name, OutputLevel_size, layers,num_parallel,layer_names, SLC, Start, OutWCC, LBLC, FullyWCC, ELBLC, EWCC, output_dict):
    cost = {}
    BestTiling = {}
    WeightsCached = {}
    Trace = {}
    try:
        for end in range(n):
            for start in range(end + 1):
                if ((start + end) % m) == worker_id:
                    temp = StackCostGenerator(WeightLevel_name, WeightLevel_size, InputLevel_name, InputLevel_size, OutputLevel_name, OutputLevel_size, layers,num_parallel,layer_names, (start, end), SLC, Start, OutWCC, LBLC, FullyWCC, ELBLC, EWCC)
                    cost[(start, end)], BestTiling[(start, end)], WeightsCached[(start, end)], Trace[(start, end)] = temp[0], temp[1], temp[2], temp[3]
                    print(f"[ {worker_id}]  Stack ({start},{end}): {cost[(start, end)]} uJ")
        worker_costs[worker_id] = cost
        worker_tilings[worker_id] = BestTiling
        worker_WC[worker_id] = WeightsCached
        worker_trace[worker_id] = Trace
    except KeyboardInterrupt:
        print(f"Worker {worker_id}: Keyboard interrupt received, terminating.")
    output_dict[worker_id] = (worker_costs[worker_id], worker_tilings[worker_id], worker_WC[worker_id],worker_trace[worker_id])

def run_deepfrack(folder_path,order,BenchMrkrLog_folder,OutputImgFile,LogFile,WeightLevel_name,WeightLevel_size,InputLevel_name,InputLevel_size,OutputLevel_name,OutputLevel_size,M):
    st = time.time()

    logo = '''
    _____                      _______               _     
    (____ \                    (_______)             | |    
    _   \ \ ____ ____ ____     _____ ____ ____  ____| |  _ 
    | |   | / _  ) _  )  _ \   |  ___) ___) _  |/ ___) | / )
    | |__/ ( (/ ( (/ /| | | |  | |  | |  ( ( | ( (___| |< ( 
    |_____/ \____)____) ||_/   |_|  |_|   \_||_|\____)_| \_)
                      |_|                                   

    '''
    print(logo)

    SLCFile = ''
    LBLCFile = ''
    FullyWCCFile = ''
    StartFile = ''
    EndFileWCC = ''
    EndFileLBLC = ''
    OutWCCFile = ''


    for file_name in os.listdir(BenchMrkrLog_folder):
        if file_name.endswith('.json'):
            name = file_name[:-5]
            path = os.path.join(BenchMrkrLog_folder, file_name)
            if name == 'ELBLC':
                EndFileLBLC = path
            elif name == 'EWCC':
                EndFileWCC = path
            elif name == 'LBLC':
                LBLCFile = path
            elif name == 'OutWCC':
                OutWCCFile = path
            elif name == 'SLC':
                SLCFile = path
            elif name == 'Start':
                StartFile = path
            elif name == 'WCC':
                FullyWCCFile = path
            else:
                print("Invalid name:", name)

    ### LOAD THE DICTIONARIES ###
    SLC = load_dictionary_from_file(SLCFile)
    LBLC = load_dictionary_from_file(LBLCFile)
    FullyWCC = load_dictionary_from_file(FullyWCCFile)
    Start = load_dictionary_from_file(StartFile)
    EWCC = load_dictionary_from_file(EndFileWCC)
    ELBLC = load_dictionary_from_file(EndFileLBLC)
    OutWCC = load_dictionary_from_file(OutWCCFile)

    ### SORT THE FILE NAMES IN THE FOLDER IN DESCENDING ORDER ###
    layers=[]

    layer_names,num_parallel=LayerOrder(order)
    for name in layer_names:
        file_path = os.path.join(folder_path, name+'.yaml')
        layers.append(file_path)

    n = len(layers)
    

    ### Now the layers are in increasing order of their layer number. ###
    stacks = []
    cost = {}
    BestTiling = {}
    WeightsCached = {}
    Trace = {}

    ### GENERATE THE COSTS OF ALL THE STACKS ###

    worker_costs = {}
    worker_tilings = {}
    worker_WC = {}
    worker_trace = {}

    with Manager() as manager:
        output_dict = manager.dict()

        processes = []
        for i in range(M):
            p = Process(target=worker, args=(
            i, n, M, worker_costs, worker_tilings, worker_WC, worker_trace, WeightLevel_name, WeightLevel_size, InputLevel_name,
            InputLevel_size, OutputLevel_name, OutputLevel_size, layers,num_parallel,layer_names, SLC, Start, OutWCC, LBLC, FullyWCC, ELBLC,
            EWCC, output_dict))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for worker_id, (w_cost, w_tiling, w_wc,w_trace) in output_dict.items():
            cost.update(w_cost)
            BestTiling.update(w_tiling)
            WeightsCached.update(w_wc)
            Trace.update(w_trace)

    for end in range(n):
        for start in range(end+1):
            stacks.append((start,end))  # start and end are the indexes of the layers in the list named layers.

    m = len(stacks)
    BestPartition = np.zeros(n)
    BestPartition[0] = cost[(0,0)]
    PartitionTracker = []
    for i in range(n):
        PartitionTracker.append(0)

    PartitionTracker[0] = 0

### DYNAMIC PROGRAMMING BASED SOLUTION TO OPTIMALLY JOIN PARTITION THE WORKLOAD INTO FUSED STACKS ###
    for i in range(0,n):
        BestPartition[i] = float('inf')
        for j in range(m):
            if (stacks[j][1] == i):
                if (stacks[j][0] > 0):
                    if ((cost[stacks[j]] + BestPartition[stacks[j][0]-1]) < BestPartition[i]):
                        BestPartition[i] = cost[stacks[j]] + BestPartition[stacks[j][0]-1]
                        PartitionTracker[i] = stacks[j]
                else:
                    if (cost[stacks[j]] < BestPartition[i]):
                        BestPartition[i] = cost[stacks[j]]
                        PartitionTracker[i] = stacks[j]                    


    print(f"Total Cost with Fused Layer Scheduling= {BestPartition[n-1]} uJ")

    SingleLayerCost = 0
    for layn in range(len(layers)):
        with open(layers[layn], 'r') as file:
            data = yaml.safe_load(file)
        # Output Dimensions:
        inputs=[]
        weights=[]
        for data_space in data['problem']['shape']['data-spaces']:
            if (data_space['name'] == 'Inputs'):
                for dim in data_space['projection']:
                    inputs.append(dim[0][0])          
            if (data_space['name'] == 'Weights'):
                for dim in data_space['projection']:
                    weights.append(dim[0][0])

        P=data['problem']['instance'][inputs[0]]
        Q=data['problem']['instance'][weights[0]] 
       
        SingleLayerCost += SLC["0"][layer_names[layn]] *num_parallel[layn]
    
    print("--------------------------------------------------------------------")
    print(f"\nSingle Layer Scheduling cost: {SingleLayerCost} uJ\n")

    Benefit = ((SingleLayerCost - BestPartition[n-1])/SingleLayerCost)*100
    print(f"Fused Layer scheduling gives an energy reduction of {Benefit}%\n")

    ### GENERATE THE STATISTICS FILE ###
    curr = n-1
    d = []
    while (curr >= 0):
        d.append(PartitionTracker[curr])
        curr = PartitionTracker[curr][0]-1

    a = len(d)
    j = 1
    distinct_tiles = []
    WCPS = []
    T=[]
    for stack in d:
        distinct_tiles.append(np.unique(BestTiling[stack])[0])

    for stack in d:
        if (WeightsCached[stack] == 'None'):
            WCPS.append('0'*(stack[1]-stack[0]+1))
        else:
            WCPS.append(WeightsCached[stack])

    for stack in d:
        T.append(Trace[stack])

    with open(LogFile, 'a') as sf: 
        sf.write(f"{d}")
        sf.write("\n")
        sf.write(f"{distinct_tiles}")
        sf.write("\n")
        sf.write(f"{WCPS}")
        sf.write("\n")
        for stack in range(a-1,-1,-1):
            sf.write(f"Fuse Stack {j}: {d[stack]} with a cost of {cost[d[stack]]} uJ\n")
            sf.write(f"-> Tiles Used: ")
            sf.write(np.array2string(np.array(BestTiling[d[stack]])))
            sf.write("\n")
            sf.write(f"-> Trace: {Trace[d[stack]]}\n")

            sf.write("Layers, whose weights were cached: ")
            if (WeightsCached[d[stack]] == 'None'):
                sf.write("None")
            else:
                for wl in range(len(WeightsCached[d[stack]])):
                    if (WeightsCached[d[stack]][wl] == '1'):
                        sf.write(f"Layer {d[stack][0] + wl} ")
            sf.write("\n")

            j += 1
            sf.write("\n")

    ### Plotting for comparison ###

    fig1, ax = plt.subplots()
    CostTypes = ['Single Layer Scheduling', 'Fused Layer Scheduling']
    EnergyValues = [SingleLayerCost/1e2,BestPartition[n-1]/1e2] # Values are in mJ
    ax.bar(CostTypes,EnergyValues)
    ax.set_xlabel('Types of Scheduling-->')
    ax.set_ylabel('Total Energy (uJ) (x100) -->')
    plt.savefig(OutputImgFile)
    et = time.time()

    ### FINAL LINES OF STATS FILE ###

    with open(LogFile, 'a') as sf2:
        sf2.write(f"Total Cost with Fused Layer Scheduling= {BestPartition[n-1]} uJ\n")
        sf2.write(f"\nSingle Layer Scheduling cost: {SingleLayerCost} uJ\n")
        sf2.write(f"Fused Layer scheduling gives an energy reduction of {Benefit}%\n")
        sf2.write(f"Total Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(et-st))}")


    print(f"Stored the statistics at {LogFile}")
    print("Total Elapsed Time For the Core to Schedule: ",time.strftime("%H:%M:%S", time.gmtime(et-st)))

def main():
    folder_path = 'BERT_Simba/BERT'
    order= 'BERT_Simba/BERT/Problem-structure.txt'
    BenchMrkrLog_folder = 'BERT_Simba/Benchmarks1'
    OutputImgFile = 'BERT_Simba/Output/Comparison_multiT.jpg'
    LogFile = 'BERT_Simba/Output/DeepFrack_logfile_MultiT.txt'
    WeightLevel_name = 'PEWeightBuffer'
    WeightLevel_size = 4096
    InputLevel_name = 'GlobalBuffer'
    InputLevel_size = 65536
    OutputLevel_name = 'GlobalBuffer'
    OutputLevel_size = 65536
    num_threads= 16
    run_deepfrack(folder_path,order,BenchMrkrLog_folder,OutputImgFile,LogFile,WeightLevel_name,WeightLevel_size,InputLevel_name,InputLevel_size,OutputLevel_name,OutputLevel_size,num_threads)

if __name__ == "__main__":
    main()