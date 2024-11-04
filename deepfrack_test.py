# this file has approach with no partial products


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
# OutWCC - for partial sum tiling and otherwise for full
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
        num_tiles=[]
            ### SCHEDULING THE LARGEST OPTIMALLY AND CHECKING FOR WEIGHT CACHING ###
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
            num_tiles.append([P//tile,Q//tile,K//tile])
        
        BestAnswer = float('inf') # For a tiling config of a stack
        ClubSize = math.ceil(n/20) # 20 is a constant that we chose
        q = math.ceil(n/ClubSize) # no of clubbed layers

        for Q in range(2**q): # Iterating over all possible Weight Caching Patterns (WCP)
            CurrAnswer = 0
            comb = decimal_to_binary_fixed_length(Q,q) # if ChosenCached[i] = 1, then the (start + i)th layer's weights are going to be cached.
            ChosenCached = ''
            for a in range(q):
                ChosenCached += comb[a]*min(ClubSize,n-a*ClubSize)
            TotalCachedWeights = 0
            for layer in range(n):
                # in attn and qkv inputs = weights 
                if (ChosenCached[layer] == '1' and layer_names[start+layer]!='QKV' and layer_names[start+layer]!='Attn'):   
                    TotalCachedWeights = WeightSizes[start+layer]*num_parallel[start+layer] # Calculate the total weights required.

            valid = False
            if (WeightLevel_name != InputLevel_name and WeightLevel_name != OutputLevel_name):
                if (TotalCachedWeights <= WeightLevel_size):
                    valid = True

            if (valid):
                arch_sizes = np.array([[WeightLevel_size - TotalCachedWeights,],[InputLevel_size,],[OutputLevel_size,]])
                                        
                ### FOR FIRST LAYER ###
                # Only two options, Start, OutWCC//.

                # an entire row in output needed for next layer
                outputs_size = (tile)**2 * num_tiles[0][1] * num_parallel[start]
                inputs_size = 0 
                weights_size = 0
                Trace=layer_names[start]# to verify the path taken
                
                # for attn and qkv inputs = weights so no weight caching in start
                if (ChosenCached[0] == '0' or layer_names[start]=='Attn' or layer_names[start]=='QKV'):
                    const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['Start']
                    if np.all(const >= arch_sizes):
                        # print('Out at 0')
                        CurrAnswer = float('inf')
                    else:
                        try:
                            MultiplicationCost = Start["1"][str(tile)]
                            AdditionCost = Start["2"][str(tile)]
                            Trace+=':Start'
                        except KeyError:
                            CurrAnswer=float('inf')      
                else:
                    const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['OutWCC']
                    if np.all(const >= arch_sizes):
                        # print('Out at 0')
                        CurrAnswer = float('inf')
                    else:
                        try:
                            MultiplicationCost = OutWCC["1"][str(tile)]
                            AdditionCost = OutWCC["2"][str(tile)]
                            Trace+=':OutWCC'
                        except KeyError:
                            CurrAnswer=float('inf')
                CurrAnswer+=(MultiplicationCost*num_tiles[0][2]+AdditionCost*(num_tiles[0][2]-1))*num_tiles[0][1]*num_tiles[0][0]*num_parallel[start]
            
            # FOR MIDDLE LAYERS
                
                for layer in range(1,n-1):
                    outputs_size = (tile)**2 * num_tiles[layer][1] * num_parallel[start+layer]
                    inputs_size = (tile)**2 * num_tiles[layer-1][2] * num_parallel[start+layer]

                    if(layer_names[start+layer]=='MHead'): # identical input
                        inputs_size = (tile)**2 * num_tiles[layer-1][2]

                    weights_size = 0
                    Trace+=f'-{layer_names[start+layer]}'
                    # for attn and qkv inputs = weights so weights always cached with inputsgh
                    if (ChosenCached[layer] == '1' or layer_names[start+layer]=='Attn' or layer_names[start+layer]=='QKV'):
                        if(layer_names[start+layer]=='QKV'):
                            # two rows produce one element in output; since next layer is disconnected does not matter
                            weights_size= inputs_size     
                        if(layer_names[start+layer]=='Attn'):
                            # entire weight block needed for one output row to go to QKV
                            weights_size= inputs_size * num_tiles[layer-1][1]               
                        const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['WCC']
                        if np.all(const >= arch_sizes):
                            # print(f'Out at {layer},{layer_names[start+layer]}1')
                            CurrAnswer = float('inf')
                        else:
                            try:
                                MultiplicationCost +=FullyWCC["1"][str(tile)]
                                AdditionCost +=FullyWCC["2"][str(tile)]
                                Trace+=f':FullyWCC{tile}'
                            except KeyError:
                                CurrAnswer=float('inf')
                    else:
                        const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['LBLC']
                        if np.all(const >= arch_sizes):
                            # print(f'Out at {layer},{layer_names[start+layer]}2')
                            CurrAnswer = float('inf')
                        else:
                            try:
                                MultiplicationCost += LBLC["1"][str(tile)]
                                AdditionCost += LBLC["2"][str(tile)]
                                Trace+=f':LBLC{tile}'
                            except KeyError:
                                CurrAnswer=float('inf')
                    CurrAnswer+=(MultiplicationCost*num_tiles[0][2]+AdditionCost*(num_tiles[0][2]-1))*num_tiles[0][1]*num_tiles[0][0]*num_parallel[start+layer]


                # FOR LAST LAYER

                outputs_size = 0
                inputs_size = (tile)**2 * num_tiles[n-2][2] * num_parallel[end-1]
                weights_size = 0
                Trace+=f'-{layer_names[end]}'

                if(layer_names[end-1]=='MHead'):
                    inputs_size = (tile)**2 * num_tiles[n-2][2]

                if (ChosenCached[n-1] == '1' or layer_names[end-1]=='Attn' or layer_names[end-1]=='QKV'):
                    if(layer_names[end-1]=='QKV'):
                        weights_size= inputs_size
                    if(layer_names[end-1]=='Attn'):
                        weights_size= inputs_size * num_tiles[n-2][1] # tiling along Q  
                    const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['EWCC']
                    if np.all(const >= arch_sizes):
                        # print(f'Out at end')
                        CurrAnswer = float('inf')
                    else:
                        try:
                            MultiplicationCost += EWCC["1"][str(tile)]
                            AdditionCost += EWCC["2"][str(tile)]
                            Trace+=f':EWCC'
                        except KeyError:
                            CurrAnswer=float('inf')
                else:
                    const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['ELBLC']
                    if np.all(const >= arch_sizes):
                        # print(f'Out at end')
                        CurrAnswer = float('inf')
                    else:
                        try:
                            MultiplicationCost += ELBLC["1"][str(tile)]
                            AdditionCost += ELBLC["2"][str(tile)]
                            Trace+=f':ELBLC'
                        except KeyError:
                            CurrAnswer=float('inf')
                CurrAnswer+=(MultiplicationCost*num_tiles[0][2]+AdditionCost*(num_tiles[0][2]-1))*num_tiles[0][1]*num_tiles[0][0]*num_parallel[end]


            if (CurrAnswer == 0):
                # print("Weights didnt fit")
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
                

            

            
                
                                




