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

def decimal_to_binary_fixed_length(number, length):
    binary = bin(number)[2:].zfill(length)
    return binary

def StackCostGenerator(
        WeightLevel_name,
        WeightLevel_size,
        InputLevel_name,
        InputLevel_size,
        OutputLevel_name,
        OutputLevel_size,
        layers,
        stack,
        SLC,
        Start,
        OutWCC,
        LBLC,
        FullyWCC,
        ELBLC,
        EWCC,
        CheatSheet
):
    # Make the Masks

    cache_names = [WeightLevel_name, InputLevel_name, OutputLevel_name]
    mask = [[1 if inner_name == outer_name else 0 for inner_name in cache_names] for outer_name in cache_names]
    mask = np.array(mask)
    factor = {'SLC':np.array([0,0,0]),'LBLC':np.array([0,1,1]),'ELBLC':np.array([0,1,0]),'WCC':np.array([1,1,1]),'EWCC':np.array([1,1,0]),'OutWCC':np.array([1,0,1]),'Start':np.array([0,0,1])}
    Dataflow_types = ['SLC','LBLC','ELBLC','WCC','EWCC','OutWCC','Start']

    with open(layers[stack[1]], 'r') as file:
        data = yaml.safe_load(file)

    # Output Dimensions:
    for data_space in data['problem']['shape']['data-spaces']:
        '''
        In the next few lines we are extracting the dimension from the workload YAML file
        '''
        if (data_space['name'] == 'Inputs'):
            for dim in data_space['projection']:
                if len(dim) == 2:
                    for i in dim:
                        if (i[1] == 'Wdilation'):
                            Kernel_Width = data['problem']['instance'][i[0]]
                            Wdilation = data['problem']['instance']['Wdilation']
                        elif (i[1] == 'Wstride'):
                            Output_Width = data['problem']['instance'][i[0]]
                            Wstride = data['problem']['instance']['Wstride']
                        elif (i[1] == 'Hdilation'):
                            Kernel_Height = data['problem']['instance'][i[0]]
                            Hdilation = data['problem']['instance']['Hdilation']
                        elif (i[1] == 'Hstride'):
                            Output_Height = data['problem']['instance'][i[0]]
                            Hstride = data['problem']['instance']['Hstride']    
    OutputWidth = Output_Width

    if (stack[0] == stack[1]):
        return SLC[str(stack[0]+1)][str(OutputWidth)], [Output_Width], 'None'
    
    ListOfLists = load_dictionary_from_file(CheatSheet)[str(OutputWidth)]
    BestTilingCost = float('inf') # For the full stack
    BestTiling = []
    BestWeightCaching = 'None'
    # ListOfTiles = []  List of all the tiles that are supposed to be used.


    for ListOfTiles in ListOfLists:
        largest = ListOfTiles[0] # Size of the largest tile. This will be the sacrificial tile
        WeightSizes = {}
        Ms = {}
        LargestTileSizes = {}
        start = stack[0]
        end = stack[1]    
        n = end - start + 1

        ### SCHEDULING THE LARGEST OPTIMALLY AND CHECKING FOR WEIGHT CACHING ###

        Tile_Width = largest
        for fileindex in range (stack[1],stack[0] -1, -1):
            file_path = layers[fileindex]
            with open(file_path, 'r') as file:
                    data = yaml.safe_load(file)

            # Output Dimensions:
            for data_space in data['problem']['shape']['data-spaces']:
                '''
                In the next few lines we are extracting the dimension from the workload YAML file
                '''
                if (data_space['name'] == 'Inputs'):
                    for dim in data_space['projection']:
                        if len(dim) == 2:
                            for i in dim:
                                if (i[1] == 'Wdilation'):
                                    Kernel_Width = data['problem']['instance'][i[0]]
                                    Wdilation = data['problem']['instance']['Wdilation']
                                elif (i[1] == 'Wstride'):
                                    Output_Width = data['problem']['instance'][i[0]]
                                    Wstride = data['problem']['instance']['Wstride']
                                elif (i[1] == 'Hdilation'):
                                    Kernel_Height = data['problem']['instance'][i[0]]
                                    Hdilation = data['problem']['instance']['Hdilation']
                                elif (i[1] == 'Hstride'):
                                    Output_Height = data['problem']['instance'][i[0]]
                                    Hstride = data['problem']['instance']['Hstride']
            Padding_Width = 0
            while (str(Tile_Width) not in LBLC[str(fileindex+1)]): # To make sure some edge cases are handled
                Tile_Width -= 1
            LargestTileSizes[fileindex] = Tile_Width
            WeightSizes[fileindex] = Kernel_Width*Kernel_Height*data['problem']['instance']['C']*data['problem']['instance']['M']
            Ms[fileindex] = data['problem']['instance']['M']
            Tile_Width = ((Tile_Width - 1) * Wstride) - (2 * Padding_Width) + (Wdilation * (Kernel_Width - 1)) + 1
            

        # NOTE LAYER NUMBER IS fileindex + 1

        BestAnswer = float('inf') # For a tiling config of a stack
        ClubSize = math.ceil(n/20) # 20 is a constant that we chose
        q = math.ceil(n/ClubSize)
        
        for Q in range(2**q): # Iterating over all possible Weight Caching Patterns (WCP)
            
            CurrAnswer = 0
            comb = decimal_to_binary_fixed_length(Q,q) # if ChosenCached[i] = 1, then the (start + i)th layer's weights are going to be cached.
            ChosenCached = ''
            for a in range(q):
                ChosenCached += comb[a]*min(ClubSize,n-a*ClubSize)
            TotalCachedWeights = 0
            for layer in range(n):
                if (ChosenCached[layer] == '1'):
                    TotalCachedWeights = WeightSizes[start+layer] # Calculate the total weights required.

            # Now check if we can cache these many weights
            valid = False
            if (WeightLevel_name != InputLevel_name and WeightLevel_name != OutputLevel_name):
                if (TotalCachedWeights <= WeightLevel_size):
                    valid = True

            if (valid):
                
                # For the sacrificial largest tile
                # This tile is called sacrificial as it will not benefit from the caching of weights
                # Note for each tile, we will have to o through the full stack.

                arch_sizes = np.array([[WeightLevel_size - TotalCachedWeights,],[InputLevel_size,],[OutputLevel_size,]])
                ### FOR START LAYER ###
                # 1 option Start

                outputs_size = (LargestTileSizes[start]**2)*Ms[start]
                inputs_size = 0
                weights_size = 0
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['Start']
                if np.any(const >= arch_sizes):
                    CurrAnswer = float('inf')
                    break
                else:
                    CurrAnswer += Start[str(start+1)][str(LargestTileSizes[start])]

                # if (OutputLevel_name == WeightLevel_name):
                #     if ((LargestTileSizes[start]**2)*Ms[start] + TotalCachedWeights <= OutputLevel_size):
                #         CurrAnswer += Start[str(start+1)][str(LargestTileSizes[start])]
                #     else:
                #         CurrAnswer += float('inf')
                # else:
                #     if ((LargestTileSizes[start]**2)*Ms[start] <= OutputLevel_size):
                #         CurrAnswer += Start[str(start+1)][str(LargestTileSizes[start])]
                #     else:
                #         CurrAnswer += float('inf')
                
                ### FOR MIDDLE LAYERS ###
                # Only one optin for the largest sacrificial layer. It has to go Layer by Layer only. If it doesnot fit, then infinte cost

                for i in range(1,n-1):
                    outputs_size = (LargestTileSizes[start+i]**2)*Ms[start+i]
                    inputs_size = (LargestTileSizes[start+i-1]**2)*Ms[start+i-1]
                    weights_size = 0
                    const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['LBLC']
                    if np.any(const >= arch_sizes):
                        CurrAnswer = float('inf')
                        break
                    else:
                        CurrAnswer += LBLC[str(start+i+1)][str(LargestTileSizes[start+i])]

                    # if (OutputLevel_name == WeightLevel_name):
                    #     if ((LargestTileSizes[start+i-1]**2)*Ms[start+i-1] + (LargestTileSizes[start+i]**2)*Ms[start+i] + TotalCachedWeights<= OutputLevel_size):
                    #         CurrAnswer += LBLC[str(start+i+1)][str(LargestTileSizes[start+i])]
                    #     else:
                    #         CurrAnswer += float('inf')
                    # else:
                    #     if ((LargestTileSizes[start+i-1]**2)*Ms[start+i-1] + (LargestTileSizes[start+i]**2)*Ms[start+i] <= OutputLevel_size):
                    #         CurrAnswer += LBLC[str(start+i+1)][str(LargestTileSizes[start+i])]
                    #     else:
                    #         CurrAnswer += float('inf')         
                                       
                ### FOR LAST LAYER ###
                # Only one option, ELBLC
                outputs_size = 0
                inputs_size = (LargestTileSizes[end-1]**2)*Ms[end-1]
                weights_size = 0
                const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['ELBLC']
                if np.any(const >= arch_sizes):
                    CurrAnswer = float('inf')
                    break
                else:
                    CurrAnswer += ELBLC[str(end+1)][str(LargestTileSizes[end])] 
                
                # if (OutputLevel_name == WeightLevel_name):
                #     if ((LargestTileSizes[end-1]**2)*Ms[end-1] + TotalCachedWeights <= OutputLevel_size):
                #         CurrAnswer += ELBLC[str(end+1)][str(LargestTileSizes[end])]
                #     else:
                #         CurrAnswer += float('inf')        
                # else:            
                #     if ((LargestTileSizes[end-1]**2)*Ms[end-1] <= OutputLevel_size):
                #         CurrAnswer += ELBLC[str(end+1)][str(LargestTileSizes[end])]
                #     else:
                #         CurrAnswer += float('inf')

    ###########################################################################################################################################

                if (CurrAnswer == float('inf')):
                    break  

                # For the others that use the benefits of weight caching.
                # The cheat sheet has such tiling where all the tiles have the same shape. In such a case, we don't have to loop over all tiles.
                # Here we can multiply one tile's val with the number of such tiles.
                
                if (len(np.unique(ListOfTiles[1:])) == 1):
                    
                    CurrAnswer_temp = 0
                    remaining_tiles = len(ListOfTiles[1:])
                    tile = ListOfTiles[1]
                    TileSizes = {}
                    Tile_Width = tile
                    for fileindex in range (stack[1],stack[0] -1, -1):
                        file_path = layers[fileindex]
                        with open(file_path, 'r') as file:
                                data = yaml.safe_load(file)

                        # Output Dimensions:
                        for data_space in data['problem']['shape']['data-spaces']:
                            '''
                            In the next few lines we are extracting the dimension from the workload YAML file
                            '''
                            if (data_space['name'] == 'Inputs'):
                                for dim in data_space['projection']:
                                    if len(dim) == 2:
                                        for i in dim:
                                            if (i[1] == 'Wdilation'):
                                                Kernel_Width = data['problem']['instance'][i[0]]
                                                Wdilation = data['problem']['instance']['Wdilation']
                                            elif (i[1] == 'Wstride'):
                                                Output_Width = data['problem']['instance'][i[0]]
                                                Wstride = data['problem']['instance']['Wstride']
                                            elif (i[1] == 'Hdilation'):
                                                Kernel_Height = data['problem']['instance'][i[0]]
                                                Hdilation = data['problem']['instance']['Hdilation']
                                            elif (i[1] == 'Hstride'):
                                                Output_Height = data['problem']['instance'][i[0]]
                                                Hstride = data['problem']['instance']['Hstride']
                        Padding_Width = 0
                        while (str(Tile_Width) not in LBLC[str(fileindex+1)]):
                            Tile_Width -= 1
                        TileSizes[fileindex] = Tile_Width
                        Tile_Width = ((Tile_Width - 1) * Wstride) - (2 * Padding_Width) + (Wdilation * (Kernel_Width - 1)) + 1


                    ### FOR FIRST LAYER ###
                    # Only two options, Start, OutWCC
                        
                    outputs_size = (TileSizes[start]**2)*Ms[start]
                    inputs_size = 0
                    weights_size = 0 
                    if (ChosenCached[0] == '1'):
                        const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['OutWCC']
                        if np.any(const >= arch_sizes):
                            CurrAnswer_temp += float('inf')
                        else:
                            CurrAnswer_temp += OutWCC[str(start+1)][str(TileSizes[start])]
                    else:
                        const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['Start']
                        if np.any(const >= arch_sizes):
                            CurrAnswer_temp += float('inf')
                        else:
                            CurrAnswer_temp += Start[str(start+1)][str(TileSizes[start])]

                    #     if (OutputLevel_name == WeightLevel_name):
                    #         if (TotalCachedWeights + (TileSizes[start]**2)*Ms[start] <= WeightLevel_size):
                    #             CurrAnswer += OutWCC[str(start+1)][str(TileSizes[start])]
                    #         else:
                    #             CurrAnswer += float('inf')
                    #     else:
                    #         CurrAnswer += OutWCC[str(start+1)][str(TileSizes[start])]
                    # else:
                    #     CurrAnswer += Start[str(start+1)][str(TileSizes[start])]
                            
                    ### FOR MIDDLE LAYERS ### 
                    # Two Options exist. Those that have been chosen to be cached will be cached. If they can't be cached then return infinte cost.
                    # The other will have to be scheduled layer by layer.
                            
                    for i in range(1,n-1):
                        outputs_size = (TileSizes[start+i]**2)*Ms[start+i] 
                        inputs_size = (TileSizes[start+i-1]**2)*Ms[start+i-1]
                        weights_size = 0
                        if (ChosenCached[i] == '1'):
                            const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['WCC']
                            if np.any(const >= arch_sizes):
                                CurrAnswer_temp += float('inf')
                            else:
                                CurrAnswer_temp += FullyWCC[str(start+i+1)][str(TileSizes[start+i])]
                        else:
                            const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['LBLC']
                            if np.any(const >= arch_sizes):
                                CurrAnswer_temp += float('inf')
                            else:
                                CurrAnswer_temp += LBLC[str(start+i+1)][str(TileSizes[start+i])]

                        #     if (OutputLevel_name == WeightLevel_name):
                        #         if (TotalCachedWeights + (TileSizes[start+i]**2)*Ms[start+i] + (TileSizes[start+i-1]**2)*Ms[start+i-1] <= WeightLevel_size):
                        #             CurrAnswer += FullyWCC[str(start+i+1)][str(TileSizes[start+i])]
                        #         else:
                        #             CurrAnswer += float('inf')          
                        #     else:
                        #         CurrAnswer += FullyWCC[str(start+i+1)][str(TileSizes[start+i])]
                        # else:
                        #     CurrAnswer += LBLC[str(start+i+1)][str(TileSizes[start+i])]
                                
                    ### FOR LAST LAYER ###
                                
                    if (ChosenCached[n-1] == '1'):
                        outputs_size = 0 
                        inputs_size = (TileSizes[end-1]**2)*Ms[end-1]
                        weights_size = 0
                        const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['EWCC']
                        if np.any(const >= arch_sizes):
                            CurrAnswer_temp += float('inf')
                        else:
                            CurrAnswer_temp += EWCC[str(end+1)][str(TileSizes[end])]
                    else:
                        const = np.dot(mask,np.array([[weights_size,],[inputs_size,],[outputs_size,]]))*factor['ELBLC']
                        if np.any(const >= arch_sizes):
                            CurrAnswer_temp += float('inf')
                        else:
                            CurrAnswer_temp += ELBLC[str(end+1)][str(LargestTileSizes[end])]

                    #     if (InputLevel_name == WeightLevel_name):
                    #         if (TotalCachedWeights + (TileSizes[end]**2)*Ms[end] <= WeightLevel_size):
                    #             CurrAnswer += EWCC[str(end+1)][str(TileSizes[end])]
                    #         else:
                    #             CurrAnswer += float('inf')
                    #     else:
                    #         CurrAnswer += EWCC[str(end+1)][str(TileSizes[end])]
                    # else:
                    #     CurrAnswer += ELBLC[str(end+1)][str(LargestTileSizes[end])]

                    CurrAnswer += (remaining_tiles)*CurrAnswer_temp  
                else: 
                    continue # CODE for uneven tiled scenario comes here.

            if (CurrAnswer == 0):
                    CurrAnswer = float('inf')

                # for tile in ListOfTiles[1:]:
                    
    ###################################################################################################################
                
            if (CurrAnswer < BestAnswer):
                BestAnswer = CurrAnswer
                BestCaching = ChosenCached
                BestListofTiles = ListOfTiles
                # print(f"({start}, {end}) Cost: {CurrAnswer} Tiling: {ListOfTiles}")

        if (BestAnswer < BestTilingCost):
            BestTilingCost = BestAnswer
            BestTiling = BestListofTiles
            if (BestCaching == "0"*n):
                BestCaching = "None"
            BestWeightCaching = BestCaching

    return BestTilingCost, BestTiling, BestWeightCaching

def GetNum(FileName):
    num = int(FileName[-7:-5])
    return num

def save_dictionary_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)

def load_dictionary_from_file(filename):
    with open(filename, 'r') as file:
        dictionary = json.load(file)
    return dictionary

def fill_rectangle(rectangle, tiles):
    """
    Function to fill a rectangle with given square tiles.
    
    Args:
    - rectangle: tuple, dimensions of the rectangle (rows, cols)
    - tiles: list of integers representing the dimensions of square tiles
    
    Returns:
    - 2D numpy array representing the filled rectangle
    """
    rows, cols = rectangle
    # Create a numpy array to represent the output rectangle
    output = np.zeros((rows, cols), dtype=int)
    
    def fill_recursively(tile_index, start_row, start_col, end_row, end_col):
        nonlocal output
        # Base case: if we've filled all the tiles
        if tile_index == len(tiles):
            return True
        
        tile_size = tiles[tile_index]
        
        # Try to place the current tile in different positions
        for r in range(start_row, end_row - tile_size + 1):
            for c in range(start_col, end_col - tile_size + 1):
                # Check if the current position is empty
                if np.all(output[r:r+tile_size, c:c+tile_size] == 0):
                    # Place the tile
                    output[r:r+tile_size, c:c+tile_size] = tile_index + 1
                    # Try to fill the rest of the rectangle recursively
                    if fill_recursively(tile_index + 1, start_row, start_col, end_row, c+tile_size):
                        return True
                    # Undo the placement if filling the rest of the rectangle fails
                    output[r:r+tile_size, c:c+tile_size] = 0
        
        return False
    
    # Start filling the rectangle recursively
    fill_recursively(0, 0, 0, rows, cols)
    
    return output

def write_output_to_file(output, filename):
    with open(filename, 'a') as file:
        file.write(np.array2string(output, separator=', '))

def worker(worker_id, n, m, worker_costs, worker_tilings, worker_WC, WeightLevel_name, WeightLevel_size, InputLevel_name, InputLevel_size, OutputLevel_name, OutputLevel_size, layers, SLC, Start, OutWCC, LBLC, FullyWCC, ELBLC, EWCC, CheatSheet, output_dict):
    cost = {}
    BestTiling = {}
    WeightsCached = {}
    try:
        for end in range(n):
            for start in range(end + 1):
                if ((start + end) % m) == worker_id:
                    temp = StackCostGenerator(WeightLevel_name, WeightLevel_size, InputLevel_name, InputLevel_size, OutputLevel_name, OutputLevel_size, layers, (start, end), SLC, Start, OutWCC, LBLC, FullyWCC, ELBLC, EWCC, CheatSheet)
                    cost[(start, end)], BestTiling[(start, end)], WeightsCached[(start, end)] = temp[0], temp[1], temp[2]
                    print(f"[ {worker_id}]  Stack ({start},{end}): {cost[(start, end)]} uJ")
        worker_costs[worker_id] = cost
        worker_tilings[worker_id] = BestTiling
        worker_WC[worker_id] = WeightsCached
    except KeyboardInterrupt:
        print(f"Worker {worker_id}: Keyboard interrupt received, terminating.")
    output_dict[worker_id] = (worker_costs[worker_id], worker_tilings[worker_id], worker_WC[worker_id])




st = time.time()
layers = []

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< INPUTS SECTION STARTS HERE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
folder_path = '/TestingDF/DeepFrack_temp/Examples/MobileNet_SimbaSystolic/MobileNet'  # Folder containing all the layers
BenchMrkrLog_folder = '/TestingDF/DeepFrack_temp/Examples/MobileNet_SimbaSystolic/BenchMrkr_log'
CheatSheet = '/TestingDF/DeepFrack_temp/CheatSheet.json'
OutputImgFile = '/TestingDF/DeepFrack_temp/Examples/MobileNet_SimbaSystolic/Plots/Comparison_multiT.jpg'
LogFile = '/TestingDF/DeepFrack_temp/Examples/MobileNet_SimbaSystolic/DeepFrack_logfile_MultiT.txt'
WeightLevel_name = 'PEWeightBuffer'
WeightLevel_size = 4096*128
InputLevel_name = 'GlobalBuffer'
InputLevel_size = 65536
OutputLevel_name = 'GlobalBuffer'
OutputLevel_size = 65536

M = 10 # Number of threads to use.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< INPUTS SECTION ENDS HERE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

for file_name in os.listdir(folder_path):
    if file_name.endswith('.yaml'):
        file_path = os.path.join(folder_path, file_name)
        layers.append(file_path)
layers.sort(key = GetNum)
n = len(layers)

### Now the layers are in increasing order of their layer number. ###
stacks = []
cost = {}
BestTiling = {}
WeightsCached = {}

### GENERATE THE COSTS OF ALL THE STACKS ###

worker_costs = {}
worker_tilings = {}
worker_WC = {}

with Manager() as manager:
    output_dict = manager.dict()

    processes = []
    for i in range(M):
        p = Process(target=worker, args=(
        i, n, M, worker_costs, worker_tilings, worker_WC, WeightLevel_name, WeightLevel_size, InputLevel_name,
        InputLevel_size, OutputLevel_name, OutputLevel_size, layers, SLC, Start, OutWCC, LBLC, FullyWCC, ELBLC,
        EWCC, CheatSheet, output_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for worker_id, (w_cost, w_tiling, w_wc) in output_dict.items():
        cost.update(w_cost)
        BestTiling.update(w_tiling)
        WeightsCached.update(w_wc)

for end in range(n):
    for start in range(end+1):
        stacks.append((start,end))  # start and end are the indexes of the layers in the list named layers.
        # print("Stack", (start,end),end = " ")
        # temp = StackCostGenerator(WeightLevel_name,WeightLevel_size, InputLevel_name, InputLevel_size, OutputLevel_name, OutputLevel_size, layers, (start,end), SLC, Start, OutWCC, LBLC, FullyWCC,ELBLC,EWCC,CheatSheet)
        # cost[(start,end)],BestTiling[(start,end)],WeightsCached[(start,end)] = temp[0], temp[1], temp[2]
        # print(":",cost[(start,end)], "uJ")

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

### FIND SINGLE LAYER COSTS FOR COMPARISON ###
OutputSizes = []
SingleLayerCost = 0
for layn in range(1,n+1):
    with open(layers[layn-1], 'r') as file:
        data = yaml.safe_load(file)
    # Output Dimensions:
    for data_space in data['problem']['shape']['data-spaces']:
        '''
        In the next few lines we are extracting the dimension from the workload YAML file
        '''
        if (data_space['name'] == 'Inputs'):
            for dim in data_space['projection']:
                if len(dim) == 2:
                    for i in dim:
                        if (i[1] == 'Wdilation'):
                            Kernel_Width = data['problem']['instance'][i[0]]
                            Wdilation = data['problem']['instance']['Wdilation']
                        elif (i[1] == 'Wstride'):
                            Output_Width = data['problem']['instance'][i[0]]
                            Wstride = data['problem']['instance']['Wstride']
                        elif (i[1] == 'Hdilation'):
                            Kernel_Height = data['problem']['instance'][i[0]]
                            Hdilation = data['problem']['instance']['Hdilation']
                        elif (i[1] == 'Hstride'):
                            Output_Height = data['problem']['instance'][i[0]]
                            Hstride = data['problem']['instance']['Hstride']
    OutputSizes.append((Output_Height,Output_Width))
    SingleLayerCost += SLC[str(layn)][str(Output_Height)]
    
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
for stack in d:
    distinct_tiles.append(np.unique(BestTiling[stack])[0])

for stack in d:
    if (WeightsCached[stack] == 'None'):
        WCPS.append('0'*(stack[1]-stack[0]+1))
    else:
        WCPS.append(WeightsCached[stack])

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
        # tiles = BestTiling[d[stack]]
        # tiles.sort(reverse=True)
        # coloured_tiling = fill_rectangle(OutputSizes[d[stack][1]], tiles)
        # write_output_to_file(coloured_tiling, LogFile)
        sf.write("Layers, whose weights were cached: ")
        if (WeightsCached[d[stack]] == 'None'):
            sf.write("None")
        else:
            for wl in range(len(WeightsCached[d[stack]])):
                if (WeightsCached[d[stack]][wl] == '1'):
                    sf.write(f"Layer {d[stack][0] + wl} ")
        sf.write("\n")
        # print(f"Fuse Stack {j}: {d[stack]} with a cost of {cost[d[stack]]} uJ")
        # print(f"Tiling for Fused Stack{j} is:")
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