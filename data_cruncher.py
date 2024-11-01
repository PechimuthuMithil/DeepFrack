import matplotlib.pyplot as plt
import numpy as np
import yaml
import json

def pad_tile(P,Q,K,tile_size):
    if (P%tile_size != 0):
        P = P + (tile_size - P%tile_size)
    if (Q%tile_size != 0):
        Q = Q + (tile_size - Q%tile_size)
    if (K%tile_size != 0):
        K = K + (tile_size - K%tile_size)
    return P,Q,K


def load_dictionary_from_file(filename):
    with open(filename, 'r') as file:
        dictionary = json.load(file)
    return dictionary

FullyWCCFile = 'BERT_Simba/Benchmarks1/WCC.json'
SLCFile = 'BERT_Simba/Benchmarks1/SLC.json'

layer="BtlCont"
WCC = load_dictionary_from_file(FullyWCCFile)
SLC = load_dictionary_from_file(SLCFile)

small = 64
file_path=f'BERT_Simba/BERT/{layer}.yaml'
energy3={}
energy1 = SLC["0"][layer]
energy2 = WCC["0"][layer]

if(energy1<energy2):
    min_energy = energy1
    at = "SLC"
else:
    min_energy = energy2
    at = "WCC"

for tile in range(2,small+1): # all sizes from 2 

        ### SCHEDULING THE LARGEST OPTIMALLY AND CHECKING FOR WEIGHT CACHING ###
        # find size of weights to be cached and tile sizes per layer then check if they can be cache
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
        
        num_tiles=[P//tile,Q//tile,K//tile]

        try:
            energy3[tile] = (WCC["1"][str(tile)]+WCC["2"][str(tile)]) * num_tiles[0] * num_tiles[1] * num_tiles[2]
            if(energy3[tile]<min_energy and energy3[tile]>0):
                min_energy=energy3[tile]
                at = f"WCC{tile}"
        except KeyError: 
            pass
    

print(min_energy,at)

# Plot energy3 vs tile
tiles = list(energy3.keys())[7:]
energies = list(energy3.values())[7:]
print(energies)
print(tiles)

plt.figure()
plt.scatter(tiles, energies, marker='o', label='Energy3')

# Add points for energy1 and energy2 at (1024, 1024)
# plt.scatter([1024], [energy1], color='red', zorder=5, label='Energy1')
# plt.scatter([1024], [energy2], color='blue', zorder=5, label='Energy2')

plt.xlabel('Tile Size')
plt.ylabel('Energy')
plt.title('Energy vs Tile Size')
plt.legend()
plt.grid(True)
plt.savefig(f"{layer}_compare.png")



