import os
import yaml
import subprocess
import json

def save_dictionary_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)

def load_dictionary_from_file(filename):
    with open(filename, 'r') as file:
        dictionary = json.load(file)
    return dictionary

def modify_yaml_file(input_file, output_file, modified_params):
    # Read the original YAML file
    with open(input_file, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Modify the specified parameters
    for param_path, new_value in modified_params.items():
        # Traverse the YAML structure to find the parameter
        keys = param_path.split('.')
        data = yaml_data
        for key in keys[:-1]:
            data = data.get(key, {})
        param_name = keys[-1]

        # Update the parameter value
        if param_name in data:
            data[param_name] = new_value

    # Write the modified YAML data to the output file
    with open(output_file, 'w') as file:
        yaml.safe_dump(yaml_data, file)



def Execute(prob, arch,arch_constraints,map_const,mapper,outputdir):
    #command = f'timeloop-mapper {arch} {prob} {map_constraints} {arch_constraints} {mapper} --output-dir {outputdir}'
    #/workspace = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    if (map_const == ''):
        command = f'timeloop-mapper {arch} {prob} {arch_constraints} {mapper} -o {outputdir}'
    else:
        command = f'timeloop-mapper {arch} {prob} {arch_constraints} {map_const} {mapper} -o {outputdir}'
    try:
        subprocess.run(command, check=True, shell = True)
        stats_file = f'{outputdir}/timeloop-mapper.stats.txt'
        with open(stats_file, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if line.strip() == "Summary Stats":
                # Read pJ/Compute
                pJ_compute_line = lines[i+22].strip()
                pJ_compute = float(pJ_compute_line.split('=')[1].strip())

                # Read Computes
                computes_line = lines[i+9].strip()
                computes = int(computes_line.split('=')[1].strip())

                # Read Cycles
                # cycles_line = lines[i+4].strip()
                # cycles = int(cycles_line.split(':')[1].strip())

                # Add to summary_stats dictionary
                break
        
        return pJ_compute*computes

    except subprocess.CalledProcessError as e:
        #print(f"Timeloop mapper command failed with error: {e}")
        return e

def GetEnergy(layers,tile,LastLayer,ArchConstFile, Arch, map_const, Mapper, OutDir):
    # layers: list of all layer files
    # tile: (width,height) of the tile
    # LastLayer: The last layer of the fusion stack. 
    CurrOutputWidth = tile[0]
    CurrOutputHeight = tile[1]
    modify_layer_info = {'problem.instance.P':CurrOutputWidth, 'problem.instance.Q':CurrOutputHeight}

    NewFile = layers[LastLayer][:-5]+'New'+'.yaml'
    modify_yaml_file(layers[LastLayer],NewFile,modify_layer_info)

    TileEnergy = Execute(NewFile,Arch,ArchConstFile,map_const,Mapper,OutDir)
    return TileEnergy


def GetNum(FileName):
    num = int(FileName[-7:-5])
    return num


layers = []
folder_path = '/workspace/Gemni_GAN/probGen_copy'  # Please note that the layers should be named 01,02,03,04...
mapper = '/workspace/DATE_final/MNIST/FP32_Gemmini/mapper/mapper.yaml'
arch = '/workspace/DATE_final/MNIST/FP32_Gemmini/arch/gemini_like.yaml'
map_constraints = '/workspace/WrapperTest/ExperimentVGG/simba_like/map_constraints/constraint.yaml'
arch_constraints_folder = '/workspace/WrapperTest/ExperimentVGG/simba_like/arch_constraints/' # Please specify the constriants as SLC.yaml, LBLC.yaml, ELBLC.yaml etc...
OutDir_partial = '/workspace/Gemni_GAN/LogFiles'
Padding_Width = 0
Padding_Height = 0
Dataflow_types = ['SLC','LBLC','ELBLC','WCC','EWCC','OutWCC','Start']

###     SORT THE FILES IN THE FOLDER    ###
for file_name in os.listdir(folder_path):
    if file_name.endswith('.yaml'):
        file_path = os.path.join(folder_path, file_name)
        layers.append(file_path)
layers.sort(key = GetNum, reverse = True)
for df in Dataflow_types:
    ###     GET THE DATA    ###
    LayerData = {}
    ln = 0
    curr_iter = 0
    for layer in layers:
        ln = GetNum(layer)
        file_path = layer
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # PARSE TREE TO GET THE PROBLEM SHAPE #
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

        DataFlow = {}
        TileArray = {}
        TileData = {}
        for tile_width in range(1,Output_Width+1):
            tile_height = tile_width # Asssuming sqaure tiles #
            OutDir = OutDir_partial+'/'+df+f'/Layer{ln}/Tile{tile_height}'
            Input_Width = ((tile_width - 1) * Wstride) - (2 * Padding_Width) + (Wdilation * (Kernel_Width - 1)) + 1
            Input_Height = ((tile_height - 1) * Hstride) - (2 * Padding_Height) + (Hdilation * (Kernel_Height - 1)) + 1

            # NOW CHECKING SIZES #
            # THIS IS NOT NECESSARY HERE AS WE CHECK IT IN THE WRAPPER TOO #
            ans = GetEnergy(layers,(tile_width,tile_height),curr_iter,arch_constraints_folder+'/'+df+'.yaml',arch,map_constraints,mapper,OutDir)
            TileData[tile_width] = ans
        LayerData[ln] = TileData
        curr_iter += 1

        # SAVE LAYER WISE #
        save_dictionary_to_file(LayerData[ln],OutDir_partial+'/'+df+f'/Layer{ln}.json')

        # DELETE TEMP FILES #
        os.remove(layers[ln][:-5]+'New'+'.yaml')

    # SAVE FULL MODEL LEVEL #
    save_dictionary_to_file(LayerData,OutDir_partial+'/'+df+'.json')

    


    