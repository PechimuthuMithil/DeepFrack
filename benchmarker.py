# only benchmarks for og file sizes + tiling of smallest size tile
# smallest size -> least no of elements -> and for all configurations in input files i.e area = 16 can be (8,2) and (2,8)

import yaml
import json
import subprocess
import time
import os


def save_dictionary_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary,file)

def load_dictionary_from_file(filename):
    with open(filename, 'r') as file:
        dictionary = json.load(file)
    return dictionary
    
def Execute(prob,arch,arch_constraints,components,map_const,mapper,outputdir):
    #command = f'timeloop-mapper {arch} {prob} {map_constraints} {arch_constraints} {mapper} --output-dir {outputdir}'
    #/workspace = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    if (map_const == ''):
        if (components == ''):
            command = f'timeloop-mapper {arch} {prob} {arch_constraints} {mapper} -o {outputdir}'
        else:
            command = f'timeloop-mapper {arch} {prob} {arch_constraints} {components}/* {mapper} -o {outputdir}'
    else:
        if (components == ''):
            command = f'timeloop-mapper {arch} {prob} {arch_constraints} {map_const} {mapper} -o {outputdir}'
        else:
            command = f'timeloop-mapper {arch} {prob} {arch_constraints} {components}/* {map_const} {mapper} -o {outputdir}'
    try:
        subprocess.run(command, check=True, shell = True)
        stats_file = f'{outputdir}/timeloop-mapper.stats.txt'
        with open(stats_file, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if line.strip() == "Summary Stats":
               
                energy_line = lines[i+5].strip()
                print("Energy line in stats file found -->", energy_line)
                energy = float(energy_line.split(':')[1][:-3].strip())
                break

        return energy

    except subprocess.CalledProcessError as e:
        print(f"Timeloop mapper command failed with error: {e}")
        return e

def modify_yaml_file(file_path, modified_params):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

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
    
    with open(file_path, 'w') as file:
        yaml.safe_dump(yaml_data, file,sort_keys=False)



def GetEnergy(tile,file,ArchConstFile, Arch, components,map_const, Mapper, OutDir,add):

    if add:
        modify_layer_info = {'problem.instance.P':tile, 'problem.instance.Q':tile}
    else:
         modify_layer_info = {'problem.instance.P':tile, 'problem.instance.Q':tile, 'problem.instance.K': tile}

    modify_yaml_file(file,modify_layer_info)
    TileEnergy = Execute(file,Arch,ArchConstFile,components,map_const,Mapper,OutDir)

    return TileEnergy

layers = []



def run_benchmarker(folder_path,mapper,arch,components,map_constraints,add_map_constraints,arch_constraints_folder,add_constraints_folder,OutDir_partial,problem_mul,problem_add):
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

    
    ###     SORT THE FILES IN THE FOLDER    ###
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.yaml'):
            file_path = os.path.join(folder_path, file_name)
            layers.append(file_path)

    Dataflow_types = ['SLC','LBLC','ELBLC','WCC','EWCC','OutWCC','Start']

    st = time.time()

    small=float('inf')
    P=[]
    Q=[]
    for layer in layers:
        file_path = layer
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # PARSE TREE TO GET THE PROBLEM SHAPE # 
        inputs=[]
        weights=[]
        for data_space in data['problem']['shape']['data-spaces']:
            if (data_space['name'] == 'Inputs'):
                for dim in data_space['projection']:
                    inputs.append(dim[0][0])          
            if (data_space['name'] == 'Weights'):
                for dim in data_space['projection']:
                    weights.append(dim[0][0])

        Output_Width=data['problem']['instance'][inputs[0]] #P
        Output_Length=data['problem']['instance'][weights[0]] #Q
        weight_mid=data['problem']['instance'][weights[1]] #K
        input_mid=data['problem']['instance'][inputs[1]]
        
        assert weight_mid == input_mid, "Mismatch"  

        small=min(Output_Width,Output_Length,weight_mid,small)
        P.append(Output_Width)
        Q.append(Output_Length)

    Data={}
    for df in Dataflow_types:
        ###     GET THE DATA    ###
        FullData = {}
        MulData={}
        AddData={}
        l=0

        # for layer in layers:
        #     ln= os.path.splitext(os.path.basename(layers[l]))[0]
        #     file_path = layer
        #     with open(file_path, 'r') as file:
        #         data = yaml.safe_load(file)

        #         OutDir = OutDir_partial+'/'+df+f'/Full/{ln}'
        #         ans = Execute(layer,arch,arch_constraints_folder+'/'+df+'.yaml',components,map_constraints,mapper,OutDir)
        #         FullData[ln] = ans
        #         l+=1

        # Data[0]=FullData
        # save_dictionary_to_file(FullData,OutDir_partial+'/'+df+f'/Full.json')


        for tile in range(64,1025):  
            OutDir = OutDir_partial+'/'+df+f'/Mul/Tile{tile}'
            ans = GetEnergy(tile,problem_mul,arch_constraints_folder+'/'+df+'.yaml',arch,components,map_constraints,mapper,OutDir,False)
            if isinstance(ans, subprocess.CalledProcessError):
                print(f"Error encountered with tile {tile}: {ans}")
            else:
                MulData[f"{tile}"] = ans

        save_dictionary_to_file(MulData,OutDir_partial+'/'+df+f'/Mul.json')
        Data[1]=MulData

        for tile in range(64,1025):
            OutDir = OutDir_partial+'/'+df+f'/Add/Tile{tile}'
            ans = GetEnergy(tile,problem_add,add_constraints_folder+'/'+df+'.yaml',arch,components,add_map_constraints,mapper,OutDir,True)
            if isinstance(ans, subprocess.CalledProcessError):
                    print(f"Error encountered with tile {tile}: {ans}")
            else:
                    AddData[f"{tile}"] = ans

        save_dictionary_to_file(AddData,OutDir_partial+'/'+df+f'/Add.json')
        Data[2]=AddData
                            
        save_dictionary_to_file(Data,OutDir_partial+'/'+df+'.json')
        print(f"SAVED BENCHMARKED DATA AT --> {OutDir_partial+'/'+df+'.json'}")
        
    et = time.time()
    print("Total Elapsed Time to Benchmark: ",time.strftime("%H:%M:%S", time.gmtime(et-st)))

def main():

    folder_path = 'BERT_Simba/BERT'
    mapper='BERT_Simba/mapper/mapper.yaml'
    arch= 'BERT_Simba/arch/simba_like.yaml'
    components='BERT_Simba/arch/components'
    map_constraints='BERT_Simba/constraints/simba_like_map_constraints.yaml'
    add_map_constraints='BERT_Simba/constraints-add/simba_like_map_constraints.yaml'
    arch_constraints_folder='BERT_Simba/constraints'
    add_constraints_folder='BERT_Simba/constraints-add'
    OutDir_partial='BERT_Simba/Benchmarks2'
    problem_mul='BERT_Simba/problem_mul.yaml'
    problem_add='BERT_Simba/problem_add.yaml'

    run_benchmarker(folder_path,mapper,arch,components,map_constraints,add_map_constraints,arch_constraints_folder,add_constraints_folder,OutDir_partial,problem_mul,problem_add)

if __name__ == "__main__":
    main()
