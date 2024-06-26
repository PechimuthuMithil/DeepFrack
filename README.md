Note: This is a dummy repository. The source code in this repository support CNNs. <-- Containers: GeminiBenchMarker2, MithilDeepFrackTesting  
TODO:   
1) Make the Hardware Benchmarker, multi threaded. That is, 7 threads for each of the 7 benchmarks.
2) The colouring algorthm that baically tiles a sqaure with tiles from a tile list is ery slow for some reason for medium size squares. Need to look into it to optimize it or parallelize it.
3) NEED TO PERFORM SENSITIVITY ANALYSIS

---
```
                             _____                      _______               _     
                            (____ \                    (_______)             | |    
                             _   \ \ ____ ____ ____     _____ ____ ____  ____| |  _ 
                            | |   | / _  ) _  )  _ \   |  ___) ___) _  |/ ___) | / )
                            | |__/ ( (/ ( (/ /| | | |  | |  | |  ( ( | ( (___| |< ( 
                            |_____/ \____)____) ||_/   |_|  |_|   \_||_|\____)_| \_)
                                              |_|                                   
```
---
# PROJECT DEEPFRACK 
DeepFrack is a novel framework developed for enhancing energy efficiency and reducing latency in deep learning workloads executed on hardware accelerators. By optimally fusing layers and implementing an asymmetric tiling strategy, DeepFrack addresses the limitations of traditional layer-by-layer scheduling. The DeepFrack project aims to build succinct and simple framework that contributes to the ongoing efforts in designing more efficient hardware accelerators for machine learning workloads.  

DeepFrack is wrapper to [Timeloop](https://timeloop.csail.mit.edu/timeloop) written in python.

This document is to serve as a guide to users for installing and using DeepFrack. This document (along with the paper) will also provide any user with additional knowledge required to go through the complete source code of DeepFrack and suggest changes. For any queries please contact Mithil Pechimuthu (pechimuthumithil@iitgn.ac.in).  

![image](https://github.com/PechimuthuMithil/DeepFrack_temp/assets/119656326/ae79b3ce-aa0f-45d5-b8d6-ec0e7591f1f4)  

The DeepFrack project consits of three modules that together make the final working tool. These consist of:
1) #### Hardware Benchmarcker: This module created the costs metric over which the DeepFrack core optimizes. It's output is fed into the DeepFrack Core.  
2) #### DeepFrack Core: This is the core mapping space explorer. The output files from go into the Statistics Generator for a fine grained analysis.  
3) #### Statistics Generator: This analyzes the output from the core, to provide the user with each component level statistics like energy/MAC for each component, total computes, total cycles.   

## Contributors
1) Tom Glint Issac
2) Joycee Mekie  
3) Mithil Pechimuthu  
   
# Installation
The files in this repository are suffecient for deploying the tool.  

## Dependencies
DeepFrack relies on Timeloop and Accelergy for it's cost metrics. Hence it is absolutely necessary for the user to have timeloop installed for running DeepFrack. One can find the the installation procedure and other useful information regarding the dependencies in the [Timeloop documentation](https://timeloop.csail.mit.edu/timeloop/installation).  
The source code for DeepFrack is in python. Hence python version >= 3.0.0 along with libraries like numpy must be present.  

# Usage
These modules are to be executed in the given sequence.  
## Hardware Benchmarcker
The Benchmarker is the module of DeepFrack that uses timeloop to generate costs (total energy by default) for each step taken by a point in the design sapce. The benchmarker takes the inputs provided to it and returns costs in a folder of .json files. The examples will make it more clear.  
### Inputs
The Bencmarker must be provided with the following YAML files.
1) Timeloop mapper YAML file.
2) YAML file to describe the architecture of the accelerator.
3) YAML file that describes the mapping constraints for timeloop.
4) A folder with seven YAML files correspoding to the variaous data flow that are possible. For example, layer by layer scheduling is a type of dataflow, only I/O cached scheduling is also another type of dataflow. The folowing table sumarizes the various types of dataflow that are possible.

    
   ![DataFlowTypes](https://github.com/PechimuthuMithil/DeepFrack_temp/assets/119656326/f0b04ded-3d74-47d0-892c-944d82c775be)

### Outputs  
The output from the hardware benchmarker is:  
1) Seven dictionaries stored in seven separate json files. The Dictionary struncture is as follows:  
```
{Layer number:{Tile Dimension: Energy Value,...},...}
```
2) We aslo store the log files from timeloop in separate folders that can be later used for obtaining a deep analysis of DeepFrack's mapping.
     
The directory structure should look something like this.  
![benchmarker_ouput](https://github.com/PechimuthuMithil/DeepFrack_temp/assets/119656326/c7801055-cf89-4a61-bbd3-930bf63dc239)

### Time Consumed
The hardware benchmarking may take some time (~hours) as we are try to qunatize a cost value for every possible step that DeepFrack may take.  

## DeepFrack Core
### Inputs
Inputs to DeepFrack are gives as paths to the following files/folders  
1) The folder that contains the yaml files of every layer in the workload.  
2) The folder containing all the .json files that store the costs generated by the Hardware Benchmarker.  
3) Output folder name where the overview statistics file along with comparison image will be stored.

### Outputs
DeepFrack will output a final log file that will show the optimal tiling, and fusion along with the total energy consumed by fusion over layer by layer scheduling. This file consists of suffecient data to describe the mapping conditions chosen. Moreover one can obtain the exact mapping on the architecture done by timeloop is present in the log files folder populated by the Hardware Benchmarker.  The Statistics file also displays the way the tiles are going to placed and the order in whixh they will be computed.         

The log file will look like the following.  
![Log_file](https://github.com/PechimuthuMithil/DeepFrack_temp/assets/119656326/ec6fb087-a1ed-4c6c-b436-30653513a987)

### Time Consumed
The DeepFrack core may also take time (~hours) proportional to the size of the search space.  

## Statitics Generator
### Inputs
This takes as input the outputs from the previous two modules. These include:  
1) The optimal partition obtained (from the log file from the DeepFrack Core)
2) The optimal weight caching pattern (from the log file from the DeepFrack Core)
3) The log files generated form the Heardware Benchmarer module.

### Outputs  
The output will include a csv file that contains the per-component detail of the optimal mapping found by the DeepFrack modules.  

### Time Consumed  
The time consumed to perform this is almost neglegible. 

## Upcoming Changes
1) Incorporate residual networks 
2) Make faster search by better pruning of search space.

## Future Works
1) Results on inception networks.  
2) Reference material for hardware designers to develop fused layer scheduling favourable hardware accelerators.  

