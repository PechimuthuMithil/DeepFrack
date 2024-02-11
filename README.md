Note: This is a dummy repository  
# DEEPFRACK 
DeepFrack is a novel framework developed for enhancing energy efficiency and reducing latency in deep learning workloads executed on hardware accelerators. By optimally fusing layers and implementing an asymmetric tiling strategy, DeepFrack addresses the limitations of traditional layer-by-layer scheduling. The DeepFrack project aims to build succinct and simple framework that contributes to the ongoing efforts in designing more efficient hardware accelerators for machine learning workloads.  

DeepFrack is wrapper to [Timeloop](https://timeloop.csail.mit.edu/timeloop) written in python.

This document is to serve as a guide to users for installing and using DeepFrack. This document (along with the paper) will also provide any user with additional knowledge required to go through the complete source code of DeepFrack and suggest changes. For any queries please contact Mithil Pechimuthu (pechimuthumithil@iitgn.ac.in).  

![image](https://github.com/PechimuthuMithil/DeepFrack_temp/assets/119656326/ae79b3ce-aa0f-45d5-b8d6-ec0e7591f1f4)  

# Installation
## Dependencies
DeepFrack relies on Timeloop for it's cost metrics. Hence it is absolutely necessary for the user to have timeloop installed for running DeepFrack. One can find the the installation procedure and other useful information regarding the dependencies in the [Timeloop documentation](https://timeloop.csail.mit.edu/timeloop/installation).  

# Usage
## Generating costs
Run the script file for generating all the costs for all tiles.  

## Input and Output
### Inputs
Inputs to DeepFrack are gives as paths to the following files/folders  
1) The folder that contains the yaml files of every layer in the workload.  
2) The yaml file denoting the architcture specificatios of the accelerator.  
3) The yaml file describing the mapping constraints.
4) The json files that store all the tile costs.
5) Output folder name.

### Outputs
DeepFrack will output a final log file that will show the optimal tiling, and fusion along with the total energy consumed by fusion over layer by layer scheduling. 

## Hardware Benchmarcker
The Benchmarker is the module of DeepFrack that uses timeloop to generate costs (total energy by default) for each step taken by a point in the design sapce. The benchmarker takes the inputs provided to it and returns costs in a folder of .json files. The examples will make it more clear.  
### Inputs
The Bencmarker must be provided with the following YAML files.
1) Timeloop mapper YAML file.
2) YAML file to describe the architecture of the accelerator.
3) YAML file that describes the mapping constraints for timeloop.
4) A folder with seven YAML files correspoding to the variaous data flow that are possible. For example, layer by layer scheduling is a type of dataflow, only I/O cached scheduling is also another type of dataflow. The folowing table sumarizes the various types of dataflow that are possible.

    
   ![DataFlowTypes](https://github.com/PechimuthuMithil/DeepFrack_temp/assets/119656326/f0b04ded-3d74-47d0-892c-944d82c775be)

Along with these YAML files, the benchmarker requires the user to provide the bit width and the sizes of the second highest (just below the DRAM) memory name and size for Inputs, Weights and Outputs. The user may also choose to provide the output folder for the outputs of the Benchmarker.  

### Outputs  
The output from the hardware benchmarker is:  
1) Seven dictionaries stored in seven separate json files. The Dictionary struncture is as follows:  
```
{Layer number:{Tile Dimension: Energy Value,...},...}
```
2) We aslo store the log files from timeloop in separate folders that can be later used for obtaining a deep analysis of DeepFrack's mapping.

### Time Consumed
The hardware benchmarking may take some time (~hours) as we are try to qunatize a cost value for every possible step that DeepFrack may take.  

