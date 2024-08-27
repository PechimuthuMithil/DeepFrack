from benchmarker_area_cycles import run_benchmarker
from DeepFrack_fast import deepfrack_fast
import argparse 
import os
import subprocess

parser = argparse.ArgumentParser(description='Run DeepFrack Benchmark')
parser.add_argument('--problems', type=str, required=True)
parser.add_argument('--mapper', type=str, required=True)
parser.add_argument('--arch', type=str, required=True)
parser.add_argument('--components', type=str, required=True)
parser.add_argument('--map_constraints', type=str, required=True)
parser.add_argument('--arch_constraints', type=str, required=True)
parser.add_argument('--benchmark_log', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--num_threads', type=int, required=True)
# parser.add_argument('--weight_level_name', type=str, required=True)
# parser.add_argument('--weigth_level_size', type=int, required=True)
# parser.add_argument('--input_level_name', type=str, required=True)
# parser.add_argument('--input_level_size', type=int, required=True)
# parser.add_argument('--output_level_name', type=str, required=True)
# parser.add_argument('--output_level_size', type=int, required=True)
# parser.add_argument('--cheat_Sheet_path', type=str, required=True)

args = parser.parse_args()

# run_benchmarker(args.problems, args.mapper, args.arch, args.components, 
#                   args.map_constraints, args.arch_constraints, args.benchmark_log,
#                   0,0)


# problems = 'Examples/AlexNet_Simba/AlexNet'
# mapper='Examples/AlexNet_Simba/simba_like/mapper/mapper.yaml'
# arch= 'Examples/AlexNet_Simba/simba_like/arch/simba_like.yaml'
# components='Examples/AlexNet_Simba/simba_like/arch/components'
# map_constraints='Examples/AlexNet_Simba/simba_like/constraints/simba_like_map_constraints.yaml'
# arch_constraints='Examples/AlexNet_Simba/simba_like/constraints'
# benchmark_log='Examples/alexnet_simba_log'

# run_benchmarker(problems, mapper, arch, components, 
#                   map_constraints, arch_constraints, benchmark_log,
#                   0,0)

cheatsheet = '/TestingDF/DeepFrack_temp/CheatSheet.json'
if not os.path.exists(cheatsheet):
    compile_process = subprocess.run([
        "g++", "-o", "CheatSheetMaker", "CheatSheetMaker.cpp",
        "-I/usr/include/jsoncpp", "-L/usr/lib/x86_64-linux-gnu", "-ljsoncpp"])
    run_process = subprocess.run(["./CheatSheetMaker"])

WeightLevel_name = 'PEWeightRegs' # at DRAM - 1 level
WeightLevel_size = 2048
InputLevel_name = 'GlobalBuffer'
InputLevel_size = 65536
OutputLevel_name = 'sharedGlobalBuffer_glb'
OutputLevel_size = 65536

OutputImgFile = args.out+"/Comparison_multiT.jpg"
LogFile=args.out+"/DeepFrack_logfile.txt"
# WeightLevel_name = args.weight_level_name
# WeightLevel_size = args.weight_level_size
# InputLevel_name = args.input_level_name
# InputLevel_size = args.input_level_size
# OutputLevel_name = args.output_level_name
# OutputLevel_size = args.output_level_size
# cheatsheet = args.cheat_sheet_path
values = deepfrack_fast(args.problems,args.benchmark_log,cheatsheet,OutputImgFile,LogFile,WeightLevel_name,WeightLevel_size,InputLevel_name,InputLevel_size,OutputLevel_name,OutputLevel_size, int(args.num_threads))

print("DEEPFRACK RUNNER SAYS", values)
        
    

    
    



