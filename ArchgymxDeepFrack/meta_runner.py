import subprocess
subprocess.run([
        "python3", "deepfrack_runner.py",
        "--problems", "Examples/AlexNet_Simba/AlexNet",
        "--mapper", "Examples/AlexNet_Simba/simba_like/mapper/mapper.yaml",
        "--arch", "Examples/AlexNet_Simba/simba_like/arch/simba_like.yaml",
        "--components", "Examples/AlexNet_Simba/simba_like/arch/components",
        "--map_constraints", "Examples/AlexNet_Simba/simba_like/constraints/simba_like_map_constraints.yaml",
        "--arch_constraints", "Examples/AlexNet_Simba/simba_like/constraints",
        "--benchmark_log", "Examples/alexnet_simba_log",
        "--out","Examples/alexnet_simba_neerja"
    ])