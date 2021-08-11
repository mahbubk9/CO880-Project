!/bin/bash                                                                                                                
#SBATCH --job-name=serial_job_test    # Job name                                                                           
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)                                          
#SBATCH --mail-user=mk788@kent.ac.uk     # Where to send mail                                                              
#SBATCH --ntasks=4                    # Run on a single CPU                                                                
#SBATCH --partition=gpu.stu                                                                                                
#SBATCH --mem=1gb                     # Job memory request                                                                 
#SBATCH --time=23:59:00               # Time limit hrs:min:sec                                                             
#SBATCH --output=serial_test_%j.log   # Standard output and error log                                                      
pwd; hostname; date



eval "$(conda shell.bash hook)"
conda activate hydraEnv          ---------the above two lines activate the "hydraEnv" conda environment
python /cluster/home/arc/mk788/mycode/dqn.py      ----- this is the actual code I want to run


date


