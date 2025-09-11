#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=$HOME/logs/slurm-%j.out
#SBATCH --error=$HOME/logs/slurm-%j.out
#SBATCH --mail-user=youremail@example.com
#SBATCH --mail-type=ALL

# Activate the virtual environment
cd SCRATCH/visual-inspection
source venv/Scripts/activate

# Train the model
python -m scripts.train