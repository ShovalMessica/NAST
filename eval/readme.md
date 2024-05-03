## Unit-Edit-Distance (UED) Calculation

The UED calculator measures the unit edit distance for audio processing tasks. Follow these steps to utilize the calculator:

### 1. Prepare the Necessary Files
Ensure you have the following files prepared:

- **Dataset TSV File:** A TSV file containing the dataset for UED calculation.
- **Configuration File:** A YAML file (`training_config.yaml`) specifying the network architecture and hyperparameters.
- **Model Checkpoint File:** A file containing the pre-trained model weights. This is optional; if not provided, the model will initialize with random weights.

### 2. Running the UED Calculator

Execute the UED calculator with the command below. Replace placeholders with actual paths and values:

```bash
python eval/ued_calculator.py \
  --dataset_tsv_path /path/to/dataset.tsv \
  --config_path /path/to/training_config.yaml \
  --use_wanb true_or_false \
  --wandb_project your_project_name \
  --wandb_entity your_entity_name
```

#### Parameters:
- `--dataset_tsv_path`: Path to the TSV file containing the dataset.
- `--config_path`: Path to the YAML configuration file.
- `--use_wandb`: Enable logging with Weights and Biases.
- `--wandb_project`: (Optional) Weights & Biases project name for logging the UED scores.
- `--wandb_entity`: (Optional) Weights & Biases entity name.

### 3. Output

The UED calculator will compute the UED scores for each audio augmentation specified in the configuration and display the results in the console. If Weights and Biases is configured, the scores will also be logged with the respective checkpoint.

#### Note:
Ensure all necessary dependencies are installed, including `wandb` if you are using Weights and Biases for logging. You can install dependencies via:

```bash
pip install -r requirements.txt
pip install wandb
```

For detailed installation instructions, refer to the setup section or the provided `requirements.txt` file.
