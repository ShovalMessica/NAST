## Speaker Probing Task

For evaluating speaker information, we utilize [textlesslib's Speaker Probing task](https://github.com/facebookresearch/textlesslib/tree/main/examples/speaker_probing). This task helps in assessing how much speaker information is retained in the generated speech representations.

### Configuring the Tokenizer Model Output

To configure the output representation of the tokenizer model, you must modify the settings in the configuration file. This adjustment controls whether the model outputs discrete or continuous representations, which are crucial for different types of speaker analysis tasks:

- **Discrete Units Representation**: Set `config[num_units]["discrete_local"]` to `True`. 

- **Continuous Global Representation**: Set `config[num_units]["continuous_global"]` to `True`.

By setting these configuration options, you can effectively control how the tokenizer processes and outputs speech data, making it suitable for your specific research needs in speaker information evaluation.

**Run the Probing Task**: Follow the detailed steps provided in the [textlesslib speaker probing guide](https://github.com/facebookresearch/textlesslib/tree/main/examples/speaker_probing) to execute the probing task.

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

#### Command-line arguments:
- `--dataset_tsv_path`: Path to the TSV file containing the dataset.
- `--config_path`: Path to the YAML configuration file.
- `--use_wandb`: (Optional) Enable logging with Weights and Biases.
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
