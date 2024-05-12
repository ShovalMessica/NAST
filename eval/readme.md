## Speaker Probing Task

For evaluating speaker information, we utilize [textlesslib's Speaker Probing task](https://github.com/facebookresearch/textlesslib/tree/main/examples/speaker_probing). 

#### Configuring the Tokenizer Model Output

To configure the output representation of the tokenizer model, you must modify the settings in the configuration file. This adjustment controls whether the model outputs discrete or continuous representations:

- **Discrete Units Representation**: Set `config[num_units]["discrete_local"]` to `True`. 

- **Continuous Global Representation**: Set `config[num_units]["continuous_global"]` to `True`.

By setting these configuration options, you can easily choose the output representation, making it suitable for your specific needs.

## Unit-Edit-Distance (UED) Calculation

The UED calculator measures the unit edit distance for audio processing tasks. Follow these steps to utilize the calculator:

### 1. Prepare the Necessary Files
Ensure you have the following files prepared:

- **Dataset TSV File:** A TSV file containing the dataset for UED calculation.
- **Configuration Files:** A YAML files (`training_config.yaml, model_config.yaml`) specifying the network and training parameters.
- **Model Checkpoint File:** A file containing the pre-trained model weights.

### 2. Running the UED Calculator
Execute the UED calculator with the command below. Replace placeholders with actual paths and values:

```bash
python eval/ued_calculator.py \
  --dataset_tsv_path /path/to/dataset.tsv \
  --training_config_path /path/to/training_config.yaml \
  --model_config_path /path/to/model_config.yaml \ 
  --model_ckpt_path /path/to/pretrained_model_checkpoints.py
```

#### Command-line arguments:
- `--dataset_tsv_path`: Path to the TSV file containing the dataset.
- `--training_config_path`: Path to the YAML training configuration file.
- `--model_config_path`: Path to the YAML model architechture configuration file.
- `--model_ckpt_path`: Path to the pre-trained model weights. 

### 3. Output

The UED calculator will compute the UED scores for each audio augmentation specified in the configuration and display the results in the console. The scores will also be logged with respect to the epoch and batch mentioned in the checkpoint path.
