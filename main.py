from models.network import Network
from utils.config import load_config
from utils.checkpoint import save_checkpoint

def create_model(model_name: str) -> Network:
    """
    Create a model instance based on the given model name.

    Args:
        model_name (str): Name of the model configuration.

    Returns:
        Network: Instantiated model.
    """
    config = load_config(model_name)
    model = Network(config)
    return model

if __name__ == "__main__":
    model_name = "network_100"
    model = create_model(model_name)
    
    # Example usage
    input_features = torch.randn(10, 768)  # Dummy input features
    output = model(input_features)
    
    # Save the model checkpoint
    save_checkpoint(model, "checkpoints/model.ckpt")
