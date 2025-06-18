from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.base import BaseModel
from src.configs.base_config import Config
from src.utils import set_seed

def prepare_testloader(config: Config) -> DataLoader:
    """
    Prepare the test dataloader for inference.
    """
    print("Setting up dataset and dataloader...")
    # --- Dataset and DataLoader ---
    # This is a placeholder; replace with your actual dataset class and arguments.
    test_dataset = YourActualDatasetClass(data_path=config.test_path, ...)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=config.pin_memory,
    )
    return test_loader


def inference(config):
    """
    Run inference on the test dataset, generate predictions,
    and save the results to a submission CSV file.
    """
    # --- Seed Setting ---
    set_seed(config)

    # --- Accelerator ---
    # Initialize accelerator with performance optimizations
    torch_dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",
            mode="max-autotune",
            use_regional_compilation=True,
    ) if hasattr(torch, "compile") else None


    accelerator = Accelerator(
        mixed_precision="fp16",  # Use "bf16" on A100/H100, "fp16" on others
        dynamo_plugin=torch_dynamo_plugin,
    )

    print("Starting inference...")
    # Let accelerate handle the device printing
    accelerator.print(f"Using device: {accelerator.device}")

    test_loader = prepare_testloader(config)

    # --- Model Loading ---
    # BEST PRACTICE: Load state_dict before preparing the model
    best_model_path = config.output_dir / "best_model_state"
    accelerator.print(f"Loading model from {best_model_path}...")
    config.pretrained = False
    model = BaseModel(config)
    
    # --- Prepare for acceleration ---
    # This will move the model to the correct device and compile it with TorchDynamo
    model, test_loader = accelerator.prepare(model, test_loader)
    accelerator.load_state(best_model_path)

    model.eval()
    all_predictions = []

    # --- Inference Loop ---
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Inference", disable=not accelerator.is_main_process)
        for inputs in progress_bar:
            # NO NEED for inputs.to(device). `accelerate` handles it.
            outputs = model(inputs)

            # Gather predictions from all processes
            gathered_predictions = accelerator.gather(outputs)
            all_predictions.append(gathered_predictions.cpu())

    # --- Process and Save Results ---
    # This block will only run on the main process to avoid duplicate work
    if accelerator.is_main_process:
        print("Processing and saving results...")
        # Concatenate all batches of predictions
        final_predictions = torch.cat(all_predictions)

        # Now you can process `final_predictions` and save to a CSV
        # e.g., create_submission_file(final_predictions, config.submission_path)
        pass

    accelerator.wait_for_everyone()
    print("Inference completed.")


if __name__ == "__main__":
    config = Config()
    inference(config)