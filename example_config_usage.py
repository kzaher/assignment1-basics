#!/usr/bin/env python3
"""
Example of how to use the existing data classes for JSON configuration
with full autocomplete support.
"""

import json
from cs336_basics.experiments.configuration import (
    LlmPretrainingConfiguration,
    OptimizerConfiguration, 
    TransformerLlmConfiguration,
    AnnealingConfiguration
)

# Method 1: Create configuration objects programmatically (full autocomplete)
def create_config_with_autocomplete():
    """Create configuration using data classes - you get full autocomplete!"""
    
    # Create optimizer config with autocomplete
    optimizer_config = OptimizerConfiguration(
        lr=1e-4,
        weight_decay=0.1,
        betas=[0.9, 0.95],
        eps=1e-8
    )
    
    # Create transformer config with autocomplete  
    transformer_config = TransformerLlmConfiguration(
        vocab_size=32000,
        context_length=2048,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        rope_theta=10000.0,
        device="cuda"
    )
    
    # Create annealing config with autocomplete
    annealing_config = AnnealingConfiguration(
        max_learning_rate=1e-3,
        min_learning_rate=1e-5, 
        warmup_iters=1000,
        cosine_cycle_iters=10000
    )
    
    # Create main config with autocomplete
    config = LlmPretrainingConfiguration(
        source_input_path="data",
        configuration_path="experiments/tiny_stories.json",
        checkpoint_persist_modulus=50,
        optimizer_configuration=optimizer_config,
        transformer_llm=transformer_config,
        annealing_configuration=annealing_config,
        batch_size=32,
        context_length=2048,
        max_l2_norm=1.0
    )
    
    return config

# Method 2: Load from JSON and get typed object (partial autocomplete after loading)
def load_config_from_json(json_path: str) -> LlmPretrainingConfiguration:
    """Load JSON and get a typed configuration object."""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert to typed object - now you have autocomplete on the result!
    config = LlmPretrainingConfiguration.from_dict(config_dict)
    return config

# Method 3: Create a template generator
def generate_config_template():
    """Generate a complete JSON template with all required fields."""
    config = create_config_with_autocomplete()
    
    # Convert to dict for JSON serialization
    # Note: You might need to add a to_dict method to your serialization module
    return config

if __name__ == "__main__":
    # Example usage
    config = create_config_with_autocomplete()
    print(f"Created config with vocab_size: {config.transformer_llm.vocab_size}")
    
    # You can access nested properties with full autocomplete:
    # config.optimizer_configuration.lr
    # config.transformer_llm.d_model  
    # config.annealing_configuration.warmup_iters
