import os
import torch
import gc
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from src.config import config

# Get device info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(config["NNINT_CKPT_DIR"], "nnInteractive_v1.0")

print("Using device:", device)
print("Available cpus:", os.cpu_count())
print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Function to check and print memory stats
def print_memory_stats():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"GPU max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

try:
    # Create session
    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=True,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    
    session.initialize_from_trained_model_folder(checkpoint_path)
    
    # Get the network from the session
    network = session.network

    torch.save(network, "/work/dlclarge2/ndirt-SegFM3D/model_checkpoints/nnint/custom_nnint.pth")
    
    # Put network in training mode
    network.train()
    
    # Create a random input tensor (simulating a datapoint)
    # Reduce size if needed to avoid memory issues
    x = torch.randn(1, 8, 96, 96, 96).to(device)
    
    # Create a random target tensor (simulating ground truth)
    # Assuming the output has the same spatial dimensions as input
    # but with a different number of channels (e.g., 1 for segmentation)
    y_target = torch.randn(1, 1, 96, 96, 96).to(device)
    
    print("Created random input and target tensors")
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y_target.shape}")
    print_memory_stats()
    
    # Create an optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    # Define a loss function
    criterion = torch.nn.MSELoss()
    
    # Training loop for a few iterations
    num_iterations = 3
    print(f"\nStarting training for {num_iterations} iterations...")
    
    for i in range(num_iterations):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = network(x)
        
        # If the output shape doesn't match the target, adjust target shape
        if y_pred.shape != y_target.shape:
            print(f"Output shape ({y_pred.shape}) doesn't match target shape ({y_target.shape})")
            print("Adjusting target shape...")
            y_target = torch.randn_like(y_pred)
            print(f"New target shape: {y_target.shape}")
        
        # Compute loss
        loss = criterion(y_pred, y_target)
        
        # Print current loss
        print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print memory stats after each iteration
        print_memory_stats()
        print()
    
    print("Training completed!")
    
    # Check if weights have changed
    # Save initial weights of first layer
    first_layer = next(network.parameters())
    initial_weight_sum = first_layer.data.sum().item()
    
    # Do one more iteration
    optimizer.zero_grad()
    y_pred = network(x)
    loss = criterion(y_pred, y_target)
    loss.backward()
    optimizer.step()
    
    # Check if weights changed
    new_weight_sum = first_layer.data.sum().item()
    print(f"\nFirst layer weight sum before: {initial_weight_sum}")
    print(f"First layer weight sum after: {new_weight_sum}")
    if initial_weight_sum != new_weight_sum:
        print("✅ Weights have changed, training was successful!")
    else:
        print("❌ Weights did not change, training failed.")
    
    # Clean up
    del y_pred, y_target, x, loss, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()
    
    # Test inference after training
    print("\nTesting inference after training...")
    
    # Put the network in evaluation mode
    network.eval()
    
    # Create a new test input
    x_test = torch.randn(1, 8, 96, 96, 96).to(device)
    
    # Run inference
    with torch.no_grad():
        y_test = network(x_test)
    
    print(f"Test input shape: {x_test.shape}")
    print(f"Test output shape: {y_test.shape}")
    
    # Final cleanup
    del network, y_test, x_test, session
    torch.cuda.empty_cache()
    gc.collect()
    print_memory_stats()

except Exception as e:
    print(f"Error: {e}")
    # Try to clean up even if there was an error
    torch.cuda.empty_cache()
    gc.collect()
    
finally:
    # Final cleanup
    print("Script completed")
    print_memory_stats()