"""
Training utilities for neural operators
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from typing import Dict, List, Optional, Tuple


def generate_training_data(solver, num_steps: int, save_fields: Dict[str, bool], 
                          output_filename: str = "training_data.npz") -> str:
    """
    Generate training data by running the simulation and saving specified fields.
    
    Args:
        solver: BaselineSolver instance
        num_steps: Number of simulation steps to run
        save_fields: Dictionary specifying which fields to save
                    (e.g., {'u': True, 'v': True, 'p': True, 'mask': True})
        output_filename: Output .npz filename
    
    Returns:
        Path to the saved .npz file
    """
    print(f"Generating training data: {num_steps} steps...")
    
    # Initialize storage for data
    data = {}
    
    # Pre-allocate arrays for efficiency
    nx, ny = solver.grid.nx, solver.grid.ny
    
    if save_fields.get('u', False):
        data['u'] = np.zeros((num_steps, nx, ny))
    if save_fields.get('v', False):
        data['v'] = np.zeros((num_steps, nx, ny))
    if save_fields.get('p', False):
        data['p'] = np.zeros((num_steps, nx, ny))
    if save_fields.get('mask', False):
        data['mask'] = np.zeros((num_steps, nx, ny))
    if save_fields.get('divergence', False):
        data['divergence'] = np.zeros((num_steps, nx, ny))
    
    # Store metadata
    data['dx'] = solver.grid.dx
    data['dy'] = solver.grid.dy
    data['dt'] = solver.dt
    data['nx'] = nx
    data['ny'] = ny
    
    # Run simulation and collect data
    for i in range(num_steps):
        # Step the simulation
        _, _, _, _ = solver.step_for_visualization(compute_vorticity=False, compute_divergence=False, compute_energy=False,
                                     compute_drag_lift=False, compute_diagnostics=False)
        
        # Collect requested fields
        if save_fields.get('u', False):
            data['u'][i] = np.array(solver.u)
        if save_fields.get('v', False):
            data['v'][i] = np.array(solver.v)
        if save_fields.get('p', False):
            data['p'][i] = np.array(solver.current_pressure)
        if save_fields.get('mask', False):
            data['mask'][i] = np.array(solver.mask)
        if save_fields.get('divergence', False):
            from solver.operators import divergence
            div = divergence(solver.u, solver.v, solver.grid.dx, solver.grid.dy)
            data['divergence'][i] = np.array(div)
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Step {i + 1}/{num_steps} completed")
    
    # Save to .npz file
    np.savez(output_filename, **data)
    print(f"Training data saved to {output_filename}")
    
    return os.path.abspath(output_filename)


def train_pressure_operator(dataset_path: str, model_class, model_params: Dict, 
                            training_params: Dict, output_path: str, cancel_flag=None) -> str:
    """
    Train a neural operator on a dataset.
    
    Args:
        dataset_path: Path to training dataset (.npz file)
        model_class: Neural operator model class
        model_params: Model initialization parameters
        training_params: Training hyperparameters (epochs, learning_rate, batch_size)
        output_path: Path to save trained model
        cancel_flag: Optional threading.Event to allow cancellation
    
    Returns:
        Path to the saved trained model
    """
    import equinox as eqx
    import optax
    
    print(f"Loading dataset from {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    
    # Print available keys for debugging
    print(f"Available keys in dataset: {list(data.keys())}")
    
    # Extract training data
    # For pressure operator: input = (divergence, mask), output = pressure
    try:
        divergence = data['divergence']
    except KeyError:
        # Compute divergence from u and v if not in dataset
        print("Divergence not in dataset, computing from u and v...")
        u = data['u']
        v = data['v']
        dx = data['dx']
        dy = data['dy']
        
        # Compute divergence using finite differences
        divergence = np.zeros_like(u)
        divergence[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx) + (v[2:, :] - v[:-2, :]) / (2 * dy)
        print(f"Computed divergence shape: {divergence.shape}")
    
    mask = data['mask']
    pressure = data['p']
    
    # Check if velocity fields are available for enhanced training
    has_velocity = 'u' in data and 'v' in data
    if has_velocity:
        u = data['u']
        v = data['v']
        u_jax = jnp.array(u)
        v_jax = jnp.array(v)
        print(f"Velocity fields available: u={u.shape}, v={v.shape}")
        # Only use velocity if model_params specifies in_channels=4
        if model_params.get('in_channels', 2) == 4:
            print("Using velocity fields (model has in_channels=4)")
        else:
            print("Model has in_channels=2, ignoring velocity fields")
            u_jax = None
            v_jax = None
            has_velocity = False
    else:
        u_jax = None
        v_jax = None
        print("Velocity fields not available, using 2 input channels (rhs, mask)")
    
    # Convert to JAX arrays
    divergence_jax = jnp.array(divergence)
    mask_jax = jnp.array(mask)
    pressure_jax = jnp.array(pressure)
    
    print(f"Dataset shape: divergence={divergence.shape}, mask={mask.shape}, pressure={pressure.shape}")
    
    # Initialize model
    key = jax.random.PRNGKey(0)
    model = model_class(**model_params, key=key)
    
    # Setup optimizer
    optimizer = optax.adam(training_params['learning_rate'])
    opt_state = optimizer.init(model)
    
    # Training loop
    num_samples = divergence.shape[0]
    batch_size = training_params['batch_size']
    epochs = training_params['epochs']
    
    def loss_fn(model, rhs_batch, mask_batch, p_true_batch, u_batch=None, v_batch=None):
        # vmap will apply model to each sample in the batch
        # For NonLinear model, pass u and v if available
        if hasattr(model, 'conv1'):  # Check if it's NonLinear (has conv layers)
            if u_batch is not None and v_batch is not None:
                pred = jax.vmap(lambda r, msk, u, v: model(r, msk, u=u, v=v))(rhs_batch, mask_batch, u_batch, v_batch)
            else:
                pred = jax.vmap(lambda r, msk: model(r, msk, u=None, v=None))(rhs_batch, mask_batch)
        else:
            pred = jax.vmap(model)(rhs_batch, mask_batch)
        return jnp.mean((pred - p_true_batch) ** 2)
    
    def train_step(model, opt_state, rhs_batch, mask_batch, p_true_batch, u_batch=None, v_batch=None):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, rhs_batch, mask_batch, p_true_batch, u_batch, v_batch)
        # Clip gradients to prevent explosion
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    print(f"Starting training: {epochs} epochs, batch_size={batch_size}")
    
    best_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Check for cancellation
        if cancel_flag is not None and cancel_flag.is_set():
            print("Training cancelled by user")
            break
        
        # Shuffle data
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, num_samples)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            # Check for cancellation during batch processing
            if cancel_flag is not None and cancel_flag.is_set():
                print("Training cancelled by user")
                break
            
            batch_indices = indices[i:i+batch_size]
            rhs_batch = divergence_jax[batch_indices]
            mask_batch = mask_jax[batch_indices]
            p_true_batch = pressure_jax[batch_indices]
            
            # Pass velocity batches if available
            if has_velocity:
                u_batch = u_jax[batch_indices]
                v_batch = v_jax[batch_indices]
                model, opt_state, loss = train_step(model, opt_state, rhs_batch, mask_batch, p_true_batch, u_batch, v_batch)
            else:
                model, opt_state, loss = train_step(model, opt_state, rhs_batch, mask_batch, p_true_batch)
            epoch_loss += loss
            num_batches += 1
        
        # If cancelled during batch loop, exit
        if cancel_flag is not None and cancel_flag.is_set():
            break
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model
            # Save checkpoint after each epoch with best loss
            checkpoint_path = output_path.replace('.eqx', f'_epoch{epoch+1}.eqx')
            eqx.tree_serialise_leaves(checkpoint_path, best_model)
            print(f"  -> New best model saved to checkpoint (loss: {best_loss:.6f})")
    
    # Save best trained model
    if best_model is not None:
        eqx.tree_serialise_leaves(output_path, best_model)
        print(f"Best trained model saved to {output_path} (loss: {best_loss:.6f})")
    else:
        print("No model saved (training cancelled before first epoch)")
        return None
    
    return os.path.abspath(output_path)


def load_trained_model(model_class, model_params: Dict, model_path: str):
    """
    Load a trained neural operator model from file.
    
    Args:
        model_class: Neural operator model class
        model_params: Model initialization parameters (in_channels is auto-detected from saved model)
        model_path: Path to saved model file
    
    Returns:
        Loaded model instance
    """
    import equinox as eqx
    
    # Try to detect in_channels by attempting to load with different values
    detected_in_channels = None
    
    # Try in_channels=2 first (most common)
    for test_channels in [2, 4]:
        test_params = model_params.copy()
        test_params['in_channels'] = test_channels
        try:
            temp_model = model_class(**test_params, key=jax.random.PRNGKey(0))
            temp_model = eqx.tree_deserialise_leaves(model_path, temp_model)
            detected_in_channels = test_channels
            print(f"Successfully loaded with in_channels={detected_in_channels}")
            break
        except Exception:
            continue
    
    if detected_in_channels is None:
        raise ValueError("Could not load model with either in_channels=2 or in_channels=4")
    
    # Update model_params with detected in_channels
    model_params['in_channels'] = detected_in_channels
    
    # Initialize model with correct parameters
    model = model_class(**model_params, key=jax.random.PRNGKey(0))
    
    # Load trained weights
    model = eqx.tree_deserialise_leaves(model_path, model)
    print(f"Loaded trained model from {model_path}")
    
    return model
