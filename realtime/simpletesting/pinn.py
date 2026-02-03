# Complete Training Script for FEniCS + Physics PINN
# Copy this entire cell to your Colab notebook and run it

# from google.colab import drive
# drive.mount('/content/drive')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# ENHANCED PINN MODEL WITH NORMALIZATION
# ============================================================================

class ParametricADEPINN(nn.Module):
    def __init__(self):
        super(ParametricADEPINN, self).__init__()
        # Original working size: 4 layers x 128 neurons
        self.net = nn.Sequential(
            nn.Linear(10, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softplus()
        )

        # Normalization ranges - will be set from data
        self.register_buffer('x_min', torch.tensor(0.0))
        self.register_buffer('x_max', torch.tensor(20000.0))
        self.register_buffer('y_min', torch.tensor(0.0))
        self.register_buffer('y_max', torch.tensor(20000.0))
        self.register_buffer('t_min', torch.tensor(0.0))
        self.register_buffer('t_max', torch.tensor(10000.0))
        self.register_buffer('cx_min', torch.tensor(0.0))
        self.register_buffer('cx_max', torch.tensor(20000.0))
        self.register_buffer('cy_min', torch.tensor(0.0))
        self.register_buffer('cy_max', torch.tensor(20000.0))
        self.register_buffer('u_min', torch.tensor(-10.0))
        self.register_buffer('u_max', torch.tensor(10.0))
        self.register_buffer('v_min', torch.tensor(-10.0))
        self.register_buffer('v_max', torch.tensor(10.0))
        self.register_buffer('d_min', torch.tensor(190.0))
        self.register_buffer('d_max', torch.tensor(3700.0))
        self.register_buffer('kappa_min', torch.tensor(0.0))
        self.register_buffer('kappa_max', torch.tensor(120.0))
        self.register_buffer('Q_min', torch.tensor(1e-6))
        self.register_buffer('Q_max', torch.tensor(3e-3))

    def set_normalization_from_data(self, data_dict):
        """Set normalization ranges from actual training data"""
        self.x_min.fill_(data_dict['x'].min())
        self.x_max.fill_(data_dict['x'].max())
        self.y_min.fill_(data_dict['y'].min())
        self.y_max.fill_(data_dict['y'].max())
        self.t_min.fill_(data_dict['t'].min())
        self.t_max.fill_(data_dict['t'].max())
        self.cx_min.fill_(data_dict['cx'].min())
        self.cx_max.fill_(data_dict['cx'].max())
        self.cy_min.fill_(data_dict['cy'].min())
        self.cy_max.fill_(data_dict['cy'].max())
        self.u_min.fill_(data_dict['u'].min())
        self.u_max.fill_(data_dict['u'].max())
        self.v_min.fill_(data_dict['v'].min())
        self.v_max.fill_(data_dict['v'].max())
        self.d_min.fill_(data_dict['d'].min())
        self.d_max.fill_(data_dict['d'].max())
        self.kappa_min.fill_(data_dict['kappa'].min())
        self.kappa_max.fill_(data_dict['kappa'].max())
        self.Q_min.fill_(data_dict['Q'].min())
        self.Q_max.fill_(data_dict['Q'].max())

    def normalize_input(self, x, y, t, cx, cy, u, v, d, kappa, Q):
        """Normalize all inputs to [-1, 1] range"""
        x_norm = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        t_norm = 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1
        cx_norm = 2 * (cx - self.cx_min) / (self.cx_max - self.cx_min) - 1
        cy_norm = 2 * (cy - self.cy_min) / (self.cy_max - self.cy_min) - 1
        u_norm = 2 * (u - self.u_min) / (self.u_max - self.u_min) - 1
        v_norm = 2 * (v - self.v_min) / (self.v_max - self.v_min) - 1
        d_norm = 2 * (d - self.d_min) / (self.d_max - self.d_min) - 1
        kappa_norm = 2 * (kappa - self.kappa_min) / (self.kappa_max - self.kappa_min) - 1
        Q_norm = 2 * (Q - self.Q_min) / (self.Q_max - self.Q_min) - 1

        return x_norm, y_norm, t_norm, cx_norm, cy_norm, u_norm, v_norm, d_norm, kappa_norm, Q_norm

    def forward(self, x, y, t, cx, cy, u, v, d, kappa, Q, normalize=True):
        """
        Forward pass with optional normalization
        normalize=True: inputs are in physical units (RECOMMENDED)
        normalize=False: inputs are already normalized to [-1,1]
        """
        if normalize:
            x_n, y_n, t_n, cx_n, cy_n, u_n, v_n, d_n, kappa_n, Q_n = \
                self.normalize_input(x, y, t, cx, cy, u, v, d, kappa, Q)
        else:
            x_n, y_n, t_n, cx_n, cy_n, u_n, v_n, d_n, kappa_n, Q_n = \
                x, y, t, cx, cy, u, v, d, kappa, Q

        inputs = torch.cat((x_n, y_n, t_n, cx_n, cy_n, u_n, v_n, d_n, kappa_n, Q_n), dim=1)
        phi_raw = self.net(inputs)

        # Soft temporal constraint - only affects t very close to t_min
        # At t=t_min (t_norm_01=0): multiplier = 0 (phi=0)
        # At t=t_min + 5% of range: multiplier ≈ 0.99 (almost no effect)
        # This ensures IC (phi=0 at t=0) without suppressing t=500+ predictions
        t_norm_01 = (t_n + 1) / 2  # Convert from [-1,1] to [0,1]

        # Steep ramp: reaches 99% at t_norm_01 = 0.05 (5% into time range)
        # For t_max=10000, this means full output by t=500
        RAMP_STEEPNESS = 100.0  # Higher = steeper ramp (faster transition)
        temporal_multiplier = 1.0 - torch.exp(-RAMP_STEEPNESS * t_norm_01)

        return temporal_multiplier * phi_raw


# ============================================================================
# DATASET CLASS
# ============================================================================

class FEniCSDataset(Dataset):
    def __init__(self, data_dict):
        """
        data_dict contains: 't', 'x', 'y', 'cx', 'cy', 'd', 'Q', 'u', 'v', 'kappa', 'phi'
        """
        self.x = torch.tensor(data_dict['x'], dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(data_dict['y'], dtype=torch.float32).view(-1, 1)
        self.t = torch.tensor(data_dict['t'], dtype=torch.float32).view(-1, 1)
        self.cx = torch.tensor(data_dict['cx'], dtype=torch.float32).view(-1, 1)
        self.cy = torch.tensor(data_dict['cy'], dtype=torch.float32).view(-1, 1)
        self.u = torch.tensor(data_dict['u'], dtype=torch.float32).view(-1, 1)
        self.v = torch.tensor(data_dict['v'], dtype=torch.float32).view(-1, 1)
        self.d = torch.tensor(data_dict['d'], dtype=torch.float32).view(-1, 1)
        self.kappa = torch.tensor(data_dict['kappa'], dtype=torch.float32).view(-1, 1)
        self.Q = torch.tensor(data_dict['Q'], dtype=torch.float32).view(-1, 1)
        self.phi = torch.tensor(data_dict['phi'], dtype=torch.float32).view(-1, 1)

        # Store raw phi (clamped to reasonable range)
        self.phi = self.phi.clamp(min=1e-15, max=1e3)

        # Compute normalized log for stable training
        # log10 scale, shifted and scaled to roughly [-1, 1] range
        self.phi_log = torch.log10(self.phi.clamp(min=1e-15))

        # Normalize log values: typical range is [-15, 0], map to [-1, 1]
        self.log_min = -15.0
        self.log_max = 0.0
        self.phi_log_normalized = 2.0 * (self.phi_log - self.log_min) / (self.log_max - self.log_min) - 1.0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx], 'y': self.y[idx], 't': self.t[idx],
            'cx': self.cx[idx], 'cy': self.cy[idx], 'u': self.u[idx],
            'v': self.v[idx], 'd': self.d[idx], 'kappa': self.kappa[idx],
            'Q': self.Q[idx],
            'phi': self.phi[idx],
            'phi_log_norm': self.phi_log_normalized[idx]
        }


# ============================================================================
# PHYSICS RESIDUAL COMPUTATION - CORRECT ADVECTION-DIFFUSION EQUATION
# ============================================================================

def compute_pde_residual_fast(model, x, y, t, cx, cy, u, v, d, kappa, Q, device):
    """
    Compute PDE residual for the CORRECT advection-diffusion equation:

    ∂φ/∂t + u·∂φ/∂x + v·∂φ/∂y = κ(∂²φ/∂x² + ∂²φ/∂y²) + S(x,y,cx,cy,d,Q)

    Where:
    - κ (kappa) is the DIFFUSION coefficient
    - d is the source diameter
    - S = Q / (π(d/2)²) · exp(-((x-cx)² + (y-cy)²) / (d/2)²)  (Gaussian source)

    Residual = φ_t + u·φ_x + v·φ_y - κ·(φ_xx + φ_yy) - S = 0
    """
    # Ensure all inputs are on the correct device and require grad
    x = x.detach().to(device).requires_grad_(True)
    y = y.detach().to(device).requires_grad_(True)
    t = t.detach().to(device).requires_grad_(True)
    cx = cx.detach().to(device)
    cy = cy.detach().to(device)
    u = u.detach().to(device)
    v = v.detach().to(device)
    d = d.detach().to(device)
    kappa = kappa.detach().to(device)
    Q = Q.detach().to(device)

    # Forward pass
    phi = model(x, y, t, cx, cy, u, v, d, kappa, Q, normalize=True)
    phi_safe = phi.clamp(min=1e-12)

    # Compute first-order gradients
    ones = torch.ones_like(phi)
    phi_t = torch.autograd.grad(phi, t, ones, create_graph=True, retain_graph=True)[0]
    phi_x = torch.autograd.grad(phi, x, ones, create_graph=True, retain_graph=True)[0]
    phi_y = torch.autograd.grad(phi, y, ones, create_graph=True, retain_graph=True)[0]

    # Compute second-order gradients (Laplacian for diffusion)
    phi_xx = torch.autograd.grad(phi_x, x, ones, create_graph=True, retain_graph=True)[0]
    phi_yy = torch.autograd.grad(phi_y, y, ones, create_graph=True)[0]

    # Laplacian
    laplacian = phi_xx + phi_yy

    # Source term: S = Q / (π(d/2)²) · exp(-((x-cx)² + (y-cy)²) / (d/2)²)
    # Simplify: radius r = d/2, so S = Q / (π·r²) · exp(-dist²/r²)
    r_squared = (d / 2.0) ** 2  # (d/2)²
    dist_squared = (x - cx) ** 2 + (y - cy) ** 2
    source = (Q / (np.pi * r_squared)) * torch.exp(-dist_squared / r_squared)

    # CORRECT PDE residual:
    # φ_t + u·φ_x + v·φ_y - κ·(φ_xx + φ_yy) - S = 0
    residual = phi_t + u * phi_x + v * phi_y - kappa * laplacian - source

    # Normalize by (phi + source) for scale invariance (both can be large)
    scale = torch.abs(phi_safe) + torch.abs(source).clamp(min=1e-12)
    residual_normalized = residual / scale

    return residual_normalized.clamp(-10, 10)  # Clamp for stability


# ============================================================================
# COMBINED TRAINING FUNCTION
# ============================================================================

def train_data_only(model, dataloader, optimizer, device):
    """
    FAST data-only training - no physics loss for speed
    """
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Stack all inputs into single tensor for efficient GPU transfer
        x = batch['x'].cuda()
        y = batch['y'].cuda()
        t = batch['t'].cuda()
        cx = batch['cx'].cuda()
        cy = batch['cy'].cuda()
        u = batch['u'].cuda()
        v = batch['v'].cuda()
        d = batch['d'].cuda()
        kappa = batch['kappa'].cuda()
        Q = batch['Q'].cuda()
        phi_true = batch['phi'].cuda()

        # Forward pass
        phi_pred = model(x, y, t, cx, cy, u, v, d, kappa, Q, normalize=True)

        # Log-ratio loss (bounded)
        log_pred = torch.log10(phi_pred.clamp(min=1e-15))
        log_true = torch.log10(phi_true.clamp(min=1e-15))
        loss = torch.mean((log_pred - log_true).clamp(-5, 5) ** 2)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_with_physics(model, dataloader, optimizer, device, physics_weight=0.01, physics_every=10):
    """
    Training with occasional physics loss - computed every N batches for speed
    """
    model.train()
    total_loss = 0
    data_loss_total = 0
    physics_loss_total = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move data to GPU - use .cuda() directly
        x = batch['x'].cuda()
        y = batch['y'].cuda()
        t = batch['t'].cuda()
        cx = batch['cx'].cuda()
        cy = batch['cy'].cuda()
        u = batch['u'].cuda()
        v = batch['v'].cuda()
        d = batch['d'].cuda()
        kappa = batch['kappa'].cuda()
        Q = batch['Q'].cuda()
        phi_true = batch['phi'].cuda()

        # Data loss
        phi_pred = model(x, y, t, cx, cy, u, v, d, kappa, Q, normalize=True)
        log_pred = torch.log10(phi_pred.clamp(min=1e-15))
        log_true = torch.log10(phi_true.clamp(min=1e-15))
        data_loss = torch.mean((log_pred - log_true).clamp(-5, 5) ** 2)

        # Physics loss - only compute every N batches (expensive!)
        physics_loss = torch.tensor(0.0, device='cuda')
        if batch_idx % physics_every == 0:
            # Use small subset for physics (32 points max)
            n_phys = min(32, x.shape[0])
            idx = torch.randperm(x.shape[0], device='cuda')[:n_phys]

            residual = compute_pde_residual_fast(
                model, x[idx], y[idx], t[idx].clamp(min=100),
                cx[idx], cy[idx], u[idx], v[idx],
                d[idx], kappa[idx], Q[idx], 'cuda'
            )
            physics_loss = torch.mean(residual ** 2)

        loss = data_loss + physics_weight * physics_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        data_loss_total += data_loss.item()
        physics_loss_total += physics_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, data_loss_total / n, physics_loss_total / n


# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

def load_fenics_data_from_scenarios(path):
    """
    Load FEniCS data from scenario folders
    Column order: [t, x, y, cx, cy, d, Q, u, v, kappa, phi]
    Loads BOTH collocation_points.npz AND ic_points.npz for full time range
    """
    all_scenarios_list = []
    n_collocation = 0
    n_ic = 0

    for folder_path in Path(path).iterdir():
        if folder_path.is_dir() and folder_path.name.startswith('scenario_'):
            for file_path in folder_path.iterdir():
                # Load collocation points (t > 0)
                if file_path.name == 'collocation_points.npz':
                    data = np.load(file_path)['data']
                    all_scenarios_list.append(data.astype(np.float32))
                    n_collocation += data.shape[0]
                # ALSO load initial condition points (t = 0)
                elif file_path.name == 'ic_points.npz':
                    data = np.load(file_path)['data']
                    all_scenarios_list.append(data.astype(np.float32))
                    n_ic += data.shape[0]

    print(f"Loaded {n_collocation:,} collocation points (t > 0)")
    print(f"Loaded {n_ic:,} initial condition points (t = 0)")

    # Stack into one array
    big_np_array = np.vstack(all_scenarios_list)
    print(f"\nTotal data points: {big_np_array.shape[0]}")

    # Extract columns (CORRECT ORDER: t, x, y, cx, cy, d, Q, u, v, kappa, phi)
    fenics_data = {
        't': big_np_array[:, 0],
        'x': big_np_array[:, 1],
        'y': big_np_array[:, 2],
        'cx': big_np_array[:, 3],
        'cy': big_np_array[:, 4],
        'd': big_np_array[:, 5],
        'Q': big_np_array[:, 6],
        'u': big_np_array[:, 7],
        'v': big_np_array[:, 8],
        'kappa': big_np_array[:, 9],
        'phi': big_np_array[:, 10]
    }

    # DATA VALIDATION: Check for NaN/Inf values
    print("\nValidating data...")
    for key, arr in fenics_data.items():
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"  WARNING: {key} has {n_nan} NaN and {n_inf} Inf values")

    # Remove rows with NaN or Inf in any column
    mask = np.ones(big_np_array.shape[0], dtype=bool)
    for i in range(big_np_array.shape[1]):
        mask &= ~np.isnan(big_np_array[:, i])
        mask &= ~np.isinf(big_np_array[:, i])

    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  Removed {n_removed} rows with NaN/Inf values")
        big_np_array = big_np_array[mask]
        fenics_data = {
            't': big_np_array[:, 0],
            'x': big_np_array[:, 1],
            'y': big_np_array[:, 2],
            'cx': big_np_array[:, 3],
            'cy': big_np_array[:, 4],
            'd': big_np_array[:, 5],
            'Q': big_np_array[:, 6],
            'u': big_np_array[:, 7],
            'v': big_np_array[:, 8],
            'kappa': big_np_array[:, 9],
            'phi': big_np_array[:, 10]
        }

    # Keep phi as-is for now (we'll filter in the next step)
    # Just remove any negative values (shouldn't exist but just in case)
    fenics_data['phi'] = np.maximum(fenics_data['phi'], 0)

    # Also ensure t > 0 to avoid issues with temporal constraint
    fenics_data['t'] = np.clip(fenics_data['t'], 1.0, None)

    # Print data ranges for debugging
    print("\nData ranges:")
    for key, arr in fenics_data.items():
        print(f"  {key:6s}: min={arr.min():.4e}, max={arr.max():.4e}, mean={arr.mean():.4e}")

    # Check for any remaining issues
    print("\nData health check:")
    print(f"  Total points: {len(fenics_data['phi'])}")
    print(f"  Phi zeros: {(fenics_data['phi'] <= 0).sum()}")
    print(f"  Phi very small (<1e-12): {(fenics_data['phi'] < 1e-12).sum()}")

    return fenics_data


# ============================================================================
# ANIMATION FUNCTION (PHYSICAL UNITS)
# ============================================================================

def animate_plume_physical(model, scenario_physical, frames=60, domain_size=20000):
    """Animate plume using PHYSICAL units with colorbar"""
    model.eval()
    import time
    # Create figure with space for colorbar
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get device from model
    device = next(model.parameters()).device

    x_range = np.linspace(0, domain_size, 100)
    y_range = np.linspace(0, domain_size, 100)
    X, Y = np.meshgrid(x_range, y_range)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    params = {k: torch.full_like(x_flat, v) for k, v in scenario_physical.items()}

    # Pre-compute to find value range for proper color scaling
    print("  Computing value range for visualization...")
    t_mid = (model.t_min.item() + model.t_max.item()) / 2  # Middle of training time range
    start = time.time()
    with torch.no_grad():
        t_test = torch.full_like(x_flat, t_mid)
        phi_test = model(x_flat, y_flat, t_test,
                        params['cx'], params['cy'], params['u'],
                        params['v'], params['d'], params['kappa'],
                        params['Q'], normalize=True)
        phi_np = phi_test.cpu().numpy()
        print(f"  Model output range: min={phi_np.min():.2e}, max={phi_np.max():.2e}, mean={phi_np.mean():.2e}")
    end = time.time()
    print(f"  Time taken to compute value range: {end - start:.2f} seconds")
    # Determine appropriate levels based on actual output
    phi_max = max(phi_np.max(), 1e-10)
    phi_min = max(phi_np[phi_np > 0].min() if (phi_np > 0).any() else 1e-15, 1e-15)
    log_min = np.log10(phi_min)
    log_max = np.log10(phi_max)
    print(f"  Log10 range: [{log_min:.1f}, {log_max:.1f}]")

    # Use log scale if range spans > 2 orders of magnitude
    use_log = (log_max - log_min) > 2

    # Set up fixed levels for consistent colorbar
    if use_log:
        levels = np.linspace(log_min, log_max, 50)
        cbar_label = "log₁₀(Concentration)"
    else:
        levels = np.linspace(0, phi_max * 1.1, 50)
        cbar_label = "Concentration"

    # Create initial plot to set up colorbar (use t=0 for initial frame)
    with torch.no_grad():
        t_flat_init = torch.full_like(x_flat, 0.0)
        phi_init = model(x_flat, y_flat, t_flat_init,
                        params['cx'], params['cy'], params['u'],
                        params['v'], params['d'], params['kappa'],
                        params['Q'], normalize=True)
        phi_grid_init = phi_init.reshape(100, 100).cpu().numpy()

    if use_log:
        phi_plot_init = np.log10(np.clip(phi_grid_init, 1e-15, None))
    else:
        phi_plot_init = phi_grid_init

    # Initial contour and colorbar (colorbar stays fixed)
    cs = ax.contourf(X, Y, phi_plot_init, levels=levels, cmap='jet', extend='both')
    cbar = fig.colorbar(cs, ax=ax, label=cbar_label, shrink=0.9)

    # Add tick labels in scientific notation for log scale
    if use_log:
        # Show actual concentration values on colorbar
        tick_locs = np.linspace(log_min, log_max, 6)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([f'$10^{{{int(t)}}}$' if t == int(t) else f'$10^{{{t:.1f}}}$' for t in tick_locs])

    # Time range for animation (start from 0 now that we have IC data)
    t_min_anim = 0.0  # Start from t=0 to show buildup
    t_max_anim = model.t_max.item()
    print(f"  Animation time range: {t_min_anim:.0f}s to {t_max_anim:.0f}s")

    def update(frame):
        ax.clear()
        # Linear time progression from 0 to t_max
        t_val = t_min_anim + (frame / (frames - 1)) * (t_max_anim - t_min_anim)
        t_flat = torch.full_like(x_flat, t_val)

        with torch.no_grad():
            phi_pred = model(x_flat, y_flat, t_flat,
                           params['cx'], params['cy'], params['u'],
                           params['v'], params['d'], params['kappa'],
                           params['Q'], normalize=True)
            phi_grid = phi_pred.reshape(100, 100).cpu().numpy()

        if use_log:
            phi_plot = np.log10(np.clip(phi_grid, 1e-15, None))
        else:
            phi_plot = phi_grid

        ax.contourf(X, Y, phi_plot, levels=levels, cmap='jet', extend='both')
        ax.scatter(scenario_physical['cx'], scenario_physical['cy'],
                  color='white', marker='x', s=150, linewidths=3, label='Source')
        ax.set_title(f"Benzene Plume Dispersion | Time: {t_val:.0f}s | Max: {phi_grid.max():.2e}", fontsize=12)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_xlim(0, domain_size)
        ax.set_ylim(0, domain_size)
        ax.set_aspect('equal')
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    plt.tight_layout()
    plt.close()
    return ani


# Training script removed for deployment
if __name__ == "__main__":
    pass