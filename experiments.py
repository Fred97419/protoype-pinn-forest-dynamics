import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.interpolate import RegularGridInterpolator


# ============================================================
# GENERAL
# ============================================================

def set_seed(seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    return device


# ============================================================
# PARAMETERS
# ============================================================

def get_parameters(device):

    params = {
        # Domain
        "Lx": 20.0,
        "Ly": 20.0,
        "T": 100.0,

        # True physical parameters
        "D_true": np.array(
            [0.015, 0.010, 0.020],
            dtype=np.float32
        ),

        "r_true": np.array(
            [0.060, 0.050, 0.080],
            dtype=np.float32
        ),

        # Initial guesses for inverse problem
        "D_initial_guess": torch.tensor(
            [0.012, 0.008, 0.014],
            dtype=torch.float32,
            device=device
        ),

        "r_initial_guess": torch.tensor(
            [0.050, 0.040, 0.070],
            dtype=torch.float32,
            device=device
        ),

        # Beer-Lambert coefficients
        "alpha_light": torch.tensor(
            [0.8, 1.0, 1.2],
            dtype=torch.float32,
            device=device
        ),

        # Initial Gaussian populations
        "A": np.array(
            [0.75, 0.65, 0.80],
            dtype=np.float32
        ),

        "mu": np.array([
            [5.0, 5.0],
            [15.0, 6.0],
            [10.0, 15.0]
        ], dtype=np.float32),

        "sigma": np.array(
            [0.8, 1.0, 0.9],
            dtype=np.float32
        )
    }

    return params


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_x(x, params):

    return 2 * (x / params["Lx"]) - 1


def normalize_y(y, params):

    return 2 * (y / params["Ly"]) - 1


def normalize_t(t, params):

    return 2 * (t / params["T"]) - 1


# ============================================================
# INITIAL CONDITIONS
# ============================================================

def initial_conditions(X, Y, params):

    A = params["A"]
    mu = params["mu"]
    sigma = params["sigma"]

    u1 = A[0] * np.exp(
        -(
            (X - mu[0, 0])**2
            + (Y - mu[0, 1])**2
        )
        / (2 * sigma[0]**2)
    )

    u2 = A[1] * np.exp(
        -(
            (X - mu[1, 0])**2
            + (Y - mu[1, 1])**2
        )
        / (2 * sigma[1]**2)
    )

    u3 = A[2] * np.exp(
        -(
            (X - mu[2, 0])**2
            + (Y - mu[2, 1])**2
        )
        / (2 * sigma[2]**2)
    )

    return np.stack(
        [u1, u2, u3],
        axis=-1
    )


# ============================================================
# REFERENCE SIMULATION
# ============================================================

def simulate_reference(
    params,
    Nx=80,
    Ny=80,
    Nt=1001
):

    Lx = params["Lx"]
    Ly = params["Ly"]
    T = params["T"]

    D_true = params["D_true"]
    r_true = params["r_true"]

    alpha_light = (
        params["alpha_light"]
        .detach()
        .cpu()
        .numpy()
    )

    # Grids
    x_grid = np.linspace(
        0,
        Lx,
        Nx
    )

    y_grid = np.linspace(
        0,
        Ly,
        Ny
    )

    t_grid = np.linspace(
        0,
        T,
        Nt
    )

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dt = t_grid[1] - t_grid[0]

    X_grid, Y_grid = np.meshgrid(
        x_grid,
        y_grid
    )

    # Solution
    U = np.zeros(
        (Nt, Ny, Nx, 3),
        dtype=np.float32
    )

    U[0] = initial_conditions(
        X_grid,
        Y_grid,
        params
    )

    # Time integration
    for n in range(Nt - 1):

        current = U[n]

        lap = np.zeros_like(
            current
        )

        lap[1:-1, 1:-1, :] = (

            (
                current[1:-1, 2:, :]
                - 2 * current[1:-1, 1:-1, :]
                + current[1:-1, :-2, :]
            ) / dx**2

            +

            (
                current[2:, 1:-1, :]
                - 2 * current[1:-1, 1:-1, :]
                + current[:-2, 1:-1, :]
            ) / dy**2
        )

        u1 = current[:, :, 0]
        u2 = current[:, :, 1]
        u3 = current[:, :, 2]

        u_total = u1 + u2 + u3

        # Beer-Lambert light attenuation
        light = np.exp(
            -(
                alpha_light[0] * u1
                + alpha_light[1] * u2
                + alpha_light[2] * u3
            )
        )

        reaction1 = (
            r_true[0]
            * u1
            * (1 - u1)
            * (1 - u_total)
            * light
        )

        reaction2 = (
            r_true[1]
            * u2
            * (1 - u2)
            * (1 - u_total)
            * light
        )

        reaction3 = (
            r_true[2]
            * u3
            * (1 - u3)
            * (1 - u_total)
            * light
        )

        reaction = np.stack(
            [
                reaction1,
                reaction2,
                reaction3
            ],
            axis=-1
        )

        U[n + 1] = (
            current
            + dt
            * (
                lap
                * D_true.reshape(
                    1,
                    1,
                    3
                )
                + reaction
            )
        )

        # Neumann boundary conditions
        # zero flux
        U[n + 1, 0, :, :] = (
            U[n + 1, 1, :, :]
        )

        U[n + 1, -1, :, :] = (
            U[n + 1, -2, :, :]
        )

        U[n + 1, :, 0, :] = (
            U[n + 1, :, 1, :]
        )

        U[n + 1, :, -1, :] = (
            U[n + 1, :, -2, :]
        )

        # Avoid negative values
        U[n + 1] = np.maximum(
            U[n + 1],
            0
        )

    reference = {
        "U_reference": U,

        "x_grid": x_grid,
        "y_grid": y_grid,
        "t_grid": t_grid,

        "X_grid": X_grid,
        "Y_grid": Y_grid
    }

    return reference


# ============================================================
# OBSERVATION DATA
# ============================================================

def generate_observation_data(
    reference,
    params,
    device,
    N_obs=500,
    noise_std=0.02
):

    Lx = params["Lx"]
    Ly = params["Ly"]
    T = params["T"]

    x_grid = reference["x_grid"]
    y_grid = reference["y_grid"]
    t_grid = reference["t_grid"]

    U_reference = (
        reference["U_reference"]
    )

    # Random observations
    x_obs = np.random.uniform(
        0,
        Lx,
        N_obs
    )

    y_obs = np.random.uniform(
        0,
        Ly,
        N_obs
    )

    t_obs = np.random.uniform(
        0,
        T,
        N_obs
    )

    points_obs = np.column_stack([
        t_obs,
        y_obs,
        x_obs
    ])

    observations = np.zeros(
        (N_obs, 3),
        dtype=np.float32
    )

    # Interpolation of reference solution
    for species in range(3):

        interpolator = (
            RegularGridInterpolator(
                (
                    t_grid,
                    y_grid,
                    x_grid
                ),
                U_reference[
                    :, :, :, species
                ]
            )
        )

        observations[
            :, species
        ] = interpolator(
            points_obs
        )

    # Add noise
    observations_noisy = (
        observations
        + np.random.normal(
            0.0,
            noise_std,
            observations.shape
        ).astype(np.float32)
    )

    X_obs = torch.tensor(
        np.column_stack([
            normalize_x(
                x_obs,
                params
            ),
            normalize_y(
                y_obs,
                params
            ),
            normalize_t(
                t_obs,
                params
            )
        ]),
        dtype=torch.float32,
        device=device
    )

    U_obs = torch.tensor(
        observations_noisy,
        dtype=torch.float32,
        device=device
    )

    return X_obs, U_obs


# ============================================================
# PDE COLLOCATION POINTS
# ============================================================

def generate_pde_data(
    params,
    device,
    N_f=8000
):

    Lx = params["Lx"]
    Ly = params["Ly"]
    T = params["T"]

    x_f = np.random.uniform(
        0,
        Lx,
        N_f
    )

    y_f = np.random.uniform(
        0,
        Ly,
        N_f
    )

    t_f = np.random.uniform(
        0,
        T,
        N_f
    )

    X_f = torch.tensor(
        np.column_stack([
            normalize_x(
                x_f,
                params
            ),
            normalize_y(
                y_f,
                params
            ),
            normalize_t(
                t_f,
                params
            )
        ]),
        dtype=torch.float32,
        device=device
    )

    return X_f


# ============================================================
# INITIAL CONDITION DATA
# ============================================================

def generate_initial_data(
    params,
    device,
    N_ic_side=30
):

    Lx = params["Lx"]
    Ly = params["Ly"]

    x_ic = np.linspace(
        0,
        Lx,
        N_ic_side
    )

    y_ic = np.linspace(
        0,
        Ly,
        N_ic_side
    )

    X_ic_grid, Y_ic_grid = (
        np.meshgrid(
            x_ic,
            y_ic
        )
    )

    x_i = X_ic_grid.ravel()
    y_i = Y_ic_grid.ravel()

    t_i = np.zeros_like(
        x_i
    )

    X_i = torch.tensor(
        np.column_stack([
            normalize_x(
                x_i,
                params
            ),
            normalize_y(
                y_i,
                params
            ),
            normalize_t(
                t_i,
                params
            )
        ]),
        dtype=torch.float32,
        device=device
    )

    U_i = torch.tensor(
        initial_conditions(
            X_ic_grid,
            Y_ic_grid,
            params
        ).reshape(
            -1,
            3
        ),
        dtype=torch.float32,
        device=device
    )

    return X_i, U_i


# ============================================================
# BOUNDARY DATA
# ============================================================

def generate_boundary_data(
    params,
    device,
    N_b=300
):

    Lx = params["Lx"]
    Ly = params["Ly"]
    T = params["T"]

    t_b = np.random.uniform(
        0,
        T,
        N_b
    )

    # Left / right
    y_lr = np.random.uniform(
        0,
        Ly,
        N_b
    )

    X_left = torch.tensor(
        np.column_stack([
            normalize_x(
                np.zeros(N_b),
                params
            ),
            normalize_y(
                y_lr,
                params
            ),
            normalize_t(
                t_b,
                params
            )
        ]),
        dtype=torch.float32,
        device=device
    )

    X_right = torch.tensor(
        np.column_stack([
            normalize_x(
                np.full(
                    N_b,
                    Lx
                ),
                params
            ),
            normalize_y(
                y_lr,
                params
            ),
            normalize_t(
                t_b,
                params
            )
        ]),
        dtype=torch.float32,
        device=device
    )

    # Bottom / top
    x_bt = np.random.uniform(
        0,
        Lx,
        N_b
    )

    X_bottom = torch.tensor(
        np.column_stack([
            normalize_x(
                x_bt,
                params
            ),
            normalize_y(
                np.zeros(N_b),
                params
            ),
            normalize_t(
                t_b,
                params
            )
        ]),
        dtype=torch.float32,
        device=device
    )

    X_top = torch.tensor(
        np.column_stack([
            normalize_x(
                x_bt,
                params
            ),
            normalize_y(
                np.full(
                    N_b,
                    Ly
                ),
                params
            ),
            normalize_t(
                t_b,
                params
            )
        ]),
        dtype=torch.float32,
        device=device
    )

    return (
        X_left,
        X_right,
        X_bottom,
        X_top
    )


# ============================================================
# GENERATE ALL DATA
# ============================================================

def generate_training_data(
    reference,
    params,
    device
):

    X_obs, U_obs = (
        generate_observation_data(
            reference,
            params,
            device
        )
    )

    X_f = generate_pde_data(
        params,
        device
    )

    X_i, U_i = (
        generate_initial_data(
            params,
            device
        )
    )

    (
        X_left,
        X_right,
        X_bottom,
        X_top
    ) = generate_boundary_data(
        params,
        device
    )

    data = {
        "X_obs": X_obs,
        "U_obs": U_obs,

        "X_f": X_f,

        "X_i": X_i,
        "U_i": U_i,

        "X_left": X_left,
        "X_right": X_right,

        "X_bottom": X_bottom,
        "X_top": X_top
    }

    return data


# ============================================================
# PINN
# ============================================================

class PINN(nn.Module):

    def __init__(
        self,
        hidden_size=64,
        hidden_layers=4
    ):

        super().__init__()

        layers = []

        # Input
        layers.append(
            nn.Linear(
                3,
                hidden_size
            )
        )

        layers.append(
            nn.Tanh()
        )

        # Hidden layers
        for _ in range(
            hidden_layers - 1
        ):

            layers.append(
                nn.Linear(
                    hidden_size,
                    hidden_size
                )
            )

            layers.append(
                nn.Tanh()
            )

        # 3 species
        layers.append(
            nn.Linear(
                hidden_size,
                3
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, X):

        return self.network(X)


# ============================================================
# AUTODIFFERENTIATION
# ============================================================

def compute_derivatives(
    model,
    X,
    params
):

    Lx = params["Lx"]
    Ly = params["Ly"]
    T = params["T"]

    X = (
        X.clone()
        .detach()
        .requires_grad_(True)
    )

    U = model(X)

    Ut_list = []
    Lap_list = []

    for i in range(3):

        u = U[:, i:i + 1]

        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(
                u
            ),
            create_graph=True
        )[0]

        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        grad_ux = torch.autograd.grad(
            outputs=u_x,
            inputs=X,
            grad_outputs=torch.ones_like(
                u_x
            ),
            create_graph=True
        )[0]

        grad_uy = torch.autograd.grad(
            outputs=u_y,
            inputs=X,
            grad_outputs=torch.ones_like(
                u_y
            ),
            create_graph=True
        )[0]

        u_xx = grad_ux[:, 0:1]
        u_yy = grad_uy[:, 1:2]

        # Chain rule due to normalization
        u_t = (
            2.0 / T
        ) * u_t

        u_xx = (
            2.0 / Lx
        )**2 * u_xx

        u_yy = (
            2.0 / Ly
        )**2 * u_yy

        Ut_list.append(
            u_t
        )

        Lap_list.append(
            u_xx + u_yy
        )

    U_t = torch.cat(
        Ut_list,
        dim=1
    )

    Lap = torch.cat(
        Lap_list,
        dim=1
    )

    return U, U_t, Lap


# ============================================================
# PDE RESIDUAL
# ============================================================

def pde_residuals(
    model,
    X_f,
    D,
    r,
    params
):

    alpha_light = (
        params["alpha_light"]
    )

    U, U_t, Lap = (
        compute_derivatives(
            model,
            X_f,
            params
        )
    )

    u1 = U[:, 0:1]
    u2 = U[:, 1:2]
    u3 = U[:, 2:3]

    u_total = (
        u1 + u2 + u3
    )

    light = torch.exp(
        -(
            alpha_light[0] * u1
            + alpha_light[1] * u2
            + alpha_light[2] * u3
        )
    )

    reaction1 = (
        r[0]
        * u1
        * (1 - u1)
        * (1 - u_total)
        * light
    )

    reaction2 = (
        r[1]
        * u2
        * (1 - u2)
        * (1 - u_total)
        * light
    )

    reaction3 = (
        r[2]
        * u3
        * (1 - u3)
        * (1 - u_total)
        * light
    )

    # PDE:
    # du/dt = D * Lap(u) + reaction

    f1 = (
        U_t[:, 0:1]
        - (
            D[0]
            * Lap[:, 0:1]
            + reaction1
        )
    )

    f2 = (
        U_t[:, 1:2]
        - (
            D[1]
            * Lap[:, 1:2]
            + reaction2
        )
    )

    f3 = (
        U_t[:, 2:3]
        - (
            D[2]
            * Lap[:, 2:3]
            + reaction3
        )
    )

    return torch.cat(
        [f1, f2, f3],
        dim=1
    )


# ============================================================
# LOSSES
# ============================================================

def data_loss(
    model,
    X_obs,
    U_obs
):

    prediction = model(
        X_obs
    )

    return torch.mean(
        (
            prediction
            - U_obs
        )**2
    )


def physics_loss(
    model,
    X_f,
    D,
    r,
    params
):

    residuals = (
        pde_residuals(
            model,
            X_f,
            D,
            r,
            params
        )
    )

    return torch.mean(
        residuals**2
    )


def initial_loss(
    model,
    X_i,
    U_i
):

    prediction = model(
        X_i
    )

    return torch.mean(
        (
            prediction
            - U_i
        )**2
    )


def normal_derivative_loss(
    model,
    X,
    spatial_index,
    scale
):

    X = (
        X.clone()
        .detach()
        .requires_grad_(True)
    )

    U = model(X)

    derivatives = []

    for i in range(3):

        u_i = U[
            :, i:i + 1
        ]

        grad_u = (
            torch.autograd.grad(
                outputs=u_i,
                inputs=X,
                grad_outputs=torch.ones_like(
                    u_i
                ),
                create_graph=True
            )[0]
        )

        du_dn_norm = (
            grad_u[
                :,
                spatial_index:
                spatial_index + 1
            ]
        )

        du_dn = (
            scale
            * du_dn_norm
        )

        derivatives.append(
            du_dn
        )

    derivatives = torch.cat(
        derivatives,
        dim=1
    )

    return torch.mean(
        derivatives**2
    )


def boundary_loss(
    model,
    X_left,
    X_right,
    X_bottom,
    X_top,
    params
):

    Lx = params["Lx"]
    Ly = params["Ly"]

    loss_left = (
        normal_derivative_loss(
            model,
            X_left,
            spatial_index=0,
            scale=2.0 / Lx
        )
    )

    loss_right = (
        normal_derivative_loss(
            model,
            X_right,
            spatial_index=0,
            scale=2.0 / Lx
        )
    )

    loss_bottom = (
        normal_derivative_loss(
            model,
            X_bottom,
            spatial_index=1,
            scale=2.0 / Ly
        )
    )

    loss_top = (
        normal_derivative_loss(
            model,
            X_top,
            spatial_index=1,
            scale=2.0 / Ly
        )
    )

    return (
        loss_left
        + loss_right
        + loss_bottom
        + loss_top
    ) / 4.0


def total_loss(
    model,
    D,
    r,
    data,
    params,
    lambda_data=1.0,
    lambda_pde=1.0,
    lambda_ic=1.0,
    lambda_bc=1.0
):

    l_data = data_loss(
        model,
        data["X_obs"],
        data["U_obs"]
    )

    l_pde = physics_loss(
        model,
        data["X_f"],
        D,
        r,
        params
    )

    l_ic = initial_loss(
        model,
        data["X_i"],
        data["U_i"]
    )

    l_bc = boundary_loss(
        model,
        data["X_left"],
        data["X_right"],
        data["X_bottom"],
        data["X_top"],
        params
    )

    total = (
        lambda_data * l_data
        + lambda_pde * l_pde
        + lambda_ic * l_ic
        + lambda_bc * l_bc
    )

    return (
        total,
        l_data,
        l_pde,
        l_ic,
        l_bc
    )


# ============================================================
# HISTORY
# ============================================================

def create_history():

    return {
        "total": [],
        "data": [],
        "pde": [],
        "ic": [],
        "bc": [],

        "D": [],
        "r": []
    }


def update_history(
    history,
    loss,
    l_data,
    l_pde,
    l_ic,
    l_bc,
    D,
    r
):

    history["total"].append(
        loss.item()
    )

    history["data"].append(
        l_data.item()
    )

    history["pde"].append(
        l_pde.item()
    )

    history["ic"].append(
        l_ic.item()
    )

    history["bc"].append(
        l_bc.item()
    )

    history["D"].append(
        D.detach()
        .cpu()
        .numpy()
        .copy()
    )

    history["r"].append(
        r.detach()
        .cpu()
        .numpy()
        .copy()
    )


# ============================================================
# PRETRAINING
# ============================================================

def pretrain(
    model,
    data,
    params,
    device,
    epochs,
    history
):

    D_fixed = torch.tensor(
        params["D_initial_guess"],
        dtype=torch.float32,
        device=device
    )

    r_fixed = torch.tensor(
        params["r_initial_guess"],
        dtype=torch.float32,
        device=device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    for epoch in range(
        epochs
    ):

        optimizer.zero_grad()

        (
            loss,
            l_data,
            l_pde,
            l_ic,
            l_bc
        ) = total_loss(
            model,
            D_fixed,
            r_fixed,
            data,
            params
        )

        loss.backward()

        optimizer.step()

        update_history(
            history,
            loss,
            l_data,
            l_pde,
            l_ic,
            l_bc,
            D_fixed,
            r_fixed
        )

        if (
            epoch % 500 == 0
            or epoch == epochs - 1
        ):

            print(
                f"Pretrain {epoch:5d}/{epochs} | "
                f"loss = {loss.item():.3e} | "
                f"D = {D_fixed.detach().cpu().numpy()} | "
                f"r = {r_fixed.detach().cpu().numpy()}"
            )


# ============================================================
# INVERSE TRAINING WITH ADAM
# ============================================================

def inverse_training(
    model,
    rho_D,
    rho_r,
    data,
    params,
    epochs,
    history
):

    optimizer = torch.optim.Adam([
        {
            "params": model.parameters(),
            "lr": 1e-4
        },
        {
            "params": [
                rho_D,
                rho_r
            ],
            "lr": 1e-6
        }
    ])

    for epoch in range(
        epochs
    ):

        optimizer.zero_grad()

        D_now = torch.exp(
            rho_D
        )

        r_now = torch.exp(
            rho_r
        )

        (
            loss,
            l_data,
            l_pde,
            l_ic,
            l_bc
        ) = total_loss(
            model,
            D_now,
            r_now,
            data,
            params
        )

        loss.backward()

        optimizer.step()

        # Values AFTER optimization step
        D_now = torch.exp(
            rho_D
        )

        r_now = torch.exp(
            rho_r
        )

        update_history(
            history,
            loss,
            l_data,
            l_pde,
            l_ic,
            l_bc,
            D_now,
            r_now
        )

        if (
            epoch % 500 == 0
            or epoch == epochs - 1
        ):

            print(
                f"Inverse {epoch:5d}/{epochs} | "
                f"loss = {loss.item():.3e} | "
                f"D = {D_now.detach().cpu().numpy()} | "
                f"r = {r_now.detach().cpu().numpy()}"
            )


# ============================================================
# L-BFGS
# ============================================================

def lbfgs_refinement(
    model,
    rho_D,
    rho_r,
    data,
    params,
    max_iter=500
):

    optimizer_lbfgs = (
        torch.optim.LBFGS(
            list(
                model.parameters()
            )
            + [
                rho_D,
                rho_r
            ],
            lr=1.0,
            max_iter=max_iter,
            history_size=100,
            line_search_fn="strong_wolfe"
        )
    )

    lbfgs_history = []

    def closure():

        optimizer_lbfgs.zero_grad()

        D_now = torch.exp(
            rho_D
        )

        r_now = torch.exp(
            rho_r
        )

        (
            loss,
            l_data,
            l_pde,
            l_ic,
            l_bc
        ) = total_loss(
            model,
            D_now,
            r_now,
            data,
            params
        )

        loss.backward()

        lbfgs_history.append({
            "total": loss.item(),
            "data": l_data.item(),
            "pde": l_pde.item(),
            "ic": l_ic.item(),
            "bc": l_bc.item()
        })

        return loss

    optimizer_lbfgs.step(
        closure
    )

    return lbfgs_history


# ============================================================
# PARAMETER RESULTS
# ============================================================

def print_parameters(
    rho_D,
    rho_r,
    params
):

    D_true = params[
        "D_true"
    ]

    r_true = params[
        "r_true"
    ]

    D_learned = (
        torch.exp(rho_D)
        .detach()
        .cpu()
        .numpy()
    )

    r_learned = (
        torch.exp(rho_r)
        .detach()
        .cpu()
        .numpy()
    )

    D_error = (
        np.abs(
            D_learned
            - D_true
        )
        / np.abs(
            D_true
        )
    )

    r_error = (
        np.abs(
            r_learned
            - r_true
        )
        / np.abs(
            r_true
        )
    )

    print()
    print("True D     :", D_true)
    print("Estimated D   :", D_learned)
    print(
        " D error    :",
        D_error
    )

    print()

    print("True r     :", r_true)
    print("Estimated r    :", r_learned)
    print(
        "r error     :",
        r_error
    )

    return (
        D_learned,
        r_learned
    )


# ============================================================
# LOSS PLOT
# ============================================================

def plot_losses(
    history,
    lbfgs_history
):

    loss_names = [
        "total",
        "data",
        "pde",
        "ic",
        "bc"
    ]

    adam_end = len(
        history["total"]
    )

    plt.figure(
        figsize=(9, 6)
    )

    for name in loss_names:

        adam_values = np.array(
            history[name]
        )

        lbfgs_values = np.array([
            h[name]
            for h
            in lbfgs_history
        ])

        all_values = (
            np.concatenate([
                adam_values,
                lbfgs_values
            ])
        )

        plt.plot(
            all_values,
            linewidth=1.5,
            label=name
        )

    plt.axvline(
        adam_end,
        linestyle="--",
        linewidth=2,
        label="L-BFGS"
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Optimization step"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "PINN losses "
        "(Adam + L-BFGS)"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# SPATIAL PREDICTION
# ============================================================

def predict_at_time(
    model,
    reference,
    params,
    device,
    t_value=50.0
):

    X_grid = reference[
        "X_grid"
    ]

    Y_grid = reference[
        "Y_grid"
    ]

    x_flat = X_grid.ravel()
    y_flat = Y_grid.ravel()

    t_flat = np.full_like(
        x_flat,
        t_value
    )

    X_eval = torch.tensor(
        np.column_stack([
            normalize_x(
                x_flat,
                params
            ),
            normalize_y(
                y_flat,
                params
            ),
            normalize_t(
                t_flat,
                params
            )
        ]),
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        prediction = (
            model(X_eval)
            .cpu()
            .numpy()
        )

    Ny, Nx = X_grid.shape

    prediction = (
        prediction.reshape(
            Ny,
            Nx,
            3
        )
    )

    return prediction


def plot_species(
    model,
    reference,
    params,
    device,
    t_value=50.0
):

    prediction = predict_at_time(
        model,
        reference,
        params,
        device,
        t_value
    )

    t_grid = reference[
        "t_grid"
    ]

    U_reference = reference[
        "U_reference"
    ]

    index_t = np.argmin(
        np.abs(
            t_grid
            - t_value
        )
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8)
    )

    for species in range(3):

        im1 = axes[
            0,
            species
        ].imshow(
            U_reference[
                index_t,
                :,
                :,
                species
            ],
            origin="lower",
            cmap="summer",
            extent=[
                0,
                params["Lx"],
                0,
                params["Ly"]
            ]
        )

        axes[
            0,
            species
        ].set_title(
            f"Reference - Species {species + 1}"
        )

        plt.colorbar(
            im1,
            ax=axes[
                0,
                species
            ]
        )

        im2 = axes[
            1,
            species
        ].imshow(
            prediction[
                :,
                :,
                species
            ],
            origin="lower",
            cmap="summer",
            extent=[
                0,
                params["Lx"],
                0,
                params["Ly"]
            ]
        )

        axes[
            1,
            species
        ].set_title(
            f"PINN - Species {species + 1}"
        )

        plt.colorbar(
            im2,
            ax=axes[
                1,
                species
            ]
        )

    fig.suptitle(
        f"Species distribution at t = {t_value}"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# COMPLETE EXPERIMENT
# ============================================================

def run_experiment(
    pretrain_epochs,
    inverse_epochs,
    lbfgs_iter
):

    start_time = time.time()

    # --------------------------------------------------------
    # 1. Seed and device
    # --------------------------------------------------------

    set_seed(42)

    device = get_device()

    # --------------------------------------------------------
    # 2. Parameters
    # --------------------------------------------------------

    params = get_parameters(
        device
    )

    # --------------------------------------------------------
    # 3. Generate reference simulation
    # --------------------------------------------------------

    print(
        "\nGenerating reference simulation"
    )

    reference = simulate_reference(
        params
    )

    # --------------------------------------------------------
    # 4. Generate ALL data
    # --------------------------------------------------------

    print(
        "Generating points"
    )

    data = generate_training_data(
        reference,
        params,
        device
    )

    # --------------------------------------------------------
    # 5. Model
    # --------------------------------------------------------

    model = PINN(
        hidden_size=64,
        hidden_layers=4
    ).to(
        device
    )

    # --------------------------------------------------------
    # 6. Inverse parameters
    # --------------------------------------------------------

    rho_D = nn.Parameter(
        torch.log(
            params[
                "D_initial_guess"
            ].clone()
        )
    )

    rho_r = nn.Parameter(
        torch.log(
            params[
                "r_initial_guess"
            ].clone()
        )
    )

    # --------------------------------------------------------
    # 7. History
    # --------------------------------------------------------

    history = create_history()

    # --------------------------------------------------------
    # 8. Pretraining
    # --------------------------------------------------------

    print(
        "\n----Pretraining with fixed parameters----  "
    )

    pretrain(
        model,
        data,
        params,
        device,
        pretrain_epochs,
        history
    )

    # --------------------------------------------------------
    # 9. Inverse training
    # --------------------------------------------------------

    print(
        "\n----Training with all parameters---- "
    )

    inverse_training(
        model,
        rho_D,
        rho_r,
        data,
        params,
        inverse_epochs,
        history
    )

    # --------------------------------------------------------
    # 10. L-BFGS
    # --------------------------------------------------------

    print(
        "\n---- L-BFGS ----"
    )

    lbfgs_history = (
        lbfgs_refinement(
            model,
            rho_D,
            rho_r,
            data,
            params,
            max_iter=lbfgs_iter
        )
    )

    # --------------------------------------------------------
    # 11. Results
    # --------------------------------------------------------

    print(
        "\n---- Parameters ----"
    )

    print_parameters(
        rho_D,
        rho_r,
        params
    )

    # --------------------------------------------------------
    # 12. Runtime
    # --------------------------------------------------------

    runtime = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Runtime: "
        f"{runtime / 60:.2f} min"
    )

    # --------------------------------------------------------
    # 13. Plots
    # --------------------------------------------------------

    plot_losses(
        history,
        lbfgs_history
    )

    plot_species(
        model,
        reference,
        params,
        device,
        t_value=50.0
    )

    return {
        "model": model,

        "rho_D": rho_D,
        "rho_r": rho_r,

        "history": history,

        "lbfgs_history":
            lbfgs_history,

        "reference":
            reference,

        "data":
            data,

        "params":
            params
    }


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Multi-species "
                "2D inverse PINN"
            )
        )
    )

    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=4000,
        help=(
            "Number of "
            "pre-training epochs"
        )
    )

    parser.add_argument(
        "--inverse-epochs",
        type=int,
        default=14000,
        help=(
            "Number of inverse "
            "training epochs"
        )
    )

    parser.add_argument(
        "--lbfgs-iter",
        type=int,
        default=500,
        help=(
            "Maximum number "
            "of L-BFGS iterations"
        )
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    run_experiment(
        pretrain_epochs=
            args.pretrain_epochs,

        inverse_epochs=
            args.inverse_epochs,

        lbfgs_iter=
            args.lbfgs_iter
    )


if __name__ == "__main__":

    main()