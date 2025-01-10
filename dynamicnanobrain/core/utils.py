# utils.py

import torch

def inverse_gain_coefficient(gammas, eta_handle, Vthres, layer_params, device='cpu'):
    """
    Calculates the unity coupling coefficient and Imax.

    Parameters
    ----------
    gammas : list or torch.Tensor
        List of gamma values.
    eta_handle : callable
        Function to handle eta calculations.
    Vthres : float
        Threshold voltage.
    layer_params : dict
        Dictionary containing layer parameters ('Rstore', 'm', 'kT', 'I_Vt').
    device : torch.device or str, optional
        Device to use for Torch tensors. Default is 'cpu'.

    Returns
    -------
    tuple
        unity_coeff (float), Imax (float)
    """
    Rstore = layer_params.get('Rstore', None)
    m = layer_params.get('m', None)
    kT = layer_params.get('kT', None)
    I_Vt = layer_params.get('I_Vt', None)

    if None in [Rstore, m, kT, I_Vt]:
        raise KeyError("One or more required parameters ('Rstore', 'm', 'kT', 'I_Vt') not found.")

    # Assuming Rsum = Rstore for simplicity
    Rsum = Rstore

    max_Vg = Vthres * Rstore / Rsum
    Iexc = (Vthres / Rsum) * 1e9  # Convert to nA

    # Calculate Isd
    Isd = transistorIV(max_Vg, Vthres, m, kT, I_Vt, device=device)

    # Calculate Iout using eta_handle
    Iout = eta_handle(Isd) * Isd

    # Calculate unity_coeff and Imax
    unity_coeff = (Iexc / Iout).item()
    Imax = Iexc.item()

    return unity_coeff, Imax

def setup_gain(gammas, device='cpu'):
    """
    Calculates the eigenvalues based on gammas.

    Parameters
    ----------
    gammas : list or torch.Tensor
        List of gamma values.
    device : torch.device or str, optional
        Device to use for Torch tensors. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Tensor of eigenvalues.
    """
    eigenvalues = -torch.tensor(gammas, dtype=torch.float32, device=device) * 1.0  # Adjust scaling as needed
    return eigenvalues

def gain_function(s, eigvals, gammas):
    """
    Calculates the gain function G11 based on frequency s, eigenvalues, and gammas.

    Parameters
    ----------
    s : numpy.ndarray
        Frequency values (complex numbers).
    eigvals : torch.Tensor
        Tensor of eigenvalues.
    gammas : list or torch.Tensor
        List of gamma values.

    Returns
    -------
    tuple
        G11 (numpy.ndarray), additional_outputs (unused, can be None)
    """
    G11 = np.zeros_like(s, dtype=np.complex128)
    for gamma, eigval in zip(gammas, eigvals.cpu().numpy()):
        G11 += gamma / (s + eigval)
    return G11, None

def transistorIV(Vg, Vt, m, kT, I_Vt, device='cpu'):
    """
    Simulates transistor current based on gate voltage.

    Parameters
    ----------
    Vg : float or torch.Tensor
        Gate voltage.
    Vt : float or torch.Tensor
        Threshold voltage.
    m : float
        Parameter m.
    kT : float
        Thermal voltage.
    I_Vt : float
        Saturation current.
    device : torch.device or str, optional
        Device to use for Torch tensors. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Transistor current.
    """
    # Convert inputs to Tensor if they are not already
    if not isinstance(Vg, torch.Tensor):
        Vg = torch.tensor(Vg, dtype=torch.float32, device=device)
    if not isinstance(Vt, torch.Tensor):
        Vt = torch.tensor(Vt, dtype=torch.float32, device=device)

    exponent = (Vg - Vt) / (m * kT)
    Id_sub = I_Vt * torch.exp(exponent)
    Id_sat = torch.full_like(Id_sub, I_Vt)  # Assuming saturation current equals I_Vt
    Id = torch.where(Vg < Vt, Id_sub, Id_sat)
    return Id
