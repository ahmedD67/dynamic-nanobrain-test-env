import torch
import torch.nn.functional as F

NV = 3  # Internal voltage degrees of freedom


class Layer:
    keys = ['Rinh', 'Rexc', 'RLED', 'Rstore', 'Cinh', 'Cexc', 'CLED', 'Cstore', 'Cgate', 'Vt', 'm', 'I_Vt', 'vt', 'Lg',
            'AB', 'CB']
    units = ['Ohm'] * 4 + ['F'] * 4 + ['F/cm'] + ['V', 'dim. less', 'nA', 'cm/s', 'um', 'uA', '1/uA']
    kT = 0.02585

    def __init__(self, N, layer_type, path_to_file, device='cpu'):
        self.device = torch.device(device)
        self.N = N
        self.layer_type = layer_type
        self.p_dict = {}
        self.p_units = {}
        self.read_parameter_file(path_to_file)
        self.linslope = self.p_dict['Cgate'] * self.p_dict['vt'] * 1e9  # nA/V
        self.gammas = self.calc_gammas()
        self.A = self.calc_A()

    def read_parameter_file(self, path_to_file):
        """
        Reads device parameters from a file, ignoring comments and extracting numerical values.

        Parameters
        ----------
        path_to_file : str
            Path to the parameter file.

        Raises
        ------
        ValueError
            If the number of parameters does not match the expected number of keys.
        """
        params = []

        with open(path_to_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Remove leading/trailing whitespace
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Skip full-line comments
                if line.startswith('#'):
                    continue

                # Split the line at the first '#' to remove inline comments
                parts = line.split('#', 1)
                value_str = parts[0].strip()

                if value_str:
                    try:
                        # Convert the numerical part to float and append to params
                        value = float(value_str)
                        params.append(value)
                    except ValueError:
                        raise ValueError(f"Invalid numerical value on line {line_num}: '{line}'")

        # Convert the list of parameters to a PyTorch tensor
        params_tensor = torch.tensor(params, device=self.device)

        # Check if the number of parameters matches the number of expected keys
        if len(params_tensor) != len(self.keys):
            raise ValueError(
                f"Expected {len(self.keys)} parameters, but found {len(params_tensor)} in '{path_to_file}'.")

        # Assign each parameter to the corresponding key and unit
        for k, key in enumerate(self.keys):
            self.p_dict[key] = params_tensor[k].item()  # Store as a Python float
            self.p_units[key] = self.units[k]

    def calc_A_mat(self, g11, g22, g13, g23, g33):
        gsum = g13 + g23 + g33
        A = torch.tensor([
            [-g11, 0.0, g11],
            [0.0, -g22, g22],
            [g13, g23, -gsum]
        ], device=self.device)
        return A

    def calc_gammas(self, Rstore=None, Cstore=None):
        Cmem = self.calc_Cmem(Cstore)
        g11 = 1e-9 / self.p_dict['Cinh'] / self.p_dict['Rinh']  # ns^-1
        g22 = 1e-9 / self.p_dict['Cexc'] / self.p_dict['Rexc']  # ns^-1
        g13 = 1e-9 / Cmem / self.p_dict['Rinh']  # ns^-1
        g23 = 1e-9 / Cmem / self.p_dict['Rexc']  # ns^-1
        g33 = 1e-9 / Cmem / (Rstore if Rstore is not None else self.p_dict['Rstore'])  # ns^-1
        gled = 1e-9 / self.p_dict['CLED'] / self.p_dict['RLED']  # ns^-1
        return torch.tensor([g11, g22, g13, g23, g33, gled], device=self.device)

    def calc_Cmem(self, Cstore=None):
        if Cstore is None:
            Cmem = self.p_dict['Cstore'] + self.p_dict['Cgate'] * self.p_dict['Lg'] * 1e-4
        else:
            Cmem = Cstore + self.p_dict['Cgate'] * self.p_dict['Lg'] * 1e-4
        return Cmem

    def calc_A(self, Rstore=None, Cstore=None):
        gammas = self.calc_gammas(Rstore, Cstore) if (Rstore is not None or Cstore is not None) else self.gammas
        return self.calc_A_mat(*gammas[:-1])

    def get_node_name(self, node_idx, layer_idx=1):
        if isinstance(layer_idx, int):
            if self.layer_type == 'hidden':
                letters = 'HKLMN'
                letter = letters[layer_idx - 1]
            elif self.layer_type == 'input':
                letters = 'IJ'
                letter = letters[layer_idx]
            else:
                letter = self.layer_type[0].upper()
        else:
            letter = f"{layer_idx}_"
        return f"{letter}{node_idx}"

    def get_names(self, layer_idx=1):
        return [self.get_node_name(n, layer_idx) for n in range(self.N)]

    def reset(self):
        pass

    def reset_B(self):
        pass

    def setup_gain(self):
        A = self.A
        eigenvalues, _ = torch.linalg.eig(A)
        gled = self.gammas[-1]
        v_add = torch.cat((eigenvalues, torch.tensor([-gled], device=self.device)))
        return v_add

    def gain(self, s, eigvals, vsat=1e7, Vt_prime=0.):
        if Vt_prime > 0.:
            raise NotImplementedError('Values of Vt_prime not equal zero not supported')

        denom = torch.ones_like(s, dtype=torch.cfloat, device=self.device)
        for l in eigvals:
            denom *= (s - l)

        prefactor = self.gammas[-1] * vsat * self.p_dict['Cgate'] * 1e-9  # Convert ns to s

        g11 = self.gammas[3] * (s + self.gammas[1]) / self.p_dict['Cexc'] * prefactor / denom
        g12 = -self.gammas[2] * (s + self.gammas[0]) / self.p_dict['Cinh'] * prefactor / denom

        return g11, g12

    def transistorIV_example(self, Vstart=-0.5, Vend=1.0, NV=200):
        Vgate = torch.linspace(Vstart, Vend, NV, device=self.device)
        I = self.transistorIV(Vgate) * 1e-3  # Convert to uA
        data = torch.stack((Vgate, I), dim=1).cpu().numpy()
        return data  # Return as NumPy array for plotting

    def eta_example(self, handle):
        I = torch.logspace(-2, 2, steps=50, device=self.device)  # in uA
        eta = handle(I * 1e3)  # ABC expects nA
        if eta.numel() < I.numel():
            eta = eta.expand_as(I)
        data = torch.stack((I, eta), dim=1).cpu().numpy()
        return data  # Return as NumPy array for plotting

class HiddenLayer(Layer):
    def __init__(self, N, output_channel, inhibition_channel, excitation_channel, path_to_file, device='cpu', Vthres=1.2, multiA=False):
        super().__init__(N, layer_type='hidden', path_to_file=path_to_file, device=device)
        self.out_channel = output_channel
        self.inh_channel = inhibition_channel
        self.exc_channel = excitation_channel
        self.channel_map = {inhibition_channel: 0, excitation_channel: 1}

        # Initialize internal variables as PyTorch tensors
        self.V = torch.zeros((NV, self.N), device=self.device)
        self.B = torch.zeros_like(self.V)
        self.dV = torch.zeros_like(self.V)
        self.I = torch.zeros(self.N, device=self.device)
        self.P = torch.zeros(self.N, device=self.device)
        self.ISD = torch.zeros_like(self.I)
        self.Vthres = Vthres
        self.Vt_vec = None
        self.multiA = multiA
        self.Bscale = torch.diag(torch.tensor([1e-18 / self.p_dict['Cinh'],
                                              1e-18 / self.p_dict['Cexc'],
                                              0.0], device=self.device))

    def inverse_gain_coefficient(self):
        """
        Calculates the unity coupling coefficient and maximum current.

        Returns
        -------
        tuple
            (unity_coeff, Imax) where unity_coeff is a scaling factor and Imax is the maximum current in nA.
        """
        Rsum = self.p_dict['Rstore'] + self.p_dict['Rexc']
        max_Vg = self.Vthres * self.p_dict['Rstore'] / Rsum
        Iexc = self.Vthres / Rsum * 1e9  # nA

        # Convert max_Vg to tensor
        max_Vg_tensor = torch.tensor(max_Vg, device=self.device)

        # Calculate Isd using tensor input
        Isd = self.transistorIV(max_Vg_tensor)

        # Calculate Iout using the ABC efficiency model
        Iout = self.eta_ABC(Isd) * Isd

        # Ensure Iout is not zero to avoid division by zero
        Iout = torch.clamp(Iout, min=1e-12)

        # Calculate unity_coeff
        unity_coeff = (Iexc / Iout).item()

        return unity_coeff, Iexc

    def generate_uniform_Adist(self, scale):
        if not self.p_dict:
            raise ValueError("Please first assign a device before generating Adist")
        A = torch.zeros((self.N, 3, 3), device=self.device)
        R_ref = self.p_dict['Rstore']
        C_ref = self.p_dict['Cstore']
        scale_RC_dist = torch.sqrt(torch.rand(self.N, device=self.device) * (scale**2 - 1.0) + 1.0)
        for k in range(self.N):
            Rstore = R_ref * scale_RC_dist[k]
            Cstore = C_ref * scale_RC_dist[k]
            A[k] = self.calc_A(Rstore, Cstore)
        self.Adist = A

    def generate_exp_Adist(self, mean):
        if not self.p_dict:
            raise ValueError("Please first assign a device before generating Adist")
        A = torch.zeros((self.N, 3, 3), device=self.device)
        R_ref = self.p_dict['Rstore']
        C_ref = self.p_dict['Cstore']
        scale_RC_dist = torch.sqrt(torch.distributions.Exponential(scale=mean).sample((self.N,)).to(self.device))
        for k in range(self.N):
            Rstore = R_ref * (1 + scale_RC_dist[k])
            Cstore = C_ref * (1 + scale_RC_dist[k])
            A[k] = self.calc_A(Rstore, Cstore)
        self.Adist = A

    def generate_poisson_Adist(self, mean):
        if not self.p_dict:
            raise ValueError("Please first assign a device before generating Adist")
        A = torch.zeros((self.N, 3, 3), device=self.device)
        R_ref = self.p_dict['Rstore']
        C_ref = self.p_dict['Cstore']
        scale_RC_dist = torch.sqrt(torch.poisson(mean, (self.N,)).to(self.device).float())
        for k in range(self.N):
            Rstore = R_ref * (1 + scale_RC_dist[k])
            Cstore = C_ref * (1 + scale_RC_dist[k])
            A[k] = self.calc_A(Rstore, Cstore)
        self.Adist = A

    def generate_Adist(self, noise=0.1, p_label='Rstore'):
        if not self.p_dict:
            raise ValueError("Please first assign a device before generating Adist")
        A = torch.zeros((self.N, 3, 3), device=self.device)
        p_ref = self.p_dict[p_label]
        Rstore_dist = torch.normal(mean=p_ref, std=noise * p_ref, size=(self.N,), device=self.device)
        Rstore_dist = torch.clamp(Rstore_dist, min=p_ref * 0.01)
        for k in range(self.N):
            A[k] = self.calc_A(Rstore_dist[k])
        self.Adist = A

    def specify_Vt(self, Vts):
        self.Vt_vec = Vts.to(self.device)

    def get_dV(self, t):
        if not self.multiA:
            self.dV = torch.matmul(self.A, self.V) + torch.matmul(self.Bscale, self.B)
        else:
            # Assuming self.Adist is (N, 3, 3) and self.V is (3, N)
            self.dV = torch.einsum('jik,kj->ij', self.Adist, self.V) + torch.matmul(self.Bscale, self.B)
        return self.dV

    def update_V(self, dt):
        self.V += dt * self.dV
        overshoots = ((self.V < -self.Vthres) & (self.V > self.Vthres)).sum().item()
        self.V = torch.clamp(self.V, -self.Vthres, self.Vthres)
        return overshoots

    def eta_ABC(self, I):
        return self.ABC(I * 1e-3, self.p_dict['AB'], self.p_dict['CB'])  # ABC expects nA

    def ABC(self, I, AB, CB):
        return I / (AB + I + CB * I**2)

    def Id_sub(self, Vg, Vt, mask):
        return self.p_dict['I_Vt'] * torch.exp((Vg - Vt[mask]) / self.p_dict['m'] / self.kT)

    def Id_sat(self, Vg, Vt, mask):
        return self.p_dict['I_Vt'] + self.linslope * (Vg - Vt[~mask])

    def Id_sub_0(self, Vg, Vt):
        """
        Subthreshold current.

        Parameters
        ----------
        Vg : torch.Tensor
            Gate voltage(s).
        Vt : torch.Tensor
            Threshold voltage(s).

        Returns
        -------
        torch.Tensor
            Subthreshold current(s).
        """
        exponent = (Vg - Vt) / self.p_dict['m'] / self.kT
        # Ensure exponent is a tensor
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent, device=self.device)
        return self.p_dict['I_Vt'] * torch.exp(exponent)

    def Id_sat_0(self, Vg, Vt):
        """
        Saturation current.

        Parameters
        ----------
        Vg : torch.Tensor
            Gate voltage(s).
        Vt : torch.Tensor
            Threshold voltage(s).

        Returns
        -------
        torch.Tensor
            Saturation current(s).
        """
        return self.p_dict['I_Vt'] + self.linslope * (Vg - Vt)

    def transistorIV(self, Vg, Vt_vec=None):
        """
        Calculates the transistor current based on the gate voltage and threshold voltage.

        Parameters
        ----------
        Vg : float, int, list, numpy.ndarray, or torch.Tensor
            Gate voltage(s).
        Vt_vec : torch.Tensor, optional
            Threshold voltage vector for each node. Defaults to None.

        Returns
        -------
        torch.Tensor
            Transistor current(s) in nA.
        """
        # Convert Vg to tensor if it's a float, int, list, or numpy array
        if isinstance(Vg, (float, int)):
            Vg = torch.tensor([Vg], device=self.device)
        elif isinstance(Vg, list):
            Vg = torch.tensor(Vg, device=self.device)
        elif not isinstance(Vg, torch.Tensor):
            raise TypeError(f"Vg must be a float, int, list, numpy.ndarray, or torch.Tensor, got {type(Vg)}")

        if Vt_vec is None:
            Vt = torch.tensor([self.p_dict['Vt']], device=self.device)
            mask = Vg < Vt
            return torch.where(mask, self.Id_sub_0(Vg, Vt), self.Id_sat_0(Vg, Vt))
        else:
            Vt = Vt_vec
            if not isinstance(Vt, torch.Tensor):
                Vt = torch.tensor(Vt, device=self.device)
            mask = Vg < Vt
            return torch.where(mask, self.Id_sub(Vg, Vt, mask), self.Id_sat(Vg, Vt, mask))

    def update_I(self, dt):
        self.ISD = self.transistorIV(self.V[2], self.Vt_vec)
        self.I += dt * self.gammas[-1] * (self.ISD - self.I)
        self.P = self.I * self.eta_ABC(self.I)

    def reset_B(self):
        self.B.zero_()

    def reset_I(self):
        self.I.zero_()

    def reset_V(self):
        self.V.zero_()

    def update_B(self, weights, C):
        if weights.channel == self.inh_channel:
            self.B[self.channel_map[weights.channel]] -= torch.matmul(weights.W, C)
        else:
            self.B[self.channel_map[weights.channel]] += torch.matmul(weights.W, C)

    def reset(self):
        self.reset_B()
        self.reset_I()
        self.reset_V()

class InputLayer(Layer):
    def __init__(self, N, path_to_file, device='cpu'):
        super().__init__(N, layer_type='input', path_to_file=path_to_file, device=device)
        self.C = torch.zeros((self.N,), device=self.device)
        self.input_func_handles = {}
        self.input_func_args = {}

    def set_input_vector_func(self, func_handle, func_args=None):
        self.input_func_handles['v'] = func_handle
        self.input_func_args['v'] = func_args

    def set_input_func_per_node(self, node, func_handle, func_args=None):
        self.input_func_handles[node] = func_handle
        self.input_func_args[node] = func_args

    def update_C(self, t):
        if 'v' in self.input_func_handles:
            try:
                if self.input_func_args['v'] is not None:
                    self.C = self.input_func_handles['v'](t, *self.input_func_args['v']).to(self.device)
                else:
                    self.C = self.input_func_handles['v'](t).to(self.device)
            except:
                pass
            self.C = self.C.flatten()
        else:
            for key in self.input_func_handles:
                try:
                    if self.input_func_args[key] is not None:
                        self.C[key] = self.input_func_handles[key](t, *self.input_func_args[key]).to(self.device)
                    else:
                        self.C[key] = self.input_func_handles[key](t).to(self.device)
                except:
                    pass
        return self.C

class OutputLayer(Layer):
    def __init__(self, N, teacher_delay=None, nsave=800, path_to_file='', device='cpu'):
        super().__init__(N, layer_type='output', path_to_file=path_to_file, device=device)
        self.C = torch.zeros(self.N, device=self.device)
        self.B = torch.zeros_like(self.C)
        self.I = torch.zeros(self.N, device=self.device)
        self.output_func_handles = {}
        self.output_func_args = {}
        self.teacher_delay = teacher_delay
        self.nsave = nsave
        self.reset_teacher_memory(self.nsave)

    def set_teacher_delay(self, delay, nsave=None):
        self.teacher_delay = delay
        if nsave is not None:
            self.nsave = nsave
        self.reset_teacher_memory(self.nsave)

    def reset_teacher_memory(self, nsave):
        self.teacher_memory = torch.zeros((nsave, self.B.numel() + 1), device=self.device)

    def set_output_func(self, func_handle, func_args=None):
        self.output_func_handles = func_handle
        self.output_func_args = func_args

    def get_output_current(self, t):
        try:
            if self.output_func_args is not None:
                self.I = self.output_func_handles(t, *self.output_func_args).to(self.device)
            else:
                self.I = self.output_func_handles(t).to(self.device)
        except:
            pass
        return self.I

    def update_B(self, weights, C):
        self.B += torch.matmul(weights.W, C)

    def reset_B(self):
        self.B.zero_()

    def lin_intp(self, x, f):
        x0 = f[0, 0]
        x1 = f[1, 0]
        p1 = (x - x0) / (x1 - x0)
        p0 = 1 - p1
        return p0 * f[0] + p1 * f[1]

    def update_C_from_B(self, t, t0=0):
        if self.teacher_delay is not None:
            self.teacher_memory[:-1] = self.teacher_memory[1:]
            self.teacher_memory[-1] = torch.cat((torch.tensor([t], device=self.device), self.B.flatten()))
            t_find = max(t - self.teacher_delay, t0)
            idx_t = -1
            while t_find < self.teacher_memory[idx_t, 0]:
                idx_t -= 1
            if t_find > t0:
                end = idx_t + 2 if idx_t + 2 < 0 else None
                teacher_signal = self.lin_intp(t_find, self.teacher_memory[idx_t:end])
            else:
                teacher_signal = self.teacher_memory[idx_t]
            B_for_update = teacher_signal[1:].reshape(self.N)
        else:
            B_for_update = self.B.clone()
        self.C = B_for_update.clone()

    def update_C(self, t):
        self.C = self.get_output_current(t).clone()

    def reset(self):
        self.reset_B()
        self.reset_teacher_memory(self.nsave)

def connect_layers(down, up, layers, channel, connect_all=False, device='cpu'):
    class Weights:
        def __init__(self, down, up, layers, channel, connect_all=False):
            self.from_layer = down
            self.to_layer = up
            self.channel = channel
            L0 = layers[down]
            L1 = layers[up]
            self.W = torch.zeros((L1.N, L0.N), device=device)
            if connect_all:
                self.W.fill_(1.0)

        def connect_nodes(self, from_node, to_node, weight=1.0):
            self.W[to_node, from_node] = weight

        def get_edges(self):
            edges = {}
            edge_list = []
            rows, cols = torch.nonzero(self.W > 0, as_tuple=True)
            for down, up in zip(cols.tolist(), rows.tolist()):
                weight = self.W[up, down].item()
                edge_list.append((down, up, weight))
            edges[self.channel] = edge_list
            return edges

        def print_W(self):
            print(f"{self.channel}:\n{self.W.cpu().numpy()}")

        def set_W(self, W):
            self.W = W.to(self.device)

        def scale_W(self, scale):
            self.W *= scale

        def ask_W(self, silent=False):
            if not silent:
                print(f'Set weights by set_W(tensor of size {self.W.shape})')
            return self.W.shape

    return Weights(down, up, layers, channel, connect_all)

def reset(layers) :
    """ Resets all layer values to 0."""
    for key in layers :
        # all layer types have this function
        layers[key].reset()