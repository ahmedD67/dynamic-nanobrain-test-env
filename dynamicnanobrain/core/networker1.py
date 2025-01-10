# layers.py

import torch
import torch.nn as nn

NV = 3  # internal voltage degrees of freedom

class Layer:
    keys = ['Rinh', 'Rexc', 'RLED', 'Rstore', 'Cinh', 'Cexc', 'CLED', 'Cstore',
            'Cgate', 'Vt', 'm', 'I_Vt', 'vt', 'Lg', 'AB', 'CB']
    units = ['Ohm'] * 4 + ['F'] * 4 + ['F/cm'] + ['V', 'dim. less', 'nA',
                                                'cm/s', 'um', 'uA', '1/uA']
    kT = 0.02585

    def __init__(self, N, layer_type, path_to_file, device):
        self.N = N
        self.layer_type = layer_type
        self.p_dict = {}
        self.p_units = {}
        self.device = device
        self.read_parameter_file(path_to_file, self.p_dict, self.p_units)
        self.linslope = self.p_dict['Cgate'] * self.p_dict['vt'] * 1e9  # nA/V
        self.gammas = self.calc_gammas()
        self.A = self.calc_A()

    def read_parameter_file(self, path_to_file, p_dict, p_units):
        """
        Reads parameters from a text file and assigns them to dictionaries without using NumPy.

        Args:
            path_to_file (str): Path to the parameter file.
            p_dict (dict): Dictionary to store parameter values.
            p_units (dict): Dictionary to store parameter units.
        """
        params = []
        try:
            with open(path_to_file, 'r') as file:
                for line_number, line in enumerate(file, start=1):
                    stripped_line = line.strip()

                    # Skip empty lines
                    if not stripped_line:
                        continue

                    # Skip full-line comments
                    if stripped_line.startswith('#'):
                        continue

                    # Handle inline comments by splitting at '#'
                    if '#' in stripped_line:
                        data_part = stripped_line.split('#')[0].strip()
                    else:
                        data_part = stripped_line

                    # If there's no data before the comment, skip the line
                    if not data_part:
                        continue

                    # Split the data part into tokens (assuming whitespace separation)
                    tokens = data_part.split()

                    for token in tokens:
                        try:
                            param = float(token)
                            params.append(param)
                        except ValueError:
                            raise ValueError(f"Invalid numerical value '{token}' on line {line_number}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{path_to_file}' does not exist.")
        except IOError as e:
            raise IOError(f"An error occurred while reading the file: {e}")

        # Ensure that the number of parameters matches the number of keys
        if len(params) < len(self.keys):
            raise ValueError(
                f"Not enough parameters in the file. Expected at least {len(self.keys)}, got {len(params)}."
            )
        elif len(params) > len(self.keys):
            print(
                f"Warning: More parameters in the file than expected. Extra parameters will be ignored."
            )

        # Assign parameters and units to the dictionaries
        for k, key in enumerate(self.keys):
            p_dict[key] = params[k]
            p_units[key] = self.units[k]

    def calc_A(self, Rstore=None, Cstore=None):
        if (Rstore is not None) and (Cstore is not None):
            new_gammas = self.calc_gammas(Rstore, Cstore)
            return self.A_mat(*new_gammas[:-1])
        elif Rstore is not None:
            new_gammas = self.calc_gammas(Rstore)
            return self.A_mat(*new_gammas[:-1])
        else:
            return self.A_mat(*self.gammas[:-1])

    def A_mat(self, g11, g22, g13, g23, g33):
        gsum = g13 + g23 + g33
        A = torch.tensor([[-g11, 0, g11],
                          [0, -g22, g22],
                          [g13, g23, -gsum]], device=self.device)
        return A

    def calc_gammas(self, Rstore=None, Cstore=None):
        Cmem = self.calc_Cmem(Cstore) if Cstore is not None else self.calc_Cmem()
        g11 = 1e-9 / self.p_dict['Cinh'] / self.p_dict['Rinh']  # ns^-1 # GHz
        g22 = 1e-9 / self.p_dict['Cexc'] / self.p_dict['Rexc']  # ns^-1 # GHz
        g13 = 1e-9 / Cmem / self.p_dict['Rinh']  # ns^-1 # GHz
        g23 = 1e-9 / Cmem / self.p_dict['Rexc']  # ns^-1 # GHz
        g33 = 1e-9 / Cmem / (Rstore if Rstore is not None else self.p_dict['Rstore'])
        gled = 1e-9 / self.p_dict['CLED'] / self.p_dict['RLED']  # ns^-1 # GHz
        return torch.tensor([g11, g22, g13, g23, g33, gled], device=self.device)

    def calc_Cmem(self, Cstore=None):
        return self.p_dict['Cstore'] + self.p_dict['Cgate'] * self.p_dict['Lg'] * 1e-4 \
            if Cstore is None else Cstore + self.p_dict['Cgate'] * self.p_dict['Lg'] * 1e-4

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


class HiddenLayer(Layer):
    def __init__(self, N, output_channel, inhibition_channel, excitation_channel,
                 device, Vthres=1.2, multiA=False, path_to_file=''):
        super().__init__(N, layer_type='hidden', path_to_file=path_to_file, device=device)
        self.out_channel = output_channel
        self.inh_channel = inhibition_channel
        self.exc_channel = excitation_channel
        self.channel_map = {inhibition_channel: 0, excitation_channel: 1}

        # Initialize tensors on the specified device
        self.V = torch.zeros((NV, self.N), device=device)
        self.B = torch.zeros_like(self.V)
        self.dV = torch.zeros_like(self.V)
        self.I = torch.zeros(self.N, device=device)
        self.P = torch.zeros(self.N, device=device)
        self.ISD = torch.zeros_like(self.I)
        self.Vthres = Vthres
        self.Vt_vec = None
        self.multiA = multiA

    def assign_device(self, device):
        self.Bscale = torch.diag(torch.tensor([
            1e-18 / self.p_dict['Cinh'],
            1e-18 / self.p_dict['Cexc'],
            0.0
        ], device=self.device))

    def A_mat(self, g11, g22, g13, g23, g33):
        gsum = g13 + g23 + g33
        A = torch.tensor([[-g11, 0, g11],
                          [0, -g22, g22],
                          [g13, g23, -gsum]], device=self.device)
        return A

    def generate_uniform_Adist(self, scale):
        if not self.p_dict:
            raise ValueError("Please first assign a device before generating Adist")
        A = torch.zeros((self.N, 3, 3), device=self.device)
        R_ref = self.p_dict['Rstore']
        C_ref = self.p_dict['Cstore']
        scale_RC_dist = torch.sqrt(torch.rand(self.N, device=self.device) * (scale ** 2 - 1.0) + 1.0)
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
        scale_RC_dist = torch.sqrt(torch.distributions.Exponential(scale=mean).sample((self.N,)))
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
        scale_RC_dist = torch.sqrt(torch.poisson(torch.full((self.N,), mean, device=self.device)).float())
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
        Rstore_dist = torch.clamp(torch.normal(mean=p_ref, std=noise * p_ref, size=(self.N,), device=self.device),
                                  min=p_ref * 0.01)
        for k in range(self.N):
            A[k] = self.calc_A(Rstore_dist[k])
        self.Adist = A

    def specify_Vt(self, Vts):
        self.Vt_vec = torch.tensor(Vts, device=self.device)

    def get_dV(self, t):
        if not self.multiA:
            self.dV = torch.matmul(self.A, self.V) + torch.matmul(self.Bscale, self.B)
        else:
            self.dV = torch.einsum('ijk,kj->ij', self.Adist, self.V) + torch.matmul(self.Bscale, self.B)
        return self.dV

    def update_V(self, dt):
        self.V += dt * self.dV
        overshoots = ((self.V < -self.Vthres) & (self.V > self.Vthres)).sum().item()
        self.V = torch.clamp(self.V, -self.Vthres, self.Vthres)
        return overshoots

    def eta_ABC(self, I):
        return self.ABC(I * 1e-3, self.p_dict['AB'], self.p_dict['CB'])

    def ABC(self, I, AB, CB):
        eta = I / (AB + I + CB * I ** 2)
        return eta

    def Id_sub(self, Vg, Vt, mask):
        return self.p_dict['I_Vt'] * torch.exp((Vg - Vt[mask]) / self.p_dict['m'] / self.kT)

    def Id_sat(self, Vg, Vt, mask):
        return self.p_dict['I_Vt'] + self.linslope * (Vg - Vt[~mask])

    def Id_sub_0(self, Vg, Vt):
        return self.p_dict['I_Vt'] * torch.exp((Vg - Vt) / self.p_dict['m'] / self.kT)

    def Id_sat_0(self, Vg, Vt):
        return self.p_dict['I_Vt'] + self.linslope * (Vg - Vt)

    def transistorIV(self, Vg, Vt_vec=None):
        if Vt_vec is None:
            Vt = self.p_dict['Vt']
            return torch.where(Vg < Vt, self.Id_sub_0(Vg, Vt), self.Id_sat_0(Vg, Vt))
        else:
            Vt = Vt_vec
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
    def __init__(self, N, path_to_file, device):
        super().__init__(N, layer_type='input', path_to_file=path_to_file, device=device)
        self.C = torch.zeros(self.N, device=device)
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
                    self.C = self.input_func_handles['v'](t, *self.input_func_args['v'])
                else:
                    self.C = self.input_func_handles['v'](t)
            except:
                pass
            self.C = self.C.flatten()
        else:
            for key, func in self.input_func_handles.items():
                try:
                    if self.input_func_args[key] is not None:
                        self.C[key] = func(t, *self.input_func_args[key])
                    else:
                        self.C[key] = func(t)
                except:
                    pass
        return self.C


class OutputLayer(Layer):
    def __init__(self, N, teacher_delay=None, nsave=800, path_to_file='', device=None):
        super().__init__(N, layer_type='output', path_to_file=path_to_file, device=device)
        self.C = torch.zeros(self.N, device=device)
        self.B = torch.zeros_like(self.C)
        self.I = torch.zeros(self.N, device=device)
        self.output_func_handles = {}
        self.output_func_args = {}
        self.teacher_delay = teacher_delay
        self.nsave = nsave
        self.teacher_memory = torch.zeros((nsave, self.B.numel() + 1), device=device)

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
                self.I = self.output_func_handles(t, *self.output_func_args)
            else:
                self.I = self.output_func_handles(t)
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
            idx_t = (self.teacher_memory[:, 0] <= t_find).nonzero(as_tuple=True)[0].max().item()
            if t_find > t0 and idx_t < self.nsave -1:
                f = self.teacher_memory[idx_t:idx_t+2]
                teacher_signal = self.lin_intp(t_find, f)
            else:
                teacher_signal = self.teacher_memory[idx_t]
            B_for_update = teacher_signal[1:].reshape(self.N)
        else:
            B_for_update = self.B.clone()
        self.C = B_for_update.clone()

    def update_C(self, t):
        self.C = self.get_output_current(t)

    def reset(self):
        self.reset_B()
        self.reset_teacher_memory(self.nsave)

def connect_layers(from_layer_name, to_layer_name, layers, channel_name, device, connect_all=False, weight_scale=0.1):
    """
    Connects two layers by creating a weight matrix.

    Parameters
    ----------
    from_layer_name : str
        Name of the source layer.
    to_layer_name : str
        Name of the target layer.
    layers : dict
        Dictionary containing all layer instances.
    channel_name : str
        Identifier for the connection channel.
    device : torch.device
        The device on which to allocate the weight tensor.
    connect_all : bool, optional
        If True, connects all neurons from the source to the target layer. Default is False.
    weight_scale : float, optional
        Scaling factor for weight initialization. Default is 0.1.

    Returns
    -------
    Weights
        An instance of the Weights class holding the weight matrix and related methods.
    """

    class Weights:
        def __init__(self, from_layer_name, to_layer_name, channel, device, connect_all=False, weight_scale=0.1):
            """
            Constructor for the Weights class.

            Parameters
            ----------
            from_layer_name : str
                Name of the source layer.
            to_layer_name : str
                Name of the target layer.
            channel : str
                Identifier for the connection channel.
            device : torch.device
                The device on which to allocate the weight tensor.
            connect_all : bool, optional
                If True, connects all neurons. Default is False.
            weight_scale : float, optional
                Scaling factor for weight initialization. Default is 0.1.
            """
            self.from_layer = from_layer_name  # Store as string key
            self.to_layer = to_layer_name      # Store as string key
            self.channel = channel

            # Retrieve layer instances using keys
            from_layer = layers[from_layer_name]
            to_layer = layers[to_layer_name]

            # Initialize the weight matrix with zeros
            # Shape: (number of neurons in target layer, number of neurons in source layer)
            self.W = torch.zeros((to_layer.N, from_layer.N), device=device)

            if connect_all:
                # Fully connect all neurons with a default weight
                self.W[:, :] = weight_scale  # You can adjust the initialization as needed

        def connect_nodes(self, from_node, to_node, weight=1.0):
            """
            Connects a specific node from the source layer to the target layer with a given weight.

            Parameters
            ----------
            from_node : int
                Index of the neuron in the source layer.
            to_node : int
                Index of the neuron in the target layer.
            weight : float, optional
                Weight value for the connection. Default is 1.0.
            """
            from_layer = layers[self.from_layer]
            to_layer = layers[self.to_layer]

            if from_node < 0 or from_node >= from_layer.N:
                raise IndexError(f"from_node index {from_node} out of range for layer '{self.from_layer}' with {from_layer.N} neurons.")
            if to_node < 0 or to_node >= to_layer.N:
                raise IndexError(f"to_node index {to_node} out of range for layer '{self.to_layer}' with {to_layer.N} neurons.")

            self.W[to_node, from_node] = weight

        def get_edges(self):
            """
            Retrieves all active connections (edges) in the weight matrix.

            Returns
            -------
            dict
                A dictionary with the channel name as the key and a list of tuples (from_node, to_node, weight) as values.
            """
            edges = {}
            edge_list = []

            # Iterate over the weight matrix to find non-zero connections
            to_nodes, from_nodes = torch.nonzero(self.W, as_tuple=True)
            for from_node, to_node in zip(from_nodes, to_nodes):
                weight = self.W[to_node, from_node].item()
                edge_list.append((from_node.item(), to_node.item(), weight))

            edges[self.channel] = edge_list
            return edges

        def print_W(self, *args):
            """
            Prints the weight matrix with the channel name.

            Parameters
            ----------
            *args : tuple
                Additional arguments (unused).
            """
            with torch.no_grad():
                print(f"{self.channel} Weights:")
                print(self.W)

        def set_W(self, W):
            """
            Manually sets the weight matrix.

            Parameters
            ----------
            W : torch.Tensor
                A tensor with the same shape as the current weight matrix.
            """
            if not isinstance(W, torch.Tensor):
                raise TypeError("W must be a torch.Tensor.")
            if W.shape != self.W.shape:
                raise ValueError(f"Shape mismatch: Provided W has shape {W.shape}, expected {self.W.shape}.")
            self.W = W.to(self.W.device)

        def scale_W(self, scale):
            """
            Scales the weight matrix by a given factor.

            Parameters
            ----------
            scale : float
                The scaling factor.
            """
            self.W = self.W * scale

        def ask_W(self, silent=False):
            """
            Returns the shape of the weight matrix.

            Parameters
            ----------
            silent : bool, optional
                If False, prints the shape. Default is False.

            Returns
            -------
            tuple
                The shape of the weight matrix.
            """
            if not silent:
                print(f"Weights '{self.channel}' shape: {self.W.shape}")
            return self.W.shape

    # Retrieve layer instances from the layers dictionary
    if from_layer_name not in layers:
        raise KeyError(f"Source layer '{from_layer_name}' not found in layers dictionary.")
    if to_layer_name not in layers:
        raise KeyError(f"Target layer '{to_layer_name}' not found in layers dictionary.")

    # Create and return a Weights instance
    return Weights(from_layer_name, to_layer_name, channel_name, device, connect_all, weight_scale)
