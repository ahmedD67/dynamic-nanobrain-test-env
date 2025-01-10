# physics.py

import torch
import numpy as np

def square_pulse(t, tlims, amplitude):
    tmp = torch.zeros(1, device='cpu')
    for ttuple in tlims:
        tmp += ((t >= ttuple[0]) & (t < ttuple[1])).float()
    return tmp * amplitude

def constant(t, amplitude):
    return torch.full((1,), amplitude, device='cpu')

class Device:
    keys = ['Rinh','Rexc','RLED','Rstore','Cinh','Cexc','CLED','Cstore',
            'Cgate','Vt','m','I_Vt','vt','Lg','AB','CB']
    units = ['Ohm'] * 4 + ['F'] * 4 + ['F/cm'] + ['V', 'dim. less', 'nA',
                                                'cm/s', 'um', 'uA', '1/uA']
    kT = 0.02585

    def __init__(self, path_to_file, device):
        self.p_dict = {}
        self.p_units = {}
        self.device = device
        self.read_parameter_file(path_to_file, self.p_dict, self.p_units)
        self.linslope = self.p_dict['Cgate'] * self.p_dict['vt'] * 1e9  # nA/V
        self.gammas = self.calc_gammas()
        self.A = self.calc_A()

    def read_parameter_file(self, path_to_file, p_dict, p_units):
        if path_to_file.endswith('.pt'):
            params = torch.load(path_to_file)
        else:
            params = torch.tensor(np.loadtxt(path_to_file))
        for k, key in enumerate(self.keys):
            p_dict[key] = params[k].item() if isinstance(params, torch.Tensor) else params[k]
            p_units[key] = self.units[k]

    def set_parameter(self, key, value):
        self.p_dict[key] = value
        self.gammas = self.calc_gammas()
        self.A = self.calc_A()

    def print_parameter(self, key):
        print(f'The parameter {key}={self.p_dict[key]} {self.p_units[key]}')

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

    def transistorIV_example(self, Vstart=-0.5, Vend=1.0, steps=200):
        Vgate = torch.linspace(Vstart, Vend, steps, device=self.device)
        I = self.transistorIV(Vgate) * 1e-3  # Convert to uA
        data = torch.stack((Vgate, I))
        return {'Vgate': data[0].cpu().numpy(), 'Current': data[1].cpu().numpy()}

    def calc_Cmem(self, Cstore=None):
        return self.p_dict['Cstore'] + self.p_dict['Cgate'] * self.p_dict['Lg'] * 1e-4 \
            if Cstore is None else Cstore + self.p_dict['Cgate'] * self.p_dict['Lg'] * 1e-4

    def calc_gammas(self, Rstore=None, Cstore=None):
        Cmem = self.calc_Cmem(Cstore) if Cstore is not None else self.calc_Cmem()
        g11 = 1e-9 / self.p_dict['Cinh'] / self.p_dict['Rinh']  # ns^-1 # GHz
        g22 = 1e-9 / self.p_dict['Cexc'] / self.p_dict['Rexc']  # ns^-1 # GHz
        g13 = 1e-9 / Cmem / self.p_dict['Rinh']  # ns^-1 # GHz
        g23 = 1e-9 / Cmem / self.p_dict['Rexc']  # ns^-1 # GHz
        g33 = 1e-9 / Cmem / (Rstore if Rstore is not None else self.p_dict['Rstore'])
        gled = 1e-9 / self.p_dict['CLED'] / self.p_dict['RLED']  # ns^-1 # GHz
        return torch.tensor([g11, g22, g13, g23, g33, gled], device=self.device)

    def A_mat(self, g11, g22, g13, g23, g33):
        gsum = g13 + g23 + g33
        A = torch.tensor([[-g11, 0, g11],
                          [0, -g22, g22],
                          [g13, g23, -gsum]], device=self.device)
        return A

    def calc_A(self, Rstore=None, Cstore=None):
        if (Rstore is not None) and (Cstore is not None):
            new_gammas = self.calc_gammas(Rstore, Cstore)
            return self.A_mat(*new_gammas[:-1])
        elif Rstore is not None:
            new_gammas = self.calc_gammas(Rstore)
            return self.A_mat(*new_gammas[:-1])
        else:
            return self.A_mat(*self.gammas[:-1])

    def setup_gain(self, gammas):
        A = self.A_mat(*gammas[:-1])
        eigvals = torch.linalg.eigvals(A)
        v_add = torch.cat((eigvals, torch.tensor([-gammas[-1]], device=self.device, dtype=torch.cfloat)))
        return v_add

    def gain(self, s, eigvals, gammas, vsat=1e7, Vt_prime=0.):
        if Vt_prime > 0.:
            raise NotImplementedError("Vt_prime != 0 not supported")
        g11, g22, g13, g23, g33, gled = gammas
        Cexc = self.p_dict['Cexc']
        Cinh = self.p_dict['Cinh']
        Cgate = self.p_dict['Cgate']
        denom = torch.ones_like(s, dtype=torch.cfloat, device=self.device)
        for l in eigvals:
            denom *= (s - l)
        prefactor = gled * vsat * Cgate * 1e-9  # Adjust units
        G11 = g23 * (s + g22) / Cexc * prefactor / denom
        G12 = -g13 * (s + g11) / Cinh * prefactor / denom
        return G11, G12

    def unity_coupling_coefficient(self, eta_handle, s=1e-3):
        sample_s = torch.tensor([s], device=self.device, dtype=torch.cfloat)
        gammas = self.gammas
        eigvals = self.setup_gain(gammas)
        G11, _ = self.gain(sample_s, eigvals, gammas)
        eta_max = eta_handle(torch.tensor([1.0], device=self.device)).max().item()
        return (eta_max * G11.real[0]).item() ** -1

    def inverse_gain_coefficient(self, eta_handle, Vthres):
        Rsum = self.p_dict['Rstore'] + self.p_dict['Rexc']
        max_Vg = Vthres * self.p_dict['Rstore'] / Rsum
        Iexc = Vthres / Rsum * 1e9  # nA
        Isd = self.transistorIV(max_Vg)
        Iout = eta_handle(Isd) * Isd
        return (Iexc / Iout).item(), Iexc.item()

    def ABC(self, I, AB, CB):
        eta = I / (AB + I + CB * I ** 2)
        return eta

    def eta_ABC(self, I):
        return self.ABC(I * 1e-3, self.p_dict['AB'], self.p_dict['CB'])

    def eta_unity(self, I):
        return torch.ones_like(I)

    def eta_example(self, handle):
        I = torch.logspace(-2, 2, steps=50, device=self.device)  # in uA
        eta = handle(I * 1e3)  # ABC expects nA
        if eta.numel() < I.numel():
            eta = eta * torch.ones_like(I)
        return {'Current (uA)': I.cpu().numpy(), 'eta, IQE': eta.cpu().numpy()}
