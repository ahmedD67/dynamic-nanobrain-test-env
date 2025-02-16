import torch
from torch import nn

class Layer(nn.Module):
    keys = ['Rinh','Rexc','RLED','Rstore','Cinh','Cexc','CLED','Cstore','Cgate','Vt','m','I_Vt','vt','Lg','AB','CB']
    kT = 0.02585
    def __init__(self, N, layer_type):
        self.device_params = torch.zeros(len(self.keys))
        self._devidx = {
            k: idx
        for idx, k in enumerate(self.keys)
        }

    def calc_gammas(self, Rstore=None, Cstore=None):
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is not None:
            Cmem = self.calc_Cmem(Cstore)
        else :
            Cmem = self.calc_Cmem()
        # System frequencies
        g11 = 1e-9/self.device_params[self._devidx['Cinh']]/self.device_params[self._devidx['Rinh']] # ns^-1 # GHz
        g22 = 1e-9/self.device_params[self._devidx['Cexc']]/self.device_params[self._devidx['Rexc']] # ns^-1 # GHz
        g13 = 1e-9/Cmem/self.device_params[self._devidx['Rinh']] # ns^-1 # GHz
        g23 = 1e-9/Cmem/self.device_params[self._devidx['Rexc']] # ns^-1 # GHz
        if Rstore is not None :
            g33 = 1e-9/Cmem/Rstore
        else :
            g33 = 1e-9/Cmem/self.device_params[self._devidx['Rstore']] # ns^-1 # GHz
        gled = 1e-9/self.device_params[self._devidx['CLED']]/self.device_params[self._devidx['RLED']] # ns^-1 # GHz

        return torch.tensor([g11,g22,g13,g23,g33,gled])
    
    def calc_Cmem(self,Cstore=None) :
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is None :
            Cmem = self.device_params[self._devidx['Cstore']] + self.device_params[self._devidx['Cgate']]*self.device_params[self._devidx['Lg']]*1e-4 
        else :
            Cmem = Cstore + self.device_params[self._devidx['Cgate']]*self.device_params[self._devidx['Lg']]*1e-4  
        return Cmem

    def forward(self, X):
        pass

class HiddenLayer(Layer):
    def __init__(self, N, output_channel, inhibition_channel, excitation_channel, 
                 device=None, Vthres=1.2, multiA=False, NV = 3):
        Layer.__init__(N, layer_type='hidden')
        self.overshoots = 0
        # Set up internal variables
        self.V = torch.zeros((NV,self.N))
        self.B = torch.zeros_like(self.V)
        self.dV= torch.zeros_like(self.V)
        # I is the current through the LED
        self.I = torch.zeros(self.N)
        # Power is the outputted light, in units of current
        self.P = torch.zeros(self.N)
        self.ISD = torch.zeros_like(self.I)
        self.Vthres = Vthres
        # Sequence of transistor threshold voltages, initialized to None
        self.Vt_vec = None 
        # Device object hold A, for example
        
        self.multiA=multiA
        self.Bscale=torch.diag([1e-18/self.device_params[self._devidx['Cinh']],
                             1e-18/self.device_params[self._devidx['Cexc']],
                             0.])
        
    def forward(self, dt):
        self.V += dt*self.dV
        # Count voltage overshoots
        overshoots = (self.V<-self.Vthres)*(self.V>self.Vthres)
        N = torch.sum(overshoots)
        # Voltage clipping (third try)
        self.V = torch.clip(self.V,-self.Vthres,self.Vthres)

        self.overshoots += N

        self.ISD = self.transistorIV(self.V[2],self.Vt_vec)
        self.I += dt*self.gammas[-1]*(self.ISD-self.I)
        # Convert current to power through efficiency function
        self.P = self.I*self.eta_ABC(self.I)
    