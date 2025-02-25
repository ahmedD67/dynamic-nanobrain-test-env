# "activations" will be contained here
# These aren't strictly activations in the same sense as ANNs
# but they act are non-linearities to process the affine transformations 
# so we go with that

import torch
@staticmethod
class Id_sub_0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Vg, Vt, I_Vt, m, kT):
        ctx.save_for_backward(Vg, Vt)
        return I_Vt * torch.exp((Vg-Vt)/m/kT)
    @staticmethod
    def backward(ctx, grad_output):
        pass

@staticmethod
class Id_sat_0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Vg, Vt, I_Vt, linslope):
        ctx.save_for_backward(Vg, Vt)
        return I_Vt + linslope * (Vg-Vt)
    @staticmethod
    def backward(ctx, grad_output):
        pass

@staticmethod
class Id_sub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Vg, Vt, I_Vt, m, kT, mask):
        ctx.save_for_backward(Vg, Vt)
        return I_Vt*torch.exp((Vg-Vt[mask])/m/kT)
    @staticmethod
    def backward(ctx, grad_output):
        pass

@staticmethod
class Id_sat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Vg, Vt, I_Vt, linslope, mask):
        ctx.save_for_backward(Vg, Vt)
        return I_Vt + linslope *(Vg-Vt[mask==False])
    @staticmethod
    def backward(ctx, grad_output):
        pass

@staticmethod
class TransistorIV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I_Vt, kT, m, mask, Vt, linslope, Vg,):
        ctx.save_for_backward(Vg, Vt)
        if t == 0:
            output = torch.where(
                Vg < Vt, Id_sub_0(Vg, Vt, I_Vt, m, kT), Id_sat_0(Vg, Vt, I_Vt, linslope)
            )
        else:
            output = torch.where(
                Vg < Vt, Id_sub(Vg, Vt, I_Vt, m, kT, mask), Id_sat(Vg, Vt, I_Vt, linslope, mask)
            )

        return output
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        pass

@staticmethod
class eta_ABC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, AB, CB):
        ctx.save_for_backward(I)
        eta = (I*1e-3)/(AB + I + CB*I**2)
        return eta

    @staticmethod
    def backward(ctx, grad_output):
        pass
