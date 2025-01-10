import torch

def evolve(t, layers, dVmax, dtmax, device='cpu'):
    max_dV = 0.0
    for layer in layers.values():
        if layer.layer_type == 'hidden':
            dV = layer.get_dV(t)
            max_dV_layer = torch.abs(dV).max().item()
            if max_dV_layer > max_dV:
                max_dV = max_dV_layer
    dt = dVmax / max_dV if max_dV > 0 else dtmax
    dt = min(dt, dtmax)
    return dt


def update(dt, t, layers, weights, overshoots=None, unity_coeff=1.0, t0=0., teacher_forcing=False, device='cpu'):
    # Update voltages and currents
    for key, layer in layers.items():
        if layer.layer_type == 'hidden':
            N = layer.update_V(dt)
            if overshoots is not None:
                overshoots[key] += N
            layer.update_I(dt)
        if layer.layer_type == 'input':
            layer.update_C(t)
        if layer.layer_type == 'output':
            if teacher_forcing:
                layer.update_C(t)
            else:
                layer.update_C_from_B(t, t0)
        layer.reset_B()

    # Update currents B according to weights
    for w in weights.values():
        from_idx = w.from_layer
        to_idx = w.to_layer
        if layers[from_idx].layer_type == 'hidden':
            C = layers[from_idx].P.clone() * unity_coeff
        else:
            C = layers[from_idx].C.clone()
        layers[to_idx].update_B(w, C)

    return
