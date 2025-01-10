import torch

class Logger:
    def __init__(self, layers, feedback=False, device='cpu'):
        self.list_data = []
        self.device = device
        self.feedback = feedback
        self.column_labels = self.column_names(layers)

    def column_names(self, layers):
        names = ['Time']
        for idx in layers.keys():
            node_list = layers[idx].get_names(idx)
            if layers[idx].layer_type == 'input':
                for node in node_list:
                    names.append(f"{node}-Pout")
            elif layers[idx].layer_type == 'hidden':
                # Voltages
                for node in node_list:
                    names.append(f"{node}-Vinh")
                    names.append(f"{node}-Vexc")
                    names.append(f"{node}-Vgate")
                # Input currents
                for node in node_list:
                    names.append(f"{node}-Iinh")
                    names.append(f"{node}-Iexc")
                # Output currents
                for node in node_list:
                    names.append(f"{node}-Iout")
                # ISD (source drain) done separately to get correct ordering
                for node in node_list:
                    names.append(f"{node}-ISD")
                for node in node_list:
                    names.append(f"{node}-Pout")
            elif layers[idx].layer_type == 'output':
                # Currents
                for node in node_list:
                    names.append(f"{node}-Pout")
                if self.feedback:
                    # Add some extra columns for the signal fed back in (C)
                    for node in node_list:
                        names.append(f"{node}-Pinp")
            else:
                raise RuntimeError('Unexpected layer_type in logger.column_names')
        return names

    def add_tstep(self, t, layers, unity_coeff=1.0):
        """
        Adds a timestep to the log by extracting relevant data from each layer.

        Parameters
        ----------
        t : float
            Current time step.
        layers : dict
            Dictionary of layer objects.
        unity_coeff : float, optional
            Scaling factor for output currents. Default is 1.0.
        """
        # Initialize the row with the current time
        row = [t]

        for idx in layers.keys():
            layer = layers[idx]

            if layer.layer_type == 'input':
                # Assuming layer.C is a 1D tensor
                curr = layer.C.flatten().tolist()
                row += curr

            elif layer.layer_type == 'hidden':
                # Assuming layer.V is a 2D tensor of shape (NV, N)
                # To emulate 'F' (column-major) flattening, transpose before flattening
                volt = layer.V.t().reshape(-1).tolist()
                row += volt

                # Assuming layer.B[:2] is a 2D tensor of shape (2, N)
                # Transpose to emulate 'F' order flattening
                curr = layer.B[:2].t().reshape(-1).tolist()
                row += curr

                # Assuming layer.I is a 1D tensor
                curr = (layer.I * unity_coeff).flatten().tolist()
                row += curr

                # Assuming layer.ISD is a 1D tensor
                curr = layer.ISD.flatten().tolist()
                row += curr

                # Assuming layer.P is a 1D tensor
                pout = (layer.P * unity_coeff).flatten().tolist()
                row += pout

            elif layer.layer_type == 'output':
                # Assuming layer.B is a 1D tensor
                curr = layer.B.flatten().tolist()
                row += curr

                if self.feedback:
                    # Assuming layer.C is a 1D tensor
                    curr = layer.C.flatten().tolist()
                    row += curr
            else:
                raise RuntimeError('Unexpected layer_type in logger.add_tstep')

        # Append the row to the logged data
        self.list_data.append(row)

    def get_timelog(self):
        # Convert to PyTorch tensor
        return torch.tensor(self.list_data, device=self.device)
