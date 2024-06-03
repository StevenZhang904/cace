###########################################################################################
# Atomic Data Class for handling molecules as graphs
# modified from MACE
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import torch
import numpy as np
from typing import Optional, Sequence

#import torch_geometric
from ..tools import torch_geometric
import torch.nn as nn
import torch.utils.data
from ..tools import voigt_to_matrix

from .neighborhood import get_neighborhood
from .utils import Configuration


def apply_pbc(masked_disp: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    # Ensure the cell is of shape [3, 3]
    cell = cell.view(3, 3)
    
    # Check for NaNs in the input
    if torch.isnan(masked_disp).any() or torch.isnan(cell).any():
        raise ValueError("Input tensors contain NaN values.")
    
    # Check if the cell matrix is singular
    if torch.det(cell) == 0:
        raise ValueError("Cell matrix is singular and cannot be inverted.")
    
    try:
        # Compute the inverse of the cell matrix
        cell_inv = torch.inverse(cell)
    except RuntimeError as e:
        raise ValueError(f"Failed to invert cell matrix: {e}")
    
    # If masked_disp is a 1-dimensional tensor, reshape it to [1, 3] for batch processing
    if masked_disp.ndimension() == 1:
        masked_disp = masked_disp.unsqueeze(0)
    
    # Map the displacements into fractional coordinates
    frac_disp = torch.matmul(masked_disp, cell_inv)
    
    # Apply the periodic boundary conditions in fractional coordinates
    frac_disp = frac_disp - torch.floor(frac_disp + 0.5)
    
    # Map the displacements back to Cartesian coordinates
    cart_disp = torch.matmul(frac_disp, cell)
    
    # Check for NaNs in the output
    if torch.isnan(cart_disp).any():
        raise ValueError("Output tensor contains NaN values.")
    
    # If the original input was 1-dimensional, return a 1-dimensional output
    if cart_disp.shape[0] == 1:
        cart_disp = cart_disp.squeeze(0)
    
    return cart_disp


def create_mask(atom_type, ratio):
    '''
    returns a mask that mask out hydrogen positions w.r.t their connected oxygen atoms,
    which are selected with given ratio 
    mask_value: 
    0: nothing
    1: in a molecule whose hydrogen has been masked
    2: is the hydrogen atom that has been masked
    '''
    num_molecules = len(atom_type) // 3
    # randomly select molecules according to ratio
    # print('num_molecules: ', num_molecules)
    num_selected = int(num_molecules * ratio)
    assert num_selected > 0, 'masking ratio is too small'
    selected_idx = np.random.choice(num_molecules, num_selected, replace=False)
    # create mask
    mask = torch.zeros_like(atom_type, dtype=torch.long)
    for i in selected_idx:
        # Here we assume that data is in repetition of (O,H,H) sequence
        mask[i*3] = 1
        if np.random.rand() < 0.5:  # break tie
            mask[i*3+1] = 1
            mask[i*3+2] = 2
        else:
            mask[i*3+2] = 1
            mask[i*3+1] = 2
    return mask

def get_rel_disp(mask, pos, cell_size):
        '''
        Params:
        mask: hydrogen mask from oxygen_positional_encoding
        pos: unmodified positions tensor
        cell_size

        Among all water molecules, using masks to identify which water mol has been selected
        by hydrogen mask, for that particular water mol, calculate the displacements between
        the unmasked atoms to the masked hydrogen atom. 

        returns: a list of displacements with shape (N, 3)
        '''
        masked_disp = torch.zeros_like(pos)
        for i in range(0, len(mask), 3):
            if mask[i] == 1:
                for k in range(3):
                    if mask[i+2] == 2: 
                        masked_disp[i][k] = pos[i+2][k] - pos[i][k]
                        masked_disp[i+1][k] = pos[i+2][k] - pos[i+1][k]
                    elif mask[i+1] == 2:
                        masked_disp[i][k] = pos[i+1][k] - pos[i][k]
                        masked_disp[i+2][k] = pos[i+1][k] - pos[i+2][k]
                    else:
                        raise ValueError('mask value error')

                # fist, make sure the displacement is within the box
                # masked_disp[i] = torch.remainder(masked_disp[i]  + cell_size[0]/2., cell_size[0]) - cell_size[0]/2.
                # masked_disp[i+1] = torch.remainder(masked_disp[i+1] + cell_size[1]/2., cell_size[1]) - cell_size[1]/2.
                # masked_disp[i+2] = torch.remainder(masked_disp[i+2] + cell_size[2]/2., cell_size[2]) - cell_size[2]/2.
                
                ### TODO: fix this, the above code will cause nan in disp
                masked_disp[i] = apply_pbc(masked_disp[i], cell_size)
                masked_disp[i+1] = apply_pbc(masked_disp[i+1], cell_size)
                masked_disp[i+2] = apply_pbc(masked_disp[i+2], cell_size)

                # normalize
                if torch.norm(masked_disp[i]) != 0:
                    masked_disp[i] = masked_disp[i] / torch.norm(masked_disp[i])
                if torch.norm(masked_disp[i+1]) != 0:
                    masked_disp[i+1] = masked_disp[i+1] / torch.norm(masked_disp[i+1])
                if torch.norm(masked_disp[i+2]) != 0:
                    masked_disp[i+2] = masked_disp[i+2] / torch.norm(masked_disp[i+2])

        masked_disp = masked_disp[mask != 2].view(-1,3) # remove masked hydrogen from it
        return masked_disp     

class AtomicData(torch_geometric.data.Data):
    atomic_numbers: torch.Tensor
    num_graphs: torch.Tensor
    num_nodes: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    n_atom_basis: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    charges: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor
    mask: torch.Tensor
    disp: torch.Tensor
    
    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges], always sender -> receiver
        atomic_numbers: torch.Tensor,  # [n_nodes]
        num_nodes: torch.Tensor, #[,]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        stress_weight: Optional[torch.Tensor],  # [,]
        virials_weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        stress: Optional[torch.Tensor],  # [1,3,3]
        virials: Optional[torch.Tensor],  # [1,3,3]
        dipole: Optional[torch.Tensor],  # [, 3]
        charges: Optional[torch.Tensor],  # [n_nodes, ]
        mask: Optional[torch.Tensor],
        disp: Optional[torch.Tensor],
    ):
        # Check shapes
        #assert num_nodes == atomic_numbers.shape[0]
        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        # assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert weight is None or len(weight.shape) == 0
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert stress_weight is None or len(stress_weight.shape) == 0
        assert virials_weight is None or len(virials_weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert dipole is None or dipole.shape[-1] == 3
        assert charges is None or charges.shape == (num_nodes,)
        assert disp is None or torch.isnan(disp).sum() == 0
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "atomic_numbers": atomic_numbers,
            "num_nodes": num_nodes,
            "weight": weight,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "virials_weight": virials_weight,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "dipole": dipole,
            "charges": charges,
            "mask": mask, 
            "disp": disp, 
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls, config: Configuration, 
        cutoff: float,
        pretrain_config: dict, 
    ) -> "AtomicData":

  
        atomic_numbers = torch.tensor(config.atomic_numbers, dtype=torch.long)

        cell = (
            torch.tensor(config.cell, dtype=torch.get_default_dtype())
            if config.cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else 1
        )

        energy_weight = (
            torch.tensor(config.energy_weight, dtype=torch.get_default_dtype())
            if config.energy_weight is not None
            else 1
        )

        forces_weight = (
            torch.tensor(config.forces_weight, dtype=torch.get_default_dtype())
            if config.forces_weight is not None
            else 1
        )

        stress_weight = (
            torch.tensor(config.stress_weight, dtype=torch.get_default_dtype())
            if config.stress_weight is not None
            else 1
        )

        virials_weight = (
            torch.tensor(config.virials_weight, dtype=torch.get_default_dtype())
            if config.virials_weight is not None
            else 1
        )

        forces = (
            torch.tensor(config.forces, dtype=torch.get_default_dtype())
            if config.forces is not None
            else None
        )
        energy = (
            torch.tensor(config.energy, dtype=torch.get_default_dtype())
            if config.energy is not None
            else None
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(config.stress, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.stress is not None
            else None
        )
        virials = (
            torch.tensor(config.virials, dtype=torch.get_default_dtype()).unsqueeze(0)
            if config.virials is not None
            else None
        )
        dipole = (
            torch.tensor(config.dipole, dtype=torch.get_default_dtype()).unsqueeze(0)
            if config.dipole is not None
            else None
        )
        charges = (
            torch.tensor(config.charges, dtype=torch.get_default_dtype())
            if config.charges is not None
            else None
        )

        
        if pretrain_config['status'] == True:
            hydrogen_mask = create_mask(atomic_numbers, ratio = pretrain_config["ratio"])
            disp = get_rel_disp(mask=hydrogen_mask, pos=torch.tensor(config.positions, dtype=torch.get_default_dtype()), cell_size=cell)
            edge_index, shifts, unit_shifts  = get_neighborhood(
                positions=config.positions[hydrogen_mask!=2], cutoff=cutoff, pbc=config.pbc, cell=config.cell
            )               
            
            return cls(
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                positions=torch.tensor(config.positions, dtype=torch.get_default_dtype())[hydrogen_mask!=2],
                shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
                unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
                cell=cell,
                atomic_numbers=atomic_numbers[hydrogen_mask!=2],
                num_nodes=atomic_numbers[hydrogen_mask!=2].shape[0],
                weight=weight,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                stress_weight=stress_weight,
                virials_weight=virials_weight,
                forces=forces[hydrogen_mask!=2],
                energy=energy,
                stress=stress,
                virials=virials,
                dipole=dipole,
                charges=charges[hydrogen_mask!=2],
                mask = hydrogen_mask[hydrogen_mask!=2], 
                disp = disp,
            ) 
        else:
            edge_index, shifts, unit_shifts  = get_neighborhood(
                positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
            ) 
            mask = torch.zeros_like(atomic_numbers, dtype=torch.long)
            disp = torch.zeros_like(torch.tensor(config.positions, dtype=torch.get_default_dtype()))
            return cls(
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
                shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
                unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
                cell=cell,
                atomic_numbers=atomic_numbers,
                num_nodes=atomic_numbers.shape[0],
                weight=weight,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                stress_weight=stress_weight,
                virials_weight=virials_weight,
                forces=forces,
                energy=energy,
                stress=stress,
                virials=virials,
                dipole=dipole,
                charges=charges,
                mask = mask, 
                disp = disp,
            )


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
