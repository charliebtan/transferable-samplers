import torch

from src.data.components.encoding import AA_CODE_CONVERSION, AA_TYPE_ENCODING_DICT, ATOM_TYPE_ENCODING_DICT

# common backbone order
BACKBONE_ORDER = ["N", "CA", "C"]  # O technically not part of backbone

# Get backbone atom codes expected for this residue
BACKBONE_CODES_SET = set(ATOM_TYPE_ENCODING_DICT[n] for n in BACKBONE_ORDER)

# side-chain heavy-atom names per amino acid
SIDECHAIN_MAP = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD", "ND2"],
    "ASP": ["CB", "CG", "OD"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE"],
    "GLY": [],  # no side chain
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "NE1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}


class Permutation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Overload me")


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False, **kwargs) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False, **kwargs) -> torch.Tensor:
        return x.flip(dims=[dim])


class PermutationBackBone(Permutation):
    def __init__(self):
        super().__init__()
        # code→3-letter AA name
        self.rev_aa = {v: k for k, v in AA_TYPE_ENCODING_DICT.items()}
        # cache: aa_key_tuple → perm_list
        self._cache: dict[tuple[int, ...], list[int]] = {}

    def forward(
        self,
        x: torch.Tensor,  # (B, L, ...)
        atom_type: torch.Tensor,  # (B, L)
        aa_type: torch.Tensor,  # (B, L)
        dim: int = 1,
        inverse: bool = False,
        **kwargs,
    ):
        x = self.permute(x, atom_type, aa_type, dim, inverse)
        return x

    def convert_aa(self, aa_type):
        convert_to_aa = lambda _x: AA_CODE_CONVERSION[self.rev_aa[_x]]

        return "".join(list(map(convert_to_aa, aa_type.tolist())))

    def make_key(self, aa_type, atom_type):
        # use N code to find the unique aa type in expanded sequence
        N_code = ATOM_TYPE_ENCODING_DICT["N"]

        # this unique describes a sequence including N-terminal and C-terminal AA variants
        # (due to varying length) but without any padding tokens
        maybe_padded_keys = [aa_type[i, atom_type[i] == N_code].detach().cpu() for i in range(len(aa_type))]

        keys = [self.convert_aa(k[k != 0]) for k in maybe_padded_keys]
        return keys

    def permute(
        self,
        x: torch.Tensor,  # (B, L, ...)
        atom_type: torch.Tensor,  # (B, L)
        aa_type: torch.Tensor,  # (B, L)
        dim: int = 1,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, None]:
        B, L = aa_type.shape[0], aa_type.shape[1]
        device = x.device
        N_code = ATOM_TYPE_ENCODING_DICT["N"]

        keys = self.make_key(aa_type, atom_type)

        # compute & cache any missing permutations
        for idx, key in enumerate(keys):
            # if in cache already skip
            if key in self._cache:
                continue

            # get the atom type tensor for this sequence
            atom_type_row = atom_type[idx]  # shape (L,)

            # find residue boundaries by locating every "N"
            segment_boundaries = (atom_type_row == N_code).nonzero(as_tuple=True)[0].tolist()

            # add a "start" at the end of the sequence so segments can be computed
            segment_boundaries.append(L)

            perm_backbone_list = []  # list for backbone atoms
            perm_other_list = []  # list for sidechain atoms
            for s, e in zip(segment_boundaries[:-1], segment_boundaries[1:]):
                # indexes corresopnding to this amino acid
                segment = torch.arange(s, e).to(device)

                # Get atom types for this segment
                segment_atom_types = atom_type_row[s:e]

                # Mask of atoms matching backbone codes
                backbone_mask = torch.tensor(
                    [code in BACKBONE_CODES_SET for code in segment_atom_types.tolist()],
                    device=segment_atom_types.device,
                )

                # Add matched heavy atoms in order
                perm_backbone_list.append(segment[backbone_mask])

                # Add unmatched atoms in order
                perm_other_list.append(segment[~backbone_mask])

            # concatenate all the backbones first, followed by all other atoms
            perm = torch.cat([*perm_backbone_list, *perm_other_list])

            self._cache[key] = perm.detach().cpu()

        # assemble the full (B, L) perm index
        perm_idx = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, key in enumerate(keys):
            perm = self._cache[key].clone().detach().to(device)
            # cached perm is padded. If aa_type is unpadded, truncate
            # to the same length
            perm_idx[i, :L] = perm[:L]

        # optionally invert
        if inverse:
            inv = torch.zeros_like(perm_idx)
            for b in range(B):
                p = perm_idx[b]
                inv[b, p] = torch.arange(L, device=device)
            perm_idx = inv

        # apply via advanced indexing
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        x_out = x[batch_idx, perm_idx]

        return x_out


class PermutationBackBoneFlip(PermutationBackBone):
    def forward(
        self,
        x: torch.Tensor,
        atom_type: torch.Tensor,
        aa_type: torch.Tensor,
        dim: int = 1,
        inverse: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if inverse:
            x = x.flip(dims=[dim])

        x = self.permute(x, atom_type, aa_type, dim=dim, inverse=inverse)
        if not inverse:
            x = x.flip(dims=[dim])

        return x


class PermutationRandom(Permutation):
    def forward(
        self,
        x: torch.Tensor,  # (B, L) or (B, L, ...)
        mask: torch.Tensor | None = None,  # (B, L) 1=real, 0=pad
        dim: int = 1,
        inverse: bool = False,
        perm: torch.Tensor | None = None,  # (B, L) or None
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = x.shape[0], x.shape[dim]
        device = x.device

        # INVERSE: undo a previous perm
        if inverse:
            if perm is None:
                raise ValueError("Must pass `perm` when inverse=True")
            # build inv[batch,i] such that inv[batch,perm[batch,i]] = i
            inv = torch.zeros_like(perm)
            for b in range(B):
                inv[b, perm[b]] = torch.arange(L, device=device)
            # apply inverse shuffle
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
            x_inv = x[batch_idx, inv]
            return x_inv, perm

        if perm is None:
            if mask is None:
                # shuffle all positions
                perm = torch.randperm(L, device=device).unsqueeze(0).repeat(B, 1)
            else:
                # shuffle only the real atoms, then append the pad indices
                perm = []
                for b in range(B):
                    real = torch.where(mask[b] == 1)[0]
                    pad = torch.where(mask[b] == 0)[0]
                    real_shuf = real[torch.randperm(real.numel(), device=device)]
                    perm.append(torch.cat([real_shuf, pad], dim=0))
                perm = torch.stack(perm, dim=0)

        # apply forward shuffle
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        x_perm = x[batch_idx, perm]
        return x_perm, perm


def shift_pos(x: torch.Tensor, mask: torch.Tensor):
    B, L, *rest = x.shape
    device = x.device

    # make a position matrix for each batch
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)

    # we expect mask to be the original mask and not flipped.
    # build a key that keeps original order for True (pos),
    # but pushes False to the beginning.
    key = pos.masked_fill(~mask.bool(), -1)  # (B, L)

    # argsort so that all True positions come first, then False
    order = key.argsort(dim=1)  # (B, L)

    # expand order to index into the last dims of x
    if rest:
        idx = order.unsqueeze(-1).repeat(1, 1, *x.shape[2:])
    else:
        idx = order

    # we expect mask to be the original and not flipped
    flipped_mask = mask[..., None].flip(dims=[1])
    return x.gather(1, idx) * flipped_mask
