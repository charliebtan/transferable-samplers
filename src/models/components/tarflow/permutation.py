import torch
from itertools import groupby

from src.data.components.encodings import AA_TYPE_ENCODING_DICT, ATOM_TYPE_ENCODING_DICT, AA_CODE_CONVERSION

# common backbone order
BACKBONE_ORDER = ["N", "CA", "C", "O"]

# side-chain heavy-atom names per amino acid
SIDECHAIN_MAP = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],             # no side chain
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
        if isinstance(dim, int):
            dim = [dim]
        return x.flip(dims=dim)


class PermutationBackBone(Permutation):
    def __init__(self):
        super().__init__()
        # invert the AA dict
        self.rev_aa = {v:k for k,v in AA_TYPE_ENCODING_DICT.items()}
        self.permutation_cache = {}

    def forward(
            self, 
            x: torch.Tensor, 
            atom_type_seq: torch.Tensor, 
            aa_type_seq: torch.Tensor, 
            dim: int = 1, 
            inverse: bool = False, 
            **kwargs
    ) -> torch.Tensor:
        return self.permute(x, atom_type_seq, aa_type_seq, dim=dim)
    
    def permute(
        self,
        x: torch.Tensor,             # (B, L, ...)
        atom_type_seq: torch.Tensor, # (B, L)
        aa_type_seq: torch.Tensor,   # (B, L)
        dim: int = 1,
    ) -> torch.Tensor:
        B, L = x.shape[0], x.shape[dim]
        device = x.device

        # pre‐allocate per‐batch index map
        perm_idx = torch.zeros((B, L), dtype=torch.long, device=device)
        N_code = ATOM_TYPE_ENCODING_DICT["N"]

        for i in range(B):
            types = atom_type_seq[i]   # (L,)
            aas   = aa_type_seq[i]     # (L,)
            # aa_key = tuple(int(v) for v in aas.tolist())
            aa_key = "".join(
                map(lambda a: AA_CODE_CONVERSION[self.rev_aa[int(a)]], tuple(k for k, _ in groupby(aas)))
            )

            if aa_key in self.permutation_cache:
                perm_list = self.permutation_cache[aa_key]
            else:            
                # find the residue start positions by locating every “N”
                starts = (types == N_code).nonzero(as_tuple=True)[0].tolist()
                starts.sort()
                # build list of index‐ranges for each residue
                segments = []
                for j, s in enumerate(starts):
                    e = starts[j+1] if j+1 < len(starts) else L
                    segments.append(list(range(s, e)))

                perm_list: list[int] = []
                for seg in segments:
                    # residue name from the first atom in segment
                    aa_code = int(aas[seg[0]].item())
                    aa_name = self.rev_aa[aa_code]

                    # desired heavy‐atom codes
                    heavy_names = BACKBONE_ORDER + SIDECHAIN_MAP.get(aa_name, [])
                    heavy_codes = [ATOM_TYPE_ENCODING_DICT[n] for n in heavy_names]

                    # collect heavy atoms in order, then everything else
                    heavy_pos = []
                    for c in heavy_codes:
                        for j in seg:
                            if types[j].item() == c:
                                heavy_pos.append(j)
                                break
                    rest = [j for j in seg if j not in heavy_pos]

                    perm_list += heavy_pos + rest

                self.permutation_cache[aa_key] = perm_list

            perm_idx[i] = torch.tensor(perm_list, device=device)

        # now gather along dim=1
        # build a batch‐wise index for advanced indexing
        idx_batch = torch.arange(B, device=device).unsqueeze(1).expand(B, L)

        return x[idx_batch, perm_idx]


class PermutationBackBoneFlip(PermutationBackBone):
    def forward(
            self, 
            x: torch.Tensor, 
            atom_type_seq: torch.Tensor, 
            aa_type_seq: torch.Tensor, 
            dim: int = 1, 
            inverse: bool = False, 
            **kwargs
    ) -> torch.Tensor:
        if isinstance(dim, int):
            dim = [dim]

        x = self.permute(x, atom_type_seq, aa_type_seq, dim=dim)
        return x.flip(dims=dim)
