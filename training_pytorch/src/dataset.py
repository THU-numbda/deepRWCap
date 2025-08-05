from __future__ import annotations
import os, math, numpy as np, torch
from torch.utils.data import Dataset


class GreenDataset(Dataset):
    _VALID_TYPES = ("greens_function", "gradient")

    def __init__(self,
                 files: list[str],
                 dataset_type: str = "greens_function",
                 N: int = 16,
                 dtype: np.dtype = np.float32,
                 n_structures: int | None = 0  # <── NEW ARG
                 ):

        if dataset_type not in self._VALID_TYPES:
            raise ValueError(f"dataset_type must be one of {self._VALID_TYPES}")
        if not files:
            raise ValueError("No files supplied")

        self.files        = sorted(files)
        self.dtype        = dtype
        self.N            = N
        self.kind         = dataset_type
        self.n_structures = n_structures  # may be None → fallback to old logic

        diel_all, gf_all = [], []
        for p in self.files:
            diel, gf = self._load_single_file(p)
            diel_all.append(diel)
            gf_all.append(gf)

        self.dielectric = np.concatenate(diel_all).astype(np.float32)
        self.gf_raw     = np.concatenate(gf_all  ).astype(np.float32)

        if self.kind == "greens_function":
            self._prep_greens_function()
        else:
            self._prep_gradient()

    # ─────────────────────────── torch dataset API ────────────────────────────
    def __len__(self):  return self.dielectric.shape[0]          # noqa: DUNDER

    def __getitem__(self, idx):                                   # noqa: DUNDER
        x = torch.from_numpy(self.dielectric[idx])
        if self.kind == "greens_function":
            y = {"greens_tensor": torch.from_numpy(self.greens_tensor[idx]),
                 "face_distribution": torch.from_numpy(self.face_distribution[idx])}
        else:
            y = {"normalised_gradient": torch.from_numpy(self.normalised_grad[idx]),
                 "face_grad_weights":   torch.from_numpy(self.face_grad_weights[idx])}
        return x, y

    # ─────────────────────────── binary-file handling ─────────────────────────
    def _load_single_file(self, path: str):
        vec = np.fromfile(path, dtype=self.dtype)
        n, block_w = map(int, vec[:2])
        if n != self.N:
            raise ValueError(f"{path}: header N={n}, expected {self.N}")

        blockn   = n // block_w
        diel_len = blockn ** 3
        gf_len   = 6 * n * n
        body     = vec[2:]
        total    = body.size

        # —— NEW: fixed struct_len from constructor ————————————————
        if self.n_structures is not None:
            struct_len = 7 * self.n_structures
            sample_len = diel_len + struct_len + gf_len
            if total % sample_len:
                raise ValueError(f"{path}: file length not compatible with "
                                 f"{self.n_structures} structures per sample")
            num_samples = total // sample_len
        else:
            # Fallback: attempt to infer (legacy behaviour)
            num_samples, struct_len = self._factor_file(total, diel_len, gf_len)
            sample_len = diel_len + struct_len + gf_len
        # ---------------------------------------------------------------------

        data = body.reshape(num_samples, sample_len)
        dielectrics = data[:, :diel_len].reshape(num_samples, 1, 1, blockn, blockn, blockn)
        gf_block    = data[:, -gf_len:].reshape(num_samples, 6, 1, n, n)
        return dielectrics, gf_block

    @staticmethod
    def _factor_file(total_elems, diel_len, gf_len):
        """Legacy inference (kept for backward compatibility)."""
        min_len = diel_len + gf_len
        for samples in GreenDataset._divisors(total_elems):
            s_len = total_elems // samples
            struct_len = s_len - min_len
            if struct_len >= 0 and struct_len % 7 == 0:
                return samples, struct_len
        raise ValueError("Could not determine per-sample structure length")

    @staticmethod
    def _divisors(n: int):
        small, large = [], []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                small.append(i)
                if i * i != n: large.append(n // i)
        yield from small; yield from reversed(large)

    # ───────────────────────────── preprocessing ─────────────────────────────
    def _prep_greens_function(self):
        face_sum = self.gf_raw.sum(axis=(2,3,4))
        grid_sum = self.gf_raw.sum(axis=(3,4), keepdims=True);  grid_sum[grid_sum==0]=1
        self.greens_tensor     = (self.gf_raw / grid_sum).astype(np.float32)
        self.face_distribution = face_sum.astype(np.float32)

    def _prep_gradient(self):
        signs = np.sign(self.gf_raw); mags = np.abs(self.gf_raw)
        face_tot = mags.sum(axis=(3,4), keepdims=True)
        self.normalised_grad   = (mags / face_tot * signs).astype(np.float32)
        weights                = face_tot.reshape(face_tot.shape[0], 6)
        face_grad = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
        self.face_grad_weights = np.concatenate((face_grad, weights.sum(axis=1, keepdims=True)), axis=1).astype(np.float32)
