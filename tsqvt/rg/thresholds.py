import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ThresholdCorrections:
    """
    Threshold corrections for RG running.
    """
    mass_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.mass_thresholds is None:
            self.mass_thresholds = {
                't': 172.7,
                'H': 125.1,
                'Z': 91.1876,
                'W': 80.379,
                'b': 4.18,
                'tau': 1.777,
                'c': 1.27,
            }

    # --- compatibility attributes / API expected by tests ---
    @property
    def thresholds(self) -> Dict[str, float]:
        """Alias que los tests esperan (dict particle->mass)."""
        return self.mass_thresholds

    def add_threshold(self, mass: float, labels: Dict[str, int]):
        """Añade un umbral; labels suele ser {'t':1}."""
        if isinstance(labels, dict) and len(labels) == 1:
            name = next(iter(labels.keys()))
            self.mass_thresholds[name] = float(mass)
        else:
            key = f"thr_{len(self.mass_thresholds)+1}"
            self.mass_thresholds[key] = float(mass)

    @classmethod
    def standard_sm(cls):
        """Constructor con umbrales estándar del SM."""
        return cls()

    @classmethod
    def with_seesaw(cls, heavy_neutrino_masses: List[float] = None):
        """Devuelve ThresholdCorrections con umbrales de seesaw añadidos."""
        tc = cls()
        if heavy_neutrino_masses is None:
            heavy_neutrino_masses = [1e12, 1e13, 1e14]
        for i, m in enumerate(heavy_neutrino_masses, start=1):
            tc.mass_thresholds[f'nuR{i}'] = float(m)
        return tc

    def compute_correction(self, gauge_group: str, scale: float) -> float:
        """Alias: compute_correction(gauge, scale) -> total correction to 1/α."""
        return self.total_correction(mu_high=1e20, mu_low=scale, gauge_group=gauge_group)

    def apply(self, couplings: Dict[str, float], scale: float) -> Dict[str, float]:
        """Aplica correcciones de umbral a un dict de couplings (α-values)."""
        corrected = couplings.copy()
        for key in list(corrected.keys()):
            if key.lower().startswith('alpha1') or key.upper() == 'U1':
                gauge = 'U1'
            elif key.lower().startswith('alpha2') or key.upper() == 'SU2':
                gauge = 'SU2'
            elif key.lower().startswith('alpha3') or key.upper() == 'SU3':
                gauge = 'SU3'
            else:
                gauge = 'U1'
            delta_inv = self.total_correction(mu_high=1e20, mu_low=scale, gauge_group=gauge)
            alpha_old = corrected[key]
            if alpha_old <= 0:
                continue
            alpha_inv_new = 1.0 / alpha_old + delta_inv
            corrected[key] = 1.0 / alpha_inv_new
        return corrected
    # --------------------------------------------------------------------

    def correction_at(self, particle: str, gauge_group: str) -> float:
        T_R = self._dynkin_index(particle, gauge_group)
        return T_R / (6 * np.pi)

    def _dynkin_index(self, particle: str, group: str) -> float:
        indices = {
            ('t', 'SU3'): 0.5,
            ('t', 'SU2'): 0.5,
            ('t', 'U1'): 4/9,
            ('b', 'SU3'): 0.5,
            ('b', 'SU2'): 0.5,
            ('b', 'U1'): 1/9,
            ('H', 'SU2'): 0.5,
            ('H', 'U1'): 0.5,
            ('W', 'SU2'): 2,
            ('Z', 'U1'): 0,
        }
        return indices.get((particle, group), 0.0)

    def ordered_thresholds(self, mu_high: float, mu_low: float) -> List[Tuple[float, str]]:
        thresholds = []
        for particle, mass in self.mass_thresholds.items():
            if mu_low < mass < mu_high:
                thresholds.append((mass, particle))
        return sorted(thresholds, reverse=True)

    def total_correction(self, mu_high: float, mu_low: float, gauge_group: str) -> float:
        total = 0.0
        for mass, particle in self.ordered_thresholds(mu_high, mu_low):
            total += self.correction_at(particle, gauge_group)
        return total
