import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GUTMatching:
    """
    GUTMatching compatible con tests: acepta gut_scale/unified_coupling kwargs.
    """
    m_gut: float = 2e16
    g_gut: float = 0.72

    def __init__(self, **kwargs):
        # Acepta tanto 'gut_scale' como 'm_gut' para compatibilidad
        if 'gut_scale' in kwargs:
            self.m_gut = float(kwargs.pop('gut_scale'))
        else:
            self.m_gut = float(kwargs.pop('m_gut', 2e16))

        # Acepta tanto 'unified_coupling' como 'g_gut' para compatibilidad
        if 'unified_coupling' in kwargs:
            self.g_gut = float(kwargs.pop('unified_coupling'))
        else:
            self.g_gut = float(kwargs.pop('g_gut', 0.72))

    def sm_couplings_at_gut(self) -> Dict[str, float]:
        # Devuelve los acoplamientos de las tres interacciones en el GUT
        return {'g1': self.g_gut, 'g2': self.g_gut, 'g3': self.g_gut}

    def alpha_gut(self) -> float:
        # Convierte g_gut en la constante de acoplamiento α = g^2 / (4π)
        return self.g_gut**2 / (4 * np.pi)

    def proton_lifetime(self) -> float:
        # Estimación simple de la vida media del protón en años
        m_p = 0.938  # masa del protón en GeV
        alpha = self.alpha_gut()
        # fórmula dimensional (constante 1e-9 incluida según tu implementación)
        tau_seconds = (self.m_gut**4) / (alpha**2 * m_p**5) * 1e-9
        # convertir segundos a años
        return tau_seconds / (365.25 * 24 * 3600)

    def verify_unification(self, couplings_mz: Dict[str, float], tolerance: float = 0.05) -> Tuple[bool, float]:
        """
        Ejecuta el running de las α desde M_Z hasta m_gut y comprueba la unificación.
        Devuelve (unificado?, desviación máxima relativa).
        Mejora: filtra valores no finitos y evita división por cero en la media.
        """
        from tsqvt.rg.running import RGRunner
        runner = RGRunner(loops=2)
        M_Z = 91.1876
        alphas_gut = []
        # Recolecta α1, α2, α3 evaluadas en la escala GUT si están presentes
        for group, key in [(1, 'alpha1'), (2, 'alpha2'), (3, 'alpha3')]:
            if key in couplings_mz:
                alpha_at_gut = runner.run_alpha(couplings_mz[key], M_Z, self.m_gut, group)
                alphas_gut.append(alpha_at_gut)

        # Si faltan acoplamientos, no se puede comprobar la unificación
        if len(alphas_gut) < 3:
            return False, float('inf')

        # Convertir a array numpy y filtrar valores no finitos (NaN, inf)
        arr = np.asarray(alphas_gut, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            # Si todos son no finitos, devolvemos fallo con desviación infinita
            return False, float('inf')

        mean_alpha = float(np.mean(arr))

        # Evitar división por cero: si la media es cero, usar desviación absoluta en lugar de relativa
        if mean_alpha == 0.0:
            max_dev = float(np.max(np.abs(arr - mean_alpha)))
        else:
            # cálculo de la desviación relativa máxima
            max_dev = float(np.max(np.abs(arr - mean_alpha) / mean_alpha))

        return (max_dev < tolerance), max_dev

    # Capa de compatibilidad esperada por los tests
    def check_unification(self, couplings: Dict[str, float], tolerance: float = 0.05) -> Dict[str, float]:
        ok, dev = self.verify_unification(couplings, tolerance=tolerance)
        return {'unified': bool(ok), 'max_deviation': float(dev)}

    def proton_decay_bound(self) -> float:
        # Simple envoltura para devolver la vida media del protón
        return float(self.proton_lifetime())

    def tsqvt_matching(self, C4: Dict[str, float]) -> Dict[str, float]:
        """
        Asigna las α_i proporcionales a las entradas de C4.
        Si la suma es <= 0, devuelve α igual para los tres grupos usando alpha_gut.
        """
        total = sum(float(v) for v in C4.values()) if C4 else 0.0
        if total <= 0:
            alpha = self.alpha_gut()
            return {'alpha_U1': alpha, 'alpha_SU2': alpha, 'alpha_SU3': alpha}
        alpha_gut = self.alpha_gut()
        return {
            'alpha_U1': (C4.get('U1', 0.0) / total) * alpha_gut,
            'alpha_SU2': (C4.get('SU2', 0.0) / total) * alpha_gut,
            'alpha_SU3': (C4.get('SU3', 0.0) / total) * alpha_gut,
        }

    def match_down(self, couplings_gut: Dict[str, float] = None) -> Dict[str, float]:
        """
        Si no se pasan couplings_gut, devuelve α_i calculadas desde self.g_gut.
        Si se pasan, convierte g1,g2,g3 (o gX) en α correspondiente.
        Ignora claves que no sigan el patrón esperado o valores no convertibles.
        """
        if couplings_gut is None:
            alpha = self.alpha_gut()
            return {'alpha1': alpha, 'alpha2': alpha, 'alpha3': alpha}
        result = {}
        for k, v in couplings_gut.items():
            # acepta claves que empiecen por 'g' o 'G' seguidas del índice
            if k.lower().startswith('g'):
                try:
                    gval = float(v)
                    idx = k[1:]
                    result[f'alpha{idx}'] = gval**2 / (4 * np.pi)
                except Exception:
                    # Si la conversión falla, se ignora esa entrada
                    pass
        return result
