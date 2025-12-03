# fl_judge.py
"""
Módulo para decidir parada en FL usando:
  - max_rounds (tope absoluto de 50 rondas)
  - early stopping sobre recall global (con patience de 20 rondas)

Uso:
    from .judge import Judge
    judge = Judge(max_rounds=50, patience=20, min_delta=1e-4)
    continue_training = judge.update_and_should_continue(round_idx, global_recall)
"""

from typing import Optional
import logging

class Judge:
    def __init__(self, max_rounds: int = 50, patience: int = 20, min_delta: float = 1e-6):
        """
        max_rounds: número máximo de rondas globales (tope absoluto: 50).
        patience: número de rondas consecutivas sin mejora en recall global antes de parar (20).
        min_delta: mínima mejora en recall para considerarla real (evita ruido).
        """
        self.max_rounds = int(max_rounds)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        # estado interno para recall
        self.best_recall: float = -float("inf")
        self.no_improve_count: int = 0

    def reset(self):
        """Resetear estado interno."""
        self.best_recall = -float("inf")
        self.no_improve_count = 0

    def update_and_should_continue(self, round_idx: int, global_recall: Optional[float]) -> bool:
        """
        Debe llamarse **después** de evaluar el modelo global (y obtener recall global).
        - round_idx: entero (ronda actual, por ejemplo 1,2,...).
        - global_recall: recall global calculado (float en [0,1]). Si es None, solo se aplica max_rounds.

        Devuelve True si **DEBE CONTINUAR**, False si **DEBE PARAR**.
        """
        # 1) tope absoluto de 50 rondas
        if round_idx >= self.max_rounds:
            logging.info(f"[Judge] Parar: round {round_idx} >= max_rounds {self.max_rounds}")
            return False

        # 2) si global_recall no está disponible, no podemos hacer early stopping -> continuar
        if global_recall is None:
            logging.debug("[Judge] global_recall no proporcionado; usar solo max_rounds.")
            return True

        # 3) early stopping basado en recall global: comparar con el mejor recall visto
        if global_recall > self.best_recall + self.min_delta:
            logging.info(f"[Judge] ✓ Mejora recall: {self.best_recall:.6f} -> {global_recall:.6f}")
            self.best_recall = global_recall
            self.no_improve_count = 0
            return True
        else:
            self.no_improve_count += 1
            logging.info(f"[Judge] ⚠ No mejora en recall: contador {self.no_improve_count}/{self.patience}")
            if self.no_improve_count >= self.patience:
                logging.info(f"[Judge] ✗ Parar por early stopping: {self.no_improve_count} rondas sin mejora en recall global.")
                return False
            return True
