from .base import Bot, RandomBot
from .heuristic import GreedyBot
from .mcts import MCTSBot


def make_bot(spec: str, seed: int | None = None) -> Bot:
    """Crée un bot depuis une chaîne.

    aleatoire | glouton | mcts | mcts:2000 | mcts:t=3
    ia | ia:800 | ia:t=2 | ia:modele=chemin.pt,sims=400,dispositif=cuda
    heur:200   (recherche Gumbel évaluée par l'heuristique, sans réseau)
    """
    name, _, arg = spec.partition(":")
    if name in ("aleatoire", "random"):
        return RandomBot(seed)
    if name in ("glouton", "greedy"):
        return GreedyBot(seed)
    if name == "mcts":
        if arg.startswith("t="):
            return MCTSBot(time_limit=float(arg[2:]), seed=seed)
        return MCTSBot(iterations=int(arg) if arg else 800, seed=seed)
    if name in ("ia", "heur"):
        from .neural import depuis_spec
        return depuis_spec(arg, seed, heuristique=(name == "heur"))
    raise ValueError(f"Bot inconnu : {spec}")


BOT_NAMES = ["aleatoire", "glouton", "mcts", "heur", "ia"]
