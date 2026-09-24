import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { compact } from "../jeu";
import type { Tableau } from "../types";

interface Serie {
  cle: string;
  titre: string;
  /** infobulle : ce que mesure le graphique et comment le lire */
  aide: string;
  xs: number[];
  ys: (number | null)[];
  format: (v: number) => string;
  /** lignes de référence (ancres d'Elo) */
  refs?: [string, number][];
}

const pct = (v: number) => `${(100 * v).toFixed(0)} %`;
const dec = (n: number) => (v: number) => v.toFixed(n).replace(".", ",");
const signe = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(Math.round(v))}`;

function bornes(vals: number[]): [number, number] {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (!isFinite(lo)) return [0, 1];
  if (hi - lo < 1e-9) { lo -= Math.abs(lo) * 0.1 || 1; hi += Math.abs(hi) * 0.1 || 1; }
  const m = (hi - lo) * 0.08;
  return [lo - m, hi + m];
}

/** Petit graphique d'une seule série ; le survol est partagé par tous (même itération). */
function MiniGraphe({ s, survol, setSurvol, droite }: { s: Serie; survol: number | null; setSurvol: (i: number | null) => void; droite?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const [taille, setTaille] = useState({ w: 300, h: 110 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setTaille({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const pts = useMemo(
    () => s.xs.map((x, k) => [x, s.ys[k]] as [number, number | null]).filter((p): p is [number, number] => p[1] != null),
    [s.xs, s.ys],
  );
  const { w, h } = taille;
  const gx = 8, gy = 8, bas = 16;
  const [lo, hi] = bornes(pts.map(p => p[1]).concat((s.refs ?? []).map(r => r[1])));
  const x0 = pts.length ? pts[0][0] : 0, x1 = pts.length ? Math.max(pts[pts.length - 1][0], x0 + 1) : 1;
  const X = (x: number) => gx + ((w - 2 * gx) * (x - x0)) / (x1 - x0);
  const Y = (v: number) => gy + ((h - gy - bas) * (hi - v)) / (hi - lo);
  const dernier = pts[pts.length - 1];
  const actif = survol != null ? pts.reduce<[number, number] | null>((m, p) => (!m || Math.abs(p[0] - survol) < Math.abs(m[0] - survol) ? p : m), null) : null;
  const affiche = actif ?? dernier;
  const ligne = pts.map(p => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");

  const bouger = (e: React.PointerEvent) => {
    if (!pts.length) return;
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const x = x0 + ((e.clientX - r.left - gx) / (w - 2 * gx)) * (x1 - x0);
    setSurvol(Math.round(x));
  };

  return (
    <figure className="graphe">
      <figcaption>
        <span className="titre" tabIndex={0} aria-describedby={`aide-${s.cle}`}>
          {s.titre}<span className="icone-aide" aria-hidden="true">i</span>
        </span>
        <span className={`bulle${droite ? " droite" : ""}`} role="tooltip" id={`aide-${s.cle}`}>
          {s.aide}
          <small>Survol de la courbe : valeur d'une itération, lue en même temps sur tous les graphiques.</small>
        </span>
        <span className="valeur">{affiche ? s.format(affiche[1]) : "—"}</span>
        {affiche && <span className="quand">it. {affiche[0]}</span>}
      </figcaption>
      <div className="zone" ref={ref}>
        {pts.length === 0 ? (
          <div className="attente">en attente de données</div>
        ) : (
          <svg width={w} height={h} onPointerMove={bouger} onPointerLeave={() => setSurvol(null)} role="img" aria-label={s.titre}>
            <line className="axe" x1={gx} x2={w - gx} y1={h - bas} y2={h - bas} />
            {(s.refs ?? []).map(([nom, v]) => (
              <g key={nom} className="ref">
                <line x1={gx} x2={w - gx} y1={Y(v)} y2={Y(v)} />
                <text x={w - gx} y={Y(v) - 3}>{nom}</text>
              </g>
            ))}
            {pts.length > 1 && (
              <>
                <polygon className="aire" points={`${X(pts[0][0])},${h - bas} ${ligne} ${X(dernier[0])},${h - bas}`} />
                <polyline className="courbe" points={ligne} />
              </>
            )}
            <circle className="point" cx={X(dernier[0])} cy={Y(dernier[1])} r={4} />
            {actif && (
              <g className="curseur">
                <line x1={X(actif[0])} x2={X(actif[0])} y1={gy} y2={h - bas} />
                <circle cx={X(actif[0])} cy={Y(actif[1])} r={4.5} />
              </g>
            )}
            <text className="borne" x={gx} y={h - 3}>it. {x0}</text>
            <text className="borne" x={w - gx} y={h - 3} textAnchor="end">it. {pts[pts.length - 1][0]}</text>
          </svg>
        )}
      </div>
    </figure>
  );
}

export function Graphes({ t }: { t: Tableau }) {
  const [survol, setSurvol] = useState<number | null>(null);
  const S = t.series;
  const it = S.iteration as number[];
  const series: Serie[] = [
    { cle: "elo", titre: "Elo (glouton = 0)", xs: t.courbe.map(p => p.iteration), ys: t.courbe.map(p => p.elo), format: signe,
      refs: Object.entries(t.ancres).filter(([k]) => k !== "glouton"),
      aide: "Force du modèle de chaque itération, mesurée par des matchs contre des adversaires de référence et "
        + "recalculée sur tous les résultats. Le bot glouton sert d'origine (0) ; les lignes horizontales situent "
        + "les autres références (heur:64 = recherche guidée par l'heuristique). +200 ≈ 76 % des points contre un "
        + "adversaire à 0, +400 ≈ 91 %. Doit monter." },
    { cle: "pph", titre: "Parties d'auto-jeu / h", xs: it, ys: S.parties_par_heure, format: v => compact(v),
      aide: "Débit de l'auto-jeu : parties que le réseau joue contre lui-même par heure pendant cette phase. "
        + "Dépend du matériel et du nombre de simulations par coup ; une chute soudaine signale un goulot (GPU, CPU)." },
    { cle: "pp", titre: "Perte de politique", xs: it, ys: S.perte_politique, format: dec(3),
      aide: "Écart (entropie croisée) entre les coups que le réseau propose d'instinct et ceux que la recherche "
        + "retient après réflexion. Plus elle baisse, mieux le réseau anticipe seul le bon coup. Elle ne tombe "
        + "jamais à 0 : la cible elle-même hésite entre plusieurs coups." },
    { cle: "pv", titre: "Perte de valeur", xs: it, ys: S.perte_valeur, format: dec(3),
      aide: "Écart (entropie croisée) entre le pronostic victoire / nulle / défaite du réseau et l'issue réelle "
        + "des parties. Plus elle baisse, mieux le réseau juge qui va gagner. Vers 1,1, il répond au hasard." },
    { cle: "vp", titre: "Précision valeur (inédites)", xs: it, ys: S.val_precision, format: pct,
      aide: "Sur des parties jamais vues à l'apprentissage : part des positions où le réseau désigne le bon "
        + "vainqueur (pour une nulle : pronostic proche de l'équilibre). 50 % ≈ pile ou face. Mesurée sur des "
        + "parties inédites, elle ne peut pas être gonflée par le par-cœur." },
    { cle: "pc", titre: "1er coup = cible (inédites)", xs: it, ys: S.val_premier_coup, format: pct,
      aide: "Sur ces mêmes parties inédites : part des positions où le coup préféré du réseau, sans aucune "
        + "recherche, est aussi le meilleur coup trouvé par la recherche. Mesure la qualité de son intuition." },
    { cle: "ma", titre: "Manches par partie", xs: it, ys: S.manches, format: dec(1),
      aide: "Durée moyenne des parties d'auto-jeu, en manches (chaque joueur pioche 3 pièces par manche). "
        + "Des parties qui raccourcissent traduisent en général un jeu plus tranchant, qui conclut plus vite." },
    { cle: "nu", titre: "Parties nulles", xs: it, ys: S.nulles, format: pct,
      aide: "Part des parties d'auto-jeu arrêtées à la limite de manches sans que personne ait posé tous ses "
        + "marqueurs Contrôle. Doit baisser à mesure que les deux camps apprennent à conclure." },
  ];
  return (
    <div className="graphes">
      {series.map((s, k) => <MiniGraphe key={s.cle} s={s} survol={survol} setSurvol={setSurvol} droite={k >= series.length / 2} />)}
    </div>
  );
}
