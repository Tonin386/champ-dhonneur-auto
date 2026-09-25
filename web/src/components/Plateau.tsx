import { memo } from "react";
import { EQUIPE, NOMS, geometrie, geometrieRetournee, hexagone, imgPiece } from "../jeu";
import type { Decor, Image, UniteImg } from "../types";

interface Props {
  decor: Decor;
  image: Image;
  /** image précédente : unités éliminées (fondu) et mise en valeur du coup */
  precedente?: Image | null;
  /** durée des animations (ms), adaptée à la vitesse de lecture */
  duree?: number;
  mini?: boolean;
  /** clé du coup affiché : relance les animations ponctuelles */
  cle?: number;
  /** jeu : cases jouables (attaque = cible ennemie), case choisie, coup survolé ou conseillé */
  cibles?: Map<number, "attaque" | "jouable">;
  choisie?: number | null;
  survol?: number[] | null;
  conseil?: number[] | null;
  /** ligne d'analyse survolée : cases de chaque coup et équipe qui le joue (0 Or, 1 Argent) */
  ligne?: { cases: number[]; equipe: number }[] | null;
  /** clic sur une case (droit = clic droit, pour l'éditeur) */
  onCase?: (i: number, droit: boolean) => void;
  /** cases qui réagissent au clic (toutes si absent) */
  cliquables?: Set<number>;
  /** plateau retourné : Argent en bas */
  retourne?: boolean;
}

const ATTAQUES = new Set(["attack"]);

function Unite({ u, decor, x, y, mini }: { u: UniteImg; decor: Decor; x: number; y: number; mini?: boolean }) {
  const [, , owner, type, coins] = u;
  const equipe = decor.equipes[owner];
  const r = 21;
  const pile = [];
  for (let k = Math.min(coins, 5) - 1; k > 0; k--)
    pile.push(<circle key={k} cy={k * 2.6} r={r + 1.5} className={`pile e${equipe}`} />);
  return (
    <g className="unite" style={{ transform: `translate(${x}px, ${y}px)` }}>
      <g className="unite-corps">
        {pile}
        <circle r={r + 2.2} className={`bord e${equipe}`} />
        <image href={imgPiece(type, equipe)} x={-r} y={-r} width={2 * r} height={2 * r} />
        {!mini && (
          <g transform={`translate(${-r * 0.82} ${r * 0.82})`}>
            <circle r={7.5} className={`pastille e${equipe}`} />
            <text className="lettre">{type}</text>
          </g>
        )}
        {coins > 1 && (
          <g transform={`translate(${r * 0.86} ${-r * 0.86})`}>
            <circle r={mini ? 11 : 9.5} className="compte" />
            <text className="nombre">{coins}</text>
          </g>
        )}
        {decor.mode === "4J" && !mini && <text className="numero" x={-r * 0.9} y={-r * 0.9}>{owner + 1}</text>}
      </g>
      <title>{`${NOMS[type]} (${EQUIPE[equipe]}), ${coins} pièce${coins > 1 ? "s" : ""}`}</title>
    </g>
  );
}

/** Flèche d'une case à l'autre, arrêtée avant le centre de l'arrivée. */
function Fleche({ G, de, a, cls, marqueur }: { G: ReturnType<typeof geometrie>; de: number; a: number; cls: string; marqueur: string }) {
  const [x1, y1] = G.centre(de);
  const [x2, y2] = G.centre(a);
  const L = Math.hypot(x2 - x1, y2 - y1);
  const k = (L - G.s * 0.62) / L;
  return <line className={cls} x1={x1} y1={y1} x2={x1 + (x2 - x1) * k} y2={y1 + (y2 - y1) * k} markerEnd={marqueur} />;
}

/** Ligne d'analyse : une flèche par déplacement, une pastille numérotée (2 pour le coup suivant…)
 *  à l'arrivée de chaque coup, à la couleur du camp ; plusieurs coups sur une case s'étagent. */
function Ligne({ G, ligne }: { G: ReturnType<typeof geometrie>; ligne: { cases: number[]; equipe: number }[] }) {
  const empiles = new Map<number, number>();
  const pastilles = ligne.flatMap(({ cases, equipe }, k) => {
    if (!cases.length) return [];
    const c = cases[cases.length - 1];
    const rang = empiles.get(c) ?? 0;
    empiles.set(c, rang + 1);
    const [x, y] = G.centre(c);
    return [{ k, equipe, c, x: x + G.s * (0.5 - 0.56 * rang), y: y - G.s * 0.52 }];
  });
  return (
    <g className="ligne-analyse" aria-hidden="true">
      {ligne.map(({ cases, equipe }, k) => cases.length > 1 && cases[0] !== cases[cases.length - 1] && (
        <Fleche key={`lf${k}`} G={G} de={cases[0]} a={cases[cases.length - 1]} cls={`fleche-ligne e${equipe}`}
          marqueur={`url(#pointe-l${equipe})`} />
      ))}
      {pastilles.map(({ k, equipe, c }) => {
        const [x, y] = G.centre(c);
        return <circle key={`la${k}`} className={`anneau-ligne e${equipe}`} cx={x} cy={y} r={G.s * 0.8} />;
      })}
      {pastilles.map(({ k, equipe, x, y }) => (
        <g key={`lp${k}`} className={`pastille-ligne e${equipe}${k === 0 ? " premier" : ""}`}>
          <circle cx={x} cy={y} r={G.s * 0.3} />
          <text x={x} y={y}>{k + 1}</text>
        </g>
      ))}
    </g>
  );
}

export const Plateau = memo(function Plateau({ decor, image, precedente, duree = 400, mini, cle, cibles, choisie, survol, conseil, ligne, onCase, cliquables, retourne }: Props) {
  const G = retourne ? geometrieRetournee(decor) : geometrie(decor);
  const a = image.a;
  const cases = a?.c ?? [];
  const cible = cases.length ? cases[cases.length - 1] : null;
  const depart = cases.length > 1 ? cases[0] : null;
  const controle = new Map(image.c);
  const presents = new Set(image.u.map(u => u[0]));
  const disparues = precedente ? precedente.u.filter(u => !presents.has(u[0])) : [];
  const style = { "--duree": `${duree}ms` } as React.CSSProperties;

  return (
    <svg
      className={`plateau${mini ? " mini" : ""}`}
      viewBox={`0 0 ${G.W} ${G.H}`}
      style={style}
      role="img"
      aria-label="Plateau de jeu"
    >
      <defs>
        <marker id="pointe" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="4" markerHeight="4" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="pointe" />
        </marker>
        <marker id="pointe-a" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="4" markerHeight="4" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="pointe attaque" />
        </marker>
        <marker id="pointe-c" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="3.4" markerHeight="3.4" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="pointe conseil" />
        </marker>
        {[0, 1].map(e => (
          <marker key={e} id={`pointe-l${e}`} viewBox="0 0 10 10" refX="6" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" className={`pointe-ligne e${e}`} />
          </marker>
        ))}
        <radialGradient id="parchemin" cx="50%" cy="40%" r="70%">
          <stop offset="0%" stopColor="var(--case-clair)" />
          <stop offset="100%" stopColor="var(--case)" />
        </radialGradient>
      </defs>
      {decor.cases.map(([nom, , , lieu], i) => {
        const [x, y] = G.centre(i);
        const cls = [
          "case",
          lieu && "lieu",
          i === cible && (a && ATTAQUES.has(a.k) ? "cible attaque" : "cible"),
          i === depart && "depart",
          (i === choisie || survol?.includes(i)) && "choisie",
        ].filter(Boolean).join(" ");
        const t = controle.get(i);
        const R = G.s * (t === undefined ? 0.58 : 0.74);
        return (
          <g key={i}>
            <polygon className={cls} points={hexagone(x, y, G.s - 1.2)} />
            {cibles?.has(i) && <polygon className={`teinte-cible ${cibles.get(i)}`} points={hexagone(x, y, G.s - 3)} />}
            {lieu && (
              <image
                key={t ?? "libre"}
                className={t === undefined ? "lieu-libre" : "marqueur"}
                href={t === undefined ? "/img/lieu.svg" : `/img/controle-${t ? "noir" : "blanc"}.png`}
                x={x - R} y={y - R} width={2 * R} height={2 * R}
              />
            )}
            {!mini && <text className="coord" x={x} y={y - G.h * 0.72}>{nom}</text>}
          </g>
        );
      })}
      {depart !== null && cible !== null && depart !== cible && (
        <Fleche key={`fl${cle}`} G={G} de={depart} a={cible} cls={`fleche${a && ATTAQUES.has(a.k) ? " attaque" : ""}`}
          marqueur={a && ATTAQUES.has(a.k) ? "url(#pointe-a)" : "url(#pointe)"} />
      )}
      {disparues.map(u => {
        const [x, y] = G.centre(u[1]);
        return (
          <g key={`f${u[0]}-${cle}`} className="fantome">
            <Unite u={u} decor={decor} x={x} y={y} mini={mini} />
          </g>
        );
      })}
      {image.u.map(u => {
        const [x, y] = G.centre(u[1]);
        return <Unite key={u[0]} u={u} decor={decor} x={x} y={y} mini={mini} />;
      })}
      {ligne && ligne.length > 0 && <Ligne G={G} ligne={ligne} />}
      {conseil && conseil.length > 1 && conseil[0] !== conseil[conseil.length - 1] && (
        <Fleche G={G} de={conseil[0]} a={conseil[conseil.length - 1]} cls="fleche conseil" marqueur="url(#pointe-c)" />
      )}
      {conseil && conseil.length === 1 && (() => {
        const [x, y] = G.centre(conseil[0]);
        return <circle className="anneau-conseil" cx={x} cy={y} r={G.s * 0.78} />;
      })()}
      {cibles && [...cibles].map(([i, t]) => {
        const [x, y] = G.centre(i);
        const occupee = image.u.some(u => u[1] === i);
        return occupee || t === "attaque"
          ? <circle key={`c${i}`} className={`cible-anneau ${t}`} cx={x} cy={y} r={G.s * 0.8} />
          : <circle key={`c${i}`} className={`cible-point ${t}`} cx={x} cy={y} r={G.s * 0.24} />;
      })}
      {onCase && decor.cases.map((_, i) => {
        if (cliquables && !cliquables.has(i)) return null;
        const [x, y] = G.centre(i);
        return (
          <polygon
            key={`z${i}`} className="zone-clic" points={hexagone(x, y, G.s - 1.2)}
            onClick={() => onCase(i, false)}
            onContextMenu={e => { e.preventDefault(); onCase(i, true); }}
          />
        );
      })}
      {cible !== null && !mini && (() => {
        const [x, y] = G.centre(cible);
        return <circle key={`on${cle}`} className={`onde${a && ATTAQUES.has(a.k) ? " attaque" : ""}`} cx={x} cy={y} r={G.s * 0.8} />;
      })()}
    </svg>
  );
});
