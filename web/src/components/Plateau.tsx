import { memo } from "react";
import { EQUIPE, NOMS, geometrie, hexagone, imgPiece } from "../jeu";
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

export const Plateau = memo(function Plateau({ decor, image, precedente, duree = 400, mini, cle }: Props) {
  const G = geometrie(decor);
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
        ].filter(Boolean).join(" ");
        const t = controle.get(i);
        const R = G.s * (t === undefined ? 0.58 : 0.74);
        return (
          <g key={i}>
            <polygon className={cls} points={hexagone(x, y, G.s - 1.2)} />
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
      {depart !== null && cible !== null && depart !== cible && (() => {
        const [x1, y1] = G.centre(depart);
        const [x2, y2] = G.centre(cible);
        const L = Math.hypot(x2 - x1, y2 - y1);
        const k = (L - G.s * 0.62) / L;
        return (
          <line
            key={`fl${cle}`}
            className={`fleche${a && ATTAQUES.has(a.k) ? " attaque" : ""}`}
            x1={x1} y1={y1} x2={x1 + (x2 - x1) * k} y2={y1 + (y2 - y1) * k}
            markerEnd={a && ATTAQUES.has(a.k) ? "url(#pointe-a)" : "url(#pointe)"}
          />
        );
      })()}
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
      {cible !== null && !mini && (() => {
        const [x, y] = G.centre(cible);
        return <circle key={`on${cle}`} className={`onde${a && ATTAQUES.has(a.k) ? " attaque" : ""}`} cx={x} cy={y} r={G.s * 0.8} />;
      })()}
    </svg>
  );
});
