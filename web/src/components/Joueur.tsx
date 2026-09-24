import { useState } from "react";
import { EQUIPE, NOMS, imgPiece, nomJoueur } from "../jeu";
import type { Conseil } from "../jouer/types";
import type { Decor, Image } from "../types";
import { points } from "./Unites";

function Piece({ c, equipe, petite, cachee }: { c?: string; equipe: number; petite?: boolean; cachee?: boolean }) {
  return (
    <span
      className={`piece e${equipe}${petite ? " petite" : ""}${cachee ? " cachee" : ""}`}
      title={cachee ? "Pièce face cachée" : NOMS[c!]}
      style={cachee ? undefined : { backgroundImage: `url(${imgPiece(c!, equipe)})` }}
    >
      {!cachee && <span className="l">{c}</span>}
    </span>
  );
}

interface Props {
  decor: Decor;
  image: Image;
  joueur: number;
  nom: string;
  /** libellé affiché tel quel sous le nom de l'équipe (sinon : nom du réseau) */
  role?: string;
  /** jeu : pièces de la main cliquables, pièce choisie */
  onPiece?: (c: string) => void;
  choisie?: string | null;
  /** pièces en main mises en valeur (ex. pièce piochée par le Moine soldat) */
  attente?: string | null;
  /** draft : valeur de chaque carte (conseil au joueur humain), masquable */
  conseil?: Conseil | null;
}

const lireMasque = () => {
  try {
    return localStorage.getItem("champ.conseil") === "0";
  } catch {
    return false;
  }
};

export function Joueur({ decor, image, joueur, nom, role, onPiece, choisie, attente, conseil }: Props) {
  const [masque, setMasque] = useState(lireMasque);
  const basculer = () => {
    setMasque(!masque);
    try { localStorage.setItem("champ.conseil", masque ? "1" : "0"); } catch { /* stockage indisponible */ }
  };
  const e = decor.equipes[joueur];
  const p = image.j[joueur];
  const trait = !image.f && image.t === joueur;
  const reste = image.m[e];
  const titre = decor.mode === "4J" ? `J${joueur + 1} ${EQUIPE[e]}` : EQUIPE[e];
  return (
    <section className={`joueur e${e}${trait ? " trait" : ""}`} aria-label={`Joueur ${titre}`}>
      <div className="identite">
        <img src={imgPiece("*", e)} alt="" className="sceau" />
        <div>
          <div className="nom">
            {titre}
            {image.i === joueur && <span className="initiative">Initiative</span>}
          </div>
          <div className="role">{role ?? nomJoueur(nom)}</div>
        </div>
      </div>
      <div className="bloc main-j">
        <div className="etiquette">Main</div>
        <div className="pieces">
          {attente && (
            <span className="piochee" title="Pièce piochée par le Moine soldat, à jouer aussitôt">
              <Piece c={attente} equipe={e} />
            </span>
          )}
          {p.h.length ? p.h.map((c, k) =>
            c === "?" ? <Piece key={k} equipe={e} cachee />
              : onPiece ? (
                <button key={k} type="button" className={`bouton-piece${choisie === c ? " choisie" : ""}`}
                  onClick={() => onPiece(c)} aria-pressed={choisie === c} title={`${NOMS[c]} : voir ses coups`}>
                  <Piece c={c} equipe={e} />
                </button>
              ) : <Piece key={k} c={c} equipe={e} />,
          ) : !attente && <span className="vide">—</span>}
        </div>
      </div>
      <div className="bloc">
        <div className="etiquette">Sac · défausse</div>
        <div className="pieces">
          <span className="sac" title="Pièces dans le sac"><img src="/img/sac.png" alt="Sac" />{p.s}</span>
          {p.x > 0 && <span className="compteur" title="Défausse face cachée"><Piece equipe={e} petite cachee />×{p.x}</span>}
          {p.d.map((c, k) => <Piece key={k} c={c} equipe={e} petite />)}
        </div>
      </div>
      <div className="bloc">
        <div className="etiquette">Réserve{p.l ? ` · ${p.l} éliminée${p.l > 1 ? "s" : ""}` : ""}</div>
        <div className="pieces">
          {Object.entries(p.v).map(([u, n]) => (
            <span key={u} className={`compteur${n ? "" : " epuise"}`}><Piece c={u} equipe={e} petite />×{n}</span>
          ))}
        </div>
      </div>
      <div className="cartes" aria-label="Unités">
        {(image.arm ?? decor.unites)[joueur].map(u => (
          <figure key={u} className="carte" title={`${decor.cartes[u]?.nom} — ${decor.cartes[u]?.tactique || decor.cartes[u]?.capacite}`}>
            <img src={`/img/cartes/${u}.jpg`} alt="" loading="lazy" />
            <figcaption><b>{u}</b> {decor.cartes[u]?.nom}</figcaption>
          </figure>
        ))}
      </div>
      {image.tir && trait && (
        <div className="bloc draft" aria-label="Cartes à choisir">
          <div className="etiquette">Mise en place avancée : cartes à choisir</div>
          <div className="cartes">
            {image.tir.dispo.map(u => {
              const c = !masque ? conseil?.cartes[u] : undefined;
              const aide = c ? `
Valeur du draft si vous la prenez : ${points(c.valeur)} points (choix suivants optimaux)
`
                + `Valeur propre ${points(c.propre)} ±${c.ic95.toFixed(1)}, synergies ${points(c.synergie)}, contres ${points(c.contre)}` : "";
              return (
              <figure key={u} className={`carte${onPiece ? " jouable" : ""}`}
                title={`${decor.cartes[u]?.nom} — ${decor.cartes[u]?.tactique || decor.cartes[u]?.capacite}${aide}`}
                onClick={onPiece ? () => onPiece(u) : undefined}>
                {c && <span className={`badge-valeur${conseil!.meilleure === u ? " meilleure" : ""}`}>{points(c.valeur)}</span>}
                <img src={`/img/cartes/${u}.jpg`} alt="" loading="lazy" />
                <figcaption><b>{u}</b> {decor.cartes[u]?.nom}</figcaption>
              </figure>
              );
            })}
          </div>
          {conseil && (
            <div className="conseil-draft">
              <button type="button" onClick={basculer} aria-pressed={!masque}>{masque ? "Afficher" : "Masquer"} la valeur des cartes</button>
              {!masque && <span title="Points du scoreur (10 = un Lieu d'avance) : valeur du draft pour vous si vous prenez la carte, d'après la dernière mesure de la valeur des unités">
                it. {conseil.iteration} · {conseil.source}</span>}
            </div>
          )}
        </div>
      )}
      <div className="bloc marqueurs" title="Marqueurs Contrôle restant à poser (0 = victoire)">
        <div className="etiquette">À poser</div>
        <div className="pieces">
          {Array.from({ length: reste }, (_, k) => (
            <img key={k} src={`/img/controle-${e ? "noir" : "blanc"}.png`} alt="" />
          ))}
          <b className="reste">{reste}</b>
        </div>
      </div>
    </section>
  );
}
