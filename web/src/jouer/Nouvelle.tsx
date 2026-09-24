import { useEffect, useRef, useState } from "react";
import { EQUIPE } from "../jeu";
import { useJeu, type Config } from "./store";
import type { InfoIA, JoueurCfg } from "./types";

/** Choix d'un joueur : Humain ou IA entraînée (niveau = simulations par décision, modèle). */
export function ChoixJoueur({ equipe, j, info, onChange, niveaux }: {
  equipe: number; j: JoueurCfg; info: InfoIA | null; onChange: (j: JoueurCfg) => void; niveaux: Record<string, string>;
}) {
  const iaOk = !!info?.disponible;
  return (
    <fieldset className={`choix-joueur e${equipe}`}>
      <legend><img src={`/img/pieces/sceau-${equipe ? "noir" : "blanc"}.png`} alt="" /> {EQUIPE[equipe]}</legend>
      <div className="bascule" role="radiogroup" aria-label={`Joueur ${EQUIPE[equipe]}`}>
        <button type="button" role="radio" aria-checked={j.type === "humain"} className={j.type === "humain" ? "on" : ""}
          onClick={() => onChange({ ...j, type: "humain" })}>Humain</button>
        <button type="button" role="radio" aria-checked={j.type === "ia"} className={j.type === "ia" ? "on" : ""} disabled={!iaOk}
          title={iaOk ? "Réseau entraîné par auto-jeu" : info?.raison} onClick={() => onChange({ ...j, type: "ia" })}>IA entraînée</button>
      </div>
      {j.type === "ia" && iaOk && (
        <div className="options-ia">
          <label>Niveau
            <select value={j.niveau} onChange={e => onChange({ ...j, niveau: +e.target.value })}>
              {Object.entries(niveaux).map(([v, l]) => <option key={v} value={v}>{l} ({v} simulations)</option>)}
            </select>
          </label>
          <label>Modèle
            <select value={j.modele ?? ""} onChange={e => onChange({ ...j, modele: e.target.value || null })}>
              <option value="">Meilleur modèle (par défaut)</option>
              {info!.modeles.map(m => <option key={m.chemin} value={m.chemin}>{m.nom}</option>)}
            </select>
          </label>
        </div>
      )}
    </fieldset>
  );
}

export function DialogueNouvelle() {
  const ouvert = useJeu(s => s.dialogue);
  const config = useJeu(s => s.config);
  const regles = useJeu(s => s.regles);
  const info = useJeu(s => s.infoIA);
  const ref = useRef<HTMLDialogElement>(null);
  const [c, setC] = useState<Config>(config);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (ouvert && !d.open) { setC(useJeu.getState().config); d.showModal(); }
    if (!ouvert && d.open) d.close();
  }, [ouvert]);

  const fermer = () => useJeu.getState().set({ dialogue: false });
  const lancer = async (e: React.FormEvent) => {
    e.preventDefault();
    await useJeu.getState().nouvelle(c);
  };
  const joueur = (k: number) => (j: JoueurCfg) => setC({ ...c, joueurs: c.joueurs.map((x, i) => (i === k ? j : x)) });
  return (
    <dialog ref={ref} className="dialogue" onClose={fermer} onClick={e => { if (e.target === ref.current) fermer(); }}>
      <form onSubmit={lancer}>
        <h2>Nouvelle partie</h2>
        <div className="deux-joueurs">
          {c.joueurs.map((j, k) => (
            <ChoixJoueur key={k} equipe={k} j={j} info={info} onChange={joueur(k)} niveaux={regles?.niveaux ?? {}} />
          ))}
        </div>
        {!info?.disponible && info?.raison && <p className="remarque">IA indisponible : {info.raison}.</p>}
        <div className="ligne-options">
          <label>Unités
            <select value={c.unites} onChange={e => setC({ ...c, unites: e.target.value })}>
              {Object.entries(regles?.scenarios ?? { premiere: "Première partie" }).map(([v, l]) => (
                <option key={v} value={v}>{l}{v.includes("/") ? ` (${v})` : ""}</option>
              ))}
            </select>
          </label>
          <label>Graine
            <input type="number" value={c.graine ?? ""} placeholder="au hasard"
              onChange={e => setC({ ...c, graine: e.target.value === "" ? null : +e.target.value })} />
          </label>
          <label className="case-a-cocher">
            <input type="checkbox" checked={c.mains_visibles} onChange={e => setC({ ...c, mains_visibles: e.target.checked })} />
            Voir la main de l'IA
          </label>
        </div>
        <p className="remarque">
          Or joue en bas du plateau, Argent en haut. Pour partir d'une position précise, utilisez l'éditeur de position.
        </p>
        <div className="actions-dialogue">
          <button type="button" onClick={fermer}>Annuler</button>
          <button type="submit" className="on">Commencer</button>
        </div>
      </form>
    </dialog>
  );
}
