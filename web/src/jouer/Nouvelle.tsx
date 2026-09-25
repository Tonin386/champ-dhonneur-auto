import { useEffect, useRef, useState } from "react";
import { EQUIPE } from "../jeu";
import { useJeu, type Config } from "./store";
import type { InfoIA, JoueurCfg, Regles } from "./types";

/** Choix d'un joueur : Humain ou IA entraînée (modèle de sa réflexion). L'IA ne joue jamais d'elle-même :
 *  elle réfléchit sans limite et c'est vous qui décidez quand elle joue. */
export function ChoixJoueur({ equipe, j, info, onChange, hybride }: {
  equipe: number; j: JoueurCfg; info: InfoIA | null; onChange: (j: JoueurCfg) => void;
  /** partie sur un vrai plateau : « Humain » désigne le joueur plateau ; on choisit aussi le coup de l'IA */
  hybride?: boolean;
}) {
  const iaOk = !!info?.disponible;
  return (
    <fieldset className={`choix-joueur e${equipe}`}>
      <legend><img src={`/img/pieces/sceau-${equipe ? "noir" : "blanc"}.png`} alt="" /> {EQUIPE[equipe]}</legend>
      <div className="bascule" role="radiogroup" aria-label={`Joueur ${EQUIPE[equipe]}`}>
        <button type="button" role="radio" aria-checked={j.type === "humain"} className={j.type === "humain" ? "on" : ""}
          onClick={() => onChange({ ...j, type: "humain" })}>{hybride ? "Joueur plateau" : "Humain"}</button>
        <button type="button" role="radio" aria-checked={j.type === "ia"} className={j.type === "ia" ? "on" : ""} disabled={!iaOk}
          title={iaOk ? (hybride ? "Vous choisissez ses coups, conseillé par sa réflexion"
            : "Réseau entraîné par auto-jeu : il réfléchit à son tour, vous décidez quand il joue") : info?.raison}
          onClick={() => onChange({ ...j, type: "ia" })}>{hybride ? "IA conseillère" : "IA entraînée"}</button>
      </div>
      {j.type === "ia" && iaOk && (
        <div className="options-ia">
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

/** Partie hybride : exactement une IA. `k` : joueur qui vient d'être changé (l'autre s'adapte). */
export function unSeulIA(js: JoueurCfg[], hybride: boolean, k: number): JoueurCfg[] {
  if (!hybride) return js;
  const ia = k >= 0 ? (js[k].type === "ia" ? k : 1 - k) : js[1].type === "ia" || js[0].type !== "ia" ? 1 : 0;
  return js.map((j, i) => ({ ...j, type: i === ia ? "ia" as const : "humain" as const }));
}

// ------------------------------------------------------------------ cartes
const melanger = <T,>(l: T[]) => {
  const m = [...l];
  for (let i = m.length - 1; i > 0; i--) {
    const k = Math.floor(Math.random() * (i + 1));
    [m[i], m[k]] = [m[k], m[i]];
  }
  return m;
};

/** Les 16 cartes Unité ; `equipe` colore une carte prise (0 Or, 1 Argent, -1 choisie sans équipe). */
function GrilleCartes({ regles, equipe, onCarte }: {
  regles: Regles; equipe: (u: string) => number | null; onCarte: (u: string) => void;
}) {
  return (
    <div className="grille-cartes">
      {Object.entries(regles.unites).map(([u, d]) => {
        const e = equipe(u);
        return (
          <button key={u} type="button" className={`carte-choix${e === null ? "" : ` prise e${e}`}`} aria-pressed={e !== null}
            onClick={() => onCarte(u)} title={`${d.nom} — ${d.tactique || d.capacite}`}>
            <img src={`/img/cartes/${u}.jpg`} alt="" loading="lazy" />
            {e !== null && e >= 0 && <img className="sceau-carte" src={`/img/pieces/sceau-${e ? "noir" : "blanc"}.png`} alt={EQUIPE[e]} />}
            <span><b>{u}</b> {d.nom}</span>
          </button>
        );
      })}
    </div>
  );
}

function Bascule<T extends string | number | null>({ valeur, options, onChange, nom }: {
  valeur: T; options: [T, string][]; onChange: (v: T) => void; nom: string;
}) {
  return (
    <div className="bascule" role="radiogroup" aria-label={nom}>
      {options.map(([v, l]) => (
        <button key={String(v)} type="button" role="radio" aria-checked={valeur === v} className={valeur === v ? "on" : ""}
          onClick={() => onChange(v)}>{l}</button>
      ))}
    </div>
  );
}

const PREMIER: [number | null, string][] = [[null, "Au hasard"], [0, EQUIPE[0]], [1, EQUIPE[1]]];

function Draft({ c, setC, regles }: { c: Config; setC: (c: Config) => void; regles: Regles }) {
  const n = regles.draft;
  const cartes = c.cartes ?? [];
  const basculer = (u: string) => {
    if (cartes.includes(u)) setC({ ...c, cartes: cartes.filter(x => x !== u) });
    else if (cartes.length < n) setC({ ...c, cartes: [...cartes, u] });
  };
  return (
    <div className="mise">
      <p className="explication">
        Mise en place avancée du livret : {n} cartes Unité, choisies une à une dans l'ordre A B B A A B B A.
        L'autre joueur prend l'Initiative.
      </p>
      <div className="ligne-reglage">
        <span>Cartes</span>
        <Bascule nom="Cartes du draft" valeur={c.cartes === null ? "hasard" : "choix"}
          options={[["hasard", "Tirées au hasard"], ["choix", "Choisies"]]}
          onChange={v => setC({ ...c, cartes: v === "hasard" ? null : cartes })} />
        {c.cartes !== null && (
          <span className="compte">
            <b className={cartes.length === n ? "complet" : ""}>{cartes.length}</b> / {n}
            <button type="button" onClick={() => setC({ ...c, cartes: melanger(Object.keys(regles.unites)).slice(0, n) })}>Tirer</button>
            <button type="button" onClick={() => setC({ ...c, cartes: [] })} disabled={!cartes.length}>Vider</button>
          </span>
        )}
      </div>
      {c.hybride && c.cartes === null && (
        <p className="remarque">Sur la table, tirez les {n} cartes Unité puis saisissez-les avec « Choisies ».</p>
      )}
      {c.cartes !== null && <GrilleCartes regles={regles} equipe={u => (cartes.includes(u) ? -1 : null)} onCarte={basculer} />}
      <div className="ligne-reglage">
        <span>Premier à choisir</span>
        <Bascule nom="Premier à choisir" valeur={c.premier} options={PREMIER} onChange={premier => setC({ ...c, premier })} />
      </div>
    </div>
  );
}

function Libre({ c, setC, regles }: { c: Config; setC: (c: Config) => void; regles: Regles }) {
  const [cible, setCible] = useState(0);
  const armees = c.armees;
  const equipe = (u: string) => (armees[0].includes(u) ? 0 : armees[1].includes(u) ? 1 : null);
  const poser = (a: string[][]) => setC({ ...c, armees: a });
  const clic = (u: string) => {
    const e = equipe(u);
    const a = armees.map(l => l.filter(x => x !== u));
    if (e !== cible) {
      if (a[cible].length >= 4) return;
      a[cible].push(u);
      // armée complète : on passe à l'autre si elle ne l'est pas
      if (a[cible].length === 4 && a[1 - cible].length < 4) setCible(1 - cible);
    }
    poser(a);
  };
  const modele = Object.entries(regles.armees).find(([, a]) => a.every((l, k) => [...l].sort().join() === [...armees[k]].sort().join()))?.[0];
  return (
    <div className="mise">
      <div className="ligne-reglage">
        <span>Armées</span>
        <div className="modeles">
          {Object.entries(regles.armees).map(([nom, a]) => (
            <button key={nom} type="button" className={modele === nom ? "on" : ""} onClick={() => poser(a.map(l => [...l]))}>{nom}</button>
          ))}
          <button type="button" onClick={() => { const m = melanger(Object.keys(regles.unites)); poser([m.slice(0, 4), m.slice(4, 8)]); }}>
            Au hasard
          </button>
          <button type="button" onClick={() => { poser([[], []]); setCible(0); }} disabled={!armees.flat().length}>Vider</button>
        </div>
      </div>
      <div className="ligne-reglage">
        <span>Ajouter à</span>
        <div className="bascule" role="radiogroup" aria-label="Armée à compléter">
          {[0, 1].map(e => (
            <button key={e} type="button" role="radio" aria-checked={cible === e} className={`e${e}${cible === e ? " on" : ""}`} onClick={() => setCible(e)}>
              {EQUIPE[e]} <b className={armees[e].length === 4 ? "complet" : ""}>{armees[e].length}/4</b>
            </button>
          ))}
        </div>
      </div>
      <GrilleCartes regles={regles} equipe={equipe} onCarte={clic} />
      <div className="ligne-reglage">
        <span>Initiative</span>
        <Bascule nom="Initiative" valeur={c.premier} options={PREMIER} onChange={premier => setC({ ...c, premier })} />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ dialogue
type Onglet = Config["mode"] | "position";

export function DialogueNouvelle() {
  const ouvert = useJeu(s => s.dialogue);
  const config = useJeu(s => s.config);
  const regles = useJeu(s => s.regles);
  const info = useJeu(s => s.infoIA);
  const ref = useRef<HTMLDialogElement>(null);
  const [c, setC] = useState<Config>(config);
  const [onglet, setOnglet] = useState<Onglet>(config.mode);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (ouvert && !d.open) {
      const cfg = useJeu.getState().config;
      setC(cfg);
      setOnglet(cfg.mode);
      d.showModal();
    }
    if (!ouvert && d.open) d.close();
  }, [ouvert]);

  const fermer = () => useJeu.getState().set({ dialogue: false });
  const n = regles?.draft ?? 8;
  const incomplet = onglet === "draft" ? c.cartes !== null && c.cartes.length !== n
    : onglet === "libre" ? c.armees.some(a => a.length !== 4) : false;
  const lancer = async (e: React.FormEvent) => {
    e.preventDefault();
    if (onglet === "position") {
      useJeu.getState().set({ config: { ...useJeu.getState().config, joueurs: c.joueurs, hybride: c.hybride }, dialogue: false });
      return void useJeu.getState().ouvrirEditeur();
    }
    if (incomplet) return;
    await useJeu.getState().nouvelle({ ...c, mode: onglet });
  };
  const joueur = (k: number) => (j: JoueurCfg) => setC({ ...c, joueurs: unSeulIA(c.joueurs.map((x, i) => (i === k ? j : x)), c.hybride, k) });
  const hybrideOk = !!info?.disponible;
  const ONGLETS: [Onglet, string][] = [["draft", "Draft"], ["libre", "Libre"], ["position", "Position"]];
  return (
    <dialog ref={ref} className="dialogue" onClose={fermer} onClick={e => { if (e.target === ref.current) fermer(); }}>
      <form onSubmit={lancer}>
        <h2>Nouvelle partie</h2>
        <div className="ligne-reglage lieu-partie">
          <span>Partie</span>
          <div className="bascule" role="radiogroup" aria-label="Où se joue la partie">
            <button type="button" role="radio" aria-checked={!c.hybride} className={c.hybride ? "" : "on"}
              onClick={() => setC({ ...c, hybride: false })}>À l'écran</button>
            <button type="button" role="radio" aria-checked={c.hybride} className={c.hybride ? "on" : ""} disabled={!hybrideOk}
              title={hybrideOk ? "Partie sur un vrai plateau : vous jouez le camp de l'IA, conseillé par son analyse" : "Il faut une IA entraînée"}
              onClick={() => setC({ ...c, hybride: true, joueurs: unSeulIA(c.joueurs, true, -1) })}>Sur plateau réel (hybride)</button>
          </div>
        </div>
        {c.hybride && (
          <p className="explication">
            La partie se joue sur la table. À son tour, l'IA réfléchit et affiche ses lignes, comme un moteur d'échecs ;
            vous choisissez son coup (une de ses lignes, ou « Jouer son meilleur coup ») et le reproduisez sur la table.
            Vous saisissez aussi les pièces tirées du sac de l'IA et les coups du joueur plateau (jamais sa main).
          </p>
        )}
        <div className="deux-joueurs">
          {c.joueurs.map((j, k) => (
            <ChoixJoueur key={k} equipe={k} j={j} info={info} onChange={joueur(k)} hybride={c.hybride} />
          ))}
        </div>
        {!c.hybride && c.joueurs.some(j => j.type === "ia") && (
          <p className="remarque">L'IA ne joue jamais d'elle-même : à son tour, elle réfléchit sans limite en affichant ses
            lignes, et joue le premier coup de sa réflexion quand vous cliquez « L'IA joue maintenant » (Espace).</p>
        )}
        {!info?.disponible && info?.raison && <p className="remarque">IA indisponible : {info.raison}.</p>}
        <div className="onglets onglets-mise" role="tablist" aria-label="Mise en place">
          {ONGLETS.map(([v, l]) => (
            <button key={v} type="button" role="tab" aria-selected={onglet === v} className={onglet === v ? "on" : ""} onClick={() => setOnglet(v)}>{l}</button>
          ))}
        </div>
        {regles && onglet === "draft" && <Draft c={c} setC={setC} regles={regles} />}
        {regles && onglet === "libre" && <Libre c={c} setC={setC} regles={regles} />}
        {onglet === "position" && (
          <div className="mise">
            <p className="explication">
              Composez une position dans l'éditeur (pièces, mains, sacs, Lieux contrôlés), à partir de la position
              affichée (position de départ pendant un draft). Vous y lancerez la partie ou son analyse.
            </p>
          </div>
        )}
        {onglet !== "position" && (
          <details className="options-avancees">
            <summary>Options</summary>
            <label>Graine
              <input type="number" value={c.graine ?? ""} placeholder="au hasard"
                onChange={e => setC({ ...c, graine: e.target.value === "" ? null : +e.target.value })} />
            </label>
            <span className="remarque">Même graine et même mise en place : mêmes tirages du sac.</span>
          </details>
        )}
        <div className="actions-dialogue">
          <button type="button" onClick={fermer}>Annuler</button>
          <button type="submit" className="on" disabled={incomplet}
            title={incomplet ? (onglet === "draft" ? `Choisissez ${n} cartes` : "Deux armées de 4 unités") : undefined}>
            {onglet === "position" ? "Ouvrir l'éditeur" : "Commencer"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
