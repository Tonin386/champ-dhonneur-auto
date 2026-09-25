import { useEffect, useMemo, useState } from "react";
import { create } from "zustand";
import { Plateau } from "../components/Plateau";
import { EQUIPE, NOMS, imgPiece } from "../jeu";
import type { Decor, Image } from "../types";
import { ChoixJoueur, unSeulIA } from "./Nouvelle";
import { useJeu } from "./store";
import type { JoueurCfg, Position, Regles } from "./types";

// ------------------------------------------------------------------ état de l'éditeur
type Zone = "main" | "defausse" | "cachee" | "reserve" | "boite";
type Compte = Record<Zone, number>;
type LieuSceau = "main" | "sac" | "cachee";

interface Ed {
  unites: string[][];
  plateau: Record<number, { j: number; u: string; k: number }>;
  controle: Record<number, number>;
  trait: number;
  initiative: number;
  manche: number;
  comptes: Record<string, Compte>[];
  sceau: LieuSceau[];
}

type Outil = { t: "unite"; j: number; u: string } | { t: "controle"; e: number | null } | { t: "gomme" };

const ZONES: [Zone, string, string][] = [
  ["main", "Main", "Pièces en main (3 au plus, Sceau compris)"],
  ["defausse", "Déf.", "Défausse face visible"],
  ["cachee", "Cach.", "Défausse face cachée"],
  ["reserve", "Rés.", "Réserve (pièces recrutables)"],
  ["boite", "Élim.", "Pièces éliminées (remises dans la boîte)"],
];

const compter = (l: string[], u: string) => l.filter(x => x === u).length;
const indexCases = (r: Regles) => new Map(r.cases.map(([nom], i) => [nom, i]));

function depuisPosition(p: Position, r: Regles): Ed {
  const idx = indexCases(r);
  const plateau: Ed["plateau"] = {};
  for (const e of p.plateau) plateau[idx.get(e.case)!] = { j: e.joueur, u: e.unite, k: e.pieces };
  const controle: Ed["controle"] = {};
  for (const [nom, t] of Object.entries(p.controle)) controle[idx.get(nom)!] = t;
  const comptes = p.unites.map((l, j) => {
    const z = p.joueurs[j];
    return Object.fromEntries(l.map(u => {
      const surPlateau = p.plateau.filter(e => e.joueur === j && e.unite === u).reduce((a, e) => a + e.pieces, 0);
      const c: Compte = {
        main: compter(z.main, u), defausse: compter(z.defausse, u), cachee: compter(z.defausse_cachee, u),
        reserve: z.reserve[u] ?? 0, boite: 0,
      };
      c.boite = Math.max(0, r.unites[u].pieces - surPlateau - compter(z.sac, u) - c.main - c.defausse - c.cachee - c.reserve);
      return [u, c];
    }));
  });
  const sceau = p.joueurs.map(z => (z.main.includes("*") ? "main" : z.defausse_cachee.includes("*") ? "cachee" : "sac") as LieuSceau);
  return { unites: p.unites, plateau, controle, trait: p.trait, initiative: p.initiative, manche: p.manche, comptes, sceau };
}

function surPlateau(e: Ed, j: number, u: string) {
  return Object.values(e.plateau).filter(x => x.j === j && x.u === u).reduce((a, x) => a + x.k, 0);
}

function sac(e: Ed, r: Regles, j: number, u: string) {
  const c = e.comptes[j][u];
  return r.unites[u].pieces - surPlateau(e, j, u) - c.main - c.defausse - c.cachee - c.reserve - c.boite;
}

function versPosition(e: Ed, r: Regles): Position {
  const noms = r.cases.map(c => c[0]);
  const rep = (u: string, n: number) => Array.from({ length: Math.max(0, n) }, () => u);
  return {
    unites: e.unites,
    plateau: Object.entries(e.plateau).map(([i, x]) => ({ case: noms[+i], joueur: x.j, unite: x.u, pieces: x.k })),
    controle: Object.fromEntries(Object.entries(e.controle).map(([i, t]) => [noms[+i], t])),
    trait: e.trait, initiative: e.initiative, manche: e.manche,
    joueurs: e.unites.map((l, j) => {
      const c = e.comptes[j];
      return {
        main: l.flatMap(u => rep(u, c[u].main)).concat(e.sceau[j] === "main" ? ["*"] : []),
        sac: l.flatMap(u => rep(u, sac(e, r, j, u))).concat(e.sceau[j] === "sac" ? ["*"] : []),
        defausse: l.flatMap(u => rep(u, c[u].defausse)),
        defausse_cachee: l.flatMap(u => rep(u, c[u].cachee)).concat(e.sceau[j] === "cachee" ? ["*"] : []),
        reserve: Object.fromEntries(l.map(u => [u, c[u].reserve])),
      };
    }),
  };
}

/** Problèmes visibles avant même l'envoi au serveur (qui valide tout). */
function problemes(e: Ed, r: Regles): string[] {
  const out: string[] = [];
  e.unites.forEach((l, j) => {
    for (const u of l) if (sac(e, r, j, u) < 0) out.push(`${EQUIPE[j]} : ${-sac(e, r, j, u)} pièce(s) ${NOMS[u]} de trop`);
    const main = l.reduce((a, u) => a + e.comptes[j][u].main, 0) + (e.sceau[j] === "main" ? 1 : 0);
    if (main > 3) out.push(`${EQUIPE[j]} : ${main} pièces en main (3 au plus)`);
    if (j === e.trait && main === 0) out.push(`${EQUIPE[j]} est au trait mais n'a aucune pièce en main`);
    const places = Object.values(e.controle).filter(t => t === j).length;
    if (places >= 6) out.push(`${EQUIPE[j]} contrôle 6 Lieux : la partie serait déjà gagnée`);
  });
  return out;
}

interface EdStore {
  ed: Ed | null;
  outil: Outil;
  joueurs: JoueurCfg[];
  hybride: boolean; // lancer une partie sur un vrai plateau depuis cette position
  set(p: Partial<EdStore>): void;
  maj(f: (e: Ed) => void): void;
}

const useEd = create<EdStore>()((set, get) => ({
  ed: null,
  outil: { t: "gomme" },
  joueurs: [],
  hybride: false,
  set: p => set(p),
  maj: f => {
    const e = structuredClone(get().ed);
    if (!e) return;
    f(e);
    set({ ed: e });
  },
}));

function decorDe(e: Ed, r: Regles): Decor {
  const lettres = [...new Set(e.unites.flat())].sort();
  return {
    mode: "2J", cols: r.cols, max_y2: r.max_y2, cases: r.cases, equipes: [0, 1], unites: e.unites,
    cartes: Object.fromEntries(lettres.map(u => [u, r.unites[u]])), graine: 0, entetes: {},
  };
}

function imageDe(e: Ed): Image {
  return {
    r: e.manche, t: e.trait, i: e.initiative,
    m: [0, 1].map(t => 6 - Object.values(e.controle).filter(x => x === t).length) as [number, number],
    u: Object.entries(e.plateau).map(([i, x]) => [+i, +i, x.j, x.u, x.k]),
    c: Object.entries(e.controle).map(([i, t]) => [+i, t] as [number, number]),
    j: e.unites.map((l, j) => ({
      h: l.flatMap(u => Array(e.comptes[j][u].main).fill(u)), s: 0, d: [], x: 0, v: {}, l: 0,
    })),
    a: null,
  };
}

// ------------------------------------------------------------------ actions
function clicCase(i: number, droit: boolean) {
  const { ed, outil, maj } = useEd.getState();
  const r = useJeu.getState().regles;
  if (!ed || !r) return;
  const lieu = r.cases[i][3];
  const x = ed.plateau[i];
  if (droit) {
    if (x) maj(e => { if (e.plateau[i].k > 1) e.plateau[i].k--; else delete e.plateau[i]; });
    else if (lieu) maj(e => { delete e.controle[i]; });
    return;
  }
  if (outil.t === "gomme") {
    if (x) maj(e => { delete e.plateau[i]; });
    else if (lieu) maj(e => { delete e.controle[i]; });
    return;
  }
  if (outil.t === "controle") {
    if (!lieu) return useJeu.getState().setErreur("Les marqueurs Contrôle se posent sur les Lieux");
    maj(e => { if (outil.e === null) delete e.controle[i]; else e.controle[i] = outil.e; });
    return;
  }
  const { j, u } = outil;
  const def = r.unites[u];
  maj(e => {
    const c = e.comptes[j][u];
    const prendre = () => {   // une pièce vient du sac, sinon de la réserve
      if (sac(e, r, j, u) > 0) return true;
      if (c.reserve > 0) { c.reserve--; return true; }
      if (c.boite > 0) { c.boite--; return true; }
      useJeu.getState().setErreur(`Plus aucune pièce ${def.nom} disponible (${def.pieces} au total)`);
      return false;
    };
    if (x && x.j === j && x.u === u) {
      if (prendre()) x.k++;
      return;
    }
    const memes = Object.entries(e.plateau).filter(([, y]) => y.j === j && y.u === u);
    if (memes.length >= def.max && !x) {
      // une seule unité de ce type : elle se déplace avec ses pièces
      const [ancienne, y] = memes[0];
      delete e.plateau[+ancienne];
      e.plateau[i] = y;
      return;
    }
    if (prendre()) e.plateau[i] = { j, u, k: 1 };
  });
}

function changerUnite(j: number, k: number, u: string) {
  const r = useJeu.getState().regles!;
  useEd.getState().maj(e => {
    const ancienne = e.unites[j][k];
    e.unites[j] = e.unites[j].map((x, i) => (i === k ? u : x));
    for (const [i, x] of Object.entries(e.plateau)) if (x.j === j && x.u === ancienne) delete e.plateau[+i];
    delete e.comptes[j][ancienne];
    e.comptes[j][u] = { main: 0, defausse: 0, cachee: 0, reserve: r.unites[u].pieces - 2, boite: 0 };
  });
}

function positionDepart(e: Ed, r: Regles) {
  const idx = indexCases(r);
  e.plateau = {};
  e.controle = {};
  r.departs.forEach((l, t) => l.forEach(n => { e.controle[idx.get(n)!] = t; }));
  e.comptes = e.unites.map(l => Object.fromEntries(l.map((u, k) => [u, {
    main: k < 3 ? 1 : 0, defausse: 0, cachee: 0, reserve: r.unites[u].pieces - 2, boite: 0,
  }])));
  e.sceau = ["sac", "sac"];
  e.manche = 1;
  e.initiative = 0;
  e.trait = 0;
}

// ------------------------------------------------------------------ composants
function Armee({ j }: { j: number }) {
  const ed = useEd(s => s.ed);
  const regles = useJeu(s => s.regles);
  if (!ed || !regles) return null;
  const autres = new Set(ed.unites[1 - j]);
  return (
    <section className={`joueur armee e${j}`} aria-label={`Armée ${EQUIPE[j]}`}>
      <div className="identite">
        <img src={imgPiece("*", j)} alt="" className="sceau" />
        <div><div className="nom">{EQUIPE[j]}</div><div className="role">{j ? "en haut du plateau" : "en bas du plateau"}</div></div>
      </div>
      <div className="bloc">
        <div className="etiquette">Unités</div>
        <div className="choix-unites">
          {ed.unites[j].map((u, k) => (
            <select key={k} value={u} onChange={e => changerUnite(j, k, e.target.value)} aria-label={`Unité ${k + 1}`}>
              {Object.entries(regles.unites).map(([l, d]) => (
                <option key={l} value={l} disabled={autres.has(l) || (ed.unites[j].includes(l) && l !== u)}>{l} — {d.nom}</option>
              ))}
            </select>
          ))}
        </div>
      </div>
      <table className="comptes">
        <thead>
          <tr>
            <th scope="col">Pièce</th>
            <th scope="col" title="Pièces sur le plateau (cliquer sur le plateau)">Plat.</th>
            {ZONES.map(([z, l, t]) => <th key={z} scope="col" title={t}>{l}</th>)}
            <th scope="col" title="Sac : le reste des pièces (calculé)">Sac</th>
          </tr>
        </thead>
        <tbody>
          {ed.unites[j].map(u => {
            const c = ed.comptes[j][u];
            const s = sac(ed, regles, j, u);
            return (
              <tr key={u}>
                <th scope="row" title={`${regles.unites[u].nom} : ${regles.unites[u].pieces} pièces`}>
                  <span className={`piece petite e${j}`} style={{ backgroundImage: `url(${imgPiece(u, j)})` }}><span className="l">{u}</span></span>
                </th>
                <td className="calc">{surPlateau(ed, j, u)}</td>
                {ZONES.map(([z]) => (
                  <td key={z}>
                    <input type="number" min={0} max={regles.unites[u].pieces} value={c[z]} aria-label={`${NOMS[u]} ${z}`}
                      onChange={e => useEd.getState().maj(x => { x.comptes[j][u][z] = Math.max(0, +e.target.value || 0); })} />
                  </td>
                ))}
                <td className={`calc${s < 0 ? " faux" : ""}`}>{s}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="bloc">
        <div className="etiquette">Sceau royal</div>
        <div className="bascule" role="radiogroup" aria-label="Sceau royal">
          {([["main", "Main"], ["sac", "Sac"], ["cachee", "Défausse cachée"]] as [LieuSceau, string][]).map(([v, l]) => (
            <button key={v} type="button" role="radio" aria-checked={ed.sceau[j] === v} className={ed.sceau[j] === v ? "on" : ""}
              onClick={() => useEd.getState().maj(e => { e.sceau[j] = v; })}>{l}</button>
          ))}
        </div>
      </div>
    </section>
  );
}

function Palette() {
  const ed = useEd(s => s.ed);
  const outil = useEd(s => s.outil);
  if (!ed) return null;
  const actif = (o: Outil) => JSON.stringify(o) === JSON.stringify(outil);
  const bouton = (o: Outil, contenu: React.ReactNode, titre: string) => (
    <button key={JSON.stringify(o)} type="button" className={`outil${actif(o) ? " on" : ""}`} aria-pressed={actif(o)} title={titre} onClick={() => useEd.getState().set({ outil: o })}>
      {contenu}
    </button>
  );
  return (
    <div className="palette" role="toolbar" aria-label="Outils de l'éditeur">
      {[0, 1].map(j => (
        <div key={j} className="rangee">
          <span className="qui">{EQUIPE[j]}</span>
          {ed.unites[j].map(u => bouton({ t: "unite", j, u },
            <span className={`piece e${j}`} style={{ backgroundImage: `url(${imgPiece(u, j)})` }}><span className="l">{u}</span></span>,
            `${NOMS[u]} (${EQUIPE[j]}) : clic = poser ou renforcer, clic droit = retirer une pièce`))}
          {bouton({ t: "controle", e: j }, <img src={`/img/controle-${j ? "noir" : "blanc"}.png`} alt="" />, `Marqueur Contrôle ${EQUIPE[j]} (sur un Lieu)`)}
        </div>
      ))}
      <div className="rangee">
        {bouton({ t: "controle", e: null }, <span>Lieu libre</span>, "Retire le marqueur Contrôle d'un Lieu")}
        {bouton({ t: "gomme" }, <span>Gomme</span>, "Retire une unité (ou un marqueur)")}
      </div>
    </div>
  );
}

export function Editeur() {
  const editeur = useJeu(s => s.editeur);
  const regles = useJeu(s => s.regles);
  const config = useJeu(s => s.config);
  const ed = useEd(s => s.ed);

  useEffect(() => {
    if (!regles) return;
    let e: Ed;
    // pendant le draft, les armées sont incomplètes : on part de la position de départ
    if (editeur && editeur.unites.every(l => l.length === 4)) e = depuisPosition(editeur, regles);
    else {
      e = depuisPosition({ unites: regles.premiere, plateau: [], controle: {}, trait: 0, initiative: 0, manche: 1,
        joueurs: [0, 1].map(() => ({ main: [], sac: [], defausse: [], defausse_cachee: [], reserve: {} })) }, regles);
      positionDepart(e, regles);
    }
    useEd.getState().set({ ed: e, joueurs: unSeulIA(config.joueurs, config.hybride, -1), hybride: config.hybride,
      outil: { t: "unite", j: 0, u: e.unites[0][0] } });
  }, [editeur, regles]); // eslint-disable-line react-hooks/exhaustive-deps

  const decor = useMemo(() => (ed && regles ? decorDe(ed, regles) : null), [ed, regles]);
  const image = useMemo(() => (ed && regles ? imageDe(ed) : null), [ed, regles]);
  if (!ed || !regles || !decor || !image) return <div className="scene"><div className="attente-partie">Chargement de l'éditeur…</div></div>;
  return (
    <div className="scene editeur">
      <div className="colonne-joueur"><Armee j={0} /></div>
      <div className="centre">
        <div className="entete-partie">
          <span className="badge">Éditeur</span>
          <span className="qui">Choisissez un outil, puis cliquez sur les cases ; clic droit pour retirer une pièce.</span>
        </div>
        <div className="table-bois">
          <Plateau decor={decor} image={image} duree={200} onCase={clicCase} />
        </div>
        <Palette />
      </div>
      <div className="colonne-joueur"><Armee j={1} /></div>
    </div>
  );
}

export function PanneauPosition() {
  const ed = useEd(s => s.ed);
  const joueurs = useEd(s => s.joueurs);
  const hybride = useEd(s => s.hybride);
  const regles = useJeu(s => s.regles);
  const info = useJeu(s => s.infoIA);
  const [texte, setTexte] = useState("");
  if (!ed || !regles) return null;
  const maj = useEd.getState().maj;
  const pb = problemes(ed, regles);
  const lancer = async (analyse: boolean) => {
    const pos = versPosition(ed, regles);
    const js: JoueurCfg[] = analyse ? joueurs.map(j => ({ ...j, type: "humain" as const })) : joueurs;
    const ok = await useJeu.getState().nouvelle({ joueurs: js, hybride: hybride && !analyse }, pos);
    if (ok && analyse) {
      useJeu.getState().set({ aides: true });      // demande explicite d'analyse : aides affichées
      if (!useJeu.getState().analyse) useJeu.getState().basculerAnalyse();
    }
  };
  const bascule = (valeur: number, f: (v: number) => void, nom: string) => (
    <div className="bascule" role="radiogroup" aria-label={nom}>
      {[0, 1].map(t => (
        <button key={t} type="button" role="radio" aria-checked={valeur === t} className={valeur === t ? "on" : ""} onClick={() => f(t)}>{EQUIPE[t]}</button>
      ))}
    </div>
  );
  const copier = async () => {
    const t = JSON.stringify(versPosition(ed, regles));
    setTexte(t);
    try { await navigator.clipboard.writeText(t); } catch { /* presse-papiers indisponible : le texte reste affiché */ }
  };
  const coller = () => {
    try {
      const p = JSON.parse(texte) as Position;
      useEd.getState().set({ ed: depuisPosition(p, regles) });
    } catch {
      useJeu.getState().setErreur("Position illisible : collez le texte obtenu avec « Copier »");
    }
  };
  return (
    <aside className="cote position-p">
      <h2>Position</h2>
      <div className="grille-reglages">
        <span>Au trait</span>{bascule(ed.trait, v => maj(e => { e.trait = v; }), "Au trait")}
        <span>Initiative</span>{bascule(ed.initiative, v => maj(e => { e.initiative = v; }), "Initiative")}
        <label htmlFor="manche-ed">Manche</label>
        <input id="manche-ed" type="number" min={1} value={ed.manche} onChange={e => maj(x => { x.manche = Math.max(1, +e.target.value || 1); })} />
      </div>
      {pb.length > 0 ? (
        <ul className="problemes">{pb.map(p => <li key={p}>{p}</li>)}</ul>
      ) : <p className="ok-position">Position cohérente.</p>}
      <label className="case-a-cocher" title={info?.disponible ? "Une IA et un joueur plateau ; pièces cachées du joueur plateau : seul compte leur total par unité" : "Il faut une IA entraînée"}>
        <input type="checkbox" checked={hybride} disabled={!info?.disponible}
          onChange={e => useEd.getState().set({ hybride: e.target.checked, joueurs: unSeulIA(joueurs, e.target.checked, -1) })} />
        Partie sur plateau réel (hybride)
      </label>
      {hybride && (
        <p className="remarque">Joueur plateau : sa main, son sac et sa défausse cachée sont inconnus de l'IA ; seul compte le
          nombre de pièces de chaque unité et de chaque zone.</p>
      )}
      <div className="deux-joueurs compact">
        {joueurs.map((j, k) => (
          <ChoixJoueur key={k} equipe={k} j={j} info={info} niveaux={regles.niveaux} durees={regles.durees} hybride={hybride}
            onChange={nj => useEd.getState().set({ joueurs: unSeulIA(joueurs.map((x, i) => (i === k ? nj : x)), hybride, k) })} />
        ))}
      </div>
      <div className="actions-position">
        <button type="button" className="on" disabled={pb.length > 0} onClick={() => lancer(false)}>Jouer à partir d'ici</button>
        <button type="button" disabled={pb.length > 0} onClick={() => lancer(true)} title="Humain contre humain, analyse activée">Analyser la position</button>
        <button type="button" onClick={() => maj(e => positionDepart(e, regles))}>Position de départ</button>
        <button type="button" onClick={() => maj(e => { e.plateau = {}; })}>Vider le plateau</button>
      </div>
      <details className="echange">
        <summary>Copier / coller une position</summary>
        <textarea value={texte} onChange={e => setTexte(e.target.value)} rows={4} spellCheck={false}
          aria-label="Position au format texte" placeholder="Position au format JSON" />
        <div className="actions-position">
          <button type="button" onClick={copier}>Copier</button>
          <button type="button" onClick={coller} disabled={!texte}>Coller</button>
        </div>
      </details>
    </aside>
  );
}
