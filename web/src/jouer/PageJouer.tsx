import { useEffect, useMemo, useRef } from "react";
import { Chronique } from "../components/Chronique";
import { Joueur, Piece } from "../components/Joueur";
import { Plateau } from "../components/Plateau";
import { EQUIPE, NOMS } from "../jeu";
import { Analyse, BarreEval, CourbeEval, libelleScore } from "./Analyse";
import { Editeur, PanneauPosition } from "./Editeur";
import { DialogueNouvelle } from "./Nouvelle";
import { DialogueParties } from "./Parties";
import { sauverPrefs, useAides, useAnalyseActive, useImageAffichee, useJeu, type Pilotage } from "./store";
import type { Legal } from "./types";

const ATTAQUES = new Set(["attack"]);

/** Nom d'une pièce ; « ? » : pièce jouée face cachée par le joueur plateau (partie hybride). */
const nomPiece = (c: string) => (c === "?" ? "Pièce cachée" : NOMS[c]);

/** Coups jouables affichés : filtrés par la pièce et la case choisies. */
export function useCoups() {
  const humain = useJeu(s => s.humain);
  const legal = useJeu(s => s.legal);
  const pos = useJeu(s => s.pos);
  const n = useJeu(s => s.images.length);
  const piece = useJeu(s => s.piece);
  const caseChoisie = useJeu(s => s.caseChoisie);
  return useMemo(() => {
    const tous = humain && pos === n - 1 ? legal : [];
    const filtres = tous.filter(a => (piece === null || a.coin === piece) && (caseChoisie === null || a.cells.includes(caseChoisie)));
    return { tous, filtres };
  }, [humain, legal, pos, n, piece, caseChoisie]);
}

/** L'IA joue quand elle a le trait, selon le pilotage choisi (IA contre IA : pas en pause). */
function usePilote() {
  const ia = useJeu(s => s.ia);
  const n = useJeu(s => s.images.length);
  const occupe = useJeu(s => s.occupe);
  const lecture = useJeu(s => s.lecture);
  const vitesse = useJeu(s => s.vitesse);
  const mode = useJeu(s => s.mode);
  const joueurs = useJeu(s => s.joueurs);
  const pos = useJeu(s => s.pos);
  const pilotage = useJeu(s => s.pilotage);
  const panne = useJeu(s => s.panne);
  useEffect(() => {
    if (!ia || occupe || panne || mode !== "jeu" || pilotage === "demande") return;
    const seule = joueurs.every(j => j.type === "ia");
    if (seule && !lecture) return;
    // « affichage » : l'IA attend qu'on revienne à la position actuelle
    if (pilotage === "affichage" && pos < n - 1) return;
    const t = setTimeout(() => useJeu.getState().coupIA(), seule ? vitesse : 450);
    return () => clearTimeout(t);
  }, [ia, n, occupe, panne, lecture, vitesse, mode, joueurs, pos, pilotage]);
}

/** Analyse automatique de la position affichée. */
function useAnalyseAuto() {
  const analyse = useAnalyseActive();
  const pos = useJeu(s => s.pos);
  const id = useJeu(s => s.id);
  const fait = useJeu(s => !!s.analyses[s.pos]);
  const enAnalyse = useJeu(s => s.enAnalyse);
  const mode = useJeu(s => s.mode);
  useEffect(() => {
    if (!analyse || !id || fait || enAnalyse !== null || mode !== "jeu") return;
    const t = setTimeout(() => useJeu.getState().analyser(pos), 150);
    return () => clearTimeout(t);
  }, [analyse, pos, id, fait, enAnalyse, mode]);
}

const aidesVisibles = () => {
  const s = useJeu.getState();
  return s.aides || !s.joueurs.some(j => j.type === "humain");
};

function useClavier() {
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const cible = (document.activeElement as HTMLElement)?.tagName;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(cible) || (e.key === " " && cible === "BUTTON")) return;
      const s = useJeu.getState();
      if (s.dialogue || s.historique) return;
      if (s.mode !== "jeu") { if (e.key.toLowerCase() === "e") s.set({ mode: "jeu" }); return; }
      const actions: Record<string, () => void> = {
        ArrowLeft: () => s.pas(-1),
        ArrowRight: () => s.pas(1),
        Home: () => s.aller(0),
        End: () => s.aller(s.images.length - 1),
        Escape: () => s.set({ piece: null, caseChoisie: null }),
        a: () => { if (aidesVisibles() && !s.hybride) s.basculerAnalyse(); },
        " ": () => { if (s.ia && !s.occupe) void s.coupIA(); },
        n: () => s.set({ dialogue: true }),
        t: s.basculerRetourne,
        e: () => void s.ouvrirEditeur(),
        f: () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()),
      };
      const a = actions[e.key] ?? actions[e.key.toLowerCase()];
      if (a) { e.preventDefault(); a(); }
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);
}

// ------------------------------------------------------------------ barre du haut
function Statut() {
  const fini = useJeu(s => s.fini);
  const resultat = useJeu(s => s.resultat);
  const trait = useJeu(s => s.trait);
  const joueurs = useJeu(s => s.joueurs);
  const humain = useJeu(s => s.humain);
  const images = useJeu(s => s.images);
  const pos = useJeu(s => s.pos);
  const attente = useJeu(s => s.attente);
  const lecture = useJeu(s => s.lecture);
  const tirage = useJeu(s => s.tirage);
  const hybride = useJeu(s => s.hybride);
  const img = images[pos];
  if (!img) return <div className="phase"><span className="etiquette">Préparation…</span></div>;
  if (tirage && pos === images.length - 1) {
    const titre = { initial: "Première main de l'IA", manche: "Pioche de l'IA", moine: "Moine soldat de l'IA" }[tirage.contexte];
    return (
      <div className="phase statut humain">
        <span className="etiquette">{titre}</span>
        <span className="detail">pièce {tirage.rang} / {tirage.total} : saisissez la pièce tirée de son sac</span>
      </div>
    );
  }
  const pause = !lecture && joueurs.every(j => j.type === "ia");
  const passe = pos < images.length - 1;
  const t = img.t;
  let etiquette: string, detail: string, cls: string;
  if (passe) {
    etiquette = `Historique · décision ${pos}`;
    detail = `manche ${img.r} · ${images.length - 1 - pos} décision(s) plus loin`;
    cls = "passe";
  } else if (fini) {
    etiquette = resultat === "1-0" ? "Victoire d'Or" : resultat === "0-1" ? "Victoire d'Argent" : "Partie nulle";
    detail = `en ${img.r} manches`;
    cls = "fin";
  } else {
    const qui = EQUIPE[trait];
    etiquette = humain
      ? (hybride ? "Joueur plateau : saisissez son coup"
        : joueurs.filter(j => j.type === "humain").length > 1 ? `${qui} : à vous de jouer` : "À vous de jouer")
      : pause ? "En pause" : `${qui} réfléchit…`;
    detail = attente || `manche ${img.r} · ${qui} au trait`;
    cls = humain ? "humain" : "ia";
  }
  return (
    <div className={`phase statut ${cls} e${t}`}>
      <span className="etiquette">{etiquette}</span>
      <span className="detail">{detail}</span>
    </div>
  );
}

function KpiEval() {
  const analyse = useAnalyseActive();
  const a = useJeu(s => s.analyses[s.pos]);
  const enAnalyse = useJeu(s => s.enAnalyse !== null);
  if (!analyse) return null;
  const txt = a ? libelleScore(a) : null;
  return (
    <div className="kpi eval" title="10 = un bastion d'avance · + Or, − Argent · #n = victoire forcée en n coups">
      <div className="label">Évaluation{enAnalyse ? " · calcul…" : ""}</div>
      <div className={`val ${txt?.cls ?? ""}`}>{txt?.texte ?? "…"}</div>
      <div className="det">{txt?.detail ?? "analyse en cours"}</div>
    </div>
  );
}

function KpiBastions() {
  const img = useImageAffichee();
  if (!img) return null;
  const b = [0, 1].map(e => img.c.filter(([, t]) => t === e).length);
  return (
    <div className="kpi" title="Lieux contrôlés ; le premier à poser ses 6 marqueurs gagne">
      <div className="label">Bastions Or – Argent</div>
      <div className="val">{b[0]} – {b[1]}</div>
      <div className="det">à poser : Or {img.m[0]} · Argent {img.m[1]}</div>
    </div>
  );
}

/** Aides à la décision (partie avec au moins un humain) : masquées par défaut. */
function BasculeAides() {
  const humains = useJeu(s => s.joueurs.some(j => j.type === "humain"));
  const aides = useJeu(s => s.aides);
  if (!humains) return null;
  const set = (v: boolean) => useJeu.getState().set(v ? { aides: true } : { aides: false, onglet: "coups" });
  return (
    <div className="bascule aides" role="radiogroup" aria-label="Aides de jeu"
      title="Aides à la décision : évaluation, meilleurs coups et flèche, victoires forcées, valeur des cartes du draft, main de l'IA">
      <span className="lib">Aides</span>
      <button type="button" role="radio" aria-checked={!aides} className={aides ? "" : "on"} onClick={() => set(false)}>Masquées</button>
      <button type="button" role="radio" aria-checked={aides} className={aides ? "on" : ""} onClick={() => set(true)}>Affichées</button>
    </div>
  );
}

function BarreJeu() {
  const analyse = useJeu(s => s.analyse);
  const aides = useAides();
  const mode = useJeu(s => s.mode);
  const hybride = useJeu(s => s.hybride !== null);
  const s = useJeu.getState();
  return (
    <header className="barre">
      <div className="marque">
        <img src="/img/embleme.png" alt="" />
        <div>
          <h1>Champ d'honneur</h1>
          <nav className="bascule nav-pages" aria-label="Pages">
            <a className="bouton" href="/">Direct</a>
            <a className="bouton on" href="/jouer" aria-current="page">Jouer</a>
          </nav>
        </div>
      </div>
      {mode === "jeu" ? <Statut /> : (
        <div className="phase statut editeur"><span className="etiquette">Éditeur de position</span>
          <span className="detail">clic : poser · clic droit : retirer une pièce</span></div>
      )}
      <div className="kpis">
        {mode === "jeu" && <><KpiBastions /><KpiEval /></>}
      </div>
      <div className="commandes">
        <button type="button" className="on" onClick={() => s.set({ dialogue: true })} title="Nouvelle partie : draft, libre ou position (N)">Nouvelle partie</button>
        <button type="button" onClick={() => { s.set({ historique: true }); void s.chargerParties(); }}
          title="Parties jouées et en cours : reprendre ou revoir">Parties</button>
        {mode === "jeu" && <BasculeAides />}
        {mode === "jeu" ? (aides && !hybride && (
          <button type="button" className={analyse ? "on" : ""} onClick={s.basculerAnalyse} aria-pressed={analyse}
            title="Analyse de la position affichée par le réseau entraîné (A)">Analyse</button>
        )) : (
          <button type="button" onClick={() => s.set({ mode: "jeu" })} title="Quitter l'éditeur (E)">Retour à la partie</button>
        )}
      </div>
    </header>
  );
}

// ------------------------------------------------------------------ scène
function EnTete() {
  const mise = useJeu(s => s.mise);
  const vivant = useJeu(s => s.pos === s.images.length - 1);
  const retourne = useJeu(s => s.retourne);
  if (!mise) return null;
  return (
    <div className="entete-partie">
      <span className={`badge${vivant ? " vivant" : ""}`}>{vivant ? "Partie" : "Historique"}</span>
      <span className="qui">{mise.libelle}</span>
      <button type="button" className={`retourner${retourne ? " on" : ""}`} aria-pressed={retourne}
        onClick={useJeu.getState().basculerRetourne} title="Retourner le plateau : Or ou Argent en bas (T)">
        ⇅ {retourne ? "Argent en bas" : "Or en bas"}
      </button>
    </div>
  );
}

/** Partie hybride : décisions de l'IA depuis la dernière saisie, à reproduire sur le plateau. */
function useCoupsIA(): string[] {
  const images = useJeu(s => s.images);
  const hybride = useJeu(s => s.hybride);
  const decor = useJeu(s => s.decor);
  return useMemo(() => {
    if (!hybride || !decor) return [];
    const out: string[] = [];
    for (let k = images.length - 1; k > 0; k--) {
      const a = images[k].a;
      // un coup suivi d'une pioche de l'IA a déjà été annoncé dans le panneau de pioche
      if (!a || decor.equipes[a.j] !== decor.equipes[hybride.ia] || a.ti) break;
      if (a.k !== "skip") out.unshift(a.d);
    }
    return out;
  }, [images, hybride, decor]);
}

function SousPlateau() {
  const img = useImageAffichee();
  const decor = useJeu(s => s.decor);
  const humain = useJeu(s => s.humain);
  const piece = useJeu(s => s.piece);
  const vivant = useJeu(s => s.pos === s.images.length - 1);
  const attente = useJeu(s => s.attente);
  const tirage = useJeu(s => s.tirage);
  const hybride = useJeu(s => s.hybride);
  const coupsIA = useCoupsIA();
  if (!img || !decor) return <div className="sous-plateau" />;
  let texte;
  if (vivant && tirage) {
    texte = tirage.coup && tirage.coup.j === hybride?.ia
      ? <span className="a-reproduire">À reproduire pour l'IA : <b>{tirage.coup.d}</b>, puis piochez dans son sac (à droite)</span>
      : <span className="attente-decision">Piochez dans le sac de l'IA et saisissez les pièces à droite</span>;
  } else if (vivant && humain && hybride && coupsIA.length) {
    texte = <span className="a-reproduire">À reproduire pour l'IA : <b>{coupsIA.join(" · ")}</b>, puis saisissez le coup du joueur plateau</span>;
  } else if (vivant && humain && attente) texte = <span className="attente-decision">{attente}</span>;
  else if (vivant && humain) {
    texte = <span className="attente-decision">
      {img.tir ? (hybride ? "Draft : saisissez la carte choisie par le joueur plateau" : "Draft : choisissez une carte Unité dans votre colonne")
        : hybride ? (piece ? `${nomPiece(piece)} : choisissez une case en surbrillance, ou un coup dans la liste` : "Choisissez à droite la pièce jouée sur la table, puis la case")
        : piece ? `${NOMS[piece]} : choisissez une case en surbrillance, ou un coup dans la liste` : "Choisissez une pièce de votre main, ou une unité sur le plateau"}
    </span>;
  } else if (img.a) texte = <span className="dernier">{EQUIPE[decor.equipes[img.a.j]]} : {img.a.d}</span>;
  else texte = <span className="dernier">Mise en place</span>;
  return <div className="sous-plateau">{texte}</div>;
}

function Resultat() {
  const img = useImageAffichee();
  if (!img?.f) return null;
  return (
    <div className={`resultat e${img.f.g ?? "n"}`} role="status">
      {img.f.g !== null && <img src={`/img/controle-${img.f.g ? "noir" : "blanc"}.png`} alt="" />}
      <span>{img.f.g === null ? "Partie nulle" : `Victoire ${img.f.g ? "d'Argent" : "d'Or"}`}</span>
      <small>en {img.r} manches</small>
    </div>
  );
}

function PlateauJeu() {
  const decor = useJeu(s => s.decor);
  const images = useJeu(s => s.images);
  const pos = useJeu(s => s.pos);
  const piece = useJeu(s => s.piece);
  const caseChoisie = useJeu(s => s.caseChoisie);
  const survol = useJeu(s => s.survol);
  const attente = useJeu(s => s.attente);
  const analyse = useAnalyseActive();
  const fleche = useJeu(s => s.fleche);
  const a = useJeu(s => s.analyses[s.pos]);
  const { tous, filtres } = useCoups();
  const retourne = useJeu(s => s.retourne);
  const img = useImageAffichee();
  const apercu = img !== images[pos];   // hybride : coup de l'IA joué, pioche en attente

  const { cibles, cliquables } = useMemo(() => {
    const cibles = new Map<number, "attaque" | "jouable">();
    const cliquables = new Set<number>();
    if (!img) return { cibles, cliquables };
    const occupant = new Map(img.u.map(u => [u[1], u[2]]));
    for (const l of tous) l.cells.forEach(c => cliquables.add(c));
    if (piece !== null || caseChoisie !== null || attente) {
      for (const l of filtres) {
        const c = l.cells[l.cells.length - 1];
        if (c === undefined) continue;
        const ennemi = occupant.has(c) && decor && decor.equipes[occupant.get(c)!] !== decor.equipes[img.t];
        cibles.set(c, ATTAQUES.has(l.kind) || (l.kind === "tactic" && ennemi) ? "attaque" : "jouable");
      }
    }
    return { cibles, cliquables };
  }, [img, tous, filtres, piece, caseChoisie, attente, decor]);

  if (!decor || !img) return null;
  const onCase = (i: number) => {
    const s = useJeu.getState();
    const arrivee = filtres.filter(l => l.cells[l.cells.length - 1] === i);
    if ((piece !== null || caseChoisie !== null || attente) && arrivee.length === 1) return void s.jouer(arrivee[0].i);
    if (arrivee.length > 1 && (piece !== null || caseChoisie !== null)) return s.set({ caseChoisie: i, onglet: "coups" });
    s.choisirCase(i);
    s.set({ onglet: "coups" });
  };
  const conseil = analyse && fleche && !survol && a?.coups?.[0]?.cases?.length ? a.coups[0].cases : null;
  return (
    <Plateau
      key={decor.graine + ":" + decor.unites.join()} decor={decor} image={img}
      precedente={apercu ? images[pos] : pos > 0 ? images[pos - 1] : null}
      duree={380} cle={apercu ? pos + 1 : pos} cibles={cibles} choisie={caseChoisie} survol={survol} conseil={conseil}
      onCase={tous.length ? onCase : undefined} cliquables={cliquables} retourne={retourne}
    />
  );
}

const PILOTAGES: [Pilotage, string, string][] = [
  ["auto", "IA : dès son tour", "L'IA joue dès que c'est son tour, même si vous regardez un coup passé : la partie continue pendant que vous analysez"],
  ["affichage", "IA : position actuelle", "L'IA attend que la position actuelle soit affichée : revenir sur un coup passé met la partie en attente"],
  ["demande", "IA : sur demande", "L'IA ne joue que lorsque vous cliquez « Coup de l'IA » (ou Espace)"],
];

function Navigation() {
  const n = useJeu(s => s.images.length - 1);
  const pos = useJeu(s => s.pos);
  const img = useJeu(s => s.images[s.pos]);
  const joueurs = useJeu(s => s.joueurs);
  const lecture = useJeu(s => s.lecture);
  const vitesse = useJeu(s => s.vitesse);
  const fini = useJeu(s => s.fini);
  const hybride = useJeu(s => s.hybride);
  const pilotage = useJeu(s => s.pilotage);
  const ia = useJeu(s => s.ia);
  const occupe = useJeu(s => s.occupe);
  const panne = useJeu(s => s.panne);
  const s = useJeu.getState();
  if (!img) return null;
  const seule = joueurs.every(j => j.type === "ia");
  const humains = joueurs.some(j => j.type === "humain");
  const avecIA = joueurs.some(j => j.type === "ia");
  // l'IA a le trait mais le pilote ne la fera pas jouer : on peut la faire jouer à la main
  const enAttente = ia && !fini && (pilotage === "demande" || (seule && !lecture) || (pilotage === "affichage" && pos < n) || panne);
  return (
    <div className="lecteur">
      <div className="boutons">
        <button type="button" onClick={() => s.aller(0)} title="Début (Origine)" aria-label="Début">⏮</button>
        <button type="button" onClick={() => s.pas(-1)} title="Décision précédente (←)" aria-label="Décision précédente">◀</button>
        {seule && !fini && pilotage !== "demande" && (
          <button type="button" className="principal"
            onClick={() => s.set({ lecture: !lecture, panne: false, ...(pilotage === "affichage" && !lecture ? { pos: n } : {}) })}>
            {lecture ? "Pause" : "Lecture"}
          </button>
        )}
        <button type="button" onClick={() => s.pas(1)} title="Décision suivante (→)" aria-label="Décision suivante">▶</button>
        <button type="button" onClick={() => s.aller(n)} title="Position actuelle (Fin)" aria-label="Position actuelle">⏭</button>
      </div>
      <div className="temps">
        <input
          type="range" min={0} max={n} value={pos} aria-label="Position dans la partie"
          onChange={e => s.aller(+e.target.value)}
          style={{ "--p": `${n ? (100 * pos) / n : 0}%` } as React.CSSProperties}
        />
        <div className="compteurs">
          <span>Manche <b>{img.r}</b></span>
          <span>Décision <b>{pos}</b> / {n}</span>
        </div>
      </div>
      {avecIA && !fini && (
        <div className="pilotage">
          <select value={pilotage} onChange={e => s.setPilotage(e.target.value as Pilotage)} aria-label="Quand l'IA joue"
            title={PILOTAGES.find(p => p[0] === pilotage)?.[2]}>
            {PILOTAGES.map(([v, l, t]) => <option key={v} value={v} title={t}>{l}</option>)}
          </select>
          {seule && pilotage !== "demande" && (
            <select value={vitesse} onChange={e => { s.set({ vitesse: +e.target.value }); sauverPrefs(); }} aria-label="Vitesse de l'IA">
              {[[1500, "Lent"], [700, "Normal"], [250, "Rapide"], [30, "Éclair"]].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          )}
          {enAttente && (
            <button type="button" className="on" disabled={occupe} onClick={() => { s.set({ panne: false }); void s.coupIA(); }}
              title="Faire jouer à l'IA sa prochaine décision (Espace)">Coup de l'IA</button>
          )}
        </div>
      )}
      {pos < n ? (
        <button type="button" className="reprendre" onClick={() => s.revenir(pos)}
          title={fini ? "Partie terminée : rejouer depuis la position affichée dans une nouvelle partie (variante)"
            : "Effacer la suite et reprendre la partie depuis la position affichée"}>{fini ? "Variante d'ici" : "Reprendre d'ici"}</button>
      ) : humains && !fini ? (
        hybride ? (
          <button type="button" onClick={s.annuler} title="Annuler la dernière saisie : coup du joueur plateau (et réponse de l'IA), ou pioche en cours">Annuler la saisie</button>
        ) : (
          <button type="button" onClick={s.annuler} disabled={n === 0} title="Annuler votre dernier coup (et la réponse de l'IA)">Annuler mon coup</button>
        )
      ) : null}
    </div>
  );
}

function Scene() {
  const decor = useJeu(s => s.decor);
  const img = useImageAffichee();
  const retourne = useJeu(s => s.retourne);
  const joueurs = useJeu(s => s.joueurs);
  const piece = useJeu(s => s.piece);
  const vivant = useJeu(s => s.pos === s.images.length - 1);
  const humain = useJeu(s => s.humain);
  const attente = useJeu(s => s.attente);
  const pieceAttente = useJeu(s => s.pieceAttente);
  const conseil = useJeu(s => s.conseil);
  const mainsVisibles = useJeu(s => s.mainsVisibles);
  const fini = useJeu(s => s.fini);
  const hybride = useJeu(s => s.hybride);
  const aides = useAides();
  if (!decor || !img) return <div className="scene"><div className="attente-partie">Préparation de la partie…</div></div>;
  const colonne = (j: number) => {
    const actif = vivant && humain && img.t === j && !attente;
    // un seul humain : la main de l'IA est cachée, sauf à la révéler (l'analyse devient omnisciente)
    const mainIA = aides && !fini && !hybride && joueurs[j]?.type === "ia" && joueurs.filter(x => x.type === "humain").length === 1;
    return (
      <div className="colonne-joueur">
        <Joueur decor={decor} image={img} joueur={j} nom="" role={joueurs[j]?.nom}
          onPiece={actif ? c => {
            const st = useJeu.getState();
            // mise en place avancée : un clic sur une carte la choisit
            const carte = img.tir ? st.legal.find(l => l.kind === "draft" && l.n === c) : undefined;
            if (carte) st.jouer(carte.i);
            else st.choisirPiece(c);
          } : undefined} choisie={actif ? piece : null}
          attente={vivant && img.t === j && pieceAttente ? pieceAttente : null}
          conseil={aides && vivant && img.tir && conseil?.joueur === j ? conseil : null}
          mains={mainIA ? { visibles: mainsVisibles, basculer: () => useJeu.getState().reglages({ mains_visibles: !mainsVisibles }) } : undefined} />
      </div>
    );
  };
  return (
    <div className="scene">
      {colonne(retourne ? 1 : 0)}
      <div className="centre">
        <EnTete />
        <div className="table-bois avec-barre">
          <BarreEval />
          <PlateauJeu />
          <Resultat />
        </div>
        <SousPlateau />
        <Navigation />
      </div>
      {colonne(retourne ? 0 : 1)}
    </div>
  );
}

// ------------------------------------------------------------------ colonne de droite
/** Partie hybride : saisie, pièce par pièce, de la pioche de l'IA. */
function PanneauPioche() {
  const t = useJeu(s => s.tirage)!;
  const hybride = useJeu(s => s.hybride)!;
  const decor = useJeu(s => s.decor);
  const occupe = useJeu(s => s.occupe);
  const s = useJeu.getState();
  const e = decor ? decor.equipes[hybride.ia] : hybride.ia;
  const titre = { initial: "Première main de l'IA", manche: "Début de manche : pioche de l'IA", moine: "Moine soldat de l'IA : pioche d'une pièce" }[t.contexte];
  return (
    <div className="pioche-ia-p" aria-live="polite">
      <h3>{titre}</h3>
      {t.coup && (
        <p className={`coup-declencheur${t.coup.j === hybride.ia ? " ia" : ""}`}>
          {t.coup.j === hybride.ia ? <>L'IA joue <code>{t.coup.n}</code> {t.coup.d} : reproduisez-le sur le plateau.</>
            : <>Coup saisi : <code>{t.coup.n}</code> {t.coup.d}</>}
        </p>
      )}
      {t.melange && <p className="alerte-sac">Sac de l'IA vide : remettez sa défausse dans le sac et mélangez.</p>}
      <p>Pièce <b>{t.rang}</b> sur {t.total} : cliquez la pièce tirée du sac de l'IA.</p>
      <div className="sac-ia" role="group" aria-label="Pièces du sac de l'IA">
        {Object.entries(t.sac).sort().map(([c, n]) => (
          <button key={c} type="button" disabled={occupe} onClick={() => s.piocher(c)} title={`${NOMS[c]} : ${n} dans le sac`}>
            <Piece c={c} equipe={e} />
            <span className="nb">×{n}</span>
            <span className="nom">{NOMS[c]}</span>
          </button>
        ))}
      </div>
      <div className="deja">
        <span>Déjà saisies :</span>
        {t.choisies.length ? t.choisies.map((c, k) => <Piece key={k} c={c} equipe={e} petite />) : <span className="vide">aucune</span>}
        <button type="button" onClick={s.recommencerPioche} disabled={!t.choisies.length}>Recommencer la pioche</button>
      </div>
    </div>
  );
}

/** Partie hybride : pièce jouée par le joueur plateau (parmi celles qu'il pourrait avoir). */
function PiecesPossibles() {
  const possibles = useJeu(s => s.possibles);
  const piece = useJeu(s => s.piece);
  const decor = useJeu(s => s.decor);
  const hybride = useJeu(s => s.hybride);
  const { tous } = useCoups();
  if (!possibles || !hybride) return null;
  const e = decor ? decor.equipes[hybride.plateau] : hybride.plateau;
  const jouables = new Set(tous.map(l => l.coin));
  const s = useJeu.getState();
  const choix = Object.keys(possibles).filter(c => jouables.has(c)).sort();
  return (
    <div className="pieces-possibles" role="group" aria-label="Pièce jouée par le joueur plateau">
      <span className="etiquette">Pièce jouée</span>
      {choix.map(c => (
        <button key={c} type="button" className={`bouton-piece${piece === c ? " choisie" : ""}`} aria-pressed={piece === c}
          onClick={() => s.choisirPiece(c)} title={`${NOMS[c]} (face visible) : ${possibles[c]} peut-être en main`}>
          <Piece c={c} equipe={e} />
        </button>
      ))}
      {jouables.has("?") && (
        <button type="button" className={`bouton-piece${piece === "?" ? " choisie" : ""}`} aria-pressed={piece === "?"}
          onClick={() => s.choisirPiece("?")} title="Face cachée : passer, recruter, prendre l'Initiative (pièce non saisie)">
          <Piece equipe={e} cachee />
        </button>
      )}
    </div>
  );
}

function ListeCoups() {
  const humain = useJeu(s => s.humain);
  const vivant = useJeu(s => s.pos === s.images.length - 1);
  const fini = useJeu(s => s.fini);
  const piece = useJeu(s => s.piece);
  const caseChoisie = useJeu(s => s.caseChoisie);
  const decor = useJeu(s => s.decor);
  const attente = useJeu(s => s.attente);
  const { tous, filtres } = useCoups();
  const tirage = useJeu(s => s.tirage);
  const hybride = useJeu(s => s.hybride);
  const s = useJeu.getState();
  if (vivant && tirage) return <PanneauPioche />;
  if (!vivant) {
    return <p className="vide">Vous consultez l'historique. <button type="button" onClick={() => s.aller(s.images.length - 1)}>Revenir à la position actuelle</button></p>;
  }
  if (fini) return <p className="vide">Partie terminée. Lancez une nouvelle partie ou parcourez le déroulé (← →) avec l'analyse.</p>;
  if (!humain) return <p className="vide">L'IA réfléchit…</p>;
  const invite = hybride ? "Saisissez le coup joué sur la table" : "";
  const nom = (i: number | null) => (i === null || !decor ? "" : decor.cases[i][0]);
  const liste = piece !== null || caseChoisie !== null ? filtres : tous;
  return (
    <>
      {invite && <p className="invite-saisie">{invite}</p>}
      <PiecesPossibles />
      <div className="filtre-coups">
        {attente ? <span>{attente}</span> : (
          <span>
            {piece ? <>Pièce <b>{nomPiece(piece)}</b></> : "Toutes les pièces"}
            {caseChoisie !== null && <> · case <b>{nom(caseChoisie)}</b></>}
            {" "}· {liste.length} coup{liste.length > 1 ? "s" : ""}
          </span>
        )}
        {(piece !== null || caseChoisie !== null) && (
          <button type="button" onClick={() => s.set({ piece: null, caseChoisie: null })} title="Échap">Tout afficher</button>
        )}
      </div>
      <ol className="coups">
        {liste.map((l: Legal) => (
          <li key={l.i}>
            <button type="button" onClick={() => s.jouer(l.i)}
              onMouseEnter={() => s.setSurvol(l.cells.length ? l.cells : null)} onMouseLeave={() => s.setSurvol(null)}
              onFocus={() => s.setSurvol(l.cells.length ? l.cells : null)} onBlur={() => s.setSurvol(null)}>
              <code>{l.n}</code><span>{l.d}</span>
            </button>
          </li>
        ))}
      </ol>
    </>
  );
}

function PanneauCote() {
  const onglet0 = useJeu(s => s.onglet);
  const hybride = useJeu(s => s.hybride);
  const aides = useAides() || hybride !== null;   // hybride : l'analyse vue par l'IA reste visible
  const onglet = aides ? onglet0 : "coups";
  const n = useCoups().tous.length;
  const s = useJeu.getState();
  return (
    <aside className="cote">
      <div className="onglets" role="tablist">
        <button role="tab" aria-selected={onglet === "coups"} className={onglet === "coups" ? "on" : ""} onClick={() => s.set({ onglet: "coups" })}>
          {hybride ? "Saisie" : "Vos coups"} <span className="n">{n || ""}</span>
        </button>
        {aides && (
          <button role="tab" aria-selected={onglet === "analyse"} className={onglet === "analyse" ? "on" : ""} onClick={() => s.set({ onglet: "analyse" })}>
            Analyse
          </button>
        )}
      </div>
      {onglet === "coups" ? <ListeCoups /> : <Analyse />}
    </aside>
  );
}

function DerouleJeu() {
  const decor = useJeu(s => s.decor);
  const images = useJeu(s => s.images);
  const pos = useJeu(s => s.pos);
  const film = useMemo(() => (decor ? { decor, images } : null), [decor, images]);
  return (
    <section className="panneau chronique-p" aria-label="Déroulé">
      <h2>Déroulé</h2>
      <Chronique film={film} pos={pos} aller={k => useJeu.getState().aller(k)} complet />
    </section>
  );
}

export function PageJouer() {
  const mode = useJeu(s => s.mode);
  const erreur = useJeu(s => s.erreur);
  const info = useJeu(s => s.info);
  const analyse = useAnalyseActive();
  const demarre = useRef(false);
  usePilote();
  useAnalyseAuto();
  useClavier();
  useEffect(() => {
    if (demarre.current) return;
    demarre.current = true;
    document.title = "Champ d'honneur — jouer";
    useJeu.getState().demarrer();
  }, []);
  useEffect(() => {
    if (!erreur && !info) return;
    const t = setTimeout(() => useJeu.getState().set({ erreur: null, info: null }), 5000);
    return () => clearTimeout(t);
  }, [erreur, info]);
  return (
    <div className={`ecran jouer${mode === "editeur" ? " edition" : ""}`}>
      <BarreJeu />
      <main className="milieu">
        {mode === "editeur" ? <Editeur /> : <Scene />}
        <div className="droite">
          {mode === "editeur" ? <PanneauPosition /> : <><DerouleJeu /><PanneauCote /></>}
        </div>
      </main>
      {mode === "jeu" && analyse && <footer className="bas"><CourbeEval /></footer>}
      <DialogueNouvelle />
      <DialogueParties />
      {erreur ? <div className="toast" role="alert">{erreur}</div> : info && <div className="toast info" role="status">{info}</div>}
    </div>
  );
}
