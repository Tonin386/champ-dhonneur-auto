import { useEffect, useMemo, useRef } from "react";
import { Chronique } from "../components/Chronique";
import { Joueur } from "../components/Joueur";
import { Plateau } from "../components/Plateau";
import { EQUIPE, NOMS } from "../jeu";
import { Analyse, BarreEval, CourbeEval, libelleScore } from "./Analyse";
import { Editeur, PanneauPosition } from "./Editeur";
import { DialogueNouvelle } from "./Nouvelle";
import { useJeu } from "./store";
import type { Legal } from "./types";

const ATTAQUES = new Set(["attack"]);

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

/** L'IA joue quand elle a le trait (IA contre IA : seulement en lecture). */
function usePilote() {
  const ia = useJeu(s => s.ia);
  const n = useJeu(s => s.images.length);
  const occupe = useJeu(s => s.occupe);
  const lecture = useJeu(s => s.lecture);
  const vitesse = useJeu(s => s.vitesse);
  const mode = useJeu(s => s.mode);
  const joueurs = useJeu(s => s.joueurs);
  const pos = useJeu(s => s.pos);
  useEffect(() => {
    if (!ia || occupe || mode !== "jeu") return;
    const seule = joueurs.every(j => j.type === "ia");
    if (seule && (!lecture || pos < n - 1)) return;
    const t = setTimeout(() => useJeu.getState().coupIA(), seule ? vitesse : 450);
    return () => clearTimeout(t);
  }, [ia, n, occupe, lecture, vitesse, mode, joueurs, pos]);
}

/** Analyse automatique de la position affichée. */
function useAnalyseAuto() {
  const analyse = useJeu(s => s.analyse);
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

function useClavier() {
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (["INPUT", "SELECT", "TEXTAREA"].includes((document.activeElement as HTMLElement)?.tagName)) return;
      const s = useJeu.getState();
      if (s.mode !== "jeu" || s.dialogue) return;
      const actions: Record<string, () => void> = {
        ArrowLeft: () => s.pas(-1),
        ArrowRight: () => s.pas(1),
        Home: () => s.aller(0),
        End: () => s.aller(s.images.length - 1),
        Escape: () => s.set({ piece: null, caseChoisie: null }),
        a: s.basculerAnalyse,
        n: () => s.set({ dialogue: true }),
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
  const img = images[pos];
  if (!img) return <div className="phase"><span className="etiquette">Préparation…</span></div>;
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
      ? (joueurs.filter(j => j.type === "humain").length > 1 ? `${qui} : à vous de jouer` : "À vous de jouer")
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
  const analyse = useJeu(s => s.analyse);
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
  const img = useJeu(s => s.images[s.pos]);
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

function KpiJoueurs() {
  const joueurs = useJeu(s => s.joueurs);
  const edite = useJeu(s => s.edite);
  if (!joueurs.length) return null;
  return (
    <div className="kpi">
      <div className="label">{edite ? "Position de l'éditeur" : "Partie"}</div>
      <div className="val petite">{joueurs[0].type === "ia" ? "IA" : "Humain"} <span className="contre">contre</span> {joueurs[1].type === "ia" ? "IA" : "Humain"}</div>
      <div className="det">Or : {joueurs[0].nom} · Argent : {joueurs[1].nom}</div>
    </div>
  );
}

function BarreJeu() {
  const analyse = useJeu(s => s.analyse);
  const mode = useJeu(s => s.mode);
  const masques = useJeu(s => s.masques);
  const mainsVisibles = useJeu(s => s.mainsVisibles);
  const joueurs = useJeu(s => s.joueurs);
  const fini = useJeu(s => s.fini);
  const s = useJeu.getState();
  const unHumain = joueurs.filter(j => j.type === "humain").length === 1;
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
        {mode === "jeu" && <><KpiJoueurs /><KpiBastions /><KpiEval /></>}
      </div>
      <div className="commandes">
        <button type="button" onClick={() => s.set({ dialogue: true })} title="Nouvelle partie (N)">Nouvelle partie</button>
        <button type="button" className={mode === "editeur" ? "on" : ""} onClick={() => (mode === "editeur" ? s.set({ mode: "jeu" }) : s.ouvrirEditeur())}
          title="Éditeur de position (E)">Éditeur</button>
        {mode === "jeu" && (
          <button type="button" className={analyse ? "on" : ""} onClick={s.basculerAnalyse} aria-pressed={analyse}
            title="Analyse de la position affichée par le réseau entraîné (A)">Analyse</button>
        )}
        {mode === "jeu" && unHumain && !fini && (
          <button type="button" className={mainsVisibles ? "on" : ""} aria-pressed={mainsVisibles}
            onClick={() => s.reglages({ mains_visibles: !mainsVisibles })}
            title={masques.length ? "Montrer la main de l'IA (triche : l'analyse devient omnisciente)" : "Cacher de nouveau la main de l'IA"}>
            Mains visibles
          </button>
        )}
      </div>
    </header>
  );
}

// ------------------------------------------------------------------ scène
function EnTete() {
  const joueurs = useJeu(s => s.joueurs);
  const edite = useJeu(s => s.edite);
  const pos = useJeu(s => s.pos);
  const n = useJeu(s => s.images.length);
  if (!joueurs.length) return null;
  return (
    <div className="entete-partie">
      <span className={`badge${pos === n - 1 ? " vivant" : ""}`}>{pos === n - 1 ? (edite ? "Position éditée" : "Partie") : "Historique"}</span>
      <span className="qui">{joueurs[0].nom} (Or) contre {joueurs[1].nom} (Argent)</span>
    </div>
  );
}

function SousPlateau() {
  const img = useJeu(s => s.images[s.pos]);
  const decor = useJeu(s => s.decor);
  const humain = useJeu(s => s.humain);
  const piece = useJeu(s => s.piece);
  const vivant = useJeu(s => s.pos === s.images.length - 1);
  const attente = useJeu(s => s.attente);
  if (!img || !decor) return <div className="sous-plateau" />;
  let texte;
  if (vivant && humain && attente) texte = <span className="attente-decision">{attente}</span>;
  else if (vivant && humain) {
    texte = <span className="attente-decision">
      {piece ? `${NOMS[piece]} : choisissez une case en surbrillance, ou un coup dans la liste` : "Choisissez une pièce de votre main, ou une unité sur le plateau"}
    </span>;
  } else if (img.a) texte = <span className="dernier">{EQUIPE[decor.equipes[img.a.j]]} : {img.a.d}</span>;
  else texte = <span className="dernier">Mise en place</span>;
  return <div className="sous-plateau">{texte}</div>;
}

function Resultat() {
  const img = useJeu(s => s.images[s.pos]);
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
  const analyse = useJeu(s => s.analyse);
  const fleche = useJeu(s => s.fleche);
  const a = useJeu(s => s.analyses[s.pos]);
  const { tous, filtres } = useCoups();
  const img = images[pos];

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
      key={decor.graine + ":" + decor.unites.join()} decor={decor} image={img} precedente={pos > 0 ? images[pos - 1] : null}
      duree={380} cle={pos} cibles={cibles} choisie={caseChoisie} survol={survol} conseil={conseil}
      onCase={tous.length ? onCase : undefined} cliquables={cliquables}
    />
  );
}

function Navigation() {
  const n = useJeu(s => s.images.length - 1);
  const pos = useJeu(s => s.pos);
  const img = useJeu(s => s.images[s.pos]);
  const joueurs = useJeu(s => s.joueurs);
  const lecture = useJeu(s => s.lecture);
  const vitesse = useJeu(s => s.vitesse);
  const fini = useJeu(s => s.fini);
  const s = useJeu.getState();
  if (!img) return null;
  const seule = joueurs.every(j => j.type === "ia");
  const humains = joueurs.some(j => j.type === "humain");
  return (
    <div className="lecteur">
      <div className="boutons">
        <button type="button" onClick={() => s.aller(0)} title="Début (Origine)" aria-label="Début">⏮</button>
        <button type="button" onClick={() => s.pas(-1)} title="Décision précédente (←)" aria-label="Décision précédente">◀</button>
        {seule && !fini && (
          <button type="button" className="principal" onClick={() => s.set({ lecture: !lecture, pos: lecture ? pos : n })}>
            {lecture ? "Pause" : "Lecture"}
          </button>
        )}
        <button type="button" onClick={() => s.pas(1)} title="Décision suivante (→)" aria-label="Décision suivante">▶</button>
        <button type="button" onClick={() => s.aller(n)} title="Position actuelle (Fin)" aria-label="Position actuelle">⏭</button>
      </div>
      <div className="temps">
        <input
          type="range" min={0} max={n} value={pos} aria-label="Position dans la partie"
          onChange={e => { s.aller(+e.target.value); if (seule) s.set({ lecture: false }); }}
          style={{ "--p": `${n ? (100 * pos) / n : 0}%` } as React.CSSProperties}
        />
        <div className="compteurs">
          <span>Manche <b>{img.r}</b></span>
          <span>Décision <b>{pos}</b> / {n}</span>
        </div>
      </div>
      {seule && (
        <select value={vitesse} onChange={e => { s.set({ vitesse: +e.target.value }); }} aria-label="Vitesse de l'IA">
          {[[1500, "Lent"], [700, "Normal"], [250, "Rapide"], [30, "Éclair"]].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      )}
      {pos < n ? (
        <button type="button" className="reprendre" onClick={() => s.revenir(pos)}
          title="Effacer la suite et reprendre la partie depuis la position affichée">Reprendre d'ici</button>
      ) : humains ? (
        <button type="button" onClick={s.annuler} disabled={n === 0} title="Annuler votre dernier coup (et la réponse de l'IA)">Annuler mon coup</button>
      ) : null}
    </div>
  );
}

function Scene() {
  const decor = useJeu(s => s.decor);
  const img = useJeu(s => s.images[s.pos]);
  const joueurs = useJeu(s => s.joueurs);
  const piece = useJeu(s => s.piece);
  const vivant = useJeu(s => s.pos === s.images.length - 1);
  const humain = useJeu(s => s.humain);
  const attente = useJeu(s => s.attente);
  const pieceAttente = useJeu(s => s.pieceAttente);
  if (!decor || !img) return <div className="scene"><div className="attente-partie">Préparation de la partie…</div></div>;
  const colonne = (j: number) => {
    const actif = vivant && humain && img.t === j && !attente;
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
          attente={vivant && img.t === j && pieceAttente ? pieceAttente : null} />
      </div>
    );
  };
  return (
    <div className="scene">
      {colonne(0)}
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
      {colonne(1)}
    </div>
  );
}

// ------------------------------------------------------------------ colonne de droite
function ListeCoups() {
  const humain = useJeu(s => s.humain);
  const vivant = useJeu(s => s.pos === s.images.length - 1);
  const fini = useJeu(s => s.fini);
  const piece = useJeu(s => s.piece);
  const caseChoisie = useJeu(s => s.caseChoisie);
  const decor = useJeu(s => s.decor);
  const attente = useJeu(s => s.attente);
  const { tous, filtres } = useCoups();
  const s = useJeu.getState();
  if (!vivant) {
    return <p className="vide">Vous consultez l'historique. <button type="button" onClick={() => s.aller(s.images.length - 1)}>Revenir à la position actuelle</button></p>;
  }
  if (fini) return <p className="vide">Partie terminée. Lancez une nouvelle partie ou parcourez le déroulé (← →) avec l'analyse.</p>;
  if (!humain) return <p className="vide">L'IA réfléchit…</p>;
  const nom = (i: number | null) => (i === null || !decor ? "" : decor.cases[i][0]);
  const liste = piece !== null || caseChoisie !== null ? filtres : tous;
  return (
    <>
      <div className="filtre-coups">
        {attente ? <span>{attente}</span> : (
          <span>
            {piece ? <>Pièce <b>{NOMS[piece]}</b></> : "Toutes les pièces"}
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
  const onglet = useJeu(s => s.onglet);
  const n = useCoups().tous.length;
  const s = useJeu.getState();
  return (
    <aside className="cote">
      <div className="onglets" role="tablist">
        <button role="tab" aria-selected={onglet === "coups"} className={onglet === "coups" ? "on" : ""} onClick={() => s.set({ onglet: "coups" })}>
          Vos coups <span className="n">{n || ""}</span>
        </button>
        <button role="tab" aria-selected={onglet === "analyse"} className={onglet === "analyse" ? "on" : ""} onClick={() => s.set({ onglet: "analyse" })}>
          Analyse
        </button>
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
  const analyse = useJeu(s => s.analyse);
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
      {erreur ? <div className="toast" role="alert">{erreur}</div> : info && <div className="toast info" role="status">{info}</div>}
    </div>
  );
}
