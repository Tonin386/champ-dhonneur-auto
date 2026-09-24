import { useEffect, useRef } from "react";
import { Barre } from "./components/Barre";
import { Chronique } from "./components/Chronique";
import { Graphes } from "./components/Graphes";
import { Joueur } from "./components/Joueur";
import { Lecteur } from "./components/Lecteur";
import { Plateau } from "./components/Plateau";
import { Cote, Mosaique } from "./components/Tables";
import { EQUIPE, nomJoueur } from "./jeu";
import { useStore } from "./store";
import type { Run } from "./types";

const PAUSE_FIN = 6000; // position finale affichée avant la partie suivante (ms)

/** Liste des entraînements (rafraîchie chaque minute) ; le plus récent est suivi d'office. */
function useRuns() {
  useEffect(() => {
    let arret = false;
    const charger = async () => {
      try {
        const r: { entrainements: Run[] } = await (await fetch("/api/entrainements")).json();
        if (arret) return;
        const s = useStore.getState();
        s.setRuns(r.entrainements);
        if (!s.run && r.entrainements.length) s.choisirRun(r.entrainements[0].nom);
      } catch {
        useStore.getState().setConnecte(false);
      }
    };
    charger();
    const t = setInterval(charger, 60000);
    return () => { arret = true; clearInterval(t); };
  }, []);
}

/** Flux Server-Sent Events : tableau de bord et coups joués en direct. */
function useFlux(run: string | null) {
  useEffect(() => {
    if (!run) return;
    const s = useStore.getState();
    const es = new EventSource(`/api/entrainements/${encodeURIComponent(run)}/flux`);
    es.onopen = () => s.setConnecte(true);
    es.onerror = () => s.setConnecte(false);
    es.addEventListener("tableau", e => s.recevoirTableau(JSON.parse((e as MessageEvent).data)));
    es.addEventListener("table", e => s.recevoirTable(JSON.parse((e as MessageEvent).data)));
    return () => es.close();
  }, [run]);
}

/** Avance la lecture ; en direct, accélère pour rattraper un retard ; enchaîne en fin de partie. */
function useMoteur() {
  const film = useStore(s => s.film);
  const pos = useStore(s => s.pos);
  const lecture = useStore(s => s.lecture);
  const vitesse = useStore(s => s.vitesse);
  const source = useStore(s => s.source);
  const perimee = useStore(s => s.source?.type === "direct" && s.tables[s.source.table]?.id !== s.film?.id);
  useEffect(() => {
    if (!film || !lecture) return;
    const { aller, suivante } = useStore.getState();
    const n = film.images.length - 1;
    const direct = source?.type === "direct";
    let t: ReturnType<typeof setTimeout> | undefined;
    if (pos < n) {
      const retard = n - pos;
      const d = direct && retard > 4 ? Math.max(50, vitesse / Math.min(12, retard / 2)) : vitesse;
      t = setTimeout(() => aller(pos + 1), d);
    } else if (film.images[n].f || !direct) {
      t = setTimeout(suivante, PAUSE_FIN);
    } else if (perimee) {
      t = setTimeout(suivante, 3000); // table effacée (nouvelle itération) sans position finale
    }
    return () => clearTimeout(t);
  }, [film, pos, lecture, vitesse, source, perimee]);
}

function useClavier() {
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (["INPUT", "SELECT", "TEXTAREA"].includes((document.activeElement as HTMLElement)?.tagName)) return;
      const s = useStore.getState();
      const n = s.film ? s.film.images.length - 1 : 0;
      const actions: Record<string, () => void> = {
        " ": s.basculerLecture,
        ArrowLeft: () => s.pas(-1),
        ArrowRight: () => s.pas(1),
        Home: () => s.aller(0),
        End: () => s.aller(n),
        d: s.allerDirect,
        m: () => s.setVue(s.vue === "mosaique" ? "plateau" : "mosaique"),
        p: () => s.setVue("plateau"),
        r: s.basculerRegie,
        f: () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()),
      };
      const a = actions[e.key] ?? actions[e.key.toLowerCase()];
      if (a) { e.preventDefault(); a(); }
      else if (/^[1-9]$/.test(e.key)) {
        const nom = Object.keys(s.tables).sort()[+e.key - 1];
        if (nom) s.regarderTable(nom);
      }
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);
}

/** Grand écran : le pointeur de la souris disparaît après quelques secondes d'inactivité. */
function useCurseur() {
  useEffect(() => {
    let t: ReturnType<typeof setTimeout>;
    const bouge = () => {
      document.body.classList.remove("repos");
      clearTimeout(t);
      t = setTimeout(() => document.body.classList.add("repos"), 4000);
    };
    bouge();
    window.addEventListener("pointermove", bouge);
    return () => { window.removeEventListener("pointermove", bouge); clearTimeout(t); };
  }, []);
}

function EnTetePartie() {
  const film = useStore(s => s.film);
  const source = useStore(s => s.source);
  if (!film) return null;
  const e = film.decor.entetes;
  const type = e.Type === "evaluation" ? "Évaluation" : "Auto-jeu";
  const direct = source?.type === "direct";
  const memeJoueur = e.Blanc === e.Noir;
  return (
    <div className="entete-partie">
      <span className={`badge${direct ? " direct" : ""}`}>{direct ? `Direct · table ${Number(source.table.replace(/\D/g, "")) + 1}` : "Rediffusion"}</span>
      <span className="quoi">
        {type}{e.Iteration ? ` · itération ${e.Iteration}` : ""}
      </span>
      <span className="qui">
        {memeJoueur ? `${nomJoueur(e.Blanc)} contre lui-même` : `${nomJoueur(e.Blanc)} (Blanc) contre ${nomJoueur(e.Noir)} (Noir)`}
      </span>
    </div>
  );
}

function Scene() {
  const film = useStore(s => s.film);
  const pos = useStore(s => s.pos);
  const vitesse = useStore(s => s.vitesse);
  const vue = useStore(s => s.vue);
  if (vue === "mosaique") return <div className="scene"><Mosaique /></div>;
  if (!film) return <div className="scene"><div className="attente-partie">En attente d'une partie…</div></div>;
  const d = film.decor;
  const img = film.images[Math.min(pos, film.images.length - 1)];
  const prec = pos > 0 ? film.images[pos - 1] : null;
  const joueurs = (equipe: number) => d.equipes.map((e, j) => [e, j]).filter(([e]) => e === equipe).map(([, j]) => j);
  const nom = (j: number) => (d.equipes[j] ? d.entetes.Noir : d.entetes.Blanc);
  return (
    <div className="scene">
      <div className="colonne-joueur">{joueurs(0).map(j => <Joueur key={j} decor={d} image={img} joueur={j} nom={nom(j)} />)}</div>
      <div className="centre">
        <EnTetePartie />
        <div className="table-bois">
          <Plateau key={film.id} decor={d} image={img} precedente={prec} duree={Math.min(520, vitesse * 0.7)} cle={pos} />
          {img.f && (
            <div className={`resultat e${img.f.g ?? "n"}`} role="status">
              {img.f.g !== null && <img src={`/img/controle-${img.f.g ? "noir" : "blanc"}.png`} alt="" />}
              <span>{img.f.g === null ? "Partie nulle" : `Victoire de ${EQUIPE[img.f.g]}`}</span>
              <small>en {img.r} manches</small>
            </div>
          )}
        </div>
        <div className="sous-plateau">
          {img.p ? <span className="attente-decision">{img.p}</span> : img.a ? <span className="dernier">{EQUIPE[d.equipes[img.a.j]]} : {img.a.d}</span> : <span className="dernier">Mise en place</span>}
        </div>
        <Lecteur />
      </div>
      <div className="colonne-joueur">{joueurs(1).map(j => <Joueur key={j} decor={d} image={img} joueur={j} nom={nom(j)} />)}</div>
    </div>
  );
}

export function App() {
  useRuns();
  const run = useStore(s => s.run);
  const runs = useStore(s => s.runs);
  const tableau = useStore(s => s.tableau);
  const source = useStore(s => s.source);
  const erreur = useStore(s => s.erreur);
  useFlux(run);
  useMoteur();
  useClavier();
  useCurseur();

  // premier choix de partie, une fois les tables en direct arrivées
  const demarre = useRef<string | null>(null);
  useEffect(() => {
    if (!tableau || source || demarre.current === run) return;
    const t = setTimeout(() => { demarre.current = run; useStore.getState().demarrer(); }, 1200);
    return () => clearTimeout(t);
  }, [tableau, source, run]);

  useEffect(() => {
    if (!erreur) return;
    const t = setTimeout(() => useStore.getState().setErreur(null), 5000);
    return () => clearTimeout(t);
  }, [erreur]);

  if (!runs.length && !run)
    return (
      <div className="vide-total">
        <img src="/img/embleme.png" alt="" />
        <h1>Champ d'honneur</h1>
        <p>Aucun entraînement trouvé dans <code>runs/</code>. Lancez « make continu » (voir le README).</p>
        <p><a href="/jouer">Jouer contre les bots</a></p>
      </div>
    );

  return (
    <div className="ecran">
      <Barre />
      <main className="milieu">
        <Scene />
        <div className="droite">
          <section className="panneau chronique-p" aria-label="Déroulé">
            <h2>Déroulé</h2>
            <Chronique />
          </section>
          <Cote />
        </div>
      </main>
      {tableau && <footer className="bas"><Graphes t={tableau} /></footer>}
      {erreur && <div className="toast" role="alert">{erreur}</div>}
    </div>
  );
}
