import { useEffect, useRef, useState } from "react";
import { EQUIPE } from "../jeu";
import { useJeu, type ResumePartie } from "./store";

type Filtre = "toutes" | "cours" | "finies";

const date = (t: number) =>
  new Date(t * 1000).toLocaleString("fr-FR", { day: "2-digit", month: "2-digit", year: "2-digit", hour: "2-digit", minute: "2-digit" });

function etat(p: ResumePartie): { texte: string; cls: string } {
  if (!p.fini) return { texte: `En cours · manche ${p.manche}`, cls: "cours" };
  if (p.resultat === "1-0") return { texte: `Victoire d'${EQUIPE[0]}`, cls: "e0" };
  if (p.resultat === "0-1") return { texte: `Victoire d'${EQUIPE[1]}`, cls: "e1" };
  return { texte: "Partie nulle", cls: "" };
}

/** Historique : parties jouées et en cours (au moins une décision), à reprendre ou revoir. */
export function DialogueParties() {
  const ouvert = useJeu(s => s.historique);
  const parties = useJeu(s => s.parties);
  const ecriture = useJeu(s => s.ecriture);
  const courante = useJeu(s => s.id);
  const ref = useRef<HTMLDialogElement>(null);
  const [filtre, setFiltre] = useState<Filtre>("toutes");
  const [aSupprimer, setASupprimer] = useState<string | null>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (ouvert && !d.open) { setASupprimer(null); d.showModal(); }
    if (!ouvert && d.open) d.close();
  }, [ouvert]);

  const fermer = () => useJeu.getState().set({ historique: false });
  const s = useJeu.getState();
  const liste = (parties ?? []).filter(p => filtre === "toutes" || (filtre === "cours" ? !p.fini : p.fini));
  const n = { toutes: parties?.length ?? 0, cours: parties?.filter(p => !p.fini).length ?? 0, finies: parties?.filter(p => p.fini).length ?? 0 };
  return (
    <dialog ref={ref} className="dialogue historique" onClose={fermer} onClick={e => { if (e.target === ref.current) fermer(); }}>
      <h2>Parties</h2>
      <div className="onglets" role="tablist" aria-label="Filtre">
        {([["toutes", "Toutes"], ["cours", "En cours"], ["finies", "Terminées"]] as [Filtre, string][]).map(([v, l]) => (
          <button key={v} type="button" role="tab" aria-selected={filtre === v} className={filtre === v ? "on" : ""} onClick={() => setFiltre(v)}>
            {l} <span className="n">{n[v]}</span>
          </button>
        ))}
      </div>
      {!ecriture && (
        <p className="remarque alerte-sac">Le serveur ne peut pas écrire l'historique : les nouvelles parties ne seront pas conservées
          (dossier « parties » absent ou en lecture seule).</p>
      )}
      {parties === null ? <p className="vide">Chargement…</p> : !liste.length ? (
        <p className="vide">{parties.length ? "Aucune partie dans cette catégorie." : "Aucune partie enregistrée : une partie l'est dès sa première décision."}</p>
      ) : (
        <ol className="liste-parties">
          {liste.map(p => {
            const e = etat(p);
            return (
              <li key={p.id} className={p.id === courante ? "courante" : ""}>
                <button type="button" className="ouvrir" onClick={() => (p.id === courante ? fermer() : void s.ouvrirPartie(p.id))}
                  title={p.fini ? "Revoir la partie (analyse, variantes)" : "Reprendre la partie"}>
                  <span className="quand">{date(p.maj)}</span>
                  <span className="qui">
                    <b className="e0">{p.noms[0]}</b> <span className="contre">contre</span> <b className="e1">{p.noms[1]}</b>
                  </span>
                  <span className="mise-p">{p.mise}</span>
                  <span className={`etat ${e.cls}`}>{e.texte}</span>
                  <span className="nb">{p.decisions} décision{p.decisions > 1 ? "s" : ""}</span>
                  {p.id === courante && <span className="badge vivant">affichée</span>}
                </button>
                {aSupprimer === p.id ? (
                  <span className="confirmer">
                    <button type="button" className="danger" onClick={() => { void s.supprimerPartie(p.id); setASupprimer(null); }}>Supprimer</button>
                    <button type="button" onClick={() => setASupprimer(null)}>Garder</button>
                  </span>
                ) : (
                  <button type="button" className="suppr" onClick={() => setASupprimer(p.id)} title="Supprimer de l'historique" aria-label="Supprimer">×</button>
                )}
              </li>
            );
          })}
        </ol>
      )}
      <div className="actions-dialogue">
        <button type="button" onClick={fermer}>Fermer</button>
      </div>
    </dialog>
  );
}
