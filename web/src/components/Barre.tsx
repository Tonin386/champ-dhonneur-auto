import { useEffect, useState } from "react";
import { compact, duree, entier, nomJoueur } from "../jeu";
import { maintenant, tableActive, useStore } from "../store";

/** Rafraîchit l'affichage chaque seconde (durées écoulées). */
export function useHorloge(ms = 1000) {
  const [, set] = useState(0);
  useEffect(() => {
    const t = setInterval(() => set(n => n + 1), ms);
    return () => clearInterval(t);
  }, [ms]);
}

const PHASES: Record<string, string> = { autojeu: "Auto-jeu", apprentissage: "Apprentissage", evaluation: "Évaluation" };

function Phase() {
  const t = useStore(s => s.tableau);
  const tables = useStore(s => s.tables);
  const decalage = useStore(s => s.decalage);
  useHorloge();
  if (!t) return null;
  const now = maintenant(decalage);
  const ph = t.phase;
  const tl = Object.values(tables);
  const actives = tl.filter(x => tableActive(x, decalage)).length;
  const inactif = now - Math.max(t.maj ?? 0, ph?.debut ?? 0, ...tl.map(x => x.maj));
  if (!ph) {
    return (
      <div className="phase">
        <span className="etiquette">Itération {t.iteration} terminée</span>
        <span className="detail">{t.maj ? `il y a ${duree(now - t.maj)}` : "en attente"}</span>
      </div>
    );
  }
  const faites = tl.reduce((a, x) => a + x.parties, 0);
  const total = tl.reduce((a, x) => a + x.total, 0) || t.config.parties_par_iteration || 0;
  const avance = ph.phase === "autojeu" && total ? Math.min(1, faites / total) : null;
  const arret = inactif > 1800 && !actives;
  return (
    <div className={`phase ${ph.phase}${arret ? " arret" : ""}`}>
      <span className="etiquette">
        {arret ? "À l'arrêt ?" : ph.amorce && ph.phase === "autojeu" ? "Amorçage" : PHASES[ph.phase] ?? ph.phase}
        <span className="it"> · it. {ph.iteration}</span>
      </span>
      {avance !== null && (
        <span className="progression" role="progressbar" aria-valuenow={Math.round(100 * avance)} aria-valuemin={0} aria-valuemax={100}>
          <span style={{ width: `${100 * avance}%` }} />
        </span>
      )}
      <span className="detail">
        {arret
          ? `aucune activité depuis ${duree(inactif)}`
          : avance !== null
            ? `${entier(faites)} / ${entier(total)} parties · ${actives} table${actives > 1 ? "s" : ""} · ${duree(now - ph.debut)}`
            : `depuis ${duree(now - ph.debut)}`}
      </span>
    </div>
  );
}

function Kpi({ label, valeur, detail }: { label: string; valeur: React.ReactNode; detail?: React.ReactNode }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className="val">{valeur}</div>
      {detail && <div className="det">{detail}</div>}
    </div>
  );
}

export function Barre() {
  const runs = useStore(s => s.runs);
  const run = useStore(s => s.run);
  const t = useStore(s => s.tableau);
  const connecte = useStore(s => s.connecte);
  const vue = useStore(s => s.vue);
  const regie = useStore(s => s.regie);
  const { choisirRun, setVue, basculerRegie } = useStore.getState();
  const pph = t?.series.parties_par_heure.at(-1);
  const elo = t?.elo_meilleur;
  return (
    <header className="barre">
      <div className="marque">
        <img src="/img/embleme.png" alt="" />
        <div>
          <h1>Champ d'honneur</h1>
          {runs.length > 1 ? (
            <select value={run ?? ""} onChange={e => choisirRun(e.target.value)} aria-label="Entraînement suivi">
              {runs.map(r => <option key={r.nom} value={r.nom}>{r.nom} (it. {r.iteration})</option>)}
            </select>
          ) : (
            <div className="sous">Entraînement {run ?? "—"}</div>
          )}
        </div>
      </div>
      <Phase />
      {t && (
        <div className="kpis">
          <Kpi label="Itération" valeur={entier(t.iteration)} detail={t.config.iterations > 0 ? `sur ${entier(t.config.iterations)}` : "sans fin"} />
          <Kpi
            label="Meilleur modèle"
            valeur={elo == null ? "—" : `${elo >= 0 ? "+" : "−"}${Math.abs(Math.round(elo))}`}
            detail={t.meilleur ? `${nomJoueur(t.meilleur)} · Elo` : "pas encore évalué"}
          />
          <Kpi label="Parties jouées" valeur={compact(t.totaux.parties)} detail={pph ? `${compact(pph)} / h` : undefined} />
          <Kpi label="Temps de calcul" valeur={duree(t.totaux.secondes)}
            detail={t.config.modele.d ? `réseau d=${t.config.modele.d} × ${t.config.modele.couches}` : undefined} />
        </div>
      )}
      <div className="commandes">
        <div className="bascule" role="group" aria-label="Vue">
          <button type="button" className={vue === "plateau" ? "on" : ""} onClick={() => setVue("plateau")} title="Grand plateau (P)">Plateau</button>
          <button type="button" className={vue === "mosaique" ? "on" : ""} onClick={() => setVue("mosaique")} title="Toutes les tables (M)">Mosaïque</button>
        </div>
        <button type="button" className={regie ? "on" : ""} onClick={basculerRegie} aria-pressed={regie}
          title="Régie automatique : enchaîne les parties en direct et les rediffusions (R)">
          Régie auto
        </button>
        <button type="button" onClick={() => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen())} title="Plein écran (F)">
          Plein écran
        </button>
        <a className="bouton lien-page" href="/jouer" title="Jouer contre l'IA entraînée, analyser une position">Jouer</a>
        <span className={`connexion${connecte ? " ok" : ""}`} title={connecte ? "Connecté au flux en direct" : "Reconnexion…"} />
      </div>
    </header>
  );
}
