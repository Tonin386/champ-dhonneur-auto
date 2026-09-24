import { useLayoutEffect, useRef, useState } from "react";
import { EQUIPE, nomJoueur } from "../jeu";
import { tableActive, useStore } from "../store";
import type { ResumePartie, TableDirect } from "../types";
import { Plateau } from "./Plateau";

const numero = (nom: string) => Number(nom.replace(/\D/g, "")) + 1;

function etatTable(t: TableDirect, active: boolean) {
  const img = t.images[t.images.length - 1];
  if (img.f) return img.f.g === null ? "Nulle" : `Victoire ${EQUIPE[img.f.g]}`;
  return active ? "En cours" : "En attente";
}

/** Vignette d'une table en direct (colonne latérale et mosaïque). */
export function VignetteTable({ t, grande }: { t: TableDirect; grande?: boolean }) {
  const source = useStore(s => s.source);
  const decalage = useStore(s => s.decalage);
  const regarder = useStore(s => s.regarderTable);
  const img = t.images[t.images.length - 1];
  const active = tableActive(t, decalage);
  const cur = source?.type === "direct" && source.table === t.table;
  return (
    <button type="button" className={`vignette${cur ? " cur" : ""}${active ? " active" : ""}${grande ? " grande" : ""}`} onClick={() => regarder(t.table)}>
      <Plateau decor={t.decor} image={img} mini duree={250} />
      <span className="legende">
        <b>Table {numero(t.table)}</b>
        <span>manche {img.r} · {etatTable(t, active)}</span>
        {grande && img.a && <span className="coup">{img.a.d}</span>}
      </span>
    </button>
  );
}

const RESULTAT: Record<string, string> = { "1-0": "Or gagne", "0-1": "Argent gagne", "1/2-1/2": "Nulle" };

function LignePartie({ p, cur }: { p: ResumePartie; cur: boolean }) {
  const regarder = useStore(s => s.regarderPartie);
  const vue = useStore(s => s.vues.has(p.fichier));
  const evalu = p.type === "evaluation";
  return (
    <li>
      <button type="button" className={`${cur ? "cur" : ""}${vue ? " vue" : ""}`} onClick={() => regarder(p.fichier)}>
        <span className="haut">
          <span className={`type${evalu ? " eval" : ""}`}>{evalu ? "Évaluation" : "Auto-jeu"}</span>
          <b>it. {p.iteration}</b>
          <span className={`res r${p.resultat.replace(/\W/g, "")}`}>{RESULTAT[p.resultat] ?? p.resultat}</span>
          <span className="manches">{p.manches} manches</span>
        </span>
        <span className="bas">
          {nomJoueur(p.blanc)} <i>{p.unites[0]}</i> <span className="contre">contre</span> {nomJoueur(p.noir)} <i>{p.unites[1]}</i>
        </span>
      </button>
    </li>
  );
}

export function Cote() {
  const tables = useStore(s => s.tables);
  const tableau = useStore(s => s.tableau);
  const source = useStore(s => s.source);
  const liste = Object.values(tables).sort((a, b) => a.table.localeCompare(b.table));
  const [onglet, setOnglet] = useState<"direct" | "rediff" | null>(null);
  const [filtre, setFiltre] = useState<"tout" | "autojeu" | "evaluation">("tout");
  const actif = onglet ?? (liste.length ? "direct" : "rediff");
  const parties = (tableau?.parties ?? []).filter(p => filtre === "tout" || p.type === filtre);
  return (
    <aside className="cote">
      <div className="onglets" role="tablist">
        <button role="tab" aria-selected={actif === "direct"} className={actif === "direct" ? "on" : ""} onClick={() => setOnglet("direct")}>
          Direct <span className="n">{liste.length}</span>
        </button>
        <button role="tab" aria-selected={actif === "rediff"} className={actif === "rediff" ? "on" : ""} onClick={() => setOnglet("rediff")}>
          Rediffusions <span className="n">{tableau?.parties.length ?? 0}</span>
        </button>
      </div>
      {actif === "direct" ? (
        liste.length ? (
          <div className="grille-vignettes">{liste.map(t => <VignetteTable key={t.table} t={t} />)}</div>
        ) : (
          <p className="vide">
            Pas de partie en direct pour l'instant : les tables apparaissent pendant la phase d'auto-jeu
            (entraînement lancé avec la diffusion en direct).
          </p>
        )
      ) : (
        <>
          <div className="filtres" role="group" aria-label="Type de partie">
            {(["tout", "autojeu", "evaluation"] as const).map(f => (
              <button key={f} type="button" className={filtre === f ? "on" : ""} onClick={() => setFiltre(f)}>
                {f === "tout" ? "Toutes" : f === "autojeu" ? "Auto-jeu" : "Évaluations"}
              </button>
            ))}
          </div>
          <ul className="parties">
            {parties.map(p => <LignePartie key={p.fichier} p={p} cur={source?.type === "partie" && source.fichier === p.fichier} />)}
            {!parties.length && <li className="vide">Aucune partie enregistrée.</li>}
          </ul>
        </>
      )}
    </aside>
  );
}

/** Nombre de colonnes et largeur des vignettes : les plus grandes possibles sans défilement. */
function colonnes(n: number, W: number, H: number): [number, number] {
  const legende = 64, ecart = 11, ratio = 1.1;
  let meilleur = 1, taille = 0;
  for (let c = 1; c <= n; c++) {
    const r = Math.ceil(n / c);
    const w = Math.min((W - ecart * (c - 1)) / c, ((H - ecart * (r - 1)) / r - legende) * ratio);
    if (w > taille) { taille = w; meilleur = c; }
  }
  return [meilleur, Math.max(120, Math.floor(taille))];
}

export function Mosaique() {
  const tables = useStore(s => s.tables);
  const ref = useRef<HTMLDivElement>(null);
  const [taille, setTaille] = useState({ w: 1600, h: 800 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setTaille({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const liste = Object.values(tables).sort((a, b) => a.table.localeCompare(b.table));
  const [cols, largeur] = colonnes(Math.max(liste.length, 1), taille.w, taille.h);
  return (
    <div className={`mosaique${liste.length ? "" : " vide"}`} ref={ref} style={{ gridTemplateColumns: `repeat(${cols}, ${largeur}px)` }}>
      {liste.length
        ? liste.map(t => <VignetteTable key={t.table} t={t} grande />)
        : "Aucune table en direct : la mosaïque se remplit pendant l'auto-jeu."}
    </div>
  );
}
