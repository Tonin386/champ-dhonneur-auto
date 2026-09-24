import { useEffect, useMemo, useState } from "react";
import { NOMS, imgPiece } from "../jeu";
import type { MesureUnites, Unites } from "../types";

// Vue « Unités » : valeur dynamique des unités mesurée à chaque itération (ia/valeurs.py),
// en points du scoreur (10 = un bastion d'avance).

export const points = (v: number, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d).replace(".", ",")}`;
const pct = (v: number) => `${Math.round(100 * v)} %`;
const fois = (v: number | undefined) => (v == null ? "—" : `×${v.toFixed(2).replace(".", ",")}`);

/** Couleur divergente : rouge (négatif) ↔ gris neutre ↔ bleu (positif), |v| rapporté à `max`. */
function divergente(v: number, max: number): string {
  const t = Math.min(1, Math.abs(v) / (max || 1)) ** 0.8;
  const [a, b] = [[0x38, 0x38, 0x35], v >= 0 ? [0x55, 0x98, 0xe7] : [0xe6, 0x67, 0x67]];
  return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * t)).join(",")})`;
}

const paire = (t: Record<string, number>, u: string, w: string) => t[u < w ? u + w : w + u] ?? 0;
const contre = (t: Record<string, number>, u: string, w: string) => (u < w ? t[u + w] ?? 0 : -(t[w + u] ?? 0));

function useUnites(run: string, maj: number | null) {
  const [data, setData] = useState<Unites | null>(null);
  useEffect(() => {
    let arret = false;
    fetch(`/api/entrainements/${encodeURIComponent(run)}/unites`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (!arret && d) setData(d); })
      .catch(() => {});
    return () => { arret = true; };
  }, [run, maj]);
  return data;
}

/** Courbe miniature, échelle commune [lo, hi] ; point sur l'itération choisie. */
function Etincelle({ ys, lo, hi, sel, ref0 }: { ys: number[]; lo: number; hi: number; sel: number; ref0?: number }) {
  const w = 100, h = 24;
  if (ys.length < 2) return <svg className="etincelle" viewBox={`0 0 ${w} ${h}`} />;
  const X = (k: number) => 2 + ((w - 4) * k) / (ys.length - 1);
  const Y = (v: number) => 2 + ((h - 4) * (hi - v)) / (hi - lo || 1);
  return (
    <svg className="etincelle" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      {ref0 != null && <line className="zero" x1={0} x2={w} y1={Y(ref0)} y2={Y(ref0)} />}
      <polyline points={ys.map((v, k) => `${X(k)},${Y(v)}`).join(" ")} />
      <circle cx={X(sel)} cy={Y(ys[sel])} r={2.2} />
    </svg>
  );
}

function TableValeurs({ m, ordre, hist, sel, unite, setUnite }: {
  m: MesureUnites; ordre: string[]; hist: MesureUnites[]; sel: number; unite: string | null; setUnite: (u: string | null) => void;
}) {
  const borne = Math.max(...ordre.map(u => Math.abs(m.unites[u].points) + m.unites[u].ic95), 1);
  const toutes = hist.flatMap(h => Object.values(h.unites).map(x => x.points));
  const [lo, hi] = [Math.min(...toutes), Math.max(...toutes)];
  const X = (v: number) => 50 + (50 * v) / borne;
  return (
    <table className="valeurs">
      <thead>
        <tr>
          <th colSpan={2}>Unité</th>
          <th title="Gain, en points, à remplacer une unité moyenne par celle-ci (barre : IC 95 %)">Valeur propre</th>
          <th className="num" />
          <th title="Valeur propre à chaque itération (même échelle pour toutes les unités)">Évolution</th>
          <th className="num" title="Draft optimal sur les sondes : fréquence à laquelle l'unité est le meilleur premier choix quand elle est tirée">1er choix</th>
          <th className="num" title="Points perdus par le premier à choisir s'il ne la prend pas en premier alors qu'elle est disponible">Regret</th>
          <th className="num" title="Probabilité que l'auto-jeu la choisisse quand elle est disponible, rapportée au choix uniforme (×1 = neutre)">Auto-jeu</th>
        </tr>
      </thead>
      <tbody>
        {ordre.map(u => {
          const v = m.unites[u];
          const g = Math.min(X(v.points), 50), d = Math.max(X(v.points), 50);
          return (
            <tr key={u} className={unite === u ? "choisie" : ""} onClick={() => setUnite(unite === u ? null : u)}
              tabIndex={0} onKeyDown={e => { if (e.key === "Enter") setUnite(unite === u ? null : u); }}>
              <td className="icone"><img src={imgPiece(u, 0)} alt="" /></td>
              <td className="nom"><b>{u}</b> {NOMS[u]}</td>
              <td className="barre-v">
                <div className="piste">
                  <span className="zero" />
                  <span className={`b ${v.points >= 0 ? "pos" : "neg"}`} style={{ left: `${g}%`, width: `${d - g}%` }} />
                  <span className="ic" style={{ left: `${X(v.points - v.ic95)}%`, width: `${X(v.points + v.ic95) - X(v.points - v.ic95)}%` }} />
                </div>
              </td>
              <td className="num val">{points(v.points)} <small>±{v.ic95.toFixed(1).replace(".", ",")}</small></td>
              <td><Etincelle ys={hist.map(h => h.unites[u]?.points ?? 0)} lo={lo} hi={hi} sel={sel} ref0={0} /></td>
              <td className="num">{pct(v.premier_choix)}</td>
              <td className="num">{v.regret_premier.toFixed(1).replace(".", ",")}</td>
              <td className="num">{fois(v.choix_autojeu)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Evolution({ hist, ordre, sel, setSel, unite }: {
  hist: MesureUnites[]; ordre: string[]; sel: number; setSel: (k: number) => void; unite: string | null;
}) {
  const [survol, setSurvol] = useState<number | null>(null);
  const w = 600, h = 220, gx = 34, gy = 10, bas = 18;
  const vals = hist.flatMap(m => Object.values(m.unites).flatMap(x => [x.points - x.ic95, x.points + x.ic95]));
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 0);
  const n = hist.length;
  const X = (k: number) => gx + ((w - gx - 24) * k) / Math.max(n - 1, 1);
  const Y = (v: number) => gy + ((h - gy - bas) * (hi - v)) / (hi - lo || 1);
  const pas = hi - lo > 20 ? 10 : 5;
  const ticks = [];
  for (let t = Math.ceil(lo / pas) * pas; t <= hi; t += pas) ticks.push(t);
  const ligne = (u: string) => hist.map((m, k) => `${X(k).toFixed(1)},${Y(m.unites[u]?.points ?? 0).toFixed(1)}`).join(" ");
  const k = survol ?? sel;
  const bouger = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * w;
    setSurvol(Math.max(0, Math.min(n - 1, Math.round(((x - gx) / (w - gx - 24)) * (n - 1)))));
  };
  const lu = unite ?? ordre[0];
  const bande = hist.map((m, i) => `${X(i)},${Y(m.unites[lu].points + m.unites[lu].ic95)}`).join(" ")
    + " " + hist.map((m, i) => `${X(i)},${Y(m.unites[lu].points - m.unites[lu].ic95)}`).reverse().join(" ");
  const tri = [...ordre].sort((a, b) => hist[k].unites[b].points - hist[k].unites[a].points);
  return (
    <div className="evolution">
      <svg viewBox={`0 0 ${w} ${h}`} onPointerMove={bouger} onPointerLeave={() => setSurvol(null)}
        onClick={() => survol != null && setSel(survol)} role="img"
        aria-label={`Valeur propre des unités par itération ; ${NOMS[lu]} en évidence`}>
        {ticks.map(t => (
          <g key={t} className={t === 0 ? "zero" : "grille"}>
            <line x1={gx} x2={w - 24} y1={Y(t)} y2={Y(t)} />
            <text x={gx - 4} y={Y(t) + 3}>{points(t, 0)}</text>
          </g>
        ))}
        {ordre.filter(u => u !== lu).map(u => <polyline key={u} className="autre" points={ligne(u)} />)}
        {n > 1 && <polygon className="bande" points={bande} />}
        <polyline className="choisie" points={ligne(lu)} />
        <circle className="point" cx={X(n - 1)} cy={Y(hist[n - 1].unites[lu].points)} r={4} />
        <text className="etiquette" x={X(n - 1) + 7} y={Y(hist[n - 1].unites[lu].points) + 4}>{lu}</text>
        <line className="curseur" x1={X(k)} x2={X(k)} y1={gy} y2={h - bas} />
        <text className="borne" x={gx} y={h - 4}>it. {hist[0].iteration}</text>
        <text className="borne" x={w - 24} y={h - 4} textAnchor="end">it. {hist[n - 1].iteration}</text>
      </svg>
      {survol != null && (
        <div className={`bulle-evo${survol > n / 2 ? " gauche" : ""}`} role="tooltip">
          <div className="t">Itération {hist[survol].iteration}</div>
          {tri.map(u => (
            <div key={u} className={u === lu ? "ligne choisie" : "ligne"}>
              <b>{points(hist[survol].unites[u].points)}</b> <span>{u} {NOMS[u]}</span>
            </div>
          ))}
          <small>Clic : afficher cette itération</small>
        </div>
      )}
    </div>
  );
}

function Chaleur({ titre, aide, ordre, valeur, unite, format }: {
  titre: string; aide: string; ordre: string[]; valeur: (u: string, w: string) => number; unite: string | null;
  format: (u: string, w: string, v: number) => string;
}) {
  const [survol, setSurvol] = useState<[string, string] | null>(null);
  const max = Math.max(...ordre.flatMap(u => ordre.filter(w => w !== u).map(w => Math.abs(valeur(u, w)))), 0.1);
  const c = 16, g = 14, n = ordre.length;
  return (
    <figure className="chaleur">
      <figcaption title={aide}>{titre}</figcaption>
      <svg viewBox={`0 0 ${g + n * c} ${g + n * c}`} onPointerLeave={() => setSurvol(null)} role="img" aria-label={titre}>
        {ordre.map((u, i) => (
          <g key={u}>
            <text className={`lettre${u === unite ? " sel" : ""}`} x={g + i * c + c / 2} y={g - 4} textAnchor="middle">{u}</text>
            <text className={`lettre${u === unite ? " sel" : ""}`} x={g - 4} y={g + i * c + c / 2 + 3} textAnchor="end">{u}</text>
          </g>
        ))}
        {ordre.map((u, i) => ordre.map((w, j) => {
          if (u === w) return null;
          const v = valeur(u, w);
          return (
            <rect key={u + w} x={g + j * c + 1} y={g + i * c + 1} width={c - 2} height={c - 2} rx={2}
              fill={divergente(v, max)}
              className={survol && survol[0] === u && survol[1] === w ? "survol" : unite && (u === unite || w === unite) ? "sel" : ""}
              onPointerEnter={() => setSurvol([u, w])}>
              <title>{format(u, w, v)}</title>
            </rect>
          );
        }))}
      </svg>
      <div className="legende-chaleur">
        <span>{points(-max)}</span><span className="degrade" /><span>{points(max)}</span>
      </div>
      <div className="lecture">{survol ? format(survol[0], survol[1], valeur(survol[0], survol[1])) : aide}</div>
    </figure>
  );
}

function Tuile({ label, valeur, detail, aide, ys, sel, ref0 }: {
  label: string; valeur: string; detail?: string; aide: string; ys?: number[]; sel?: number; ref0?: number;
}) {
  const lo = ys && Math.min(...ys, ref0 ?? Infinity), hi = ys && Math.max(...ys, ref0 ?? -Infinity);
  return (
    <div className="kpi tuile" title={aide}>
      <div className="label">{label}</div>
      <div className="val">{valeur}</div>
      {detail && <div className="det">{detail}</div>}
      {ys && ys.length > 1 && <Etincelle ys={ys} lo={lo!} hi={hi!} sel={sel ?? ys.length - 1} ref0={ref0} />}
    </div>
  );
}

export function VueUnites({ run, maj }: { run: string; maj: number | null }) {
  const data = useUnites(run, maj);
  const [choix, setChoix] = useState<number | null>(null); // itération affichée (null : la dernière)
  const [unite, setUnite] = useState<string | null>(null);
  const hist = data?.historique ?? [];
  const sel = choix != null && choix < hist.length ? choix : hist.length - 1;
  const m = hist[sel];
  const ordre = useMemo(() => (m ? Object.keys(m.unites).sort((a, b) => m.unites[b].points - m.unites[a].points) : []), [m]);
  if (!data) return <div className="vue-unites vide">Chargement…</div>;
  if (!m) return <div className="vue-unites vide">Aucune mesure de la valeur des unités pour l'instant (<code>unites.jsonl</code> est écrit à la fin de chaque itération).</div>;
  const dr = data.draft.filter(d => d.parties_draft > 0);
  const tauxDr = dr.map(d => d.victoires_choisit / d.parties_draft);
  const drSel = dr.find(d => d.iteration === m.iteration) ?? dr[dr.length - 1];
  return (
    <div className="vue-unites">
      <section className="panneau p-valeurs">
        <h2>
          Valeur des unités
          <span className="it">
            itération {m.iteration}
            {choix != null && <button type="button" onClick={() => setChoix(null)}>dernière</button>}
          </span>
        </h2>
        <TableValeurs m={m} ordre={ordre} hist={hist} sel={sel} unite={unite} setUnite={setUnite} />
        <dl className="glossaire">
          <dt>Valeur propre</dt>
          <dd>Points gagnés à avoir cette unité dans son armée à la place d'une unité moyenne (0), synergies et contres mis à part.
            Le trait fin donne l'intervalle de confiance à 95 %. Échelle : 10 points = un Lieu contrôlé d'avance.</dd>
          <dt>1er choix</dt>
          <dd>Dans le draft optimal calculé sur les sondes du réseau : part des tirages de 8 cartes contenant l'unité où elle est
            le meilleur premier choix. 100 % : toujours à prendre en premier quand elle sort.</dd>
          <dt>Regret</dt>
          <dd>Points que perd en moyenne le premier à choisir s'il ne prend pas cette unité en premier alors qu'elle est disponible
            (0 : c'est toujours le meilleur choix ; plus c'est grand, moins elle mérite le premier choix).</dd>
          <dt>Auto-jeu</dt>
          <dd>Fréquence à laquelle le réseau la choisit réellement en auto-jeu quand elle est proposée, rapportée à un choix au hasard :
            ×1 = neutre, ×2 = choisie deux fois plus souvent, ×0,5 = deux fois moins.</dd>
        </dl>
        <p className="note">Cliquez une unité pour la suivre dans les graphiques.</p>
      </section>
      <div className="tuiles">
        <Tuile label="Avantage du 1er à choisir" valeur={points(m.avantage_premier_choix)} detail="draft optimal, en points"
          aide="Valeur du draft optimal (sondes du réseau) pour le premier à choisir, avantage de commencer du second compris. Négatif : mieux vaut choisir en second."
          ys={hist.map(h => h.avantage_premier_choix)} sel={sel} ref0={0} />
        <Tuile label="Victoires du 1er à choisir" valeur={drSel ? pct(drSel.victoires_choisit / drSel.parties_draft) : "—"}
          detail={drSel ? `auto-jeu · ${drSel.parties_draft} drafts` : "pas encore de draft"}
          aide="Part des parties d'auto-jeu avec draft gagnées par le premier à choisir (les nulles comptent comme non gagnées). Doit tendre vers 50 % si le draft est équilibré."
          ys={tauxDr} ref0={0.5} />
        <Tuile label="Avantage de commencer" valeur={points(m.commencer)} detail="pour le second à choisir"
          aide="Valeur, en points, de commencer la partie (le second à choisir prend l'Initiative), à armées égales."
          ys={hist.map(h => h.commencer)} sel={sel} ref0={0} />
        <Tuile label="Modèle d'armée (R²)" valeur={m.r2.toFixed(2).replace(".", ",")}
          detail={`valeurs propres seules : ${m.r2_effets_propres.toFixed(2).replace(".", ",")}`}
          aide="Part de la variance des évaluations du réseau expliquée par le modèle valeurs propres + synergies + contres (détail : valeurs propres seules)."
          ys={hist.map(h => h.r2)} sel={sel} />
      </div>
      <section className="panneau p-evolution">
        <h2>Évolution {unite ? `· ${NOMS[unite]}` : `· ${NOMS[ordre[0]]}`}<span className="it">bande : IC 95 %</span></h2>
        <Evolution hist={hist} ordre={ordre} sel={sel} setSel={k => setChoix(k === hist.length - 1 ? null : k)} unite={unite} />
      </section>
      <section className="panneau p-chaleur">
        <Chaleur titre="Synergies entre alliés" ordre={ordre} unite={unite} valeur={(u, w) => paire(m.synergies, u, w)}
          aide="Points gagnés à avoir les deux unités dans la même armée, au-delà de leurs valeurs propres. Survolez une case."
          format={(u, w, v) => `${NOMS[u]} + ${NOMS[w]} : ${points(v)} points`} />
        <Chaleur titre="Contres (ligne face à colonne)" ordre={ordre} unite={unite} valeur={(u, w) => contre(m.contres, u, w)}
          aide="Points gagnés par l'unité de la ligne quand l'adversaire a celle de la colonne. Bleu : la ligne l'emporte."
          format={(u, w, v) => `${NOMS[u]} face à ${NOMS[w]} : ${points(v)} points`} />
      </section>
    </div>
  );
}
