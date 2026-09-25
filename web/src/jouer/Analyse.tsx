import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { EQUIPE } from "../jeu";
import { sauverPrefs, useAnalyseActive, useJeu } from "./store";
import type { LimiteAnalyse, Analyse as TAnalyse } from "./types";

/** Texte principal, détail et couleur d'une évaluation (point de vue d'Or : + Or, − Argent). */
export function libelleScore(a: TAnalyse): { texte: string; detail: string; cls: string } {
  if (a.fini) {
    const detail = a.fini === "1-0" ? "Victoire d'Or" : a.fini === "0-1" ? "Victoire d'Argent" : "Partie nulle";
    return { texte: a.texte ?? a.fini, detail, cls: a.fini === "1-0" ? "or" : a.fini === "0-1" ? "argent" : "" };
  }
  if (a.texte === undefined) return { texte: "…", detail: a.indisponible ?? "", cls: "" };
  const cls = (a.score ?? 0) > 0.05 ? "or" : (a.score ?? 0) < -0.05 ? "argent" : "";
  const [symbole, libelle] = a.appreciation ?? ["", ""];
  return { texte: a.texte, detail: a.mat ? libelle : `${symbole} ${libelle}`, cls };
}

/** Dernière évaluation connue à la position affichée ou avant (l'IA au trait n'est pas analysée). */
function useEvalAffichee(): TAnalyse | null {
  const analyses = useJeu(s => s.analyses);
  const pos = useJeu(s => s.pos);
  return useMemo(() => {
    for (let k = pos; k >= 0 && k >= pos - 3; k--) {
      const a = analyses[k];
      if (a && a.gain_or !== undefined) return a;
    }
    return null;
  }, [analyses, pos]);
}

/** Barre verticale à gauche du plateau : Argent en haut, Or en bas (comme sur le plateau). */
export function BarreEval() {
  const analyse = useAnalyseActive();
  const retourne = useJeu(s => s.retourne);
  const a = useEvalAffichee();
  if (!analyse) return null;
  const or = a?.gain_or ?? 0.5;
  const txt = a ? libelleScore(a).texte : "";
  // texte du côté du camp qui mène ; plateau retourné : Or en haut
  const coteOr = retourne ? "haut" : "bas";
  const cote = or >= 0.5 ? coteOr : coteOr === "bas" ? "haut" : "bas";
  return (
    <div className={`barre-eval${retourne ? " retournee" : ""}`} role="meter" aria-label="Évaluation" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={Math.round(100 * or)} aria-valuetext={txt}
      title={`Part d'Or (${retourne ? "en haut" : "en bas"}) et d'Argent : chances de gain estimées`}>
      <div className="part-or" style={{ height: `${100 * or}%` }} />
      <span className={`texte-barre ${cote} ${or >= 0.5 ? "sur-or" : "sur-argent"}`}>{txt}</span>
    </div>
  );
}

/** Limites proposées pour l'analyse progressive (clé du <select> : "type:valeur"). */
const LIMITES: { groupe: string; options: [LimiteAnalyse, string][] }[] = [
  { groupe: "Durée", options: [1, 3, 10, 30, 60, 300].map(v => [{ type: "duree", valeur: v }, v < 60 ? `${v} s` : `${v / 60} min`]) },
  { groupe: "Profondeur", options: [8, 10, 12, 14, 16].map(v => [{ type: "profondeur", valeur: v }, `Profondeur ${v}`]) },
  { groupe: "Simulations", options: [400, 1600, 6400].map(v => [{ type: "simulations", valeur: v }, `${v.toLocaleString("fr-FR")} simulations`]) },
  { groupe: "Sans limite", options: [[{ type: "infini", valeur: 0 }, "Infinie (jusqu'à l'arrêt)"]] },
];
const cle = (l: LimiteAnalyse) => `${l.type}:${l.valeur}`;
const nombre = (n: number, d = 0) => n.toLocaleString("fr-FR", { maximumFractionDigits: d, minimumFractionDigits: d });

/** Avancement vers la limite, dans [0, 1] (null : analyse infinie). */
function avancement(a: TAnalyse, l: LimiteAnalyse): number | null {
  if (!a.en_cours || a.arretee) return 1;
  if (l.type === "duree") return Math.min(1, (a.secondes ?? 0) / l.valeur);
  if (l.type === "profondeur") return a.mains !== undefined ? null : Math.min(1, (a.profondeur ?? 0) / l.valeur);
  if (l.type === "simulations") return Math.min(1, (a.simulations ?? 0) / l.valeur);
  return null;
}

export function Analyse() {
  const analyse = useAnalyseActive();
  const a = useJeu(s => s.analyses[s.pos]);
  const pos = useJeu(s => s.pos);
  const enCours = useJeu(s => s.enAnalyse !== null && s.enAnalyse === s.pos);
  const limite = useJeu(s => s.limite);
  const fleche = useJeu(s => s.fleche);
  const humain = useJeu(s => s.humain);
  const vivant = useJeu(s => s.pos === s.images.length - 1);
  const iaReflechit = useJeu(s => s.occupe && s.ia);
  const s = useJeu.getState();

  if (!analyse) {
    return (
      <div className="analyse vide">
        <p>
          Le réseau entraîné évalue la position affichée, propose ses meilleurs coups avec la suite qu'il
          attend, et signale les victoires forcées.
        </p>
        <button type="button" className="on" onClick={s.basculerAnalyse}>Activer l'analyse (A)</button>
        <Legende />
      </div>
    );
  }
  const txt = a ? libelleScore(a) : null;
  const coups = a?.coups ?? [];   // réponse « indisponible » : pas de coups
  const reseau = a?.source === "reseau";
  const av = a && reseau ? avancement(a, limite) : null;
  const vitesse = a?.secondes ? (a.simulations ?? 0) / a.secondes : 0;
  return (
    <div className="analyse">
      <div className="tete-analyse">
        <div className={`gros-score ${txt?.cls ?? ""}`}>{txt?.texte ?? "…"}</div>
        <div className="det-score">
          <b>{txt?.detail || (enCours ? "Calcul…" : iaReflechit ? "L'IA réfléchit : l'analyse reprend après son coup" : "")}</b>
          {a && !a.fini && (
            <span>
              Bastions Or {a.bastions[0]} · Argent {a.bastions[1]}
              {reseau ? ` · vu par ${EQUIPE[a.observateur ?? a.trait]}` : ""}
              {a.mains ? ` · main du joueur plateau inconnue : moyenne sur ${a.mains} mains possibles` : ""}
            </span>
          )}
          {a?.indisponible && <span className="indispo">{a.indisponible}</span>}
          {a?.mat?.coup && <span>Coup gagnant : <code>{a.mat.coup}</code></span>}
        </div>
      </div>
      {reseau && a && (
        <div className={`recherche${enCours ? " active" : ""}`}>
          <div className="mesures">
            {a.profondeur ? (
              <span title="Passes d'approfondissement terminées : chaque profondeur double le nombre de simulations">
                Profondeur <b>{a.profondeur}</b>
              </span>
            ) : null}
            {a.horizon ? (
              <span title="Coups anticipés par les simulations (pièces jouées), en moyenne et au plus loin">
                horizon <b>{nombre(a.horizon, 1)}</b> coups (max {a.horizon_max})
              </span>
            ) : null}
            <span><b>{nombre(a.simulations ?? 0)}</b> simulations</span>
            {a.secondes ? <span>{nombre(a.secondes, 1)} s{vitesse ? ` · ${nombre(vitesse)}/s` : ""}</span> : null}
            {enCours ? <span className="etat">calcul en cours…</span> : a.arretee ? <span className="etat">arrêtée</span> : null}
          </div>
          <div className="jauge" aria-hidden="true">
            <div className={av === null ? "infinie" : ""} style={av === null ? undefined : { width: `${100 * av}%` }} />
          </div>
        </div>
      )}
      <div className="reglages-analyse">
        <label>Limite
          <select value={cle(limite)} onChange={e => {
            const [type, v] = e.target.value.split(":");
            s.setLimite({ type: type as LimiteAnalyse["type"], valeur: +v });
          }}>
            {LIMITES.map(g => (
              <optgroup key={g.groupe} label={g.groupe}>
                {g.options.map(([l, t]) => <option key={cle(l)} value={cle(l)}>{t}</option>)}
              </optgroup>
            ))}
          </select>
        </label>
        {enCours ? (
          <button type="button" onClick={() => s.arreterAnalyse(true)} title="Arrêter l'analyse : son résultat reste affiché">Arrêter</button>
        ) : a && reseau ? (
          <button type="button" onClick={() => s.analyser(pos)} title="Relancer l'analyse de cette position depuis le début">Relancer</button>
        ) : null}
        <label className="case-a-cocher">
          <input type="checkbox" checked={fleche} onChange={e => { s.set({ fleche: e.target.checked }); sauverPrefs(); }} /> Flèche du meilleur coup
        </label>
      </div>
      {a?.joue && (
        <p className={`coup-joue${a.joue.meilleur ? " bon" : a.joue.ecart >= 5 || a.joue.mat_manque ? " faute" : ""}`}
          title="Coup joué ensuite dans la partie, évalué par cette analyse">
          Coup joué : <code>{a.joue.coup}</code> <b>{a.joue.texte}</b>{" "}
          {a.joue.meilleur ? "— le meilleur coup de l'analyse"
            : a.joue.mat_manque ? `— victoire forcée manquée (${a.texte})`
              : `— ${a.joue.rang}ᵉ choix, ${nombre(a.joue.ecart, 1)} point${a.joue.ecart >= 2 ? "s" : ""} de moins que le meilleur`}
        </p>
      )}
      <ol className="meilleurs">
        {coups.map((c, k) => (
          <li key={k}>
            <button type="button" disabled={!(humain && vivant)}
              title={humain && vivant ? "Jouer ce coup" : c.description}
              onClick={() => s.jouer(c.i)}
              onMouseEnter={() => s.setSurvol(c.cases.length ? c.cases : null)} onMouseLeave={() => s.setSurvol(null)}>
              <span className={`s ${c.score > 0.05 ? "or" : c.score < -0.05 ? "argent" : ""}`}>{c.texte}</span>
              <code className="c">{c.coup}</code>
              <span className="p" title={`${nombre(c.visites)} simulations : plus un coup est exploré, plus son score est sûr`}>
                {c.part !== undefined ? `${Math.round(100 * c.part)} %` : `${Math.round(100 * c.probabilite)} %`}
              </span>
              <span className="d">{c.description}</span>
              {c.ligne.length > 1 && <span className="l">{c.ligne.slice(1).join("  ")}</span>}
            </button>
          </li>
        ))}
        {a && !coups.length && !a.fini && !a.indisponible && !a.mains && <li className="vide">Aucun coup à proposer.</li>}
      </ol>
      <Legende />
    </div>
  );
}

function Legende() {
  return (
    <details className="legende-eval">
      <summary>Lire l'évaluation</summary>
      <dl>
        <dt>+12,4</dt><dd>Or mène : position aussi favorable que 1,24 bastion d'avance (10 points = un Lieu contrôlé de plus que l'adversaire).</dd>
        <dt>−3,0</dt><dd>Argent mène (score négatif).</dd>
        <dt>#3 / #-2</dt><dd>Victoire forcée : Or (#3) ou Argent (#-2) pose son dernier marqueur en au plus 3 (2) coups, quoi que fasse l'adversaire et quels que soient les tirages du sac. Cherchée jusqu'à la fin de la manche.</dd>
        <dt>= ⩲ ± +−</dt><dd>Égalité, léger, net, décisif avantage d'Or (⩱ ∓ −+ pour Argent), comme aux échecs.</dd>
        <dt>Profondeur</dt><dd>L'analyse approfondit par passes successives, comme un moteur d'échecs : chaque profondeur double le nombre de simulations. L'horizon est le nombre de coups que les simulations anticipent (en moyenne, et au plus loin).</dd>
        <dt>Ordre</dt><dd>Les coups sont classés comme l'IA choisit le sien : d'abord les plus explorés, puis le meilleur score. Le premier est le coup que l'IA jouerait. Le pourcentage est la part des simulations consacrées au coup : un score peu exploré est moins sûr.</dd>
        <dt>Ligne</dt><dd>Suite la plus explorée par la recherche après le coup proposé ; un coup = une pièce jouée, avec ses effets enchaînés (&gt;).</dd>
      </dl>
    </details>
  );
}

/** Courbe de l'évaluation au fil de la partie (positions analysées) ; cliquer pour y aller. */
export function CourbeEval() {
  const analyses = useJeu(s => s.analyses);
  const n = useJeu(s => s.images.length);
  const pos = useJeu(s => s.pos);
  const ref = useRef<HTMLDivElement>(null);
  const [taille, setTaille] = useState({ w: 800, h: 80 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setTaille({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const pts = useMemo(
    () => Object.values(analyses).filter(a => a.score !== undefined).map(a => [a.pos, Math.max(-40, Math.min(40, a.score!))] as [number, number]).sort((a, b) => a[0] - b[0]),
    [analyses],
  );
  const { w, h } = taille;
  const gx = 10, gy = 6;
  const X = (x: number) => gx + ((w - 2 * gx) * x) / Math.max(1, n - 1);
  const Y = (v: number) => h / 2 - ((h / 2 - gy) * v) / 40;
  const ligne = pts.map(([x, v]) => `${X(x).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  const clic = (e: React.PointerEvent) => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    useJeu.getState().aller(Math.round(((e.clientX - r.left - gx) / (w - 2 * gx)) * (n - 1)));
  };
  return (
    <figure className="graphe courbe-eval">
      <figcaption>
        <span className="titre">Évaluation au fil de la partie (+ Or, − Argent ; 10 = un bastion)</span>
        <span className="quand">{pts.length} position{pts.length > 1 ? "s" : ""} analysée{pts.length > 1 ? "s" : ""}</span>
      </figcaption>
      <div className="zone" ref={ref}>
        <svg width={w} height={h} onPointerDown={clic} role="img" aria-label="Courbe d'évaluation">
          <defs>
            <clipPath id="haut-eval"><rect x={0} y={0} width={w} height={h / 2} /></clipPath>
            <clipPath id="bas-eval"><rect x={0} y={h / 2} width={w} height={h / 2} /></clipPath>
          </defs>
          <line className="axe" x1={gx} x2={w - gx} y1={h / 2} y2={h / 2} />
          {pts.length > 1 && (
            <>
              <polygon className="aire-or" clipPath="url(#haut-eval)" points={`${X(pts[0][0])},${h / 2} ${ligne} ${X(pts[pts.length - 1][0])},${h / 2}`} />
              <polygon className="aire-argent" clipPath="url(#bas-eval)" points={`${X(pts[0][0])},${h / 2} ${ligne} ${X(pts[pts.length - 1][0])},${h / 2}`} />
              <polyline className="courbe" points={ligne} />
            </>
          )}
          <line className="curseur-pos" x1={X(pos)} x2={X(pos)} y1={gy} y2={h - gy} />
          <text className="borne" x={gx} y={gy + 8}>Or</text>
          <text className="borne" x={gx} y={h - 2}>Argent</text>
        </svg>
      </div>
    </figure>
  );
}
