import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { EQUIPE } from "../jeu";
import { auTraitIA, sauverPrefs, useAnalyseActive, useJeu, useTourIA } from "./store";
import type { CoupAnalyse, LimiteAnalyse, Analyse as TAnalyse } from "./types";

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

/** Dernière évaluation connue à la position affichée ou avant (quelques positions en arrière). */
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

/** Chances de gain (une nulle compte pour moitié), Or d'un côté, Argent de l'autre, dans l'ordre
 *  des colonnes des joueurs (plateau retourné : Argent à gauche). */
export function Chances({ gainOr, compact }: { gainOr: number; compact?: boolean }) {
  const retourne = useJeu(s => s.retourne);
  const or = Math.round(100 * gainOr);
  return (
    <div className={`chances${compact ? " compact" : ""}${retourne ? " retournee" : ""}`} role="meter"
      aria-label="Chances de gain d'Or" aria-valuemin={0} aria-valuemax={100} aria-valuenow={or}
      title="Chances de gain estimées par le réseau (une partie nulle compte pour moitié)">
      {!compact && (
        <div className="libelles">
          <span className="or">Or <b>{or} %</b></span>
          <span className="quoi">chances de gain</span>
          <span className="argent"><b>{100 - or} %</b> Argent</span>
        </div>
      )}
      <div className="jauge-chances"><div className="part-or-h" style={{ width: `${100 * gainOr}%` }} /></div>
    </div>
  );
}

/** L'IA a le trait : c'est vous qui décidez quand elle joue. Elle joue le premier coup de sa
 *  réflexion, telle qu'elle en est (sans réflexion encore : une courte recherche). `large` : avec
 *  la description du coup. */
export function JouerMaintenant({ large }: { large?: boolean }) {
  const tour = useTourIA();
  const fini = useJeu(s => s.fini);
  const occupe = useJeu(s => s.occupe);
  const hybride = useJeu(s => s.hybride !== null);
  const c = useJeu(s => s.analyses[s.images.length - 1]?.coups?.[0]);
  if (!tour || fini) return null;
  return (
    <button type="button" className={`on jouer-maintenant${large ? " large" : ""}`} disabled={occupe}
      onClick={() => void useJeu.getState().coupIA()}
      title={`L'IA joue le premier coup de sa réflexion, telle qu'elle en est (Espace)${c ? ` : ${c.description}` : ""}`}>
      <span className="quoi">{occupe ? "L'IA joue…" : hybride ? "Jouer son meilleur coup" : "L'IA joue maintenant"}</span>
      {c && !occupe && <code>{c.coup}</code>}
      {large && c && !occupe && <span className="desc">{c.description}</span>}
      {!occupe && <kbd>Espace</kbd>}
    </button>
  );
}

const signe = (x: number, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "±") + nombre(Math.abs(x), d);
const clsScore = (x: number) => (x > 0.05 ? "or" : x < -0.05 ? "argent" : "");

/** Une ligne de l'analyse : rang, score, écart au meilleur (pour le joueur au trait), coup,
 *  part de la réflexion, suite attendue ; survolée, elle est montrée sur le plateau. */
function LigneCoup({ c, rang, meilleur, pour, jouable, reflexion }: {
  c: CoupAnalyse; rang: number; meilleur: CoupAnalyse; pour: number; jouable: boolean; reflexion: boolean;
}) {
  const s = useJeu.getState();
  const mat = Math.abs(c.score) >= 99 || Math.abs(meilleur.score) >= 99;
  const ecart = (c.score - meilleur.score) * (pour === 0 ? 1 : -1);   // ≤ 0 : moins bon pour le joueur au trait
  const part = c.part ?? c.probabilite;
  const gain = c.gain_or === undefined ? null : Math.round(100 * (pour === 0 ? c.gain_or : 1 - c.gain_or));
  const survoler = (oui: boolean) => s.set(oui ? { survol: c.cases.length ? c.cases : null, survolLigne: rang - 1 }
    : { survol: null, survolLigne: null });
  return (
    <li className={`ligne-coup${rang === 1 ? " premier" : ""}`}>
      {/* aria-disabled plutôt que disabled : un bouton désactivé ne reçoit pas le survol (ligne sur le plateau) */}
      <button type="button" aria-disabled={!jouable} onClick={() => { if (jouable) s.jouer(c.i); }}
        title={jouable ? (reflexion ? "Jouer ce coup pour l'IA" : "Jouer ce coup") : "Survolez : la ligne est montrée sur le plateau"}
        onMouseEnter={() => survoler(true)} onMouseLeave={() => survoler(false)}
        onFocus={() => survoler(true)} onBlur={() => survoler(false)}>
        <span className="rang">{rang}</span>
        <span className={`s ${clsScore(c.score)}`}>{c.texte}</span>
        <span className="ecart">
          {rang === 1 ? (reflexion ? "coup de l'IA" : "meilleur") : mat || c.visites === 0 ? "" : signe(ecart)}
        </span>
        <span className="c"><code>{c.coup}</code><span className="d" title={c.description}>{c.description}</span></span>
        <span className="p" title={`${nombre(c.visites)} simulations : plus un coup est exploré, plus son score est sûr`}>
          <span className="jauge-part"><span style={{ width: `${Math.max(3, 100 * part)}%` }} /></span>
          {Math.round(100 * part)} %
        </span>
        <span className="pv">
          {c.ligne.slice(1).map((m, j) => (
            <span key={j} className={`m e${c.equipes?.[j + 1] ?? ""}`} title={c.ligne_desc?.[j + 1]}>
              <i>{j + 2}</i>{m}
            </span>
          ))}
        </span>
        {gain !== null && (
          <span className="g" title="Chances de gain du joueur au trait après ce coup">{gain} % gain</span>
        )}
      </button>
    </li>
  );
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
  const iaJoue = useJeu(s => s.occupe && auTraitIA(s));
  const tourIA = useTourIA();
  const hybride = useJeu(s => s.hybride !== null);
  const equipes = useJeu(s => s.decor?.equipes);
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
  const trait = a?.trait ?? 0;
  const pour = equipes ? equipes[trait] : trait;   // équipe au trait : sens des écarts
  // réflexion de l'IA : la position actuelle, l'IA au trait (hybride : on choisit aussi son coup)
  const reflexion = tourIA && vivant && !a?.fini;
  return (
    <div className={`analyse${reflexion ? " reflexion" : ""}`}>
      {reflexion && (
        <div className="tete-reflexion">
          <div className="titre-reflexion">
            <span className={`pouls${enCours ? " actif" : ""}`} aria-hidden="true" />
            <b>Réflexion de l'IA · {EQUIPE[pour]}</b>
            <span>{hybride ? "cliquez une ligne pour la jouer" : "elle jouera la ligne 1 quand vous le déciderez"}</span>
          </div>
          <JouerMaintenant large />
        </div>
      )}
      <div className="tete-analyse">
        <div className={`gros-score ${txt?.cls ?? ""}`}>{txt?.texte ?? "…"}</div>
        <div className="det-score">
          <b>{txt?.detail || (iaJoue ? "L'IA joue…" : enCours ? "Calcul…" : "")}</b>
          {a && !a.fini && (
            <span>
              {reseau ? `vu par ${EQUIPE[a.observateur ?? a.trait]} · ` : ""}bastions {a.bastions[0]} – {a.bastions[1]}
              {a.mains ? ` · main du joueur plateau inconnue : moyenne sur ${a.mains} mains possibles` : ""}
            </span>
          )}
          {a?.indisponible && <span className="indispo">{a.indisponible}</span>}
          {a?.mat?.coup && <span>Coup gagnant : <code>{a.mat.coup}</code></span>}
        </div>
      </div>
      {a && !a.fini && a.gain_or !== undefined && <Chances gainOr={a.gain_or} />}
      {a?.joue && (
        <p className={`coup-joue${a.joue.meilleur ? " bon" : a.joue.ecart >= 5 || a.joue.mat_manque ? " faute" : ""}`}
          title="Coup joué ensuite dans la partie, évalué par cette analyse">
          Coup joué : <code>{a.joue.coup}</code> <b>{a.joue.texte}</b>{" "}
          {a.joue.meilleur ? "— le meilleur coup de l'analyse"
            : a.joue.mat_manque ? `— victoire forcée manquée (${a.texte})`
              : `— ${a.joue.rang}ᵉ choix, ${nombre(a.joue.ecart, 1)} point${a.joue.ecart >= 2 ? "s" : ""} de moins que le meilleur`}
        </p>
      )}
      {coups.length > 0 && (
        <div className="titre-lignes">
          <span>{reflexion ? "Lignes de l'IA" : "Meilleures lignes"}</span>
          <span className="aide-lignes">survolez une ligne pour la voir sur le plateau</span>
        </div>
      )}
      <ol className="meilleurs">
        {coups.map((c, k) => (
          <LigneCoup key={k} c={c} rang={k + 1} meilleur={coups[0]} pour={pour} jouable={humain && vivant} reflexion={reflexion} />
        ))}
        {a && !coups.length && !a.fini && !a.indisponible && !a.mains && <li className="vide">Aucun coup à proposer.</li>}
      </ol>
      <div className="pied-analyse">
        {reseau && a && (
          <div className={`recherche${enCours ? " active" : ""}`}>
            <div className="mesures">
              {a.profondeur ? (
                <span title="Passes d'approfondissement terminées : chaque profondeur double le nombre de simulations">
                  profondeur <b>{a.profondeur}</b>
                </span>
              ) : null}
              {a.horizon ? (
                <span title="Coups anticipés par les simulations (pièces jouées), en moyenne et au plus loin">
                  horizon <b>{nombre(a.horizon, 1)}</b> (max {a.horizon_max})
                </span>
              ) : null}
              <span><b>{nombre(a.simulations ?? 0)}</b> sim.</span>
              {a.secondes ? <span>{nombre(a.secondes, 1)} s{vitesse ? ` · ${nombre(vitesse)}/s` : ""}</span> : null}
              {enCours ? <span className="etat">en cours</span> : a.arretee ? <span className="etat">arrêtée</span> : null}
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
          <label className="case-a-cocher" title="Flèche bleue du meilleur coup sur le plateau">
            <input type="checkbox" checked={fleche} onChange={e => { s.set({ fleche: e.target.checked }); sauverPrefs(); }} /> Flèche
          </label>
        </div>
        <Legende />
      </div>
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
        <dt>Chances</dt><dd>Chances de gain estimées par le réseau pour chaque camp (une partie nulle compte pour moitié) ; sous chaque coup, celles du joueur au trait après ce coup.</dd>
        <dt>−2,7</dt><dd>Écart d'un coup avec le premier, pour le joueur au trait : ce qu'il perd en le préférant.</dd>
        <dt>Profondeur</dt><dd>L'analyse approfondit par passes successives, comme un moteur d'échecs : chaque profondeur double le nombre de simulations. L'horizon est le nombre de coups que les simulations anticipent (en moyenne, et au plus loin).</dd>
        <dt>Ordre</dt><dd>Les coups sont classés comme l'IA choisit le sien : d'abord les plus explorés, puis le meilleur score. Le premier est le coup que l'IA jouerait : c'est lui qu'elle joue quand vous cliquez « L'IA joue maintenant » (Espace). La jauge et le pourcentage sont la part des simulations consacrées au coup : un score peu exploré est moins sûr.</dd>
        <dt>Ligne</dt><dd>Suite la plus explorée par la recherche après le coup proposé, numérotée, en couleur du camp qui joue chaque coup (Or, Argent) ; un coup = une pièce jouée, avec ses effets enchaînés (&gt;). Survolée, la ligne est dessinée sur le plateau avec les mêmes numéros.</dd>
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
  const ici = analyses[pos];
  const txt = ici && ici.score !== undefined ? libelleScore(ici) : null;
  const clic = (e: React.PointerEvent) => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    useJeu.getState().aller(Math.round(((e.clientX - r.left - gx) / (w - 2 * gx)) * (n - 1)));
  };
  return (
    <figure className="graphe courbe-eval">
      <figcaption>
        <span className="titre">Évaluation au fil de la partie (+ Or, − Argent ; 10 = un bastion)</span>
        {txt && <span className={`valeur-courbe ${txt.cls}`}>décision {pos} : <b>{txt.texte}</b></span>}
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
          {pts.map(([x, v]) => (
            <circle key={x} className={`point-eval ${v > 0.05 ? "or" : v < -0.05 ? "argent" : ""}${x === pos ? " ici" : ""}`}
              cx={X(x)} cy={Y(v)} r={x === pos ? 4 : 2.4} />
          ))}
          <line className="curseur-pos" x1={X(pos)} x2={X(pos)} y1={gy} y2={h - gy} />
          <text className="borne" x={gx} y={gy + 8}>Or</text>
          <text className="borne" x={gx} y={h - 2}>Argent</text>
        </svg>
      </div>
    </figure>
  );
}
