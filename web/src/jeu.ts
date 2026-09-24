// Constantes et géométrie du jeu, partagées par les composants.
import type { Decor } from "./types";

export const EQUIPE = ["Blanc", "Noir"] as const;

export const NOMS: Record<string, string> = {
  A: "Archer", B: "Berserk", C: "Cavalerie", D: "Porte étendard", E: "Éclaireur", F: "Fantassin",
  G: "Garde royale", H: "Cavalerie légère", K: "Capitaine", L: "Lancier", M: "Mercenaire",
  N: "Chevalier", P: "Piquier", R: "Moine soldat", S: "Soldat", X: "Arbalétrier", "*": "Sceau royal",
};

export const imgPiece = (c: string, equipe: number) =>
  c === "*" ? `/img/pieces/sceau-${equipe ? "noir" : "blanc"}.png` : `/img/pieces/${c}.png`;

export interface Geometrie {
  s: number;
  h: number;
  W: number;
  H: number;
  centre: (i: number) => [number, number];
}

const CACHE = new WeakMap<Decor, Geometrie>();

/** Hexagones « à sommet plat » en colonnes ; Blanc en bas, Noir en haut. */
export function geometrie(d: Decor): Geometrie {
  const g0 = CACHE.get(d);
  if (g0) return g0;
  const s = 40;
  const h = (Math.sqrt(3) / 2) * s;
  const pad = 6;
  const pts = d.cases.map(([, x, y2]) => [pad + s + x * 1.5 * s, pad + h + (d.max_y2 - y2) * h] as [number, number]);
  const g = { s, h, W: pad * 2 + s * 2 + (d.cols - 1) * 1.5 * s, H: pad * 2 + h * 2 + d.max_y2 * h, centre: (i: number) => pts[i] };
  CACHE.set(d, g);
  return g;
}

export function hexagone(cx: number, cy: number, s: number): string {
  const p: string[] = [];
  for (let k = 0; k < 6; k++) {
    const t = (Math.PI / 3) * k;
    p.push(`${(cx + s * Math.cos(t)).toFixed(1)},${(cy + s * Math.sin(t)).toFixed(1)}`);
  }
  return p.join(" ");
}

export const nomJoueur = (nom: string | undefined) =>
  !nom ? "?" : nom === "heuristique" ? "Heuristique" : nom.replace(/^iter_0*(\d+)$/, "Réseau it. $1");

export function duree(s: number): string {
  if (!isFinite(s) || s < 0) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(1).replace(".", ",")} h`;
  return `${(s / 86400).toFixed(1).replace(".", ",")} j`;
}

export const entier = (n: number | null | undefined) =>
  n == null ? "—" : Math.round(n).toLocaleString("fr-FR");

export function compact(n: number | null | undefined): string {
  if (n == null) return "—";
  const a = Math.abs(n);
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 0 : 1).replace(".", ",")} M`;
  if (a >= 1e4) return `${(n / 1e3).toFixed(a >= 1e5 ? 0 : 1).replace(".", ",")} k`;
  return entier(n);
}
