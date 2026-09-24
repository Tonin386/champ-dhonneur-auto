import { useEffect, useMemo, useRef } from "react";
import { EQUIPE } from "../jeu";
import type { Film } from "../types";

interface Props {
  film: Pick<Film, "decor" | "images"> | null;
  pos: number;
  aller: (pos: number) => void;
  /** tout le déroulé (partie jouée) ou seulement jusqu'au coup affiché (spectateur : sans dévoiler la suite) */
  complet?: boolean;
}

/** Déroulé de la partie ; cliquer sur un coup pour y revenir. */
export function Chronique({ film, pos, aller, complet }: Props) {
  const boite = useRef<HTMLOListElement>(null);

  const lignes = useMemo(() => {
    if (!film) return [];
    const out: { k: number; r: number; nouvelle: boolean; e: number; n: string; d: string }[] = [];
    let r = -1;
    film.images.forEach((img, k) => {
      if (!img.a) return;
      const prec = film.images[k - 1];
      out.push({ k, r: prec.r, nouvelle: prec.r !== r, e: film.decor.equipes[img.a.j], n: img.a.n, d: img.a.d });
      r = prec.r;
    });
    return out;
  }, [film]);

  useEffect(() => {
    const b = boite.current;
    if (!b) return;
    if (!complet) { b.scrollTop = b.scrollHeight; return; }   // le coup affiché est le dernier de la liste
    b.querySelector(".cur")?.scrollIntoView({ block: "nearest" });
  }, [pos, film, complet]);

  if (!film) return null;
  return (
    <ol className="chronique" ref={boite} aria-label="Déroulé de la partie">
      {lignes.filter(l => complet || l.k <= pos).map(l => (
        <li
          key={l.k}
          className={`e${l.e}${l.k === pos ? " cur" : ""}${l.k > pos ? " apres" : ""}${l.nouvelle ? " manche" : ""}`}
          data-manche={l.nouvelle ? `Manche ${l.r}` : undefined}
          onClick={() => aller(l.k)}
        >
          <span className="pastille" aria-label={EQUIPE[l.e]} />
          <code>{l.n}</code>
          <span className="desc">{l.d}</span>
        </li>
      ))}
      {!lignes.some(l => l.k <= pos) && <li className="vide">La partie commence…</li>}
    </ol>
  );
}
