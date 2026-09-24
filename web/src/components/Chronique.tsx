import { useEffect, useMemo, useRef } from "react";
import { EQUIPE } from "../jeu";
import { useStore } from "../store";

/** Déroulé de la partie jusqu'au coup affiché (sans dévoiler la suite) ; cliquer pour revenir. */
export function Chronique() {
  const film = useStore(s => s.film);
  const pos = useStore(s => s.pos);
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
    const b = boite.current;   // le coup affiché est toujours le dernier de la liste
    if (b) b.scrollTop = b.scrollHeight;
  }, [pos, film]);

  if (!film) return null;
  return (
    <ol className="chronique" ref={boite} aria-label="Déroulé de la partie">
      {lignes.filter(l => l.k <= pos).map(l => (
        <li
          key={l.k}
          className={`e${l.e}${l.k === pos ? " cur" : ""}${l.nouvelle ? " manche" : ""}`}
          data-manche={l.nouvelle ? `Manche ${l.r}` : undefined}
          onClick={() => { useStore.setState({ lecture: false }); useStore.getState().aller(l.k); }}
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
