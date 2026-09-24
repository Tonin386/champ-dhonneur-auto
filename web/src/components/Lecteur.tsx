import { useStore } from "../store";

const VITESSES: [number, string][] = [
  [1600, "Lent"], [800, "Normal"], [400, "Rapide"], [150, "Très rapide"], [40, "Éclair"],
];

export function Lecteur() {
  const film = useStore(s => s.film);
  const pos = useStore(s => s.pos);
  const lecture = useStore(s => s.lecture);
  const vitesse = useStore(s => s.vitesse);
  const source = useStore(s => s.source);
  const { aller, pas, basculerLecture, allerDirect, setVitesse } = useStore.getState();
  if (!film) return null;
  const n = film.images.length - 1;
  const img = film.images[pos];
  const direct = source?.type === "direct";
  const auBord = direct && pos >= n && !film.fini;
  return (
    <div className="lecteur">
      <div className="boutons">
        <button type="button" onClick={() => { aller(0); }} title="Début (Origine)" aria-label="Début">⏮</button>
        <button type="button" onClick={() => pas(-1)} title="Coup précédent (←)" aria-label="Coup précédent">◀</button>
        <button type="button" className="principal" onClick={basculerLecture} title="Lecture / pause (espace)">
          {lecture ? "Pause" : "Lecture"}
        </button>
        <button type="button" onClick={() => pas(1)} title="Coup suivant (→)" aria-label="Coup suivant">▶</button>
        <button type="button" onClick={() => aller(n)} title="Fin (Fin)" aria-label="Fin">⏭</button>
      </div>
      <div className="temps">
        <input
          type="range" min={0} max={n} value={pos} aria-label="Position dans la partie"
          onChange={e => { useStore.setState({ lecture: false }); aller(+e.target.value); }}
          style={{ "--p": `${n ? (100 * pos) / n : 0}%` } as React.CSSProperties}
        />
        <div className="compteurs">
          <span>Manche <b>{img.r}</b></span>
          <span>Décision <b>{pos}</b> / {n}{direct && !film.fini ? "+" : ""}</span>
        </div>
      </div>
      <select value={vitesse} onChange={e => setVitesse(+e.target.value)} aria-label="Vitesse de lecture">
        {VITESSES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
      <button
        type="button"
        className={`direct${auBord && lecture ? " actif" : ""}`}
        onClick={allerDirect}
        title="Rejoindre le direct (D)"
      >
        <span className="point" /> Direct
      </button>
    </div>
  );
}
