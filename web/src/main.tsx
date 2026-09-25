import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/uncial-antiqua/latin-400.css";
import "@fontsource/alegreya-sans/latin-400.css";
import "@fontsource/alegreya-sans/latin-500.css";
import "@fontsource/alegreya-sans/latin-700.css";
import "./styles.css";
import { App } from "./App";
import { Garde } from "./components/Garde";
import { PageJouer } from "./jouer/PageJouer";

// même application pour les deux pages : spectateur de l'entraînement (/) et partie (/jouer)
const jouer = window.location.pathname.replace(/\/+$/, "") === "/jouer";

createRoot(document.getElementById("racine")!).render(
  <StrictMode>
    <Garde>{jouer ? <PageJouer /> : <App />}</Garde>
  </StrictMode>,
);
