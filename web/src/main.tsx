import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/uncial-antiqua/latin-400.css";
import "@fontsource/alegreya-sans/latin-400.css";
import "@fontsource/alegreya-sans/latin-500.css";
import "@fontsource/alegreya-sans/latin-700.css";
import "./styles.css";
import { App } from "./App";

createRoot(document.getElementById("racine")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
