import { Component, type ReactNode } from "react";

/** Une erreur d'affichage ne laisse plus une page vierge : message et rechargement (la partie
 *  en cours est reprise au rechargement, voir jouer/store.ts). */
export class Garde extends Component<{ children: ReactNode }, { erreur: Error | null }> {
  state = { erreur: null as Error | null };

  static getDerivedStateFromError(erreur: Error) {
    return { erreur };
  }

  componentDidCatch(erreur: Error, info: { componentStack?: string | null }) {
    console.error("Erreur d'affichage", erreur, info.componentStack);
  }

  render() {
    const e = this.state.erreur;
    if (!e) return this.props.children;
    return (
      <div className="vide-total garde" role="alert">
        <h1>Erreur d'affichage</h1>
        <p>La page a rencontré une erreur. La partie est conservée : rechargez pour la reprendre.</p>
        <code>{e.message}</code>
        <div className="actions-dialogue">
          <button type="button" onClick={() => this.setState({ erreur: null })}>Réessayer sans recharger</button>
          <button type="button" className="on" onClick={() => window.location.reload()}>Recharger</button>
        </div>
      </div>
    );
  }
}
