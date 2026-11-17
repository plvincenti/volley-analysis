import { RefObject, useState } from "react";
import { Gesture, Rally, Team } from "./api";
import { ACTION_LABELS, formatTime, ORIGIN_LABELS, RESULT_LABELS } from "./labels";

interface RallyPanelProps {
  rally: Rally;
  actions: string[];
  results: string[];
  currentTime: number;
  showOpponent: boolean;
  addFormRef: RefObject<HTMLFormElement | null>;
  onToggleOpponent: () => void;
  onWinner: (winner: Team) => void;
  onValidate: () => void;
  onToggleDeleted: () => void;
  onSplit: () => void;
  onStart: () => void;
  onEnd: () => void;
  onServingTeam: (team: Team) => void;
  onSeek: (time: number) => void;
  onAddGesture: (gesture: { team: Team; action: string; result: string }) => void;
  onUpdateGesture: (id: number, changes: Partial<Gesture>) => void;
  onDeleteGesture: (id: number) => void;
}

export function RallyPanel(props: RallyPanelProps) {
  const { rally, actions, results, currentTime, showOpponent } = props;
  const [newAction, setNewAction] = useState("attaque");
  const [newResult, setNewResult] = useState("reussi");
  const [newTeam, setNewTeam] = useState<Team>("Corbas");
  const gestures = rally.gestures.filter((g) => showOpponent || g.team === "Corbas");
  const deleted = rally.status === "supprime";

  return (
    <section className="rally-panel">
      <header className="rally-header">
        <h2>
          Échange #{rally.label} <small>set {rally.set_number}</small>
        </h2>
        <p>
          {rally.video} · {formatTime(rally.serve_s)} → {formatTime(rally.end_s)} · service{" "}
          {rally.serving_team}
        </p>
        <p className="rally-header-score">
          Score après l'échange : Corbas {rally.score.corbas} - {rally.score.adversaire}{" "}
          Adversaire
        </p>
      </header>

      <div className="panel-block">
        <h3>Point</h3>
        <div className="winner-buttons">
          {(["Corbas", "Adversaire"] as Team[]).map((team) => (
            <button
              key={team}
              className={`winner-button winner-${team} ${rally.winner === team ? "active" : ""}`}
              onClick={() => props.onWinner(team)}
              disabled={deleted}
            >
              {team} <kbd>{team === "Corbas" ? "C" : "A"}</kbd>
            </button>
          ))}
        </div>
        {rally.auto_winner !== rally.winner && (
          <p className="hint">Détecté automatiquement : {rally.auto_winner ?? "inconnu"}</p>
        )}
      </div>

      <div className="panel-block">
        <div className="block-title">
          <h3>Gestes</h3>
          <label className="small">
            <input type="checkbox" checked={showOpponent} onChange={props.onToggleOpponent} />
            Voir l'adversaire
          </label>
        </div>
        {gestures.length === 0 && <p className="hint">Aucun geste détecté.</p>}
        <table className="gestures">
          <tbody>
            {gestures.map((gesture) => (
              <tr key={gesture.id} className={`team-${gesture.team}`}>
                <td>
                  <button
                    className="time-button"
                    title="Aller à ce geste"
                    onClick={() => props.onSeek(Math.max(0, gesture.time_s - 0.5))}
                  >
                    {formatTime(gesture.time_s)}
                  </button>
                </td>
                {showOpponent && <td className="small">{gesture.team}</td>}
                <td>
                  <select
                    value={gesture.action}
                    onChange={(event) => {
                      props.onUpdateGesture(gesture.id, { action: event.target.value });
                      event.target.blur();
                    }}
                  >
                    {actions.map((action) => (
                      <option key={action} value={action}>
                        {ACTION_LABELS[action]}
                      </option>
                    ))}
                  </select>
                </td>
                <td>
                  <select
                    className={`result-select result-${gesture.result}`}
                    value={gesture.result}
                    onChange={(event) => {
                      props.onUpdateGesture(gesture.id, { result: event.target.value });
                      event.target.blur();
                    }}
                  >
                    {results.map((result) => (
                      <option key={result} value={result}>
                        {RESULT_LABELS[result]}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="small origin" title={gesture.edited ? "corrigé à la main" : ""}>
                  {ORIGIN_LABELS[gesture.origin] ?? gesture.origin}
                  {gesture.edited ? " ✎" : ""}
                </td>
                <td>
                  <button
                    className="icon-button"
                    title="Supprimer ce geste"
                    onClick={() => props.onDeleteGesture(gesture.id)}
                  >
                    ✕
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <form
          ref={props.addFormRef}
          className="add-gesture"
          onSubmit={(event) => {
            event.preventDefault();
            props.onAddGesture({
              team: newTeam,
              action: newAction,
              result: newResult,
            });
            (document.activeElement as HTMLElement | null)?.blur();
          }}
        >
          <span className="small">À {formatTime(currentTime)}</span>
          <select value={newAction} onChange={(e) => setNewAction(e.target.value)}>
            {actions.map((action) => (
              <option key={action} value={action}>
                {ACTION_LABELS[action]}
              </option>
            ))}
          </select>
          <select value={newResult} onChange={(e) => setNewResult(e.target.value)}>
            {results.map((result) => (
              <option key={result} value={result}>
                {RESULT_LABELS[result]}
              </option>
            ))}
          </select>
          {showOpponent && (
            <select value={newTeam} onChange={(e) => setNewTeam(e.target.value as Team)}>
              <option value="Corbas">Corbas</option>
              <option value="Adversaire">Adversaire</option>
            </select>
          )}
          <button type="submit">
            Ajouter <kbd>G</kbd>
          </button>
        </form>
      </div>

      <div className="panel-block">
        <h3>Découpage</h3>
        <div className="split-buttons">
          <button
            onClick={props.onSplit}
            disabled={deleted || currentTime <= rally.serve_s || currentTime >= rally.end_s}
            title="Le second échange commence à l'instant courant de la vidéo"
          >
            Couper ici, à {formatTime(currentTime)} <kbd>X</kbd>
          </button>
          <button
            onClick={props.onStart}
            disabled={deleted || currentTime >= rally.end_s}
          >
            Début ici <kbd>D</kbd>
          </button>
          <button
            onClick={props.onEnd}
            disabled={deleted || currentTime <= rally.serve_s}
          >
            Fin ici <kbd>F</kbd>
          </button>
        </div>
        <div className="serving-team">
          <span className="small">Service :</span>
          {(["Corbas", "Adversaire"] as Team[]).map((team) => (
            <button
              key={team}
              className={rally.serving_team === team ? `active winner-${team}` : ""}
              onClick={() => props.onServingTeam(team)}
              disabled={deleted}
            >
              {team}
            </button>
          ))}
        </div>
      </div>

      <div className="panel-block actions-row">
        <button className="primary" onClick={props.onValidate} disabled={deleted}>
          {rally.status === "valide" ? "Validé ✓ — suivant" : "Valider et suivant"} <kbd>Entrée</kbd>
        </button>
        <button className="danger" onClick={props.onToggleDeleted}>
          {deleted ? "Restaurer l'échange" : "Pas un échange (supprimer)"}
        </button>
      </div>
    </section>
  );
}
