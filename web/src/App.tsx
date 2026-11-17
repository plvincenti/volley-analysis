import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Gesture, Match, Rally, Team } from "./api";
import { Player, rallyStart, rallyStop } from "./Player";
import { RallyList } from "./RallyList";
import { RallyPanel } from "./RallyPanel";
import { StatsTable } from "./StatsTable";

const RATES = [0.25, 0.5, 0.75, 1, 1.5, 2];

const SHORTCUTS: [string, string][] = [
  ["Espace", "lecture / pause"],
  ["← →", "reculer / avancer d'1 s (Maj : 1 image)"],
  ["↑ ↓", "échange précédent / suivant"],
  ["C / A", "point Corbas / adversaire"],
  ["Entrée", "valider et passer au suivant"],
  ["G", "ajouter le geste du formulaire à l'instant courant"],
  ["X", "couper l'échange en deux à l'instant courant"],
  ["D / F", "début / fin de l'échange à l'instant courant"],
  ["R", "revoir l'échange depuis le début"],
  ["L", "lecture en boucle"],
  ["- / +", "ralentir / accélérer"],
];

export default function App() {
  const [match, setMatch] = useState<Match | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [rally, setRally] = useState<Rally | null>(null);
  const [onlyToReview, setOnlyToReview] = useState(false);
  const [showOpponent, setShowOpponent] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [loop, setLoop] = useState(true);
  const [rate, setRate] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [version, setVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const addFormRef = useRef<HTMLFormElement>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  const reloadMatch = useCallback(() => {
    api
      .match()
      .then((loaded) => {
        setMatch(loaded);
        setVersion((v) => v + 1);
      })
      .catch(report);
  }, []);

  useEffect(reloadMatch, [reloadMatch]);

  useEffect(() => {
    if (selectedId !== null || !match?.rallies.length) return;
    const first = match.rallies.find((r) => r.status === "a_revoir") ?? match.rallies[0];
    setSelectedId(first.id);
  }, [match, selectedId]);

  useEffect(() => {
    if (selectedId === null) return;
    api.rally(selectedId).then(setRally).catch(report);
  }, [selectedId]);

  const visible = useMemo(
    () =>
      (match?.rallies ?? []).filter(
        (r) => !onlyToReview || r.status === "a_revoir" || r.id === selectedId,
      ),
    [match, onlyToReview, selectedId],
  );

  const apply = (promise: Promise<Rally>) =>
    promise
      .then((updated) => {
        setRally(updated);
        reloadMatch();
      })
      .catch(report);

  const move = (delta: number) => {
    const index = visible.findIndex((r) => r.id === selectedId);
    const next = visible[index + delta];
    if (next) setSelectedId(next.id);
  };

  const seek = (time: number) => {
    const video = videoRef.current;
    if (video) video.currentTime = Math.max(0, time);
  };

  const setWinner = (winner: Team) => rally && apply(api.updateRally(rally.id, { winner }));

  const validateAndNext = () => {
    if (!rally) return;
    api
      .updateRally(rally.id, { status: "valide" })
      .then(() => {
        reloadMatch();
        move(1);
      })
      .catch(report);
  };

  const toggleDeleted = () =>
    rally &&
    apply(
      api.updateRally(rally.id, { status: rally.status === "supprime" ? "a_revoir" : "supprime" }),
    );

  /** Instant lu sur la vidéo elle-même, s'il tombe dans l'échange affiché. */
  const videoTime = () => {
    const time = videoRef.current?.currentTime;
    if (!rally || time === undefined || time < rallyStart(rally) || time > rallyStop(rally)) {
      setError("La vidéo n'est pas sur l'échange affiché : relancer l'échange (R) puis réessayer");
      return null;
    }
    return time;
  };

  const split = () => {
    const time = videoTime();
    if (!rally || time === null || time <= rally.serve_s || time >= rally.end_s) return;
    apply(api.splitRally(rally.id, time));
  };

  const setStart = () => {
    const time = videoTime();
    if (rally && time !== null && time < rally.end_s) apply(api.updateRally(rally.id, { serve_s: time }));
  };

  const setEnd = () => {
    const time = videoTime();
    if (rally && time !== null && time > rally.serve_s) apply(api.updateRally(rally.id, { end_s: time }));
  };

  const addGesture = (gesture: { team: Team; action: string; result: string }) => {
    const time = videoTime();
    if (rally && time !== null) apply(api.addGesture(rally.id, { ...gesture, time_s: time }));
  };

  const handleKey = useRef<(event: KeyboardEvent) => void>(() => undefined);
  handleKey.current = (event: KeyboardEvent) => {
    const target = event.target as HTMLElement;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName) || event.metaKey || event.ctrlKey) {
      return;
    }
    const video = videoRef.current;
    const frame = 1 / (match?.videos.find((v) => v.name === rally?.video)?.fps ?? 30);
    const actions: Record<string, () => void> = {
      " ": () => (video?.paused ? video.play() : video?.pause()),
      ArrowLeft: () => video && seek(video.currentTime - (event.shiftKey ? frame : 1)),
      ArrowRight: () => video && seek(video.currentTime + (event.shiftKey ? frame : 1)),
      ArrowUp: () => move(-1),
      ArrowDown: () => move(1),
      c: () => setWinner("Corbas"),
      a: () => setWinner("Adversaire"),
      Enter: validateAndNext,
      g: () => addFormRef.current?.requestSubmit(),
      x: split,
      d: setStart,
      f: setEnd,
      r: () => {
        if (!rally || !video) return;
        seek(rallyStart(rally));
        video.play().catch(() => undefined);
      },
      l: () => setLoop((value) => !value),
      "-": () => setRate((value) => RATES[Math.max(0, RATES.indexOf(value) - 1)]),
      "+": () => setRate((value) => RATES[Math.min(RATES.length - 1, RATES.indexOf(value) + 1)]),
    };
    const run = actions[event.key.length === 1 ? event.key.toLowerCase() : event.key];
    if (run) {
      event.preventDefault();
      run();
    }
  };

  useEffect(() => {
    const listener = (event: KeyboardEvent) => handleKey.current(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  if (!match) {
    return <div className="loading">{error ?? "Chargement du match…"}</div>;
  }

  const kept = match.rallies.filter((r) => r.status !== "supprime");
  const validated = kept.filter((r) => r.status === "valide").length;
  const setScores = [...new Set(kept.map((r) => r.set_number))].map((set) => {
    const last = kept.filter((r) => r.set_number === set).at(-1)!;
    return { set, ...last.score };
  });

  return (
    <div className="app">
      <header className="top-bar">
        <h1>Revue du match</h1>
        <div className="set-scores">
          {setScores.map((s) => (
            <span key={s.set}>
              Set {s.set} : Corbas {s.corbas} - {s.adversaire}
            </span>
          ))}
        </div>
        <div className="progress">
          {validated} / {kept.length} échanges validés
          <progress value={validated} max={kept.length} />
        </div>
        <button onClick={() => setShowStats((v) => !v)}>
          {showStats ? "Masquer les statistiques" : "Statistiques Corbas"}
        </button>
      </header>

      {error && (
        <div className="error" onClick={() => setError(null)}>
          {error} (cliquer pour fermer)
        </div>
      )}
      {showStats && (
        <div className="stats-panel">
          <StatsTable version={version} />
        </div>
      )}

      <main className="layout">
        <RallyList
          rallies={visible}
          selectedId={selectedId}
          onlyToReview={onlyToReview}
          onToggleOnlyToReview={() => setOnlyToReview((v) => !v)}
          onSelect={setSelectedId}
        />

        <div className="center">
          {rally && (
            <Player
              rally={rally}
              videoRef={videoRef}
              loop={loop}
              rate={rate}
              currentTime={currentTime}
              showOpponent={showOpponent}
              onTime={setCurrentTime}
            />
          )}
          <div className="player-options">
            <label>
              <input type="checkbox" checked={loop} onChange={() => setLoop((v) => !v)} />
              En boucle
            </label>
            <label>
              Vitesse
              <select value={rate} onChange={(e) => setRate(Number(e.target.value))}>
                {RATES.map((r) => (
                  <option key={r} value={r}>
                    ×{r}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <dl className="shortcuts">
            {SHORTCUTS.map(([keys, description]) => (
              <div key={keys}>
                <dt>{keys}</dt>
                <dd>{description}</dd>
              </div>
            ))}
          </dl>
        </div>

        {rally && (
          <RallyPanel
            rally={rally}
            actions={match.actions}
            results={match.results}
            currentTime={currentTime}
            showOpponent={showOpponent}
            addFormRef={addFormRef}
            onToggleOpponent={() => setShowOpponent((v) => !v)}
            onWinner={setWinner}
            onValidate={validateAndNext}
            onToggleDeleted={toggleDeleted}
            onSplit={split}
            onStart={setStart}
            onEnd={setEnd}
            onServingTeam={(team) => apply(api.updateRally(rally.id, { serving_team: team }))}
            onSeek={seek}
            onAddGesture={addGesture}
            onUpdateGesture={(id, changes: Partial<Gesture>) =>
              apply(api.updateGesture(id, changes))
            }
            onDeleteGesture={(id) => apply(api.deleteGesture(id))}
          />
        )}
      </main>
    </div>
  );
}
