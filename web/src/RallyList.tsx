import { useEffect, useRef } from "react";
import { RallySummary } from "./api";
import { formatTime } from "./labels";

interface RallyListProps {
  rallies: RallySummary[];
  selectedId: number | null;
  onlyToReview: boolean;
  onToggleOnlyToReview: () => void;
  onSelect: (id: number) => void;
}

export function RallyList({
  rallies,
  selectedId,
  onlyToReview,
  onToggleOnlyToReview,
  onSelect,
}: RallyListProps) {
  const selectedRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);

  return (
    <aside className="rally-list">
      <label className="rally-filter">
        <input type="checkbox" checked={onlyToReview} onChange={onToggleOnlyToReview} />
        Seulement les échanges à revoir
      </label>
      {rallies.map((rally, i) => (
        <div key={rally.id}>
          {(i === 0 || rallies[i - 1].set_number !== rally.set_number) && (
            <h3 className="set-title">Set {rally.set_number}</h3>
          )}
          <button
            ref={rally.id === selectedId ? selectedRef : undefined}
            className={[
              "rally-item",
              rally.id === selectedId ? "selected" : "",
              `status-${rally.status}`,
            ].join(" ")}
            onClick={() => onSelect(rally.id)}
          >
            <span className="rally-number">#{rally.label}</span>
            <span className="rally-time">
              {rally.video} · {formatTime(rally.serve_s)}
            </span>
            <span className={`winner-chip winner-${rally.winner ?? "aucun"}`}>
              {rally.winner === "Corbas" ? "C" : rally.winner === "Adversaire" ? "A" : "?"}
            </span>
            <span className="rally-score">
              {rally.score.corbas}-{rally.score.adversaire}
            </span>
            <span className="rally-status">
              {rally.status === "valide" ? "✓" : rally.status === "supprime" ? "✕" : ""}
            </span>
          </button>
        </div>
      ))}
    </aside>
  );
}
