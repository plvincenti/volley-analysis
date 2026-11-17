import { useEffect, useState } from "react";
import { api, StatRow } from "./api";
import { ACTION_LABELS } from "./labels";

/** Gestes de Corbas dans les échanges non supprimés ; `version` force le rechargement. */
export function StatsTable({ version }: { version: number }) {
  const [rows, setRows] = useState<StatRow[]>([]);
  useEffect(() => {
    api.stats().then(setRows).catch(() => setRows([]));
  }, [version]);

  return (
    <table className="stats">
      <thead>
        <tr>
          <th>Geste</th>
          <th>Total</th>
          <th>Réussis</th>
          <th>Gagnants</th>
          <th>Ratés</th>
          <th>Inconnus</th>
          <th>Réussite</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.action}>
            <td>{ACTION_LABELS[row.action]}</td>
            <td>{row.total}</td>
            <td>{row.reussi}</td>
            <td>{row.gagnant}</td>
            <td>{row.rate}</td>
            <td>{row.inconnu}</td>
            <td>{row.reussite_pct === null ? "–" : `${row.reussite_pct} %`}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
