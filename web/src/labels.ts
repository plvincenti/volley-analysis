export const ACTION_LABELS: Record<string, string> = {
  service: "Service",
  reception: "Réception",
  passe: "Passe",
  attaque: "Attaque",
  contre: "Contre",
};

export const RESULT_LABELS: Record<string, string> = {
  reussi: "Réussi",
  rate: "Raté",
  gagnant: "Gagnant",
  inconnu: "Inconnu",
};

export const ORIGIN_LABELS: Record<string, string> = {
  touche: "touche vue",
  service: "service détecté",
  filet: "déduit d'un passage du filet",
  sans_touche: "ballon tombé sans être touché",
  manuel: "saisi à la main",
};

/** Temps dans la vidéo, comme `5:02.2`. */
export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = (seconds - minutes * 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${rest}`;
}
