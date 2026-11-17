export type Team = "Corbas" | "Adversaire";
export type Status = "a_revoir" | "valide" | "supprime";

export interface Video {
  name: string;
  position: number;
  fps: number;
  duration_s: number;
}

export interface RallySummary {
  id: number;
  label: string;
  video: string;
  set_number: number;
  serve_s: number;
  end_s: number;
  serving_team: Team;
  auto_winner: Team | null;
  winner: Team | null;
  status: Status;
  note: string;
  corbas_gestures: number;
  score: { corbas: number; adversaire: number };
}

export interface Gesture {
  id: number;
  rally_id: number;
  time_s: number;
  team: Team;
  action: string;
  result: string;
  track_id: number;
  origin: string;
  edited: number;
}

export interface Rally extends RallySummary {
  gestures: Gesture[];
}

export interface Match {
  videos: Video[];
  rallies: RallySummary[];
  actions: string[];
  results: string[];
  teams: Team[];
}

export interface StatRow {
  action: string;
  total: number;
  reussi: number;
  rate: number;
  gagnant: number;
  inconnu: number;
  reussite_pct: number | null;
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`${method} ${url} : ${response.status} ${await response.text()}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  match: () => request<Match>("GET", "/api/match"),
  rally: (id: number) => request<Rally>("GET", `/api/rallies/${id}`),
  updateRally: (
    id: number,
    changes: Partial<
      Pick<Rally, "winner" | "status" | "note" | "serve_s" | "end_s" | "serving_team">
    >,
  ) => request<Rally>("PATCH", `/api/rallies/${id}`, changes),
  splitRally: (id: number, time_s: number) =>
    request<Rally>("POST", `/api/rallies/${id}/split`, { time_s }),
  addGesture: (
    rallyId: number,
    gesture: { time_s: number; team: Team; action: string; result: string },
  ) => request<Rally>("POST", `/api/rallies/${rallyId}/gestures`, gesture),
  updateGesture: (
    id: number,
    changes: Partial<Pick<Gesture, "time_s" | "team" | "action" | "result">>,
  ) => request<Rally>("PATCH", `/api/gestures/${id}`, changes),
  deleteGesture: (id: number) => request<Rally>("DELETE", `/api/gestures/${id}`),
  stats: () => request<StatRow[]>("GET", "/api/stats"),
};

export const mediaUrl = (video: string) => `/media/${encodeURIComponent(video)}`;
