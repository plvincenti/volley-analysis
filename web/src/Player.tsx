import { RefObject, useEffect } from "react";
import { Gesture, mediaUrl, Rally } from "./api";
import { ACTION_LABELS, formatTime, RESULT_LABELS } from "./labels";

// Lecture d'un échange : un peu avant le service, un peu après la fin détectée.
export const PRE_ROLL_S = 2;
export const POST_ROLL_S = 2;

export const rallyStart = (rally: Rally) => Math.max(0, rally.serve_s - PRE_ROLL_S);
export const rallyStop = (rally: Rally) => rally.end_s + POST_ROLL_S;

interface PlayerProps {
  rally: Rally;
  videoRef: RefObject<HTMLVideoElement | null>;
  loop: boolean;
  rate: number;
  currentTime: number;
  showOpponent: boolean;
  onTime: (time: number) => void;
}

export function Player({
  rally,
  videoRef,
  loop,
  rate,
  currentTime,
  showOpponent,
  onTime,
}: PlayerProps) {
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const start = () => {
      video.currentTime = rallyStart(rally);
      video.play().catch(() => undefined);
    };
    const src = mediaUrl(rally.video);
    if (video.getAttribute("src") !== src) {
      video.setAttribute("src", src);
      video.addEventListener("loadedmetadata", start, { once: true });
    } else {
      start();
    }
    // Seul le changement d'échange relance la lecture, pas ses corrections.
  }, [rally.id]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.defaultPlaybackRate = rate;
    video.playbackRate = rate;
  }, [rate, videoRef]);

  const handleTime = () => {
    const video = videoRef.current;
    if (!video) return;
    if (loop && !video.paused && video.currentTime > rallyStop(rally)) {
      video.currentTime = rallyStart(rally);
    }
    onTime(video.currentTime);
  };

  const seek = (time: number) => {
    const video = videoRef.current;
    if (video) video.currentTime = time;
  };

  return (
    <div className="player">
      <video
        ref={videoRef}
        className="video"
        controls
        muted
        playsInline
        onTimeUpdate={handleTime}
        onSeeked={handleTime}
      />
      <Timeline
        rally={rally}
        currentTime={currentTime}
        showOpponent={showOpponent}
        onSeek={seek}
      />
    </div>
  );
}

interface TimelineProps {
  rally: Rally;
  currentTime: number;
  showOpponent: boolean;
  onSeek: (time: number) => void;
}

function Timeline({ rally, currentTime, showOpponent, onSeek }: TimelineProps) {
  const start = rallyStart(rally);
  const stop = rallyStop(rally);
  const percent = (time: number) => `${(100 * (time - start)) / (stop - start)}%`;
  const rows: [string, Gesture[]][] = [
    ["Corbas", rally.gestures.filter((g) => g.team === "Corbas")],
  ];
  if (showOpponent) {
    rows.push(["Adversaire", rally.gestures.filter((g) => g.team === "Adversaire")]);
  }

  const handleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    onSeek(start + ((event.clientX - box.left) / box.width) * (stop - start));
  };

  return (
    <div className="timeline" onClick={handleClick}>
      <div
        className="timeline-rally"
        style={{ left: percent(rally.serve_s), width: `calc(${percent(rally.end_s)} - ${percent(rally.serve_s)})` }}
      />
      {rows.map(([team, gestures]) => (
        <div key={team} className="timeline-row">
          <span className="timeline-team">{team}</span>
          {gestures.map((gesture) => (
            <button
              key={gesture.id}
              className={`timeline-marker result-${gesture.result}`}
              style={{ left: percent(gesture.time_s) }}
              title={`${formatTime(gesture.time_s)} ${ACTION_LABELS[gesture.action]} : ${RESULT_LABELS[gesture.result]}`}
              onClick={(event) => {
                event.stopPropagation();
                onSeek(Math.max(start, gesture.time_s - 0.5));
              }}
            >
              {ACTION_LABELS[gesture.action][0]}
            </button>
          ))}
        </div>
      ))}
      {currentTime >= start && currentTime <= stop && (
        <div className="timeline-cursor" style={{ left: percent(currentTime) }} />
      )}
    </div>
  );
}
