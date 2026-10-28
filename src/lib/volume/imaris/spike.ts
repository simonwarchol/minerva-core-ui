/**
 * Dev page (/imaris-spike.html): opens an Imaris file with upstream
 * kiln-render's KilnViewer through ImarisDataProvider, switches channel sets
 * in place, and trains Gaussian splats beside it (shown on their own canvas).
 * `window.viewer` / `window.provider` / `window.splats` for poking at it.
 */
import { KilnViewer } from "kiln-upstream";
import { generateSplats } from "../splats/splat-builder";
import { SplatView } from "../splats/splat-view";
import { ImarisDataProvider, MAX_CHANNELS } from "./imaris-provider";

const DEFAULT_URL =
  "https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/LSP22408.ims";

/**
 * Small, fast builds for trying it out (minerva's export uses 64–96 views of
 * 512–768 px and 1500–3000 iterations).
 */
const QUICK = {
  views: 24,
  resolution: 384,
  iterations: 400,
  targetCount: 200_000,
  evaluateViews: 4,
};

const byId = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;
const volumeCanvas = byId<HTMLCanvasElement>("volume");
const splatCanvas = byId<HTMLCanvasElement>("splats");
const status = byId<HTMLOutputElement>("status");
const channelButtons = byId<HTMLDivElement>("channels");
const splatControls = byId<HTMLDivElement>("splatControls");
const splatStatus = byId<HTMLOutputElement>("splatStatus");

const url = new URLSearchParams(location.search).get("url") ?? DEFAULT_URL;
const t0 = performance.now();
const seconds = (from = t0) => ((performance.now() - from) / 1000).toFixed(1);

const provider = new ImarisDataProvider(url);
Object.assign(window, { provider });

function button(label: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement("button");
  b.textContent = label;
  b.onclick = onClick;
  return b;
}

try {
  const viewer = await KilnViewer.create(volumeCanvas, provider, {
    pyramid: "legacy",
  });
  // Minerva's defaults for thick multichannel fluorescence.
  viewer.renderer.densityScale = 0.1;
  viewer.camera.setOrbitState([0.6, 0.4, 2.2, 0, 0, 0]);
  const splats = new SplatView(viewer, volumeCanvas, splatCanvas);
  Object.assign(window, { viewer, splats });
  const meta = viewer.metadata;
  const opened = seconds();

  // Channel sets of up to four, switched with setSourceSelection.
  const count = provider.allChannels.length;
  for (let first = 0; first < count; first += MAX_CHANNELS) {
    const set = Array.from(
      { length: Math.min(MAX_CHANNELS, count - first) },
      (_, i) => first + i,
    );
    const names = set.map((c) => provider.allChannels[c]?.name ?? c);
    channelButtons.append(
      button(names.join(", "), () =>
        viewer.setSourceSelection({ sourceChannels: set }),
      ),
    );
  }

  const show = (which: "volume" | "splats") => {
    splatCanvas.hidden = which === "volume" || !splats.cloud;
  };
  let training: AbortController | null = null;
  const train = async (perChannel: boolean) => {
    training?.abort();
    const abort = new AbortController();
    training = abort;
    const start = performance.now();
    let part = "";
    try {
      const cloud = await generateSplats(
        viewer,
        () =>
          new ImarisDataProvider(url, { channels: provider.activeChannels }),
        {
          ...QUICK,
          perChannel,
          signal: abort.signal,
          onTile: (i, n) => {
            part = n > 1 ? `Channel ${i + 1}/${n} · ` : "";
          },
          onProgress: (p) => {
            splatStatus.textContent = `${part}${p.label} (${seconds(start)} s)`;
          },
        },
      );
      splats.setCloud(cloud);
      show("splats");
      splatStatus.textContent += ` · ${cloud.count.toLocaleString()} Gaussians${cloud.channelComposite ? `, per channel (${cloud.channelComposite})` : ""}`;
    } catch (e) {
      if (abort.signal.aborted) return;
      console.error(e);
      splatStatus.textContent = `Training failed: ${e instanceof Error ? e.message : String(e)}`;
    } finally {
      if (training === abort) training = null;
    }
  };
  splatControls.append(
    button("Train splats", () => void train(false)),
    button("Train per channel", () => void train(true)),
    button("Cancel", () => {
      training?.abort();
      splatStatus.textContent = "Cancelled";
    }),
    button("Show volume", () => show("volume")),
    button("Show splats", () => show("splats")),
  );

  const report = () => {
    const s = viewer.streamingManager.getStats();
    const shown = provider.activeChannels
      .map((c) => provider.allChannels[c]?.name ?? c)
      .join(", ");
    status.textContent = [
      `${meta.name}: ${meta.dimensions.join("×")} ${meta.dtype}, ${count} channels, ${meta.maxLod + 1} levels (opened in ${opened} s)`,
      `Showing: ${shown}`,
      `Bricks ${s.loadedCount}/${s.desiredCount}, ${s.pendingCount} pending · ${(s.totalBytesDownloaded / 1e6).toFixed(1)} MB in ${s.requestCount} requests · ${seconds()} s`,
    ].join("\n");
  };
  report();
  setInterval(report, 500);
} catch (e) {
  console.error(e);
  status.textContent = `Failed: ${e instanceof Error ? e.message : String(e)}`;
}
