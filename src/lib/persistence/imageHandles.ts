/**
 * Keys for Dexie `handles` (same `minerva-stories` DB as stories). Document `Image.source` stores `handleKey` only in JSON.
 */
export function imageHandleStorageKey(
  storyId: string,
  imageId: string,
): string {
  return `story:${storyId}:image:${imageId}`;
}

/** Segmentation mask handle for the same story image row. */
export function maskHandleStorageKey(storyId: string, imageId: string): string {
  return `story:${storyId}:mask:${imageId}`;
}
