import styled from "styled-components";
import { useDocumentStore } from "@/lib/stores/documentStore";

const Block = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--theme-glass-edge) 55%, transparent);
`;

const Label = styled.div`
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: color-mix(in srgb, var(--theme-light-contrast-color) 52%, transparent);
`;

const Row = styled.label`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: #c9d1d9;
  cursor: pointer;
`;

const Slider = styled.input`
  flex: 1;
  min-width: 0;
`;

const RemoveButton = styled.button`
  background: #2a2a2a;
  border: 1px solid #444;
  color: #e6edf3;
  padding: 5px 8px;
  font-size: 11px;
  cursor: pointer;
  border-radius: 3px;
  align-self: flex-start;

  &:hover {
    background: #3a1c1c;
    border-color: #5a2828;
    color: #f0d0d0;
  }
`;

/**
 * All segmentation-mask UI lives here in the right-hand config panel:
 * opacity, outline toggle, and the remove action. Removal calls the doc
 * store directly — `main.tsx`'s mask-hydration effect listens to `images`
 * and clears the in-memory mask loader entries when no mask remains.
 */
export function MaskControls() {
  const images = useDocumentStore((s) => s.images);
  const updateImageMask = useDocumentStore((s) => s.updateImageMask);
  const clearImageMask = useDocumentStore((s) => s.clearImageMask);

  const im = images[0];
  const mask = im?.mask;
  if (!im || !mask) return null;

  return (
    <Block>
      <Label>Segmentation mask</Label>
      <Row>
        Opacity
        <Slider
          type="range"
          min={0}
          max={100}
          value={Math.round(mask.opacity * 100)}
          onChange={(e) =>
            updateImageMask(im.id, {
              opacity: Number(e.target.value) / 100,
            })
          }
        />
        <span>{Math.round(mask.opacity * 100)}%</span>
      </Row>
      <Row>
        <input
          type="checkbox"
          checked={mask.outlines}
          onChange={(e) =>
            updateImageMask(im.id, { outlines: e.target.checked })
          }
        />
        Cell outlines
      </Row>
      <RemoveButton type="button" onClick={() => void clearImageMask(im.id)}>
        Remove mask
      </RemoveButton>
    </Block>
  );
}
