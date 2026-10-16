import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent } from 'react';
import type { Matrix, Region, RGBImage } from '../types';
import { pointerPixel, selectionRegion } from '../state';

type Point = { x: number; y: number };
export interface PixelCanvasProps {
  rgb?: RGBImage; values?: Matrix; upper?: number; label: string;
  support?: Matrix; saliency?: Matrix; opacity?: number; grid?: boolean; interactive?: boolean;
  mode?: 'pixel' | 'patch' | 'rectangle'; onSelect?: (region: Region) => void;
}
export default function PixelCanvas({ rgb, values, upper, label, saliency, support, opacity = .55, grid = false, interactive = false, mode = 'patch', onSelect }: PixelCanvasProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [anchor, setAnchor] = useState<Point | null>(null);
  const [point, setPoint] = useState<Point | null>(null);
  const width = values?.[0]?.length ?? rgb?.[0]?.[0]?.length ?? 28;
  const height = values?.length ?? rgb?.[0]?.length ?? 28;
  const selected = anchor && point ? selectionRegion(anchor, point, mode) : null;
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const image = context.createImageData(width, height);
    const max = upper ?? (values ? Math.max(...values.flat(), 1e-8) : 1);
    const saliencyMax = saliency ? Math.max(...saliency.flat(), 1e-8) : 1;
    const supportMax = support ? Math.max(...support.flat().map(Math.abs), 1e-8) : 1;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4;
      const fraction = values ? Math.max(0, Math.min(1, values[y][x] / Math.max(max, 1e-8))) : 0;
      // Labeled sequential scale: deep teal → pale lime, zero stays dark.
      const color = values ? [20 + fraction * 186, 52 + fraction * 189, 57 + fraction * 72] : rgb?.map(channel => channel[y][x] * 255) ?? [20, 52, 57];
      const blend = support ? Math.abs(support[y][x]) / supportMax * opacity : saliency ? saliency[y][x] / saliencyMax * opacity : 0;
      const overlay = support ? support[y][x]>=0 ? [235,86,62] : [60,133,210] : [255,83,80];
      for (let channel = 0; channel < 3; channel++) image.data[index + channel] = color[channel] * (1 - blend) + overlay[channel] * blend;
      image.data[index + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [rgb, values, upper, width, height, saliency, support, opacity]);
  function pixel(event: PointerEvent<HTMLDivElement>) {
    return pointerPixel(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());
  }
  function down(event: PointerEvent<HTMLDivElement>) {
    if (!interactive) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const next = pixel(event); setAnchor(next); setPoint(next);
  }
  function up(event: PointerEvent<HTMLDivElement>) {
    if (!interactive || !anchor) return;
    onSelect?.(selectionRegion(anchor, pixel(event), mode));
    setAnchor(null); setPoint(null);
  }
  function key(event: KeyboardEvent<HTMLDivElement>) {
    if (!interactive || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Enter', ' '].includes(event.key)) return;
    event.preventDefault(); event.stopPropagation();
    const previous = point ?? { x: 12, y: 12 };
    if (event.key === 'Enter' || event.key === ' ') {
      onSelect?.(selectionRegion(anchor ?? previous, previous, mode)); setAnchor(null); return;
    }
    const next = { x: Math.max(0, Math.min(27, previous.x + (event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0))), y: Math.max(0, Math.min(27, previous.y + (event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0))) };
    if (mode === 'rectangle' && event.shiftKey && !anchor) setAnchor(previous);
    if (!event.shiftKey) setAnchor(null);
    setPoint(next);
  }
  const cursorRegion = selected ?? (interactive && point ? selectionRegion(point, point, mode) : null);
  const box = cursorRegion && ('rectangle' in cursorRegion ? cursorRegion.rectangle : { ...cursorRegion.pixels[0], width: 1, height: 1 });
  return <div className={`pixel-canvas ${interactive ? 'is-interactive' : ''}`} role={interactive ? 'group' : 'img'} aria-label={interactive ? `${label}. Use arrow keys to select pixels, Enter to apply; hold Shift for a rectangle.` : label} tabIndex={interactive ? 0 : undefined}
    onPointerDown={down} onPointerMove={event => { if (interactive) setPoint(pixel(event)); }} onPointerUp={up}
    onPointerCancel={() => { setAnchor(null); setPoint(null); }} onPointerLeave={() => { if (!anchor) setPoint(null); }} onKeyDown={key}>
    <canvas ref={canvas} width={width} height={height} aria-hidden="true" />
    {grid && <span className="pixel-grid" style={{ backgroundSize: `${100 / width}% ${100 / height}%` }} />}
    {box && <span className="pixel-selection" style={{ left: `${box.x / 28 * 100}%`, top: `${box.y / 28 * 100}%`, width: `${box.width / 28 * 100}%`, height: `${box.height / 28 * 100}%` }} />}
  </div>;
}
