declare module 'tiff.js' {
  export default class Tiff {
    constructor(options: { buffer: ArrayBuffer });
    toCanvas(): HTMLCanvasElement;
  }
} 