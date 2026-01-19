// 48kHz Float32 の入力を 16bit PCM (48k相当) モノに変換して main-thread へ送る
// 実際の 48k→16k のダウンサンプルはサーバー側で bot.py と同一ロジックで実施
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() { super(); this._buf = []; }
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0]; // 1ch想定
    // Float32 [-1,1] → Int16 little-endian
    const pcm16 = new Int16Array(ch.length);
    for (let i = 0; i < ch.length; i++) {
      let s = Math.max(-1, Math.min(1, ch[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    // 軽量に都度送信（サーバー側で 160ms 程度でまとめて flush）
    this.port.postMessage(pcm16);
    return true;
  }
}
registerProcessor("pcm-worklet", PCMWorkletProcessor);
