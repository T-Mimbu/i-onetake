const SR = 48000;         // 解析側と合わせる
const TAKE_SEC = 3.0;     // 3秒

const logEl = document.getElementById('log');
const resultEl = document.getElementById('result');
const recBtn = document.getElementById('recBtn');
const dlLink = document.getElementById('dlLink');

let audioCtx, mediaStream, processor, source;
let chunks = [];

function log(msg){ logEl.textContent += msg + "\n"; }

function floatTo16BitPCM(float32Array){
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Uint8Array(buffer);
}

function writeWav(samples, sampleRate){
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const data = floatTo16BitPCM(samples);

  const buffer = new ArrayBuffer(44 + data.length);
  const view = new DataView(buffer);

  function writeString(v, offset){
    for (let i = 0; i < v.length; i++) view.setUint8(offset+i, v.charCodeAt(i));
  }
  let off = 0;
  writeString('RIFF', off); off += 4;
  view.setUint32(off, 36 + data.length, true); off += 4; // file length - 8
  writeString('WAVE', off); off += 4;
  writeString('fmt ', off); off += 4;
  view.setUint32(off, 16, true); off += 4;       // PCM chunk size
  view.setUint16(off, 1, true); off += 2;        // PCM
  view.setUint16(off, numChannels, true); off += 2;
  view.setUint32(off, sampleRate, true); off += 4;
  view.setUint32(off, byteRate, true); off += 4;
  view.setUint16(off, blockAlign, true); off += 2;
  view.setUint16(off, bytesPerSample * 8, true); off += 2;
  writeString('data', off); off += 4;
  view.setUint32(off, data.length, true); off += 4;

  new Uint8Array(buffer).set(data, 44);
  return new Blob([buffer], { type: 'audio/wav' });
}

async function startRecording(){
  logEl.textContent = '';
  resultEl.textContent = '';
  chunks = [];

  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: SR }, video: false });
  source = audioCtx.createMediaStreamSource(mediaStream);

  // ScriptProcessorNodeはdeprecatedだが簡便で互換性が高い
  const bufferSize = 2048;
  processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
  source.connect(processor);
  processor.connect(audioCtx.destination);

  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input)); // コピーを保持
  };

  log('録音開始…「いーー」を一定に3秒、小さめ音量、息を当てない');
  await new Promise(res => setTimeout(res, TAKE_SEC * 1000));
  await stopRecording();
}

async function stopRecording(){
  processor.disconnect();
  source.disconnect();
  mediaStream.getTracks().forEach(t => t.stop());
  audioCtx.close();

  // 結合
  let total = chunks.reduce((s, a) => s + a.length, 0);
  const full = new Float32Array(total);
  let off = 0;
  for(const a of chunks){ full.set(a, off); off += a.length; }

  // 正規化（解析側も正規化するが一応）
  const peak = full.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-12);
  const gain = Math.min(Math.pow(10, -3/20) / peak, 20.0);
  for(let i=0;i<full.length;i++) full[i] = Math.max(-1, Math.min(1, full[i]*gain));

  const wavBlob = writeWav(full, SR);
  const url = URL.createObjectURL(wavBlob);
  dlLink.href = url; dlLink.download = 'i_take.wav';
  dlLink.style.display = 'inline';
  dlLink.textContent = '録音WAVをダウンロード';

  await sendToAPI(wavBlob);
}

async function sendToAPI(wavBlob){
  const apiUrl = document.getElementById('apiUrl').value.trim();
  const profile = document.getElementById('profile').value;
  const knownHeight = document.getElementById('knownHeight').value;

  const fd = new FormData();
  fd.append('file', wavBlob, 'i_take.wav');
  fd.append('profile', profile);
  if(knownHeight) fd.append('known_height_cm', knownHeight);

  log('解析中…');
  const res = await fetch(apiUrl, { method: 'POST', body: fd });
  if(!res.ok){
    resultEl.textContent = `API Error: ${res.status} ${res.statusText}`;
    return;
  }
  const json = await res.json();
  resultEl.textContent =
`=== 解析結果 ===
Profile    : ${json.profile}
F0(median) : ${json.f0_median_hz ?? '取得不可'} Hz (σ=${json.f0_std_hz ?? 'N/A'})
VTL        : ${json.vtl_cm ?? '取得不可'} cm  (CV=${json.vtl_cv ?? 'N/A'}, 有効=${json.vtl_n_keep})
推定身長   : ${json.height_cm ?? 'N/A'} cm
${(json.alarms||[]).map(a => '⚠ '+a.msg).join('\n')}`;
}

recBtn.addEventListener('click', async ()=>{
  recBtn.disabled = true;
  try { await startRecording(); }
  catch(e){ resultEl.textContent = '録音に失敗: ' + e; }
  finally { recBtn.disabled = false; }
});
