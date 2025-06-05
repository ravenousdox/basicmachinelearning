# --------------------------- training.py ---------------------------
"""
Live‑visualised training:
  • Records vowel samples (a/e/i/o/u) or baseline noise (n)
  • Shows spectrogram + captured formants while you speak
  • Trains a 2‑D k‑NN on [F1, F2] and saves:
        knn_model.pkl         – pickled model
        formant_samples.npy   – Nx2 array of formants
        formant_labels.npy    – labels for each sample
        noise_spectrum.npy    – baseline magnitude spectrum (optional)
Run:  python training.py
"""
import sounddevice as sd, numpy as np, scipy.signal as signal, pickle, queue, threading, time, tkinter as tk
from tkinter import simpledialog
from sklearn.neighbors import KNeighborsClassifier
import librosa, matplotlib.pyplot as plt, matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')

# ---------- parameters ----------
SAMPLE_RATE = 16_000
BUFFER_SIZE = 1024
FFT_SIZE    = 1024
HOP_LENGTH  = 512
LPC_ORDER   = 12
N_FORMANTS  = 2
RECORD_BLOCKS = 32      # ≈2 s window
ENERGY_THR  = 1e-3

vowels = list('aeiou')
audio_q = queue.Queue()
pred_q  = queue.Queue()

# ---------- helpers ----------

def audio_cb(indata, *_) -> None:
    audio_q.put(indata[:, 0].copy())

def lpc_formants(x: np.ndarray):
    x *= np.hamming(len(x))
    if x.dot(x) < ENERGY_THR:
        return None
    A = librosa.lpc(x, order=LPC_ORDER)
    roots = [c for c in np.roots(A) if np.imag(c) >= 0 and abs(c) < 1]
    freqs = sorted(np.arctan2(np.imag(roots), np.real(roots)) * SAMPLE_RATE / (2 * np.pi))
    return freqs[:N_FORMANTS] if len(freqs) >= 2 else None

def flush_audio():
    while not audio_q.empty():
        try: audio_q.get_nowait()
        except queue.Empty: break

def record_blockset(n: int = RECORD_BLOCKS):
    flush_audio(); frames = [audio_q.get() for _ in range(n)]; return np.concatenate(frames)

# ---------- live visualisation ----------
spec_buf = np.full((FFT_SIZE // 2 + 1, 120), -120.0)
fig, (ax_spec, ax_form) = plt.subplots(2, 1, figsize=(9, 7)); plt.subplots_adjust(hspace=0.45)
img_spec = ax_spec.imshow(spec_buf, aspect='auto', origin='lower', extent=[0, 1, 0, SAMPLE_RATE // 2], cmap='magma', vmin=-90, vmax=-20)
ax_spec.set_title('Real‑Time Spectrogram'); ax_spec.set_ylabel('Freq (Hz)'); ax_spec.set_xticks([])
scatter = ax_form.scatter([], [], c='lime'); ax_form.set_xlim(200, 1200); ax_form.set_ylim(800, 3500); ax_form.invert_yaxis()
ax_form.set_xlabel('F1 (Hz)'); ax_form.set_ylabel('F2 (Hz)'); ax_form.set_title('Captured Formants')


def viz_thread():
    global spec_buf
    while True:
        frame = audio_q.get()
        _, _, Sxx = signal.spectrogram(frame, fs=SAMPLE_RATE, window='hann', nperseg=FFT_SIZE, noverlap=FFT_SIZE-HOP_LENGTH)
        col = 10 * np.log10(Sxx[:, 0] + 1e-10)
        spec_buf = np.roll(spec_buf, -1, axis=1)
        spec_buf[:, -1] = col


def viz_update(_):
    img_spec.set_data(spec_buf)
    img_spec.set_clim(spec_buf.max() - 80, spec_buf.max())
    try:
        f1, f2 = pred_q.get_nowait(); scatter.set_offsets(np.vstack([scatter.get_offsets(), [f1, f2]]))
    except queue.Empty: pass
    return img_spec, scatter

threading.Thread(target=viz_thread, daemon=True).start()
FuncAnimation(fig, viz_update, interval=60, cache_frame_data=False)

# ---------- interaction loop ----------
root = tk.Tk(); root.withdraw()
print('Commands:  n = noise  |  a/e/i/o/u = vowel  |  q = quit')
X, Y = [], []; noise_mag = None

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE, callback=audio_cb):
    while True:
        lbl = simpledialog.askstring('Label', '(a/e/i/o/u), n, or q')
        if lbl is None or lbl.lower() == 'q': break
        lbl = lbl.lower().strip()
        if lbl == 'n':
            print('Recording noise…'); time.sleep(0.3); ns = record_blockset()
            ns /= np.sqrt(np.mean(ns**2) + 1e-8)
            noise_mag = np.abs(np.fft.rfft(ns * np.hamming(len(ns))))
            np.save('noise_spectrum.npy', noise_mag); print('Noise saved.')
            continue
        if lbl not in vowels: print('Invalid'); continue
        print(f'Recording {lbl}…'); time.sleep(0.3); x = record_blockset()
        if noise_mag is not None:
            win = x * np.hamming(len(x)); spec = np.fft.rfft(win); mag = np.abs(spec); pha = np.angle(spec)
            mag = np.maximum(mag - noise_mag[:len(mag)], 1e-6)
            x = np.fft.irfft(mag * np.exp(1j * pha), n=len(x))
        x /= np.sqrt(np.mean(x**2) + 1e-8)
        form = lpc_formants(x)
        if form:
            X.append(form); Y.append(lbl); pred_q.put(form); print('Captured', form)
        else:
            print('Low‑energy / invalid')

plt.close(fig)
if X:
    X, Y = np.array(X), np.array(Y)
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, Y)
    pickle.dump(knn, open('knn_model.pkl', 'wb'))
    np.save('formant_samples.npy', X); np.save('formant_labels.npy', Y)
    print('Saved', len(X), 'samples ✅')
else:
    print('No data saved')