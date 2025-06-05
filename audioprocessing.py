import sounddevice as sd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import KNeighborsClassifier
import queue
import threading
import time
import tkinter as tk
from tkinter import simpledialog
import librosa
import matplotlib
matplotlib.use('TkAgg')

# === PARAMETERS ===
SAMPLE_RATE = 16000
BUFFER_SIZE  = 1024
FFT_SIZE     = 1024
HOP_LENGTH   = 512
LPC_ORDER    = 12
N_FORMANTS   = 2
RECORD_BLOCKS = 32
ENERGY_THRESHOLD = 1e-3

# === GLOBALS ===
audio_q = queue.Queue()
prediction_queue = queue.Queue()
vowel_labels = ['a', 'e', 'i', 'o', 'u']
vowel_data = []
vowel_targets = []
vowel_classifier = None
noise_baseline = None
noise_baseline_mag = None

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())

# === DSP HELPERS ===
def spectral_subtract(signal_frame: np.ndarray, noise_mag: np.ndarray) -> np.ndarray:
    windowed = signal_frame * np.hamming(len(signal_frame))
    spectrum = np.fft.rfft(windowed)
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    mag_sub = np.maximum(mag - noise_mag[:len(mag)], 1e-6)
    reconstructed = mag_sub * np.exp(1j * phase)
    return np.fft.irfft(reconstructed, n=len(signal_frame))

def lpc_formants(sig: np.ndarray, sr: int) -> list[float]:
    sig = sig * np.hamming(len(sig))
    energy = np.sum(sig**2)
    if energy < ENERGY_THRESHOLD:
        return [0.0, 0.0]
    try:
        A = librosa.lpc(sig, order=LPC_ORDER)
        rts = np.roots(A)
        rts = [r for r in rts if np.imag(r) >= 0 and np.abs(r) < 1.0]
        ang = np.arctan2(np.imag(rts), np.real(rts))
        freqs = sorted(ang * (sr / (2*np.pi)))
        return freqs[:N_FORMANTS] if len(freqs) >= N_FORMANTS else [0.0, 0.0]
    except Exception as e:
        print("LPC Error:", e)
        return [0.0, 0.0]

# === CLASSIFIER ===
def train_classifier():
    global vowel_classifier
    vowel_classifier = KNeighborsClassifier(n_neighbors=3)
    vowel_classifier.fit(vowel_data, vowel_targets)

# === VISUALISATION ===
fig, (ax_spec, ax_fspace) = plt.subplots(2, 1, figsize=(9, 7))
plt.subplots_adjust(hspace=0.45)

spec_buf = np.full((FFT_SIZE // 2 + 1, 120), -120.0, dtype=np.float32)
img_spec = ax_spec.imshow(spec_buf, aspect='auto', origin='lower',
                          extent=[0, 1, 0, SAMPLE_RATE // 2], cmap='magma', vmin=-90, vmax=-20)
ax_spec.set_title('Real‑Time Spectrogram')
ax_spec.set_ylabel('Freq (Hz)')
ax_spec.set_xticks([])

scatter = ax_fspace.scatter([], [], c='lime')
ax_fspace.set_xlim(200, 1200)
ax_fspace.set_ylim(800, 3500)
ax_fspace.set_xlabel('F1 (Hz)')
ax_fspace.set_ylabel('F2 (Hz)')
ax_fspace.invert_yaxis()
text_pred = ax_fspace.text(220, 3300, '', color='yellow', fontsize=14)
ax_fspace.set_title('Formant Space (live)')

# === PROCESSING THREAD ===
def audio_worker():
    global spec_buf
    while True:
        frame = audio_q.get()
        if len(frame) < FFT_SIZE:
            continue
        _, _, Sxx = signal.spectrogram(frame, fs=SAMPLE_RATE, window='hann',
                                       nperseg=FFT_SIZE, noverlap=FFT_SIZE-HOP_LENGTH)
        col = 10 * np.log10(Sxx[:, 0] + 1e-10)
        spec_buf = np.roll(spec_buf, -1, axis=1)
        spec_buf[:, -1] = col
        try:
            if noise_baseline_mag is not None:
                frame = spectral_subtract(frame, noise_baseline_mag)
            frame /= np.sqrt(np.mean(frame**2) + 1e-8)
            formants = lpc_formants(frame, SAMPLE_RATE)
            if len(formants) == 2:
                f1, f2 = formants
                if vowel_classifier:
                    pred = vowel_classifier.predict([[f1, f2]])[0]
                else:
                    pred = '?'
                prediction_queue.put((f1, f2, pred))
        except Exception as e:
            pass

# === MATPLOTLIB ANIMATION UPDATE ===
def update(_):
    img_spec.set_data(spec_buf)
    img_spec.set_clim(vmin=spec_buf.max() - 80, vmax=spec_buf.max())
    try:
        f1, f2, label = prediction_queue.get_nowait()
        scatter.set_offsets([[f1, f2]])
        text_pred.set_text(f'Pred: {label}')
    except queue.Empty:
        pass
    return img_spec, scatter, text_pred

# === TRAINING DIALOG ===
def flush_audio_queue():
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break

def record_sample(blocks=RECORD_BLOCKS):
    flush_audio_queue()
    buf = []
    for _ in range(blocks):
        frame = audio_q.get()
        _, _, Sxx = signal.spectrogram(frame, fs=SAMPLE_RATE, window='hann',
                                       nperseg=FFT_SIZE, noverlap=FFT_SIZE-HOP_LENGTH)
        col = 10 * np.log10(Sxx[:, 0] + 1e-10)
        spec_buf[:, -1] = col
        buf.append(frame)
    return np.concatenate(buf)

if __name__ == '__main__':
    root = tk.Tk(); root.withdraw()
    print("Add training samples:")
    print(" - Type a/e/i/o/u to record vowel")
    print(" - Type n to record noise baseline")
    print(" - Type q to quit")

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE, callback=audio_callback):
        while True:
            lbl = simpledialog.askstring('Vowel', 'Enter vowel (a/e/i/o/u), n=noise, or q=quit:')
            if lbl is None or lbl.lower() == 'q':
                break
            lbl = lbl.strip().lower()
            if lbl == 'n':
                print('Recording noise baseline...')
                noise_sample = record_sample()
                noise_sample /= np.sqrt(np.mean(noise_sample**2) + 1e-8)
                noise_windowed = noise_sample * np.hamming(len(noise_sample))
                noise_fft = np.abs(np.fft.rfft(noise_windowed))
                noise_baseline_mag = noise_fft
                print('Noise baseline captured.')

                # Plot the noise spectrum
                plt.figure("Noise Baseline Spectrum")
                freqs = np.fft.rfftfreq(len(noise_sample), 1/SAMPLE_RATE)
                plt.plot(freqs, 20*np.log10(noise_fft + 1e-10), color='gray')
                plt.title("Baseline Noise Spectrum")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.grid(True)
                plt.show(block=False)
                continue
            if lbl not in vowel_labels:
                print('Invalid label.'); continue
            print(f'Recording {lbl}… hold vowel steady.')
            time.sleep(0.3)
            sample = record_sample()
            if noise_baseline_mag is not None:
                sample = spectral_subtract(sample, noise_baseline_mag)
            sample /= np.sqrt(np.mean(sample**2) + 1e-8)
            fmts = lpc_formants(sample, SAMPLE_RATE)
            if len(fmts) == 2 and sum(f > 100 for f in fmts) >= 1:
                vowel_data.append(fmts)
                vowel_targets.append(lbl)
                print('Captured', fmts)
            else:
                print('Ignored low-energy or invalid formant frame.')
        if vowel_data:
            train_classifier()
            print('Classifier trained on', len(vowel_data), 'samples.')
        else:
            print('No data captured – live formants only.')

    threading.Thread(target=audio_worker, daemon=True).start()
    ani = FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE, callback=audio_callback):
        plt.show()
