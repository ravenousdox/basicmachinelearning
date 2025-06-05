# --------------------------- inference.py ---------------------------
"""Real‑time vowel inference (loads knn_model.pkl & optional noise_spectrum.npy)"""
import sounddevice as sd, numpy as np, scipy.signal as signal, pickle, queue, threading, librosa, matplotlib.pyplot as plt, matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')
SAMPLE_RATE=16_000; BUFFER=1024; FFT=1024; HOP=512; LPC_ORDER=12; THR=1e-3

knn = pickle.load(open('knn_model.pkl','rb'))
try: noise_mag = np.load('noise_spectrum.npy')
except FileNotFoundError: noise_mag = None

spec_buf = np.full((FFT//2+1,120),-120.0); audio_q=queue.Queue(); pred_q=queue.Queue()

def audio_cb(indata,*_): audio_q.put(indata[:,0].copy())

def lpc_formants(x):
    x*=np.hamming(len(x));
    if x.dot(x)<THR: return None
    A=librosa.lpc(x,order=LPC_ORDER); roots=[c for c in np.roots(A) if np.imag(c)>=0 and abs(c)<1]
    f=sorted(np.arctan2(np.imag(roots),np.real(roots))*SAMPLE_RATE/(2*np.pi))
    return f[:2] if len(f)>=2 else None

def worker():
    global spec_buf
    while True:
        x=audio_q.get(); _,_,Sxx=signal.spectrogram(x,SAMPLE_RATE,'hann',FFT,FFT-HOP)
        spec_buf=np.roll(spec_buf,-1,1); spec_buf[:,-1]=10*np.log10(Sxx[:,0]+1e-10)
        if noise_mag is not None:
            win=x*np.hamming(len(x)); spec=np.fft.rfft(win); mag=np.abs(spec); pha=np.angle(spec)
            mag=np.maximum(mag-noise_mag[:len(mag)],1e-6); x=np.fft.irfft(mag*np.exp(1j*pha),n=len(x))
        x/=np.sqrt(np.mean(x**2)+1e-8)
        f=lpc_formants(x)
        if f: pred_q.put((f,knn.predict([f])[0]))

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,6))
img=ax1.imshow(spec_buf,aspect='auto',origin='lower',extent=[0,1,0,SAMPLE_RATE//2],cmap='magma'); ax1.set_title('Spectrogram'); ax1.set_ylabel('Freq (Hz)'); ax1.set_xticks([])
scat=ax2.scatter([],[],c='lime'); txt=ax2.text(220,3300,'',color='yellow')
ax2.set_xlim(200,1200); ax2.set_ylim(800,3500); ax2.invert_yaxis(); ax2.set_xlabel('F1 (Hz)'); ax2.set_ylabel('F2 (Hz)'); ax2.set_title('Formant Space')

def update(_):
    img.set_data(spec_buf); img.set_clim(spec_buf.max()-80,spec_buf.max())
    try:
        (f1,f2),lbl=pred_q.get_nowait(); scat.set_offsets([[f1,f2]]); txt.set_text(f'Pred: {lbl}')
    except queue.Empty: pass
    return img,scat,txt

threading.Thread(target=worker,daemon=True).start()
FuncAnimation(fig,update,interval=60,cache_frame_data=False)
with sd.InputStream(channels=1,samplerate=SAMPLE_RATE,blocksize=BUFFER,callback=audio_cb):
    plt.show()
