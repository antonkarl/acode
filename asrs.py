import soundfile as sf
import glob,os
from scipy import signal
import numpy as np
import torch, torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from faster_whisper import WhisperModel
from pydub import AudioSegment
from util import *


# ----------------------------
# 
# --- speech recognition ---
#
# ----------------------------


def readwav(a_f):
    wav, sr = sf.read(a_f, dtype=np.float32)
    if len(wav.shape) == 2:
        wav = wav.mean(1)
    if sr != 16000:
        wlen = int(wav.shape[0] / sr * 16000)
        wav = signal.resample(wav, wlen)
    return wav
    

# read a diarisation
# into list of [L]abel, [S]tart, [E]nd, [T]ranscript
def get_segments(seg_file):
    with open(seg_file,'r') as handle:
        f = handle.read().splitlines()
    f = [l.split('\t') for l in f]
    segments = [{'l':l[0], 's':float(l[1]), 'e':float(l[2]), 't':''} for l in f]
    return segments


# handle asrs
def asr_one(file_paths,asr_type,seg_type):

    wav_file = file_paths['wav']
    seg_file = file_paths[seg_type]
    save_dir, save_base = os.path.split(seg_file.replace('/diarisation/','/asr/'))
    
    asr_func = None
    
    if asr_type.lower() == 'w2v2':
        save_file = os.path.join(save_dir,fn(save_base)+'-w2v2.txt')
        asr_func = recognise_w2v2
    elif asr_type.lower() == 'whisper':
        save_file = os.path.join(save_dir,fn(save_base)+'-whisper.txt')
        asr_func = recognise_fasterwhisper
    
    if asr_func:
        if os.path.exists(save_file):
            print(f'Specified ASR {asr_type} already existed, did not rerun.')
        else:
            asr_func(wav_file,seg_file,save_file)
    else:
        print(f'ASR type {asr_type} is not recognised')
    return save_file



# use w2v2 to recognise words in audio segments
#   based on a pre-existing segmentation
def recognise_w2v2(wav_file,seg_file, save_file):

    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w2v2path = './localmodels/LVL/wav2vec2-large-xlsr-53-icelandic-ep30-967h'
    w2v2model= Wav2Vec2ForCTC.from_pretrained(w2v2path,use_safetensors=True).to(device)
    w2v2processor = Wav2Vec2Processor.from_pretrained(w2v2path,use_safetensors=True)

    sr = 16000
    whole_audio = readwav(wav_file)
    segments = get_segments(seg_file)
    
    for i,seg in enumerate(segments):
    
        audio_seg = whole_audio[int(seg['s']*sr):int(seg['e']*sr)]
        try:
            with torch.inference_mode():
                input_values = w2v2processor(audio_seg,sampling_rate=16000).input_values[0]
                input_values = torch.tensor(input_values, device=device).unsqueeze(0)
                logits = w2v2model(input_values).logits
                pred_ids = torch.argmax(logits, dim=-1)
                dec = w2v2processor.batch_decode(pred_ids)
                xcp = dec[0]
        #   print(seg['s'],seg['e'],xcp)
        
            segments[i] = {'l':seg['l'], 's':seg['s'], 'e':seg['e'], 't':xcp}
            
        except RuntimeError:
            print(f'Skipping segment {seg["s"]} - {seg["e"]}')
    
    segments = [f'{seg["l"]}\t{seg["s"]}\t{seg["e"]}\t{seg["t"]}' for seg in segments]
    with open(save_file,'w') as handle:
        handle.write('\n'.join(segments))
        


# use whisper to recognise words in audio segments
#   based on a pre-existing segmentation
def recognise_whisper(wav_file,seg_file,save_file):
    whisperpath = './localmodels/LVL/whisper-large-icelandic-10k-steps-1000h'
    whisperprocessor = WhisperProcessor.from_pretrained(whisperpath)
    whispermodel = WhisperForConditionalGeneration.from_pretrained(whisperpath)
    
    sr = 16000
    whole_audio = readwav(wav_file)
    segments = get_segments(seg_file)
    
    print(f'Audio {round(len(whole_audio)/16000)} seconds')
    for i,seg in enumerate(segments):
    
        audio_seg = whole_audio[int(seg['s']*sr):int(seg['e']*sr)]
    
        input_features = whisperprocessor(audio_seg, sampling_rate=sr, return_tensors="pt").input_features
        predicted_ids = whispermodel.generate(input_features)
        dec = whisperprocessor.batch_decode(predicted_ids, skip_special_tokens=True,language_id='is')
        xcp = dec[0]
        
        print(seg['s'],seg['e'],xcp)
        segments[i] = {'l':seg['l'], 's':seg['s'], 'e':seg['e'], 't':xcp}
        
    segments = [f'{seg["l"]}\t{seg["s"]}\t{seg["e"]}\t{seg["t"]}' for seg in segments]
    with open(save_file,'w') as handle:
        handle.write('\n'.join(segments))



# use fasterwhisper to recognise words in audio segments
#   based on a pre-existing segmentation
def recognise_fasterwhisper(wav_file,seg_file,save_file):
    whisperpath = './localmodels/LVL/whisper-large-icelandic-62640-steps-967h-ct2'
    wdevice = "cuda" if torch.cuda.is_available() else "cpu"
    
    whispermodel = WhisperModel(model_size_or_path=whisperpath, device=wdevice, local_files_only=True)
    
    sr = 16000
    whole_audio = readwav(wav_file)
    segments = get_segments(seg_file)
    
    # weird fasterwhisper segment format
    cts = [t for ts in [[seg['s'],seg['e']] for seg in segments] for t in ts]
    assert len(cts) == 2*len(segments)
    print(f'Audio {round(len(whole_audio)/sr)} seconds')
    
    xcps, info = whispermodel.transcribe(audio = whole_audio, language = "is", no_repeat_ngram_size = 5, clip_timestamps = cts)
    
    whisper_segs = []
    for xcp in xcps:
        xid, xs, xe, xt = xcp.id, xcp.start, xcp.end, xcp.text
        #seg = segments[xid-1]
        print(xs, xe, xt, sep='\t')
        whisper_segs.append([xid,xs,xe,xt])
        #print(seg['s'],seg['e'])
        #segments[xid-1] = {'l':seg['l'], 's':seg['s'], 'e':seg['e'], 't':xt}
        
    for i,seg in enumerate(segments):
        wh_matches = [xt for xid,xs,xe,xt in whisper_segs if round(xs,2)>=round(seg['s'],2) and round(xe,2)<=round(seg['e'],2)]
        if len(wh_matches) == 0:
            print(f'No words recognised in segment {seg["s"]} - {seg["e"]}')
        else:
            wh_matches = ' '.join(wh_matches)
            segments[i] = {'l':seg['l'], 's':seg['s'], 'e':seg['e'], 't':wh_matches}
        
        
    segments = [f'{seg["l"]}\t{seg["s"]}\t{seg["e"]}\t{seg["t"]}' for seg in segments]
    with open(save_file,'w') as handle:
        handle.write('\n'.join(segments))





