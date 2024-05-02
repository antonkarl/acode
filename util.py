import glob, os
from pydub import AudioSegment

# prepare a 16khz mono wav file for VAD,
#  compatible with both LDC and Pyannote.
# (also fine for reaper pitch, that's not picky)
# ¡ do not use this function for any stereo 
#    files with 1 speaker per channel.
#   in that case do VAD for each channel separately !
def wav16mono(wav_path, temp_dir = './tmp/'):
    export_path = os.path.join(temp_dir,os.path.basename(wav_path))
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    wav_data = AudioSegment.from_wav(wav_path)
    wav_data = wav_data.set_channels(1)
    wav_data = wav_data.set_frame_rate(16000)
    wav_data.export(export_path, format="wav")
    return os.path.abspath(export_path)   



def fn(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

