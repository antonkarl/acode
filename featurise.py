import os, subprocess, glob
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment, Timeline, Annotation, notebook
import torchaudio
import numpy as np
from util import *


    
# read segments from a tsv whose first 3 columns are
# speaker_id, start_time, end_time
def read_segments(seg_path):
    with open(seg_path, 'r') as handle:
        segments = handle.read().splitlines()
    segments = [l.split('\t') for l in segments]

    annot = Annotation()
    for l in segments:
        annot[Segment(float(l[1]),float(l[2]))] = l[0]
    return annot


# TODO:
#  also track pauses following questions
#    e.g. pauses following an Interviewer seg of at least 1.second. \TODO
# gather measurements from a labelled pyannote Annotation
def compile_segments(pya):
    main_speaker = pya.chart()[0][0]
    speeches = []
    pauses = []   
    seg_ends = [(s.end, l) for s,t,l in pya.itertracks(yield_label=True)]
    seg_ends =  sorted(seg_ends, key=lambda x: x[0])
    for segment,track,label in pya.itertracks(yield_label=True):
        if label==main_speaker:
            if speeches: # check preceding pause if this isn't the first segment
                past_turns = [(e,l) for e,l in seg_ends if e<= segment.start]
                last_ended_turn = past_turns[-1]
                if last_ended_turn[1] == main_speaker:
                    pause_dur = segment.start - last_ended_turn[0]
                    pauses.append(pause_dur)
            speeches.append(segment)
            
    return speeches, pauses
    
    

    
# ----------------------------
# 
# --- pitch ---
#
# ----------------------------

# TODO:
# praat instead of reaper.


# returns f0 data as list of Time, F0 if exists, voicing indicator
# bounds set following
# https://www.ling.upenn.edu/courses/Spring_2018/ling620/Nevler2017.pdf
def get_reaper(wav_path, reaper_path, maxf0='300', minf0='75'):
    
    f0_data = subprocess.run([reaper_path, "-i", wav_path, '-f', '/dev/stdout', '-x', maxf0, '-m', minf0, '-a'],capture_output=True).stdout
    f0_data = f0_data.decode()
    f0_data = f0_data.split('EST_Header_End\n')[1].splitlines()
    f0_data = [l.split(' ') for l in f0_data] 
    f0_data = [l for l in f0_data if len(l) == 3] # the last line or 2 lines are other info, different format
    f0_data = [ [float(t), float(f), float(v)] for t,v,f in f0_data]
    return f0_data


def h2st(hertz,factor):
    return 12 * np.log2(hertz/factor)


# from the whole audio file's pitch track,
# extract pitch for the specified intervals
# then convert it to semitones 
def pitch_data(pitches,speeches):
    
    # segment pitch tracks in hz
    hz = [[f for t,f,v in pitches if v==1 and s.overlaps(t)] for s in speeches]
    hz = [s for s in hz if s] # remove segments with no voiced speech
    
    # 90th percentile of each segment in hz
    hz90s = [np.percentile(hzs,90) for hzs in hz]
    
    # 10th percentile of each semgnet in hz
    hz10s = [np.percentile(hzs,10) for hzs in hz]
    
    # 10th percentile hz for this speaker globally
    hz10_all = np.percentile([x for seg in hz for x in seg],10)
    
    
    #TODO: 
    # which is correct, global or self?
    
    # convert each segment's 90th percentile to semitones 
    #  using speaker's overall 10th percentile as reference point
    st_global = [h2st(hz90,hz10_all) for hz90 in hz90s]
    
    # using each segment's specific 10th percentile as reference point
    st_self = [h2st(hz90,hz10) for hz90,hz10 in zip(hz90s,hz10s)]
    
    return st_global, st_self
    


   

def compile_features(speeches, pauses, ranges):
    time_span = speeches[-1].end - speeches[0].start
    # speech_dur = pya.chart()[0][1]
    speech_dur = sum([s.duration for s in speeches])
    avg_speech_dur = np.mean([s.duration for s in speeches])
    avg_pause_dur = np.mean(pauses)
    n_pause_per_minute = len(pauses)/(time_span/60)
    percent_speech = speech_dur/time_span*100
    
    #TODO which one:
    avg_pitch_range_global = np.mean(ranges[0])
    avg_pitch_range_perseg = np.mean(ranges[1])


    return avg_speech_dur, avg_pause_dur, n_pause_per_minute,\
     percent_speech, avg_pitch_range_global, avg_pitch_range_perseg
     



def featurise_one(wav_file,segment_file,pitch_extracter):
    print(f'Finding features for sample {fn(wav_file)}')
    tmp_wav = wav16mono(wav_file)
    pitch_track = get_reaper(tmp_wav, pitch_extracter)
    
    segments = read_segments(segment_file)
    speeches, pauses = compile_segments(segments)
    ranges_a, ranges_b = pitch_data(pitch_track, speeches)
    sd, pd, np, cs, r_a,r_b = compile_features(speeches, pauses, (ranges_a,ranges_b))

    return [sd, pd, np, cs, r_a,r_b]
    
 
      
