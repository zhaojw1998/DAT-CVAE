import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pretty_midi as pyd
import sys
sys.path.append('./harmonization_model_ver8')
from util_tools.format_converter import melody_data2matrix, melody_matrix2data, chord_data2matrix, chord_matrix2data
import os
import random
import copy



class Nottingham(Dataset):
    def __init__(self, dataset, length, step_size, chord_fomat='chroma', shift_high=6, shift_low=-5, mask_ratio=0.2):
        super(Nottingham, self).__init__()
        self.dataset = dataset
        #print(self.dataset.shape)
        self.__length = length  # 128 for melody and 32 for chord
        self.__chunk_melodies = []
        self.__chunk_chords = []
        self.__step_size = step_size  # 16
        self.chord_fomat = chord_fomat
        self.shift_high = shift_high
        self.shift_low = shift_low
        self.chunking()
        self.mask_ratio = mask_ratio

    def __clipping(self, melody, chord):
        """
        input is (N, 130) and (N, 12)
        outputs a list of (32, 130) matrices, and a list of (12, 130) matrices.
        """
        len_m = melody.shape[0]
        len_c = chord.shape[0]
        clipped_melodies, clipped_chords = [], []
        if len_m > len_c:
            chord = np.pad(chord, ((0, len_m - len_c), (0, 0)), 'constant')
        elif len_c > len_m:
            melody = np.pad(melody, ((0, len_c - len_m), (0, 0)), 'constant')
            melody[-(len_c - len_m):, 129] = 1
        len_m = melody.shape[0]
        len_c = chord.shape[0]
        assert (len_m == len_c)
        for i in range(0, len_m, self.__step_size):
            if ((i + self.__length) < len_c):
                melody_clip = melody[i:i + self.__length]
                chord_clip = chord[i:i + self.__length][::4]
                if self.chord_fomat == 'pr':
                    chord_matrix = np.zeros(chord_clip.shape)
                    onsets = np.abs(chord_clip[1:, :] - chord_clip[:-1, :]).sum(-1)
                    onsets = np.concatenate((np.array([1]), onsets))
                    start = 0
                    for j in range(1, len(onsets)):
                        if onsets[j] > 0:
                            chord_matrix[start, np.where(chord_clip[start]==1)] = j - start
                            start = j
                    chord_matrix[start, np.where(chord_clip[start]==1)] = len(onsets) - start
                    clipped_melodies.append(melody_clip)
                    clipped_chords.append(chord_matrix)
                elif self.chord_fomat == 'chroma':
                    clipped_melodies.append(melody_clip)
                    clipped_chords.append(chord_clip)
        return clipped_melodies, clipped_chords

    def chunking(self):
        for melody, chord in tqdm(zip(self.dataset[0], self.dataset[1])):
            # melody.shape = (N, 130), chord.shape = (N, 12)
            m, c = self.__clipping(melody, chord)
            self.__chunk_melodies += m
            self.__chunk_chords += c
        assert (len(self.__chunk_melodies) == len(self.__chunk_chords))

    def mask_melody(self, melody):
        # melody: (128, 28)
        melody = copy.deepcopy(melody)
        pitch_mask = 17
        onsets, pitch_idx = np.nonzero(melody[:, :12]) #[[T, T, T, T], [idx, idx, idx, idx]]
        pitch_alter = []
        if len(onsets) == 0:
            #print(onsets, flush=True)
            #print(pitch_idx, flush=True)
            return melody
        selection = np.random.choice(a=range(len(onsets)), size=min(int(self.mask_ratio*len(onsets))+1, len(onsets)), replace=False)
        #print(mask_onsets)
        mask_onsets = onsets[selection]
        shift = random.randint(-5, 6)
        """for idx, onset in enumerate(mask_onsets):
            prob = random.random()
            if prob < 1:
                pitch_alter.append(pitch_mask)
            elif prob >= 0.8 and prob < 0.9:
                pitch_alter.append(random.randint(0, 11))
            else:
                pitch_alter.append(pitch_idx[idx])"""
        #print(mask_onsets, np.argmax(melody[mask_onsets, :16], axis=-1))
        #print(mask_onsets, pitch_alter)
        """melody[mask_onsets, :12] = 0.
        melody[mask_onsets, pitch_alter] = 1."""
        melody[mask_onsets, :12] = np.roll(melody[mask_onsets, :12], shift, axis=-1)
        melody[mask_onsets, pitch_mask] = 1.
        #print(mask_onsets, pitch_alter)
        #print(mask_onsets, np.argmax(melody[mask_onsets, :18], axis=-1))
        return melody

    def truncate_melody(self, melody):
        #melody: (128, 130)
        onsets, pitch = np.nonzero(melody[:, :128])
        chroma_idx = pitch % 12
        register_idx = pitch // 12
        part_1 = np.zeros((melody.shape[0], 15))
        part_4 = np.zeros((melody.shape[0], 10))
        part_1[onsets, chroma_idx] = 1.
        part_4[onsets, register_idx] = 1.
        return np.concatenate((part_1, melody[:, -2:], np.zeros((melody.shape[0], 1)), part_4), axis=-1) 

    
    def __len__(self):
        # consider data augmentation here
        return len(self.__chunk_chords) * (self.shift_high - self.shift_low + 1)
    
    def __getitem__(self, idx):
        no = idx // (self.shift_high - self.shift_low + 1)
        shift = idx % (self.shift_high - self.shift_low + 1) + self.shift_low

        melody = self.__chunk_melodies[no]  #128, 130
        melody = np.concatenate((np.roll(melody[:, :128], shift, -1), melody[:, -2:]), axis=-1)
        melody = self.truncate_melody(melody)    #128, 28
        masked_melody = self.mask_melody(melody)

        chord = self.__chunk_chords[no] 
        chord = np.roll(chord, shift, -1)
        #chord = np.pad(chord, ((0, 0), (4*12, 128-5*12)), 'constant')
        chord = target_to_3dtarget(pr_mat = chord,
                                    max_note_count=6,
                                    max_pitch=11,
                                    min_pitch=0, 
                                    pitch_pad_ind=14,
                                    pitch_sos_ind=12,
                                    pitch_eos_ind=13)
        return chord, melody, masked_melody

    
def target_to_3dtarget(pr_mat, max_note_count=11, max_pitch=107, min_pitch=22,
                       pitch_pad_ind=88, dur_pad_ind=2,
                       pitch_sos_ind=86, pitch_eos_ind=87):
    """
    :param pr_mat: (32, 12) matrix. pr_mat[t, p] indicates a note of pitch p,
    started at time step t, has a duration of pr_mat[t, p] time steps.
    :param max_note_count: the maximum number of notes in a time step,
    including <sos> and <eos> tokens.
    :param max_pitch: the highest pitch in the dataset.
    :param min_pitch: the lowest pitch in the dataset.
    :param pitch_pad_ind: see return value.
    :param dur_pad_ind: see return value.
    :param pitch_sos_ind: sos token.
    :param pitch_eos_ind: eos token.
    :return: pr_mat3d is a (32, max_note_count, 6) matrix. In the last dim,
    the 0th column is for pitch, 1: 6 is for duration in binary repr. Output is
    padded with <sos> and <eos> tokens in the pitch column, but with pad token
    for dur columns.
    """
    pitch_range = max_pitch - min_pitch + 1  # including pad
    pr_mat3d = np.ones((32, max_note_count, 1), dtype=int)
    
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(32, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        if cur_idx[t] == max_note_count-1:
            continue
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        #binary = np.binary_repr(int(pr_mat[t, p]) - 1, width=5)
        #temp = np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        #pr_mat3d[t, cur_idx[t], 1: 6] = temp
        cur_idx[t] += 1
    pr_mat3d[np.arange(0, 32), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d

def grid_to_pr_and_notes(est_pitch, bpm=60., start=0., max_simu_note=6, pitch_eos=129, num_step=32, min_pitch=0):
        est_pitch = est_pitch[:, :, 0]#.cpu().detach().numpy() #(32, max_simu_note-1), NO BATCH HERE
        if est_pitch.shape[1] == max_simu_note:
            est_pitch = est_pitch[:, 1:]
        
        #print(est_pitch.shape)
        #print(est_pitch)
        harmonic_rhythm = 1. - (est_pitch[:, 0]==pitch_eos) * 1.
        #print(harmonic_rhythm)

        pr = np.zeros((32, 128), dtype=int)
        alpha = 0.25 * 60 / bpm
        notes = []
        for t in range(num_step):
            for n in range(max_simu_note-1):
                note = est_pitch[t, n]
                if note == pitch_eos:
                    break
                pitch = note + min_pitch
                duration = 1
                for j in range(t+1, num_step):
                    if harmonic_rhythm[j] == 1:
                        break
                    duration +=1
                pr[t, pitch] = min(duration, 32 - t)
                notes.append(
                    pyd.Note(100, int(pitch), start + t * alpha,
                                    start + (t + duration) * alpha))
        return pr, notes



if __name__ == "__main__":
    dataset = np.load('C:/Users/zhaoj/OneDrive - National University of Singapore/Computer Music Research/data.npy', allow_pickle=True)
    dataset = Nottingham(dataset, 128, 16, 'pr', 0, 0, mask_ratio=1.)
    #print(len(dataset))
    for i in range(5):
        idx = np.random.randint(len(dataset))
        chord, melody, masked_melody = dataset.__getitem__(idx)
        #print(chord.shape, melody.shape, masked_melody.shape)

        #print(melody[:, :12], masked_melody[idx])
        #print(np.nonzero(melody[:, :17])[-1], np.nonzero(masked_melody[:, :18])[-1])
        #print(np.nonzero(melody[:, :12])[0] == np.nonzero((np.sum(melody, axis=-1) == 2)*1.)[0])
        print(np.sum(melody, axis=-1), np.sum(masked_melody, axis=-1))
        continue