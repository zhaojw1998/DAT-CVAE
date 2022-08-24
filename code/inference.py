import numpy as np
import os
import pretty_midi as pyd
import torch
import sys
from model import VAE
from util_tools.format_converter import melody_data2matrix, melody_matrix2data, chord_data2matrix, chord_matrix2data
from torch.distributions import kl_divergence, Normal
from nottingham_dataset import Nottingham
import copy

def chord_grid2data(est_pitch, bpm=60., start=0., max_simu_note=6, pitch_eos=129, num_step=32, min_pitch=0):
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
                pitch = note + 12*4
                duration = 1
                for j in range(t+1, num_step):
                    if harmonic_rhythm[j] == 1:
                        break
                    duration +=1
                pr[t, pitch] = min(duration, 32 - t)
                notes.append(
                    pyd.Note(100, int(pitch), start + t * alpha,
                                    start + (t + duration) * alpha))
        chord = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
        chord.notes = notes
        return chord

def melody_matrix2data(melody_matrix, tempo=120, start_time=0.0, get_list=False):
    HOLD_PITCH = 12
    REST_PITCH = 13
    #melodyMatrix = melody_matrix[:, :ROLL_SIZE]
    chroma = np.concatenate((melody_matrix[:, :12], melody_matrix[:, 15: 17]), axis=-1)
    register = melody_matrix[:, -10:]
    #print(chroma.shape)
    melodySequence = np.argmax(chroma, axis=-1)
    #print(melodySequence)
    
    melody_notes = []
    minStep = 60 / tempo / 4
    onset_or_rest = [i for i in range(len(melodySequence)) if not melodySequence[i]==HOLD_PITCH]
    onset_or_rest.append(len(melodySequence))

    for idx, onset in enumerate(onset_or_rest[:-1]):
        if melodySequence[onset] == REST_PITCH:
            continue
        else:
            pitch = melodySequence[onset] + 12 * np.argmax(register[onset])
            #print(pitch)
            start = onset * minStep
            end = onset_or_rest[idx+1] * minStep
            noteRecon = pyd.Note(velocity=100, pitch=pitch, start=start_time+start, end=start_time+end)
            melody_notes.append(noteRecon)
    if get_list:
        return melody_notes
    else:  
        melody = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
        melody.notes = melody_notes
        return melody


def get_gt(chord, melody):
    #chord: (num_step, max_simu_note, 1), numpy
    #melody: (num_step, 28), numpy
    chord_recon = chord_grid2data(chord, 30, pitch_eos=13)
    melody_recon = melody_matrix2data(melody, 120)
    music = pyd.PrettyMIDI(initial_tempo=120)
    music.instruments.append(melody_recon)
    music.instruments.append(chord_recon)
    return music

def shift(original_melody, p_shift):
    melody = copy.deepcopy(original_melody).cpu().detach().numpy()[0]
    onsets, pitch = np.nonzero(melody[:, :12])
    onsets, register = np.nonzero(melody[:, -10:])
    onset130 = pitch + register*12
    onset130 += p_shift
    onset12 = onset130 % 12
    register = onset130 // 12
    melody[onsets, :] = 0
    melody[onsets, onset12] = 1.
    melody[onsets, register+17] = 1.
    return torch.from_numpy(melody).float().cuda().unsqueeze(0)


def reconstruct(chord, melody):
    #chord: (1, num_step, max_simu_note, 1), torch.LongTensor, cuda()
    #melody: (1, num_step*4, 28), torch.FloatTensor, cuda()
    lengths = model.get_len_index_tensor(chord)  # lengths: (B, num_step)
    chord = model.index_tensor_to_multihot_tensor(chord)
    chord = model.enc_note_embedding(chord)    #(B, num_step, max_simu_note, note_emb_size)
    mel_ebd = model.enc_note_embedding(melody)    #(B, num_step*4, note_emb_size)
    melody_beat_summary = mel_ebd[:, ::4, :] + mel_ebd[:, 1::4, :] + mel_ebd[:, 2::4, :] + mel_ebd[:, 3::4, :]
    dist, mu, = model.encoder(chord, lengths, melody_beat_summary)
    z = dist.mean
    pitch_outs = model.decoder(z, melody_beat_summary, 
                                    inference=True, x=None, lengths=None, 
                                    teacher_forcing_ratio1=0., teacher_forcing_ratio2=0.)
    pitch_outs = pitch_outs.max(-1, keepdim=True)[1]
    pitch_outs = pitch_outs.cpu().detach().numpy()
    chord_track = chord_grid2data(pitch_outs[0], bpm=120//4, start=0, pitch_eos=13)
    
    melody = melody.cpu().detach().numpy()[0]
    melody_track = melody_matrix2data(melody, tempo=120)

    music = pyd.PrettyMIDI()
    music.instruments.append(melody_track)
    music.instruments.append(chord_track)
    return music

def melody_control(chord, melody, new_melody):
    #chord: (B, num_step, max_simu_note, 1), torch.LongTensor, cuda()
    #melody: (B, num_step*4, 28), torch.FloatTensor, cuda()
    #new_melody: (B, num_step*4, 28), torch.FloatTensor, cuda()

    lengths = model.get_len_index_tensor(chord)  # lengths: (B, num_step)
    chord = model.index_tensor_to_multihot_tensor(chord)
    chord = model.enc_note_embedding(chord)    #(B, num_step, max_simu_note, note_emb_size)
    mel_ebd = model.enc_note_embedding(melody)    #(B, num_step*4, note_emb_size)
    melody_beat_summary = mel_ebd[:, ::4, :] + mel_ebd[:, 1::4, :] + mel_ebd[:, 2::4, :] + mel_ebd[:, 3::4, :]
    new_mel_ebd = model.enc_note_embedding(new_melody)    #(B, num_step*4, note_emb_size)
    new_melody_beat_summary = new_mel_ebd[:, ::4, :] + new_mel_ebd[:, 1::4, :] + new_mel_ebd[:, 2::4, :] + new_mel_ebd[:, 3::4, :]
    dist, mu, = model.encoder(chord, lengths, melody_beat_summary)
    z = dist.mean
    pitch_outs = model.decoder(z, new_melody_beat_summary, 
                                    inference=True, x=None, lengths=None, 
                                    teacher_forcing_ratio1=0., teacher_forcing_ratio2=0.)
    pitch_outs = pitch_outs.max(-1, keepdim=True)[1]
    pitch_outs = pitch_outs.cpu().detach().numpy()
    chord_track = chord_grid2data(pitch_outs[0], bpm=120//4, start=0, pitch_eos=13)
    
    new_melody = new_melody.cpu().detach().numpy()[0]
    melody_track = melody_matrix2data(new_melody, tempo=120)

    music = pyd.PrettyMIDI()
    music.instruments.append(melody_track)
    music.instruments.append(chord_track)
    return music

def melody_prior_control(new_melody):
    #new_melody: (B, num_step*4, 28), torch.FloatTensor, cuda()
    new_mel_ebd = model.enc_note_embedding(new_melody)    #(B, num_step*4, note_emb_size)
    new_melody_beat_summary = new_mel_ebd[:, ::4, :] + new_mel_ebd[:, 1::4, :] + new_mel_ebd[:, 2::4, :] + new_mel_ebd[:, 3::4, :]
    z = Normal(torch.zeros(128), torch.ones(128)).rsample().unsqueeze(0).cuda()
    pitch_outs = model.decoder(z, new_melody_beat_summary, 
                                    inference=True, x=None, lengths=None, 
                                    teacher_forcing_ratio1=0., teacher_forcing_ratio2=0.)
    pitch_outs = pitch_outs.max(-1, keepdim=True)[1]
    pitch_outs = pitch_outs.cpu().detach().numpy()
    chord_track = chord_grid2data(pitch_outs[0], bpm=120//4, start=0, pitch_eos=13)
    
    new_melody = new_melody.cpu().detach().numpy()[0]
    melody_track = melody_matrix2data(new_melody, tempo=120)

    music = pyd.PrettyMIDI()
    music.instruments.append(melody_track)
    music.instruments.append(chord_track)
    return music


import utils
config_fn = './code/model_config.json'
#train_hyperparams = utils.load_params_dict('train_hyperparams', config_fn)
model_params = utils.load_params_dict('model_params', config_fn)
data_repr_params = utils.load_params_dict('data_repr', config_fn)
#project_params = utils.load_params_dict('project', config_fn)
#dataset_path = utils.load_params_dict('dataset_paths', config_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(max_simu_note=data_repr_params['max_simu_note'],
        max_pitch=data_repr_params['max_pitch'],
        min_pitch=data_repr_params['min_pitch'],
        pitch_sos=data_repr_params['pitch_sos'],
        pitch_eos=data_repr_params['pitch_eos'],
        pitch_pad=data_repr_params['pitch_pad'],
        num_step=data_repr_params['num_time_step'],

        note_emb_size=model_params['note_emb_size'],
        enc_notes_hid_size=model_params['enc_notes_hid_size'],
        enc_time_hid_size=model_params['enc_time_hid_size'],
        z_size=model_params['z_size'],
        dec_emb_hid_size=model_params['dec_emb_hid_size'],
        dec_time_hid_size=model_params['dec_time_hid_size'],
        dec_notes_hid_size=model_params['dec_notes_hid_size'],
        discr_nhead = model_params["discr_nhead"],
        discr_hid_size = model_params["discr_hid_size"], 
        discr_dropout = model_params["discr_dropout"], 
        discr_nlayer = model_params["discr_nlayer"],

        device=device
        )


weight_path = './code/ad-ptvae_param.pt'
params = torch.load(weight_path)
if 'model_state_dict' in params:
    params = params['model_state_dict']
model.load_state_dict(params)
model.cuda()
model.eval()



dataset = np.load('C:/Users/zhaoj/OneDrive - National University of Singapore/Computer Music Research/data.npy', allow_pickle=True).T
np.random.seed(0)
np.random.shuffle(dataset)
anchor = int(dataset.shape[0] * 0.95)
val_data = dataset[anchor:, :]
val_set = Nottingham(dataset=val_data.T, 
                        length=128, 
                        step_size=16, 
                        chord_fomat='pr', shift_high=0, shift_low=0)
print(len(val_set))
WRITE_PATH = './code/demo_generate'
if not os.path.exists(WRITE_PATH):
    os.makedirs(WRITE_PATH)


chord_1, _, melody_1, _ = val_set.__getitem__(338)
music = get_gt(chord_1, melody_1)
music.write(os.path.join(WRITE_PATH, 'gt_1.mid'))
chord_1 = torch.from_numpy(chord_1).long().cuda().unsqueeze(0)
melody_1 = torch.from_numpy(melody_1).float().cuda().unsqueeze(0)
music = reconstruct(chord_1, melody_1)
music.write(os.path.join(WRITE_PATH, 'recon_1.mid'))

chord_2, _, melody_2, _ = val_set.__getitem__(2749)
music = get_gt(chord_2, melody_2)
music.write(os.path.join(WRITE_PATH, 'gt_2.mid'))
chord_2 = torch.from_numpy(chord_2).long().cuda().unsqueeze(0)
melody_2 = torch.from_numpy(melody_2).float().cuda().unsqueeze(0)
music = reconstruct(chord_2, melody_2)
music.write(os.path.join(WRITE_PATH, 'recon_2.mid'))

chord_3, _, melody_3, _ = val_set.__getitem__(3413)
music = get_gt(chord_3, melody_3)
music.write(os.path.join(WRITE_PATH, 'gt_3.mid'))
chord_3 = torch.from_numpy(chord_3).long().cuda().unsqueeze(0)
melody_3 = torch.from_numpy(melody_3).float().cuda().unsqueeze(0)
music = reconstruct(chord_3, melody_3)
music.write(os.path.join(WRITE_PATH, 'recon_3.mid'))

chord_4, _, melody_4, _ = val_set.__getitem__(5126)
music = get_gt(chord_4, melody_4)
music.write(os.path.join(WRITE_PATH, 'gt_4.mid'))
chord_4 = torch.from_numpy(chord_4).long().cuda().unsqueeze(0)
melody_4 = torch.from_numpy(melody_4).float().cuda().unsqueeze(0)
music = reconstruct(chord_4, melody_4)
music.write(os.path.join(WRITE_PATH, 'recon_4.mid'))


midi = pyd.PrettyMIDI('./code/modal change/recon_1.mid')
melody = melody_data2matrix(midi.instruments[0], midi.get_downbeats())
melody = val_set.truncate_melody(melody)
melody_1_modal_change = torch.from_numpy(melody).float().cuda().unsqueeze(0)

midi = pyd.PrettyMIDI('./code/modal change/recon_2.mid')
melody = melody_data2matrix(midi.instruments[0], midi.get_downbeats())
melody = val_set.truncate_melody(melody)
melody_2_modal_change = torch.from_numpy(melody).float().cuda().unsqueeze(0)

midi = pyd.PrettyMIDI('./code/modal change/recon_3.mid')
melody = melody_data2matrix(midi.instruments[0], midi.get_downbeats())
melody = val_set.truncate_melody(melody)
melody_3_modal_change = torch.from_numpy(melody).float().cuda().unsqueeze(0)

midi = pyd.PrettyMIDI('./code/modal change/recon_4.mid')
melody = melody_data2matrix(midi.instruments[0], midi.get_downbeats())
melody = val_set.truncate_melody(melody)
melody_4_modal_change = torch.from_numpy(melody).float().cuda().unsqueeze(0)




music = melody_control(chord_1, melody_1, shift(melody_1, 6))
music.write(os.path.join(WRITE_PATH, 'control_1_transpose.mid'))

music = melody_control(chord_2, melody_2, shift(melody_2, 6))
music.write(os.path.join(WRITE_PATH, 'control_2_transpose.mid'))

music = melody_control(chord_3, melody_3, shift(melody_3, 6))
music.write(os.path.join(WRITE_PATH, 'control_3_transpose.mid'))

music = melody_control(chord_4, melody_4, shift(melody_4, 6))
music.write(os.path.join(WRITE_PATH, 'control_4_transpose1.mid'))



music = melody_control(chord_1, melody_1, melody_1_modal_change)
music.write(os.path.join(WRITE_PATH, 'control_1_modal_change.mid'))

music = melody_control(chord_2, melody_2, melody_2_modal_change)
music.write(os.path.join(WRITE_PATH, 'control_2_modal_change.mid'))

music = melody_control(chord_3, melody_3, melody_3_modal_change)
music.write(os.path.join(WRITE_PATH, 'control_3_modal_change.mid'))

music = melody_control(chord_4, melody_4, melody_4_modal_change)
music.write(os.path.join(WRITE_PATH, 'control_4_modal_change.mid'))




music = melody_prior_control(melody_1)
music.write(os.path.join(WRITE_PATH, 'control_1_prior.mid'))

music = melody_prior_control(melody_2)
music.write(os.path.join(WRITE_PATH, 'control_2_prior.mid'))

music = melody_prior_control(melody_3)
music.write(os.path.join(WRITE_PATH, 'control_3_prior.mid'))

music = melody_prior_control(melody_4)
music.write(os.path.join(WRITE_PATH, 'control_4_prior.mid'))




music = melody_control(chord_1, melody_1, melody_2)
music.write(os.path.join(WRITE_PATH, 'control_1c+2m.mid'))

music = melody_control(chord_2, melody_2, melody_1)
music.write(os.path.join(WRITE_PATH, 'control_2c+1m.mid'))

music = melody_control(chord_1, melody_1, melody_3)
music.write(os.path.join(WRITE_PATH, 'control_1c+3m.mid'))

music = melody_control(chord_3, melody_3, melody_1)
music.write(os.path.join(WRITE_PATH, 'control_3c+1m.mid'))

music = melody_control(chord_1, melody_1, melody_4)
music.write(os.path.join(WRITE_PATH, 'control_1c+4m.mid'))

music = melody_control(chord_4, melody_4, melody_1)
music.write(os.path.join(WRITE_PATH, 'control_4c+1m.mid'))

music = melody_control(chord_2, melody_2, melody_3)
music.write(os.path.join(WRITE_PATH, 'control_2c+3m.mid'))

music = melody_control(chord_3, melody_3, melody_2)
music.write(os.path.join(WRITE_PATH, 'control_3c+2m.mid'))

music = melody_control(chord_2, melody_2, melody_4)
music.write(os.path.join(WRITE_PATH, 'control_2c+4m.mid'))

music = melody_control(chord_4, melody_4, melody_2)
music.write(os.path.join(WRITE_PATH, 'control_4c+2m.mid'))

music = melody_control(chord_3, melody_3, melody_4)
music.write(os.path.join(WRITE_PATH, 'control_3c+4m.mid'))

music = melody_control(chord_4, melody_4, melody_3)
music.write(os.path.join(WRITE_PATH, 'control_4c+3m.mid'))

